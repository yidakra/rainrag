"""Single-video upload sessions: transcribe → index → scoped Q&A.

This module powers the "upload your own video" mode. Each uploaded video gets:

* an ephemeral working directory under ``video_upload.tmp_root``,
* an ephemeral Qdrant collection ``{prefix}{session_id}`` that holds ONLY that
  video's chunks (guaranteeing answers are scoped to the upload), and
* a background job that runs transcription (via the livevtt venv subprocess),
  then the normal RainRAG ingest → embed → index pipeline into that collection.

Sessions are ephemeral: a TTL reaper drops the collection and deletes the
working directory once a session is older than ``session_ttl_seconds`` (or when
explicitly deleted). Transcription runs one job per visible GPU — the device
count is detected at startup — so uploads beyond that queue rather than
oversubscribing GPU memory.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

from rainrag.config import Config
from rainrag.embed import Embedder
from rainrag.index import QdrantIndexer
from rainrag.ingest import Ingester


# Status values (kept as plain strings so they serialise directly to JSON).
STATUS_QUEUED = "queued"
STATUS_TRANSCRIBING = "transcribing"
STATUS_INDEXING = "indexing"
STATUS_READY = "ready"
STATUS_ERROR = "error"


class _GpuPool:
    """Hands out GPU slots to transcription jobs, one job per device.

    Replaces a single global lock: with two GPUs visible, two uploads transcribe
    at once instead of one waiting out the other. Jobs beyond the device count
    still queue, so GPU memory is never oversubscribed.
    """

    def __init__(self, device_indices: list[int]):
        self.devices = list(device_indices)
        self._available = list(device_indices)
        self._condition = threading.Condition()

    @property
    def size(self) -> int:
        return len(self.devices)

    def acquire(self, timeout: float) -> int | None:
        """Take a free device, or return None if none freed up within timeout."""
        with self._condition:
            if not self._available:
                self._condition.wait(timeout)
            if not self._available:
                return None
            return self._available.pop(0)

    def release(self, device_index: int) -> None:
        with self._condition:
            self._available.append(device_index)
            self._condition.notify()


# Language codes the ingest pipeline indexes natively. Anything else Whisper
# detects still transcribes fine (embeddings are multilingual) but is indexed
# under the default language.
_INDEXABLE_LANGUAGES = ("ru", "en")


class SessionCancelledError(RuntimeError):
    """Raised inside a session's worker thread when the session goes away.

    Either the user cancelled it or the TTL reaper collected it; in both cases
    the session no longer exists, so there is nobody to report an error to.
    """


@dataclass
class VideoSession:
    """State for one uploaded-video session (safe to serialise to the client)."""

    id: str
    filename: str
    status: str = STATUS_QUEUED
    stage: str = "queued"
    percent: float = 0.0
    collection_name: str = ""
    num_documents: int = 0
    duration_seconds: float | None = None
    error: str | None = None
    # Language Whisper detected (or the configured override) and its confidence.
    language: str | None = None
    language_probability: float | None = None
    # Sessions ahead of this one in the GPU queue; 0 means it is transcribing.
    queue_position: int = 0
    # Pixel dimensions of the upload, so the client can size its player to the
    # real aspect ratio instead of assuming 16:9. None when ffprobe is absent.
    width: int | None = None
    height: int | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Internal state (not sent to the client).
    cancelled: bool = False
    session_dir: str = ""
    video_path: str = ""
    vtt_path: str = ""

    def public_dict(self) -> dict:
        """Return a client-facing view (omits local filesystem paths)."""
        d = asdict(self)
        for k in ("session_dir", "video_path", "vtt_path", "cancelled"):
            d.pop(k, None)
        return d


class VideoSessionManager:
    """Owns the lifecycle of all active upload sessions in this process."""

    def __init__(self, config: Config):
        self.config = config
        self.cfg = config.video_upload
        self._sessions: dict[str, VideoSession] = {}
        self._lock = threading.Lock()
        # Sessions waiting on (or holding) a GPU, oldest first, so each one can
        # report how many uploads are ahead of it instead of a blank progress bar.
        self._gpu_queue: list[str] = []
        # Live transcriber subprocesses, so cancelling frees the GPU immediately.
        self._procs: dict[str, subprocess.Popen] = {}
        # One transcription per GPU, across however many the box actually has.
        self._gpu_pool = _GpuPool(self._resolve_devices())

        # Dedicated embedder/indexer for session indexing. The embedder loads
        # its own model lazily on first use; the indexer needs an explicit
        # connection to Qdrant before any collection operation.
        self._embedder = Embedder(config)
        self._indexer = QdrantIndexer(config)
        self._indexer.connect()

        # An ingester configured with web metadata DISABLED — uploaded videos
        # have no library hash, so web-metadata lookups must not run (and must
        # not skip the video for "missing" metadata).
        ingest_config = config.model_copy(deep=True)
        ingest_config.web_metadata.enabled = False
        self._ingester = Ingester(ingest_config)

        Path(self.cfg.tmp_root).mkdir(parents=True, exist_ok=True)

        # Sessions live in this process's memory, so anything left on disk or in
        # Qdrant from a previous run is unreachable and would leak forever.
        if self.cfg.sweep_orphans_on_start:
            self._sweep_orphans()

        # Background TTL reaper.
        self._stop = threading.Event()
        self._reaper = threading.Thread(target=self._reaper_loop, daemon=True)
        self._reaper.start()

    # ------------------------------------------------------------------ public

    def create_session(self, src_path: Path, original_filename: str) -> VideoSession:
        """Register a session for an already-saved upload and start processing.

        Args:
            src_path: Path to the uploaded file (will be moved into the session dir).
            original_filename: Display name for the upload.
        """
        session_id = uuid.uuid4().hex
        session_dir = Path(self.cfg.tmp_root) / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(original_filename).suffix or src_path.suffix or ".mp4"
        video_path = session_dir / f"source{suffix}"
        shutil.move(str(src_path), str(video_path))

        # Extensionless symlink so VTTParser.find_source_video + ffprobe can read
        # duration/metadata regardless of the original container extension.
        extless = session_dir / "source"
        try:
            if not extless.exists():
                extless.symlink_to(video_path.name)
        except OSError:
            pass  # non-fatal; we just lose duration metadata for odd filesystems

        session = VideoSession(
            id=session_id,
            filename=original_filename,
            collection_name=f"{self.cfg.collection_prefix}{session_id}",
            session_dir=str(session_dir),
            video_path=str(video_path),
        )
        with self._lock:
            self._sessions[session_id] = session

        thread = threading.Thread(target=self._run, args=(session_id,), daemon=True)
        thread.start()
        logger.info(f"[video-session {session_id}] created for {original_filename!r}")
        return session

    def get(self, session_id: str) -> VideoSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[VideoSession]:
        with self._lock:
            return list(self._sessions.values())

    def delete(self, session_id: str) -> bool:
        """Cancel any running work, then drop the collection + working dir.

        Deleting a session that is still transcribing kills the transcriber so
        the GPU is handed to the next queued upload right away rather than after
        the abandoned video finishes.
        """
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                session.cancelled = True
            proc = self._procs.get(session_id)
            if session_id in self._gpu_queue:
                self._gpu_queue.remove(session_id)
        if session is None:
            return False
        if proc is not None and proc.poll() is None:
            logger.info(f"[video-session {session_id}] cancelling transcription")
            with contextlib.suppress(OSError):
                proc.kill()
        self._teardown(session)
        logger.info(f"[video-session {session_id}] deleted")
        return True

    def shutdown(self) -> None:
        self._stop.set()

    # --------------------------------------------------------------- internals

    def _sweep_orphans(self) -> None:
        """Remove session collections and working dirs left over from a prior run.

        Session state is held in process memory, so after a restart the previous
        run's collections and upload directories can never be reached or cleaned
        by the TTL reaper. Only names carrying the configured prefix are touched,
        so the main archive collection is never at risk.
        """
        prefix = self.cfg.collection_prefix
        if not prefix:
            logger.warning("collection_prefix is empty; skipping orphan sweep to stay safe")
            return

        # Stale ephemeral collections.
        try:
            stale = [name for name in self._indexer.list_collections() if name.startswith(prefix)]
            for name in stale:
                # Guard against a prefix that would match the live archive.
                if name == self.config.qdrant.collection_name:
                    continue
                self._indexer.drop_collection(name)
            if stale:
                logger.info(f"Orphan sweep: dropped {len(stale)} stale session collection(s)")
        except Exception as exc:  # noqa: BLE001 - startup must not fail on cleanup
            logger.warning(f"Orphan sweep (collections) failed: {exc}")

        # Stale working directories, including any partial uploads.
        try:
            tmp_root = Path(self.cfg.tmp_root)
            removed = 0
            for entry in tmp_root.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
                elif entry.suffix == ".upload":
                    entry.unlink(missing_ok=True)
                    removed += 1
            if removed:
                logger.info(f"Orphan sweep: removed {removed} stale session file(s)/dir(s)")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Orphan sweep (directories) failed: {exc}")

    def _update(self, session_id: str, **fields) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            for k, v in fields.items():
                setattr(session, k, v)
            session.updated_at = time.time()

    def _run(self, session_id: str) -> None:
        """Full pipeline for one session: transcribe → ingest/embed/index."""
        session = self.get(session_id)
        if session is None:
            return
        try:
            self._probe_dimensions(session)
            self._transcribe(session)
            self._check_cancelled(session_id)
            self._index(session)
            self._update(session_id, status=STATUS_READY, stage="ready", percent=100.0)
            logger.info(f"[video-session {session_id}] ready")
        except SessionCancelledError:
            # The session was deleted; _update is already a no-op for it and the
            # working files were removed by delete()/the reaper.
            logger.info(f"[video-session {session_id}] cancelled")
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            # Cancelling mid-stage makes the work fail in whatever way that stage
            # fails — a killed subprocess, a dropped collection. If the session is
            # already gone it was cancelled, not broken; don't log a scare-traceback.
            if self._is_cancelled(session_id):
                logger.info(f"[video-session {session_id}] cancelled")
                return
            logger.exception(f"[video-session {session_id}] failed")
            self._update(session_id, status=STATUS_ERROR, stage="error", error=str(exc))

    # ------------------------------------------------------------- GPU queueing

    def _resolve_devices(self) -> list[int]:
        """Decide which GPUs transcription may use.

        Defaults to every CUDA device the transcription environment can see, so
        a two-GPU box runs two uploads concurrently without being configured to.
        An explicit ``device_indices`` list overrides detection; if detection is
        unavailable we fall back to the single configured device rather than
        guessing high and failing every job assigned to a device that isn't there.
        """
        if self.cfg.device != "cuda":
            return [self.cfg.device_index]

        detected = self._probe_cuda_device_count()
        configured = list(self.cfg.device_indices)

        if configured:
            if detected is None:
                return configured
            usable = [index for index in configured if index < detected]
            if not usable:
                logger.warning(
                    f"Configured device_indices {configured} exceed the {detected} visible "
                    f"CUDA device(s); falling back to device 0"
                )
                return [0]
            if len(usable) < len(configured):
                logger.warning(
                    f"Only {detected} CUDA device(s) visible; using {usable} of {configured}"
                )
            return usable

        if not detected:
            logger.warning(
                f"Could not detect CUDA devices; transcription will use device "
                f"{self.cfg.device_index} only"
            )
            return [self.cfg.device_index]

        logger.info(f"Video upload: {detected} CUDA device(s) detected for transcription")
        return list(range(detected))

    def _probe_cuda_device_count(self) -> int | None:
        """Ask the transcription environment how many CUDA devices it can see.

        Has to be a subprocess: the RainRAG environment has no ctranslate2, and
        what matters is what the *transcriber* sees (its own CUDA_VISIBLE_DEVICES).
        Returns None when the probe cannot run, which is treated as "unknown".
        """
        try:
            result = subprocess.run(
                [
                    self.cfg.livevtt_python,
                    "-c",
                    "import ctranslate2; print(ctranslate2.get_cuda_device_count())",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(f"CUDA device probe failed: {exc}")
            return None
        if result.returncode != 0:
            logger.warning(f"CUDA device probe exited {result.returncode}: {result.stderr.strip()}")
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            logger.warning(f"CUDA device probe returned {result.stdout.strip()!r}")
            return None

    def _is_cancelled(self, session_id: str) -> bool:
        """Whether the session has been deleted or flagged for cancellation."""
        with self._lock:
            session = self._sessions.get(session_id)
        return session is None or session.cancelled

    def _check_cancelled(self, session_id: str) -> None:
        """Raise if the session was deleted while its worker thread was running."""
        if self._is_cancelled(session_id):
            raise SessionCancelledError(session_id)

    def _queue_position(self, session_id: str) -> int:
        """Number of sessions ahead in the GPU queue (0 = holds it or is next)."""
        with self._lock:
            try:
                return self._gpu_queue.index(session_id)
            except ValueError:
                return 0

    def _probe_dimensions(self, session: VideoSession) -> None:
        """Record the upload's pixel size so the client can size its player.

        Best-effort: a missing or failing ffprobe just leaves the dimensions
        unset, and the client falls back to a 16:9 assumption.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    session.video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug(f"[video-session {session.id}] ffprobe unavailable: {exc}")
            return

        if result.returncode != 0:
            logger.debug(f"[video-session {session.id}] ffprobe failed: {result.stderr.strip()}")
            return

        parts = result.stdout.strip().split("x")
        if len(parts) < 2:
            return
        try:
            width, height = int(parts[0]), int(parts[1])
        except ValueError:
            return
        if width <= 0 or height <= 0:
            return

        self._update(session.id, width=width, height=height)
        logger.info(f"[video-session {session.id}] source is {width}x{height}")

    def _transcribe(self, session: VideoSession) -> None:
        """Run the single-file transcriber subprocess under the livevtt venv."""
        video_path = Path(session.video_path)
        # Written without a language suffix: with auto-detection the language is
        # only known once transcription finishes, and the ingester reads it back
        # off this filename. Renamed to source.<lang>.vtt below.
        vtt_path = Path(session.session_dir) / "source.vtt"
        progress_file = Path(session.session_dir) / "progress.json"

        script = Path(self.cfg.transcribe_script)
        if not script.is_absolute():
            script = Path.cwd() / script

        self._update(session.id, status=STATUS_TRANSCRIBING, stage="waiting_for_gpu", percent=0.0)
        # Safety cap so a hung transcription can't hold its GPU slot forever.
        timeout = max(1800, self.cfg.session_ttl_seconds)

        with self._lock:
            self._gpu_queue.append(session.id)
        device_index: int | None = None
        try:
            # Poll for a slot rather than blocking on one, so a queued session can
            # still report its position and notice cancellation while it waits.
            while device_index is None:
                self._check_cancelled(session.id)
                self._update(session.id, queue_position=self._queue_position(session.id))
                device_index = self._gpu_pool.acquire(timeout=1.0)

            cmd = [
                self.cfg.livevtt_python,
                str(script),
                str(video_path),
                "--output",
                str(vtt_path),
                "--model",
                self.cfg.model,
                "--compute-type",
                self.cfg.compute_type,
                "--device",
                self.cfg.device,
                "--device-index",
                str(device_index),
                "--language",
                self.cfg.language,
                "--detection-segments",
                str(self.cfg.language_detection_segments),
                "--beam-size",
                str(self.cfg.beam_size),
                "--progress-file",
                str(progress_file),
            ]
            if not self.cfg.multilingual:
                cmd.append("--no-multilingual")

            self._update(session.id, stage="transcribing", percent=0.0, queue_position=0)
            logger.info(f"[video-session {session.id}] transcribing on GPU {device_index}")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            with self._lock:
                self._procs[session.id] = proc
            start = time.time()
            try:
                while proc.poll() is None:
                    if time.time() - start > timeout:
                        proc.kill()
                        raise RuntimeError("transcription timed out")
                    try:
                        self._check_cancelled(session.id)
                    except SessionCancelledError:
                        proc.kill()
                        raise
                    self._poll_progress(session.id, progress_file)
                    time.sleep(1.0)
            finally:
                out = proc.stdout.read() if proc.stdout else ""
                with self._lock:
                    self._procs.pop(session.id, None)
            if proc.returncode != 0:
                # delete() kills the transcriber directly, which lands here as a
                # non-zero exit. Check for that before blaming the transcriber.
                self._check_cancelled(session.id)
                self._poll_progress(session.id, progress_file)  # capture any error stage
                tail = "\n".join((out or "").splitlines()[-15:])
                raise RuntimeError(f"transcriber exited {proc.returncode}: {tail}")
        finally:
            with self._lock:
                if session.id in self._gpu_queue:
                    self._gpu_queue.remove(session.id)
            if device_index is not None:
                self._gpu_pool.release(device_index)

        if not vtt_path.exists():
            raise RuntimeError("transcription produced no VTT output")
        self._update(session.id, vtt_path=str(self._apply_language_suffix(vtt_path, progress_file)))

    def _apply_language_suffix(self, vtt_path: Path, progress_file: Path) -> Path:
        """Rename source.vtt to source.<lang>.vtt so ingest picks up the language.

        Whisper can return any language code; the ingester only distinguishes
        ru/en, so anything else keeps the bare name and is indexed as the
        default. Retrieval is unaffected — the embedding model is multilingual.
        """
        language = self._detected_language(progress_file)
        if language not in _INDEXABLE_LANGUAGES:
            return vtt_path
        target = vtt_path.with_name(f"source.{language}.vtt")
        try:
            vtt_path.replace(target)
        except OSError as exc:
            logger.warning(f"Could not apply language suffix to {vtt_path}: {exc}")
            return vtt_path
        return target

    def _detected_language(self, progress_file: Path) -> str | None:
        """Read the language the transcriber reported, if any."""
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        language = data.get("detected_language")
        return str(language).lower() if language else None

    def _poll_progress(self, session_id: str, progress_file: Path) -> None:
        """Read the transcriber's JSON progress snapshot and mirror it to state."""
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        stage = data.get("stage")
        percent = data.get("percent")
        if stage == "error":
            return  # surfaced by the non-zero exit path with fuller context
        fields: dict = {}
        if isinstance(percent, (int, float)):
            fields["percent"] = float(percent)
        if stage in ("loading_model", "transcribing"):
            fields["stage"] = stage
        if data.get("total_seconds"):
            fields["duration_seconds"] = float(data["total_seconds"])
        # Report the detected language as soon as it is known so the UI can show
        # it while transcription is still running.
        if data.get("detected_language"):
            fields["language"] = str(data["detected_language"]).lower()
        probability = data.get("language_probability")
        if isinstance(probability, (int, float)):
            fields["language_probability"] = float(probability)
        if fields:
            self._update(session_id, **fields)

    def _index(self, session: VideoSession) -> None:
        """Ingest the VTT → embed → index into the session's ephemeral collection."""
        self._update(session.id, status=STATUS_INDEXING, stage="indexing", percent=100.0)

        documents = self._ingester.process_file(Path(session.vtt_path))
        if not documents:
            # Valid VTT but no speech cues (silent video) → nothing to answer over.
            raise RuntimeError("no speech detected in the video")

        embeddings = self._embedder.generate_embeddings(documents, show_progress=False)
        self._indexer.ensure_collection(session.collection_name)
        self._indexer.index_documents(
            embeddings, documents, collection_name=session.collection_name
        )

        duration = documents[0].duration_seconds or session.duration_seconds
        self._update(session.id, num_documents=len(documents), duration_seconds=duration)
        logger.info(
            f"[video-session {session.id}] indexed {len(documents)} chunks "
            f"into {session.collection_name}"
        )

    def _teardown(self, session: VideoSession) -> None:
        """Drop the ephemeral collection and remove the working directory."""
        try:
            if session.collection_name:
                self._indexer.drop_collection(session.collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[video-session {session.id}] collection drop failed: {exc}")
        try:
            if session.session_dir:
                shutil.rmtree(session.session_dir, ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[video-session {session.id}] dir cleanup failed: {exc}")

    def _reaper_loop(self) -> None:
        ttl = self.cfg.session_ttl_seconds
        while not self._stop.wait(timeout=60.0):
            now = time.time()
            with self._lock:
                expired = [s for s in self._sessions.values() if now - s.created_at > ttl]
                for s in expired:
                    self._sessions.pop(s.id, None)
            for s in expired:
                logger.info(f"[video-session {s.id}] expired (TTL); cleaning up")
                self._teardown(s)
