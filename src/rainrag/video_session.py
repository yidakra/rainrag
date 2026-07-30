"""Single-video upload sessions: transcribe → index → scoped Q&A.

This module powers the "upload your own video" mode. Each uploaded video gets:

* an ephemeral working directory under ``video_upload.tmp_root``,
* an ephemeral Qdrant collection ``{prefix}{session_id}`` that holds ONLY that
  video's chunks (guaranteeing answers are scoped to the upload), and
* a background job that runs transcription (via the livevtt venv subprocess),
  then the normal RainRAG ingest → embed → index pipeline into that collection.

Sessions are ephemeral: a TTL reaper drops the collection and deletes the
working directory once a session is older than ``session_ttl_seconds`` (or when
explicitly deleted). Transcription is serialised across sessions by a global
GPU lock so concurrent uploads queue instead of exhausting GPU memory.
"""

from __future__ import annotations

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

# Serialises GPU-bound transcription across all sessions in this process.
_GPU_LOCK = threading.Lock()


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
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Filesystem paths (not sent to the client).
    session_dir: str = ""
    video_path: str = ""
    vtt_path: str = ""

    def public_dict(self) -> dict:
        """Return a client-facing view (omits local filesystem paths)."""
        d = asdict(self)
        for k in ("session_dir", "video_path", "vtt_path"):
            d.pop(k, None)
        return d


class VideoSessionManager:
    """Owns the lifecycle of all active upload sessions in this process."""

    def __init__(self, config: Config):
        self.config = config
        self.cfg = config.video_upload
        self._sessions: dict[str, VideoSession] = {}
        self._lock = threading.Lock()

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
        """Drop the session's collection + working dir and forget it."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        self._teardown(session)
        logger.info(f"[video-session {session_id}] deleted")
        return True

    def shutdown(self) -> None:
        self._stop.set()

    # --------------------------------------------------------------- internals

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
            self._transcribe(session)
            self._index(session)
            self._update(session_id, status=STATUS_READY, stage="ready", percent=100.0)
            logger.info(f"[video-session {session_id}] ready")
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            logger.exception(f"[video-session {session_id}] failed")
            self._update(session_id, status=STATUS_ERROR, stage="error", error=str(exc))

    def _transcribe(self, session: VideoSession) -> None:
        """Run the single-file transcriber subprocess under the livevtt venv."""
        video_path = Path(session.video_path)
        vtt_path = Path(session.session_dir) / "source.ru.vtt"
        progress_file = Path(session.session_dir) / "progress.json"

        script = Path(self.cfg.transcribe_script)
        if not script.is_absolute():
            script = Path.cwd() / script

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
            str(self.cfg.device_index),
            "--language",
            self.cfg.language,
            "--beam-size",
            str(self.cfg.beam_size),
            "--progress-file",
            str(progress_file),
        ]

        self._update(session.id, status=STATUS_TRANSCRIBING, stage="waiting_for_gpu", percent=0.0)
        # Safety cap so a hung transcription can't hold the GPU lock forever.
        timeout = max(1800, self.cfg.session_ttl_seconds)

        with _GPU_LOCK:
            self._update(session.id, stage="transcribing", percent=0.0)
            logger.info(f"[video-session {session.id}] transcribing: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            start = time.time()
            try:
                while proc.poll() is None:
                    if time.time() - start > timeout:
                        proc.kill()
                        raise RuntimeError("transcription timed out")
                    self._poll_progress(session.id, progress_file)
                    time.sleep(1.0)
            finally:
                out = proc.stdout.read() if proc.stdout else ""
            if proc.returncode != 0:
                self._poll_progress(session.id, progress_file)  # capture any error stage
                tail = "\n".join((out or "").splitlines()[-15:])
                raise RuntimeError(f"transcriber exited {proc.returncode}: {tail}")

        if not vtt_path.exists():
            raise RuntimeError("transcription produced no VTT output")
        self._update(session.id, vtt_path=str(vtt_path))

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
