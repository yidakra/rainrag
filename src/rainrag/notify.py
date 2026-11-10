"""Event-driven indexing of one video: the queue and worker behind ``POST /ingest/notify``.

The transcoder sends a video hash when a video is ready. The worker then
indexes that one video within minutes, instead of waiting for the six-hour
incremental sweep (``scripts/run_incremental_update.sh``), which stays the
safety net.

Per-hash states, persisted in ``notify.queue_path`` so a restart loses nothing:

- ``queued``: accepted, not processed yet (also the state during a retry).
- ``waiting_for_vtt``: the archive folder has no VTT file for the hash yet.
  rainrag does not transcribe; dir2mcp writes the VTT files later.
- ``files_pending``: the chunks are in Qdrant, so search already finds the
  video, but ``docs.jsonl`` and the manifest are not written yet because the
  sweep holds its lock. The sweep deletes every point that is not in
  ``docs.jsonl``, so the worker writes the files as soon as the lock frees and
  then upserts once more to repair any delete that happened in between.
- ``indexed``: terminal success.
- ``failed``: terminal, ``notify.max_wait_hours`` after the notification.

Document ids derive from the absolute VTT path, so the worker sees the archive
at the same mount path as the sweep and a later sweep overwrites, never
duplicates. The worker leaves the embedding cache alone: the next sweep sees
the new content hashes as misses and embeds them again, which is cheap.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from rainrag.config import Config
from rainrag.ingest import Document, FileManifest, WebMetadataLoader


if TYPE_CHECKING:
    import numpy as np

    from rainrag.embed import Embedder


STATE_QUEUED = "queued"
STATE_WAITING_FOR_VTT = "waiting_for_vtt"
STATE_FILES_PENDING = "files_pending"
STATE_INDEXED = "indexed"
STATE_FAILED = "failed"
TERMINAL_STATES = frozenset({STATE_INDEXED, STATE_FAILED})

# Backoff after an error (Qdrant down, a VTT that parses to nothing) is capped here.
MAX_BACKOFF_SECONDS = 900.0


def _iso(ts: float) -> str:
    """Format an epoch timestamp as ISO 8601 UTC, for the queue file and the API."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


class NotifyQueue:
    """Persistent, thread-safe per-hash state, written via tmp+rename."""

    def __init__(self, path: str | Path) -> None:
        """Load the queue file at ``path`` (an unreadable file starts empty, with a warning)."""
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        """Read the queue file; a missing or broken file is an empty queue."""
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"notify queue at {self.path} is unreadable, starting empty: {exc}")
            return {}
        if not isinstance(raw, dict):
            logger.warning(f"notify queue at {self.path} is not an object, starting empty")
            return {}
        return {k: v for k, v in raw.items() if isinstance(v, dict)}

    def _save_locked(self) -> None:
        """Write the queue atomically; the caller holds ``self._lock``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self._entries, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def enqueue(self, video_hash: str, path: str | None, now: float) -> dict[str, Any]:
        """Record a notification; a hash that is already pending keeps its one entry."""
        with self._lock:
            entry = self._entries.get(video_hash)
            if entry is not None and entry.get("state") not in TERMINAL_STATES:
                if path and not entry.get("path"):
                    entry["path"] = path
                    self._save_locked()
                return dict(entry)
            # New, or a re-notification of a finished hash (a re-transcoded video):
            # start over. Every step is idempotent, so this is safe.
            entry = {
                "video_hash": video_hash,
                "path": path,
                "state": STATE_QUEUED,
                "notified_at": _iso(now),
                "notified_ts": now,
                "updated_at": _iso(now),
                "next_attempt_ts": now,
                "attempts": 0,
                "error": None,
                "vtt_files": [],
                "doc_count": 0,
                "searchable_at": None,
                "indexed_at": None,
                "failed_at": None,
            }
            self._entries[video_hash] = entry
            self._save_locked()
            logger.info(f"[notify {video_hash}] queued")
            return dict(entry)

    def get(self, video_hash: str) -> dict[str, Any] | None:
        """A copy of the entry for ``video_hash``, or None when the hash is unknown."""
        with self._lock:
            entry = self._entries.get(video_hash)
            return dict(entry) if entry is not None else None

    def update(self, video_hash: str, now: float, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into the entry, stamp ``updated_at`` and persist."""
        with self._lock:
            entry = self._entries[video_hash]
            previous = entry.get("state")
            entry.update(fields)
            entry["updated_at"] = _iso(now)
            self._save_locked()
            if "state" in fields and fields["state"] != previous:
                logger.info(f"[notify {video_hash}] {previous} -> {fields['state']}")
            return dict(entry)

    def next_due(self, now: float) -> str | None:
        """The oldest non-terminal hash whose next attempt is due, or None."""
        with self._lock:
            due = [
                e
                for e in self._entries.values()
                if e.get("state") not in TERMINAL_STATES and e.get("next_attempt_ts", 0) <= now
            ]
        if not due:
            return None
        due.sort(key=lambda e: e.get("notified_ts", 0))
        return str(due[0]["video_hash"])

    def next_wakeup(self) -> float | None:
        """The earliest ``next_attempt_ts`` among non-terminal entries, or None when idle."""
        with self._lock:
            pending = [
                float(e.get("next_attempt_ts", 0))
                for e in self._entries.values()
                if e.get("state") not in TERMINAL_STATES
            ]
        return min(pending) if pending else None

    def __len__(self) -> int:
        """Number of entries, terminal ones included."""
        with self._lock:
            return len(self._entries)


class SweepLock:
    """The ``flock`` the incremental sweep holds for its whole run, on the same path."""

    def __init__(self, path: str | Path, timeout_seconds: float) -> None:
        """Lock on ``path``; ``acquire`` gives up after ``timeout_seconds``."""
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def acquire(self) -> Iterator[bool]:
        """Yield True while the lock is held, or False when the sweep kept it past the timeout."""
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        held = False
        try:
            deadline = time.monotonic() + self.timeout_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.2)
            yield held
        finally:
            if held:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _signature(paths: list[Path]) -> list[tuple[str, int, int]]:
    """(path, mtime_ns, size) per file: the identity of the parsed content, in file order."""
    out = []
    for path in paths:
        try:
            st = path.stat()
            out.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(path), -1, -1))
    return out


class _Prepared:
    """One video's parsed documents and their embeddings, kept between worker cycles.

    ``signature`` records the files' (path, mtime_ns, size) at parse time. A
    cycle that finds a different signature re-parses, so a VTT rewritten while
    the hash waited (for example during a long sweep) is never written stale.
    """

    def __init__(
        self,
        files: list[tuple[Path, list[Document]]],
        embeddings: np.ndarray | None,
        signature: list[tuple[str, int, int]],
    ) -> None:
        """Hold the per-file documents, the embeddings of all of them, and the files' signature."""
        self.files = files
        self.docs: list[Document] = [doc for _, docs in files for doc in docs]
        self.embeddings = embeddings
        self.signature = signature


class NotifyWorker:
    """One daemon thread that drives the state machine for notified hashes, oldest first.

    The ingest, embed, upsert and lock callables are injectable so unit tests
    need neither a model nor Qdrant.
    """

    def __init__(
        self,
        config: Config,
        queue: NotifyQueue,
        *,
        ingest: Callable[[Path], list[Document]],
        embed: Callable[[list[Document]], np.ndarray],
        upsert: Callable[[np.ndarray, list[Document]], Any],
        lock: Callable[[], AbstractContextManager[bool]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Wire the worker to its queue and to the four pipeline steps."""
        self.config = config
        self.queue = queue
        self._ingest = ingest
        self._embed = embed
        self._upsert = upsert
        self._lock = lock
        self._clock = clock
        self._prepared: dict[str, _Prepared] = {}
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # Lifecycle

    def start(self) -> None:
        """Start the daemon thread (a no-op when it already runs)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="notify-worker", daemon=True)
        self._thread.start()
        logger.info(f"notify worker started ({len(self.queue)} entries in {self.queue.path})")

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the thread to finish its current step and wait for it."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        logger.info("notify worker stopped")

    def is_running(self) -> bool:
        """True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def notify(self, video_hash: str, path: str | None = None) -> dict[str, Any]:
        """Queue a hash (idempotent) and wake the thread; returns the entry."""
        entry = self.queue.enqueue(video_hash, path, self._clock())
        self._wake.set()
        return entry

    def status(self, video_hash: str) -> dict[str, Any] | None:
        """The persisted entry for ``video_hash``, or None."""
        return self.queue.get(video_hash)

    def _run(self) -> None:
        """Thread body: process due hashes; otherwise sleep until the next one or a notify."""
        while not self._stop.is_set():
            try:
                busy = self.run_once()
            except Exception:  # noqa: BLE001
                # run_once records step errors in the queue; this catches the queue
                # file itself failing to write. The thread must outlive that.
                logger.exception("notify worker cycle failed; retrying after poll_seconds")
                busy = False
            if not busy:
                wake = self.queue.next_wakeup()
                poll = self.config.notify.poll_seconds
                timeout = poll if wake is None else max(0.05, min(poll, wake - self._clock()))
                self._wake.wait(timeout)
                self._wake.clear()

    # State machine

    def run_once(self) -> bool:
        """Process the oldest due hash; returns False when nothing is due."""
        video_hash = self.queue.next_due(self._clock())
        if video_hash is None:
            return False
        try:
            self._step(video_hash)
        except Exception as exc:  # noqa: BLE001
            self._retry_later(video_hash, f"{type(exc).__name__}: {exc}")
        return True

    def expected_vtts(self, video_hash: str) -> list[Path]:
        """The VTT paths the sweep would index for this hash, one per configured language."""
        folder = Path(self.config.paths.archive_root) / WebMetadataLoader.hash_to_archive_dir(
            video_hash
        )
        return [folder / f"{video_hash}.{lang}.vtt" for lang in self.config.notify.vtt_languages]

    def _step(self, video_hash: str) -> None:
        """Advance one hash by one state."""
        entry = self.queue.get(video_hash)
        if entry is None:
            return
        now = self._clock()
        notified = float(entry.get("notified_ts", now))
        cfg = self.config.notify
        if now - notified > cfg.max_wait_hours * 3600:
            self._prepared.pop(video_hash, None)
            self.queue.update(
                video_hash,
                now,
                state=STATE_FAILED,
                failed_at=_iso(now),
                error=entry.get("error")
                or f"gave up after {cfg.max_wait_hours:g} h in state {entry['state']}",
            )
            logger.warning(f"[notify {video_hash}] failed: {self.queue.get(video_hash)['error']}")
            return

        expected = self.expected_vtts(video_hash)
        present = [p for p in expected if p.is_file()]
        if not present or (
            len(present) < len(expected) and now - notified < cfg.partial_after_seconds
        ):
            detail = f"{len(present)} of {len(expected)} VTT files present"
            self.queue.update(
                video_hash,
                now,
                state=STATE_WAITING_FOR_VTT,
                vtt_files=[str(p) for p in present],
                next_attempt_ts=now + cfg.poll_seconds,
            )
            logger.debug(f"[notify {video_hash}] waiting: {detail}")
            return

        prepared = self._prepared.get(video_hash)
        if prepared is not None and prepared.signature != _signature(present):
            # The files changed since the last parse (a rewrite during a long
            # wait). Parse again, and upsert the fresh chunks right away: the
            # points in the index come from the old content until then.
            prepared = None
            if entry.get("searchable_at"):
                entry = self.queue.update(video_hash, now, searchable_at=None)
            logger.info(f"[notify {video_hash}] VTT files changed since the last parse; re-parsing")
        if prepared is None:
            prepared = self._prepare(present)
            self._prepared[video_hash] = prepared
        if not prepared.docs:
            # Not cached: the next cycle parses the files again, so metadata or
            # speech that arrives later is picked up instead of the empty result.
            self._prepared.pop(video_hash, None)
            self._retry_later(
                video_hash, "the VTT files produced no documents (no speech, or metadata missing)"
            )
            return
        assert prepared.embeddings is not None

        if not entry.get("searchable_at"):
            self._upsert(prepared.embeddings, prepared.docs)
            now = self._clock()
            entry = self.queue.update(
                video_hash,
                now,
                searchable_at=_iso(now),
                doc_count=len(prepared.docs),
                vtt_files=[str(p) for p in present],
                error=None,
            )
            logger.info(f"[notify {video_hash}] {len(prepared.docs)} chunks upserted, searchable")

        with self._lock() as held:
            if not held:
                self.queue.update(
                    video_hash,
                    self._clock(),
                    state=STATE_FILES_PENDING,
                    next_attempt_ts=self._clock() + cfg.poll_seconds,
                )
                return
            if prepared.signature != _signature(present):
                # The files changed while the hash waited for the lock. Drop the
                # parse; the next cycle re-parses, re-upserts and writes the fresh one.
                self._prepared.pop(video_hash, None)
                self.queue.update(
                    video_hash,
                    self._clock(),
                    searchable_at=None,
                    next_attempt_ts=self._clock(),
                )
                logger.info(f"[notify {video_hash}] VTT files changed before the write; re-parsing")
                return
            self._write_files(prepared)

        # A sweep may have finished between the first upsert and the lock and
        # deleted the points as "not in docs.jsonl". Now they are, so put them back.
        self._upsert(prepared.embeddings, prepared.docs)
        now = self._clock()
        self._prepared.pop(video_hash, None)
        self.queue.update(video_hash, now, state=STATE_INDEXED, indexed_at=_iso(now), error=None)

    def _prepare(self, vtts: list[Path]) -> _Prepared:
        """Parse each VTT into documents (as the sweep does) and embed them all."""
        signature = _signature(vtts)
        files = [(path, self._ingest(path)) for path in vtts]
        prepared = _Prepared(files, None, signature)
        if prepared.docs:
            prepared.embeddings = self._embed(prepared.docs)
        return prepared

    def _write_files(self, prepared: _Prepared) -> None:
        """Append the documents to docs.jsonl and record the files in the manifest.

        Runs with the sweep lock held. A file the manifest already knows as
        unchanged is skipped: its lines are in docs.jsonl already. A modified
        file is appended but its manifest entry is left as it was, so the next
        sweep re-parses it and drops the superseded lines (it removes every
        line whose id belongs to a modified file before it writes the new ones).
        """
        docs_path = Path(self.config.paths.docs_output)
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = FileManifest(self.config.incremental.manifest_path)
        if not manifest.load() and manifest.manifest_path.exists():
            # A manifest that exists but cannot be read must not be replaced by
            # a near-empty one: the next sweep would then treat every file as
            # new and duplicate the corpus. Fail this attempt; the job retries.
            raise RuntimeError(f"cannot read the manifest {manifest.manifest_path}")
        written = 0
        with open(docs_path, "a", encoding="utf-8") as out:
            for vtt, docs in prepared.files:
                if not docs:
                    continue
                state = manifest.classify_file(vtt)
                if state == "unchanged":
                    continue
                for doc in docs:
                    out.write(doc.model_dump_json() + "\n")
                    written += 1
                if state == "new":
                    manifest.update_entry(vtt, [doc.id for doc in docs])
            out.flush()
            os.fsync(out.fileno())
        manifest.save()
        logger.info(f"appended {written} documents to {docs_path} and saved the manifest")

    def _retry_later(self, video_hash: str, error: str) -> None:
        """Record an error and schedule the next attempt with exponential backoff."""
        entry = self.queue.get(video_hash)
        if entry is None:
            return
        attempts = int(entry.get("attempts", 0)) + 1
        delay = min(self.config.notify.poll_seconds * 2 ** (attempts - 1), MAX_BACKOFF_SECONDS)
        now = self._clock()
        self.queue.update(
            video_hash, now, attempts=attempts, error=error, next_attempt_ts=now + delay
        )
        logger.warning(
            f"[notify {video_hash}] attempt {attempts} failed, retry in {delay:g}s: {error}"
        )


def build_worker(config: Config, embedder_factory: Callable[[], Embedder]) -> NotifyWorker:
    """The production worker: real ingester, a lazily built embedder and a lazy Qdrant client."""
    from rainrag.index import QdrantIndexer
    from rainrag.ingest import Ingester

    ingester = Ingester(config)
    holder: dict[str, Any] = {}

    def embed(docs: list[Document]) -> np.ndarray:
        """Embed with the factory's embedder, created on first use."""
        if "embedder" not in holder:
            holder["embedder"] = embedder_factory()
        return holder["embedder"].generate_embeddings(docs, show_progress=False)

    def upsert(embeddings: np.ndarray, docs: list[Document]) -> int:
        """Scoped upsert into the configured collection, connecting on first use."""
        if "indexer" not in holder:
            indexer = QdrantIndexer(config)
            indexer.connect()
            holder["indexer"] = indexer
        return holder["indexer"].index_documents(embeddings, docs)

    lock = SweepLock(config.notify.lock_path, config.notify.lock_timeout_seconds)
    return NotifyWorker(
        config,
        NotifyQueue(config.notify.queue_path),
        ingest=ingester.process_file,
        embed=embed,
        upsert=upsert,
        lock=lock.acquire,
    )
