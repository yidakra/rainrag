"""Event-driven indexing: POST /ingest/notify, the persistent queue and the worker."""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag import api as api_module  # noqa: E402
from src.rainrag.api import app  # noqa: E402
from src.rainrag.config import Config  # noqa: E402
from src.rainrag.ingest import Document, FileManifest, Ingester, VTTParser  # noqa: E402
from src.rainrag.notify import (  # noqa: E402
    STATE_FAILED,
    STATE_FILES_PENDING,
    STATE_INDEXED,
    STATE_QUEUED,
    STATE_WAITING_FOR_VTT,
    NotifyQueue,
    NotifyWorker,
    QueueNotOwnedError,
    SweepLock,
    build_worker,
)


HASH = "0c14229efcba436a4c22f8d96f67e0cb93bc9076"
HASH_B = "b" * 40


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class _FakeWorker:
    """Stands in for NotifyWorker behind the routes: records calls, no thread."""

    def __init__(self, running: bool = True) -> None:
        """Start with no entries; ``running`` is what ``is_running`` reports."""
        self.running = running
        self.entries: dict[str, dict] = {}
        self.calls: list[tuple[str, str | None]] = []
        self.queue = SimpleNamespace(owned=True)

    def is_running(self) -> bool:
        """Report the configured liveness."""
        return self.running

    def notify(self, video_hash: str, path: str | None = None) -> dict:
        """Record the call; a known hash keeps its entry, like the real queue."""
        self.calls.append((video_hash, path))
        return self.entries.setdefault(
            video_hash, {"video_hash": video_hash, "state": STATE_QUEUED, "path": path}
        )

    def status(self, video_hash: str) -> dict | None:
        """The recorded entry, or None."""
        return self.entries.get(video_hash)


def _api_config(enabled: bool = True) -> SimpleNamespace:
    """The slice of the config the notify routes read."""
    return SimpleNamespace(notify=SimpleNamespace(enabled=enabled))


@pytest.fixture
def client():
    """A TestClient over the app; startup work is skipped by the autouse fixture."""
    with TestClient(app) as c:
        yield c


def test_notify_route_accepts_a_hash_with_202_and_queued(client):
    """A valid notification is acknowledged at once and reaches the worker."""
    worker = _FakeWorker()
    with (
        patch("src.rainrag.api.config", _api_config()),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        r = client.post("/ingest/notify", json={"video_hash": HASH.upper(), "path": "/x"})
    assert r.status_code == 202, r.text
    assert r.json() == {"video_hash": HASH, "state": "queued"}
    assert worker.calls == [(HASH, "/x")]


def test_notify_route_rejects_a_malformed_hash_with_400(client):
    """Anything that is not 40 hex characters is a client error, before the worker is asked."""
    worker = _FakeWorker()
    with (
        patch("src.rainrag.api.config", _api_config()),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        r = client.post("/ingest/notify", json={"video_hash": "not-a-hash"})
        short = client.post("/ingest/notify", json={"video_hash": "abc123"})
    assert r.status_code == 400
    assert short.status_code == 400
    assert worker.calls == []


def test_notify_route_requires_the_bearer_token(client, monkeypatch):
    """With RAINRAG_AUTH_TOKEN set, a request without the token is 401 and with it 202."""
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s3cret")
    worker = _FakeWorker()
    with (
        patch("src.rainrag.api.config", _api_config()),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        missing = client.post("/ingest/notify", json={"video_hash": HASH})
        wrong = client.post(
            "/ingest/notify", json={"video_hash": HASH}, headers={"Authorization": "Bearer nope"}
        )
        ok = client.post(
            "/ingest/notify", json={"video_hash": HASH}, headers={"Authorization": "Bearer s3cret"}
        )
        status = client.get(f"/ingest/notify/{HASH}")
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 202
    assert status.status_code == 401


def test_notify_routes_accept_the_webhook_token_and_nothing_else_does(client, monkeypatch):
    """The dedicated token works on both notify routes, the API token still works there, and
    the dedicated token is refused on any other route."""
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s3cret")
    monkeypatch.setenv("RAINRAG_NOTIFY_TOKEN", "hook-only")
    worker = _FakeWorker()
    cfg = SimpleNamespace(notify=SimpleNamespace(enabled=True, token_env="RAINRAG_NOTIFY_TOKEN"))
    hook = {"Authorization": "Bearer hook-only"}
    with (
        patch("src.rainrag.api.config", cfg),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        posted = client.post("/ingest/notify", json={"video_hash": HASH}, headers=hook)
        status = client.get(f"/ingest/notify/{HASH}", headers=hook)
        master = client.post(
            "/ingest/notify", json={"video_hash": HASH}, headers={"Authorization": "Bearer s3cret"}
        )
        elsewhere = client.get("/similar-videos", params={"video_id": HASH}, headers=hook)
    assert posted.status_code == 202
    assert status.status_code != 401
    assert master.status_code == 202
    assert elsewhere.status_code == 401


def test_without_a_webhook_token_only_the_api_token_opens_the_notify_routes(client, monkeypatch):
    """When the dedicated token is not set, the routes behave as before."""
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s3cret")
    monkeypatch.delenv("RAINRAG_NOTIFY_TOKEN", raising=False)
    worker = _FakeWorker()
    cfg = SimpleNamespace(notify=SimpleNamespace(enabled=True, token_env="RAINRAG_NOTIFY_TOKEN"))
    with (
        patch("src.rainrag.api.config", cfg),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        wrong = client.post(
            "/ingest/notify",
            json={"video_hash": HASH},
            headers={"Authorization": "Bearer hook-only"},
        )
    assert wrong.status_code == 401


def test_webhook_token_closes_the_notify_routes_without_an_api_token(client, monkeypatch):
    """With only the webhook token set, the notify routes still need it: the API's
    "no API token, allow all" fallback does not open them."""
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("RAINRAG_NOTIFY_TOKEN", "hook-only")
    worker = _FakeWorker()
    cfg = SimpleNamespace(notify=SimpleNamespace(enabled=True, token_env="RAINRAG_NOTIFY_TOKEN"))
    with (
        patch("src.rainrag.api.config", cfg),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        anonymous = client.post("/ingest/notify", json={"video_hash": HASH})
        wrong = client.post(
            "/ingest/notify",
            json={"video_hash": HASH},
            headers={"Authorization": "Bearer guess"},
        )
        status_anonymous = client.get(f"/ingest/notify/{HASH}")
        hook = client.post(
            "/ingest/notify",
            json={"video_hash": HASH},
            headers={"Authorization": "Bearer hook-only"},
        )
    assert anonymous.status_code == 401
    assert wrong.status_code == 401
    assert status_anonymous.status_code == 401
    assert hook.status_code == 202


def test_without_any_token_the_notify_routes_stay_open(client, monkeypatch):
    """With neither token set, the notify routes follow the rest of the API (open)."""
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAINRAG_NOTIFY_TOKEN", raising=False)
    worker = _FakeWorker()
    cfg = SimpleNamespace(notify=SimpleNamespace(enabled=True, token_env="RAINRAG_NOTIFY_TOKEN"))
    with (
        patch("src.rainrag.api.config", cfg),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        anonymous = client.post("/ingest/notify", json={"video_hash": HASH})
    assert anonymous.status_code == 202


def test_notify_route_is_503_when_disabled_or_without_a_worker(client):
    """notify.enabled false, no worker, or a dead worker thread all answer 503."""
    with (
        patch("src.rainrag.api.config", _api_config(enabled=False)),
        patch("src.rainrag.api.notify_worker", _FakeWorker()),
    ):
        disabled = client.post("/ingest/notify", json={"video_hash": HASH})
    with (
        patch("src.rainrag.api.config", _api_config()),
        patch("src.rainrag.api.notify_worker", None),
    ):
        no_worker = client.post("/ingest/notify", json={"video_hash": HASH})
    with (
        patch("src.rainrag.api.config", _api_config()),
        patch("src.rainrag.api.notify_worker", _FakeWorker(running=False)),
    ):
        dead = client.get(f"/ingest/notify/{HASH}")
    assert disabled.status_code == 503
    assert no_worker.status_code == 503
    assert dead.status_code == 503


def test_notify_status_route_returns_the_entry_or_404(client):
    """GET gives the state and timestamps of a notified hash and 404 for an unknown one."""
    worker = _FakeWorker()
    worker.entries[HASH] = {
        "video_hash": HASH,
        "state": STATE_INDEXED,
        "notified_at": "2026-11-09T10:00:00+00:00",
        "searchable_at": "2026-11-09T10:01:00+00:00",
        "indexed_at": "2026-11-09T10:01:05+00:00",
        "doc_count": 7,
        "notified_ts": 1.0,
        "next_attempt_ts": 2.0,
    }
    with (
        patch("src.rainrag.api.config", _api_config()),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        found = client.get(f"/ingest/notify/{HASH}")
        unknown = client.get(f"/ingest/notify/{HASH_B}")
        bad = client.get("/ingest/notify/zzz")
    assert found.status_code == 200
    body = found.json()
    assert body["state"] == "indexed"
    assert body["doc_count"] == 7
    assert body["indexed_at"] == "2026-11-09T10:01:05+00:00"
    assert "next_attempt_ts" not in body
    assert unknown.status_code == 404
    assert bad.status_code == 400


def test_duplicate_notifications_keep_one_queue_entry(tmp_path):
    """The real queue is idempotent: a pending hash notified twice is still one entry."""
    queue = NotifyQueue(tmp_path / "q.json")
    first = queue.enqueue(HASH, None, now=100.0)
    second = queue.enqueue(HASH, "/folder", now=200.0)
    assert len(queue) == 1
    assert second["notified_at"] == first["notified_at"]
    assert second["path"] == "/folder"  # a later path fills the blank, nothing else moves
    assert second["state"] == STATE_QUEUED


# ---------------------------------------------------------------------------
# Queue persistence
# ---------------------------------------------------------------------------


def test_queue_state_survives_a_new_instance(tmp_path):
    """What one NotifyQueue wrote, a fresh instance on the same path reads back."""
    path = tmp_path / "data" / "notify-queue.json"
    queue = NotifyQueue(path)
    queue.enqueue(HASH, "/folder", now=1000.0)
    queue.update(HASH, 1030.0, state=STATE_WAITING_FOR_VTT, next_attempt_ts=1060.0)
    queue.release()  # the old process exits, which drops its owner lock

    reloaded = NotifyQueue(path)
    entry = reloaded.get(HASH)
    assert entry is not None
    assert entry["state"] == STATE_WAITING_FOR_VTT
    assert entry["path"] == "/folder"
    assert entry["next_attempt_ts"] == 1060.0
    assert reloaded.next_due(1059.0) is None
    assert reloaded.next_due(1060.0) == HASH
    assert not path.with_name(path.name + ".tmp").exists()


def test_queue_ignores_a_broken_file_and_orders_by_notification_time(tmp_path):
    """A corrupt file starts empty; next_due picks the oldest notification first."""
    path = tmp_path / "q.json"
    path.write_text("{not json", encoding="utf-8")
    queue = NotifyQueue(path)
    assert len(queue) == 0
    queue.enqueue(HASH_B, None, now=20.0)
    queue.enqueue(HASH, None, now=10.0)
    assert queue.next_due(now=30.0) == HASH
    queue.update(HASH, 30.0, state=STATE_INDEXED)
    assert queue.next_due(now=30.0) == HASH_B
    queue.update(HASH_B, 30.0, state=STATE_FAILED)
    assert queue.next_due(now=30.0) is None
    assert queue.next_wakeup() is None


def test_a_finished_hash_notified_again_starts_over(tmp_path):
    """A re-transcoded video (its hash already indexed) is queued again from scratch."""
    queue = NotifyQueue(tmp_path / "q.json")
    queue.enqueue(HASH, None, now=1.0)
    queue.update(HASH, 2.0, state=STATE_INDEXED, indexed_at="x", doc_count=3)
    again = queue.enqueue(HASH, None, now=3.0)
    assert again["state"] == STATE_QUEUED
    assert again["indexed_at"] is None
    assert again["doc_count"] == 0
    assert len(queue) == 1


# ---------------------------------------------------------------------------
# Worker state machine, with fakes
# ---------------------------------------------------------------------------


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        """Start at ``now``."""
        self.now = now

    def __call__(self) -> float:
        """The current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move time forward."""
        self.now += seconds


class _Fakes:
    """Fake ingest, embed, upsert and lock; the lock is held when ``locked`` is True."""

    def __init__(self) -> None:
        """Start with the lock free and nothing recorded."""
        self.locked = False
        self.upserts: list[list[str]] = []
        self.events: list[str] = []
        self.ingested: list[Path] = []
        self.docs_per_file = 2

    def ingest(self, path: Path) -> list[Document]:
        """Two documents per VTT, with path-derived ids like the real parser."""
        self.ingested.append(path)
        self.events.append(f"ingest:{path.name}")
        return [
            Document(
                id=VTTParser.generate_id(path, i),
                path=str(path),
                language=path.suffixes[0].lstrip("."),
                text=f"chunk {i} of {path.name}",
                length=20,
                is_chunk=True,
                chunk_index=i,
            )
            for i in range(self.docs_per_file)
        ]

    def embed(self, docs: list[Document]) -> np.ndarray:
        """A unit vector per document."""
        self.events.append(f"embed:{len(docs)}")
        return np.ones((len(docs), 4), dtype=np.float32)

    def upsert(self, embeddings: np.ndarray, docs: list[Document]) -> int:
        """Record the ids that went to the (fake) index."""
        assert embeddings.shape[0] == len(docs)
        self.upserts.append([d.id for d in docs])
        self.events.append("upsert")
        return len(docs)

    @contextmanager
    def lock(self):
        """Held means the sweep has it: yield False; otherwise yield True."""
        self.events.append("lock" if not self.locked else "lock-busy")
        yield not self.locked


def _worker_config(test_config: Config, tmp_path: Path, **notify: object) -> Config:
    """The test config with notify settings suited to a fake-driven test."""
    fields = {
        "enabled": True,
        "queue_path": str(tmp_path / "data" / "notify-queue.json"),
        "poll_seconds": 30.0,
        "max_wait_hours": 24.0,
        "partial_after_seconds": 600.0,
        "lock_path": str(tmp_path / "sweep.lock"),
        "lock_timeout_seconds": 0.2,
    }
    fields.update(notify)
    return test_config.model_copy(update={"notify": test_config.notify.model_copy(update=fields)})


def _make_worker(config: Config, fakes: _Fakes, clock: _Clock) -> NotifyWorker:
    """A worker over a fresh queue at the config's queue path, wired to the fakes."""
    return NotifyWorker(
        config,
        NotifyQueue(config.notify.queue_path),
        ingest=fakes.ingest,
        embed=fakes.embed,
        upsert=fakes.upsert,
        lock=fakes.lock,
        clock=clock,
    )


def _write_vtts(worker: NotifyWorker, video_hash: str, text: str, languages=("ru", "en")) -> list:
    """Create the expected VTT files for ``video_hash`` in the configured archive."""
    written = []
    for path in worker.expected_vtts(video_hash):
        if path.suffixes[0].lstrip(".") in languages:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            written.append(path)
    return written


def _lines(path: Path) -> list[dict]:
    """The JSON objects in a JSONL file."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_worker_waits_for_the_vtts_then_indexes(test_config, tmp_path, sample_vtt_ru):
    """No VTT: waiting_for_vtt and a re-check in poll_seconds. VTTs present: indexed."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)

    worker.notify(HASH, "/transcoder/folder")
    assert worker.run_once() is True
    entry = worker.status(HASH)
    assert entry["state"] == STATE_WAITING_FOR_VTT
    assert entry["next_attempt_ts"] == clock.now + 30.0
    assert fakes.events == []
    assert worker.run_once() is False  # not due yet

    clock.advance(30.0)
    assert worker.run_once() is True
    assert worker.status(HASH)["state"] == STATE_WAITING_FOR_VTT

    _write_vtts(worker, HASH, sample_vtt_ru)
    clock.advance(30.0)
    assert worker.run_once() is True
    entry = worker.status(HASH)
    assert entry["state"] == STATE_INDEXED
    assert entry["doc_count"] == 4
    assert entry["searchable_at"] and entry["indexed_at"]
    assert entry["error"] is None
    assert sorted(Path(p).name for p in entry["vtt_files"]) == [f"{HASH}.en.vtt", f"{HASH}.ru.vtt"]
    # Both VTTs parsed, embedded once, upserted, files written under the lock, upserted again.
    assert fakes.events == [
        f"ingest:{HASH}.ru.vtt",
        f"ingest:{HASH}.en.vtt",
        "embed:4",
        "upsert",
        "lock",
        "upsert",
    ]
    assert fakes.upserts[0] == fakes.upserts[1]
    assert worker.run_once() is False


def test_worker_upserts_before_it_writes_the_files(test_config, tmp_path, sample_vtt_ru):
    """Search works first; docs.jsonl and the manifest follow, with the sweep's ids."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    docs_path = Path(config.paths.docs_output)
    manifest_path = Path(config.incremental.manifest_path)
    docs_path.write_text('{"id": "older", "text": "kept"}\n', encoding="utf-8")

    vtts = _write_vtts(worker, HASH, sample_vtt_ru)
    worker.notify(HASH)
    worker.run_once()

    assert fakes.events.index("upsert") < fakes.events.index("lock")
    lines = _lines(docs_path)
    assert lines[0] == {"id": "older", "text": "kept"}  # appended, never rewritten
    assert [d["id"] for d in lines[1:]] == fakes.upserts[0]
    manifest = FileManifest(manifest_path)
    assert manifest.load()
    assert set(manifest.entries) == {str(p.absolute()) for p in vtts}
    for path in vtts:
        assert manifest.entries[str(path.absolute())].doc_ids == [
            VTTParser.generate_id(path, 0),
            VTTParser.generate_id(path, 1),
        ]
        assert manifest.classify_file(path) == "unchanged"


def test_lock_held_leaves_files_pending_then_finishes_with_a_second_upsert(
    test_config, tmp_path, sample_vtt_ru
):
    """A running sweep holds the lock: the video is searchable but files wait; then done."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    fakes.locked = True
    worker = _make_worker(config, fakes, clock)
    docs_path = Path(config.paths.docs_output)

    _write_vtts(worker, HASH, sample_vtt_ru)
    worker.notify(HASH)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_FILES_PENDING
    assert entry["searchable_at"] is not None
    assert entry["indexed_at"] is None
    assert len(fakes.upserts) == 1
    assert not docs_path.exists()
    assert entry["next_attempt_ts"] == clock.now + 30.0

    # Still held on the next cycle: no second ingest or embed, no file write.
    clock.advance(30.0)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_FILES_PENDING
    assert len(fakes.upserts) == 1
    assert fakes.events.count("embed:4") == 1

    # The sweep finished and (as it does) deleted points not in docs.jsonl.
    fakes.locked = False
    clock.advance(30.0)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_INDEXED
    assert len(fakes.upserts) == 2
    assert fakes.upserts[0] == fakes.upserts[1]
    assert [d["id"] for d in _lines(docs_path)] == fakes.upserts[0]
    assert fakes.events.count("embed:4") == 1


def test_files_pending_survives_a_restart(test_config, tmp_path, sample_vtt_ru):
    """A new worker over the same queue file re-parses and completes a pending hash."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    fakes.locked = True
    worker = _make_worker(config, fakes, clock)
    _write_vtts(worker, HASH, sample_vtt_ru)
    worker.notify(HASH)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_FILES_PENDING

    fakes.locked = False
    clock.advance(30.0)
    worker.queue.release()  # the old process exits, which drops its owner lock
    restarted = _make_worker(config, fakes, clock)  # fresh in-memory cache, same queue file
    assert restarted.run_once() is True
    assert restarted.status(HASH)["state"] == STATE_INDEXED
    assert fakes.events.count("embed:4") == 2  # parsed again after the restart
    assert len(fakes.upserts) == 2  # the first upsert is not repeated: searchable_at was set


def test_partial_vtts_wait_then_index_what_is_present(test_config, tmp_path, sample_vtt_ru):
    """Only the ru VTT exists: wait until partial_after_seconds, then index it alone."""
    config = _worker_config(test_config, tmp_path, partial_after_seconds=120.0)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    _write_vtts(worker, HASH, sample_vtt_ru, languages=("ru",))
    worker.notify(HASH)

    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_WAITING_FOR_VTT
    assert [Path(p).name for p in entry["vtt_files"]] == [f"{HASH}.ru.vtt"]

    clock.advance(120.0)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_INDEXED
    assert entry["doc_count"] == 2
    assert fakes.ingested == [worker.expected_vtts(HASH)[0]]


def test_worker_fails_the_hash_after_max_wait(test_config, tmp_path):
    """VTTs that never appear: waiting_for_vtt until max_wait_hours, then failed with a reason."""
    config = _worker_config(test_config, tmp_path, max_wait_hours=1.0)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    worker.notify(HASH)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_WAITING_FOR_VTT

    clock.advance(3600.0)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_WAITING_FOR_VTT  # not past the limit yet

    clock.advance(31.0)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_FAILED
    assert entry["failed_at"] is not None
    assert "gave up after 1 h" in entry["error"]
    assert fakes.events == []
    assert worker.run_once() is False


def test_an_error_is_retried_with_backoff_and_kept_in_the_entry(
    test_config, tmp_path, sample_vtt_ru
):
    """Qdrant down: the step records the error, backs off (30, 60, ...) and succeeds later."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    _write_vtts(worker, HASH, sample_vtt_ru)
    worker.notify(HASH)

    def broken_upsert(embeddings, docs):
        """Fail like a connection error."""
        raise ConnectionError("qdrant unreachable")

    worker._upsert = broken_upsert
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_QUEUED
    assert entry["attempts"] == 1
    assert "qdrant unreachable" in entry["error"]
    assert entry["next_attempt_ts"] == clock.now + 30.0

    clock.advance(30.0)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["attempts"] == 2
    assert entry["next_attempt_ts"] == clock.now + 60.0

    worker._upsert = fakes.upsert
    clock.advance(60.0)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_INDEXED
    assert entry["error"] is None


def test_vtts_without_documents_are_retried_not_indexed(test_config, tmp_path, sample_vtt_ru):
    """A VTT that parses to nothing (silent, or metadata missing) is retried, not marked indexed."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    fakes.docs_per_file = 0
    worker = _make_worker(config, fakes, clock)
    _write_vtts(worker, HASH, sample_vtt_ru)
    worker.notify(HASH)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] == STATE_QUEUED
    assert "no documents" in entry["error"]
    assert fakes.upserts == []


def test_write_files_skips_unchanged_and_leaves_a_modified_entry_stale(
    test_config, tmp_path, sample_vtt_ru
):
    """A file the manifest already has as unchanged is not appended again; a modified one is
    appended but its manifest entry stays, so the next sweep re-parses and dedupes it."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    docs_path = Path(config.paths.docs_output)
    ru, en = _write_vtts(worker, HASH, sample_vtt_ru)

    # The sweep already has both files.
    manifest = FileManifest(config.incremental.manifest_path)
    manifest.update_entry(ru, ["old-ru"])
    manifest.update_entry(en, ["old-en"])
    manifest.save()
    docs_path.write_text('{"id": "old-ru"}\n{"id": "old-en"}\n', encoding="utf-8")

    # Then the ru file was rewritten (bigger), the en file is unchanged.
    ru.write_text(sample_vtt_ru + "\n4\n00:00:15.000 --> 00:00:20.000\nЕщё.\n", encoding="utf-8")

    worker.notify(HASH)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_INDEXED
    ids = [d["id"] for d in _lines(docs_path)]
    assert ids == ["old-ru", "old-en", VTTParser.generate_id(ru, 0), VTTParser.generate_id(ru, 1)]
    after = FileManifest(config.incremental.manifest_path)
    assert after.load()
    assert after.entries[str(ru.absolute())].doc_ids == ["old-ru"]  # stale on purpose
    assert after.classify_file(ru) == "modified"
    assert after.classify_file(en) == "unchanged"


def test_sweep_lock_yields_false_while_another_process_holds_the_flock(tmp_path):
    """The real lock: a held flock on the same path means False within the timeout."""
    path = tmp_path / "rainrag-incremental.lock"
    lock = SweepLock(path, timeout_seconds=0.3)
    with lock.acquire() as held:
        assert held is True

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with lock.acquire() as held:
            assert held is False
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    with lock.acquire() as held:
        assert held is True


def test_worker_thread_processes_a_notification(test_config, tmp_path, sample_vtt_ru):
    """start() runs the loop in a daemon thread and stop() ends it."""
    config = _worker_config(test_config, tmp_path, poll_seconds=0.05)
    fakes = _Fakes()
    worker = _make_worker(config, fakes, _Clock())
    worker._clock = time.time
    _write_vtts(worker, HASH, sample_vtt_ru)
    worker.start()
    assert worker.is_running()
    try:
        worker.notify(HASH)
        deadline = time.monotonic() + 5.0
        while worker.status(HASH)["state"] != STATE_INDEXED:
            assert time.monotonic() < deadline, worker.status(HASH)
            time.sleep(0.02)
    finally:
        worker.stop(timeout=2.0)
    assert not worker.is_running()
    assert len(fakes.upserts) == 2


# ---------------------------------------------------------------------------
# Same ids as the sweep
# ---------------------------------------------------------------------------


def test_notified_vtt_produces_the_sweeps_document_ids(test_config, tmp_path, sample_vtt_ru):
    """process_file on a notified VTT yields the ids the sweep's full ingest writes."""
    config = _worker_config(test_config, tmp_path)
    worker = _make_worker(config, _Fakes(), _Clock())
    (ru,) = _write_vtts(worker, HASH, sample_vtt_ru, languages=("ru",))

    ingester = Ingester(config)
    notified_docs = ingester.process_file(ru)
    assert notified_docs, "the sample VTT must parse"

    # The sweep's way: walk archive_root and write docs.jsonl.
    Ingester(config).ingest(incremental=False)
    sweep_ids = [d["id"] for d in _lines(Path(config.paths.docs_output))]

    assert [d.id for d in notified_docs] == sweep_ids
    for doc in notified_docs:
        expected = VTTParser.generate_id(ru, doc.chunk_index if doc.is_chunk else None)
        assert doc.id == expected


def test_build_worker_wires_the_real_ingester_and_a_lazy_embedder(test_config, tmp_path):
    """build_worker parses with Ingester.process_file and only builds the embedder on use."""
    config = _worker_config(test_config, tmp_path)
    created: list[str] = []

    def factory():
        """Count factory calls; never called in this test."""
        created.append("embedder")
        raise AssertionError("the embedder must be created lazily")

    worker = build_worker(config, factory)
    assert isinstance(worker.queue, NotifyQueue)
    assert worker.queue.path == Path(config.notify.queue_path)
    assert created == []
    assert worker.expected_vtts(HASH)[0].name == f"{HASH}.ru.vtt"
    assert str(worker.expected_vtts(HASH)[0].parent).endswith(
        "0c/14/22/9e/fc/ba/43/6a/4c/22/f8/d9/6f/67/e0/cb/93/bc/90/76"
    )


def test_notify_embedder_shares_the_query_engines_model(monkeypatch):
    """In the API the worker's Embedder reuses the loaded SentenceTransformer, not a second copy."""
    model = object()
    config = SimpleNamespace(
        embedding=SimpleNamespace(provider="local"),
        paths=SimpleNamespace(embeddings_cache="/nonexistent"),
    )
    monkeypatch.setattr(api_module, "config", config)
    monkeypatch.setattr(api_module, "query_engine", SimpleNamespace(embedding_model=model))
    with patch("rainrag.embed.Embedder") as embedder_cls:
        embedder = api_module._notify_embedder()
    embedder_cls.assert_called_once_with(config)
    assert embedder.model is model


def test_an_empty_parse_is_parsed_again_on_the_next_cycle(test_config, tmp_path, sample_vtt_ru):
    """A parse that yields no documents is not cached: the next cycle parses the files again."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    fakes.docs_per_file = 0
    worker = _make_worker(config, fakes, clock)
    _write_vtts(worker, HASH, sample_vtt_ru)
    worker.notify(HASH)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_QUEUED
    first_parses = len(fakes.ingested)

    fakes.docs_per_file = 2  # the metadata arrived
    clock.advance(31)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_INDEXED
    assert len(fakes.ingested) == 2 * first_parses


def test_a_vtt_rewritten_while_the_lock_was_held_is_parsed_again_before_the_write(
    test_config, tmp_path, sample_vtt_ru
):
    """A file that changed during files_pending is re-parsed and re-upserted; the stale parse is never written."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    ru, en = _write_vtts(worker, HASH, sample_vtt_ru)
    fakes.locked = True
    worker.notify(HASH)
    worker.run_once()
    assert worker.status(HASH)["state"] == STATE_FILES_PENDING
    assert len(fakes.upserts) == 1
    parses_before = len(fakes.ingested)

    # The transcriber rewrote the Russian file while the sweep held the lock.
    ru.write_text(sample_vtt_ru + "\n4\n00:00:15.000 --> 00:00:20.000\nЕщё.\n", encoding="utf-8")
    st = ru.stat()
    os.utime(ru, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    fakes.locked = False
    clock.advance(31)
    worker.run_once()  # parses again, upserts the fresh chunks, writes, upserts once more
    assert worker.status(HASH)["state"] == STATE_INDEXED
    assert len(fakes.ingested) == 2 * parses_before
    assert len(fakes.upserts) == 3
    events = fakes.events
    busy = events.index("lock-busy")
    reparse = events.index("ingest:" + ru.name, busy)
    assert reparse < events.index("upsert", busy) < events.index("lock", busy)


def test_an_unreadable_manifest_fails_the_attempt_and_writes_nothing(
    test_config, tmp_path, sample_vtt_ru
):
    """A manifest file that exists but cannot be read is left alone; the attempt is retried."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    _write_vtts(worker, HASH, sample_vtt_ru)
    manifest_path = Path(config.incremental.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{not json", encoding="utf-8")
    docs_path = Path(config.paths.docs_output)

    worker.notify(HASH)
    worker.run_once()
    entry = worker.status(HASH)
    assert entry["state"] != STATE_INDEXED
    assert "cannot read the manifest" in entry["error"]
    assert manifest_path.read_text(encoding="utf-8") == "{not json"
    assert not docs_path.exists() or docs_path.read_text(encoding="utf-8") == ""


def test_notify_token_env_must_not_be_the_api_token():
    """token_env RAINRAG_AUTH_TOKEN would make the webhook token the API token."""
    from src.rainrag.config import NotifyConfig

    with pytest.raises(ValueError, match="must not be RAINRAG_AUTH_TOKEN"):
        NotifyConfig(token_env="RAINRAG_AUTH_TOKEN")
    assert NotifyConfig(token_env="RAINRAG_NOTIFY_TOKEN").token_env == "RAINRAG_NOTIFY_TOKEN"
    assert NotifyConfig().token_env == "RAINRAG_NOTIFY_TOKEN"


def test_a_second_queue_on_the_same_file_stands_by_until_the_owner_exits(tmp_path):
    """Two processes on one queue file (a replacement pod while the old one stops):
    only the owner reads or writes; the other takes over, with the owner's entries."""
    path = tmp_path / "notify-queue.json"
    owner = NotifyQueue(path)
    owner.enqueue(HASH, None, 1000.0)
    standby = NotifyQueue(path)
    assert owner.owned
    assert not standby.owned
    with pytest.raises(QueueNotOwnedError):
        standby.enqueue(HASH, None, 1001.0)
    with pytest.raises(QueueNotOwnedError):
        standby.get(HASH)
    assert not standby.claim()

    owner.release()
    assert standby.claim()
    assert standby.get(HASH)["state"] == STATE_QUEUED
    standby.release()


def test_notify_routes_are_503_with_retry_after_while_the_queue_is_owned_elsewhere(
    client, monkeypatch
):
    """A standby process answers 503 and asks the sender to retry."""
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAINRAG_NOTIFY_TOKEN", raising=False)
    worker = _FakeWorker()
    worker.queue = SimpleNamespace(owned=False)
    cfg = SimpleNamespace(
        notify=SimpleNamespace(enabled=True, token_env="RAINRAG_NOTIFY_TOKEN", poll_seconds=30)
    )
    with (
        patch("src.rainrag.api.config", cfg),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        posted = client.post("/ingest/notify", json={"video_hash": HASH})
        status = client.get(f"/ingest/notify/{HASH}")
    assert posted.status_code == 503
    assert posted.headers.get("retry-after") == "30"
    assert status.status_code == 503
    assert worker.calls == []


def test_a_standby_queue_refuses_update_and_len(tmp_path):
    """Every read and write on a standby queue raises QueueNotOwnedError, not KeyError."""
    path = tmp_path / "notify-queue.json"
    owner = NotifyQueue(path)
    owner.enqueue(HASH, None, 1000.0)
    standby = NotifyQueue(path)
    with pytest.raises(QueueNotOwnedError):
        standby.update(HASH, 1001.0, state=STATE_WAITING_FOR_VTT)
    with pytest.raises(QueueNotOwnedError):
        len(standby)
    owner.release()


def test_notify_route_maps_a_lost_queue_to_503(client, monkeypatch):
    """Ownership dropping between the route's check and the call is 503, not 500."""
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAINRAG_NOTIFY_TOKEN", raising=False)
    worker = _FakeWorker()

    def lost(*_args, **_kwargs):
        # the class the api module imported (the tests reach notify.py by two paths)
        raise api_module.QueueNotOwnedError("released during shutdown")

    worker.notify = lost
    worker.status = lost
    cfg = SimpleNamespace(
        notify=SimpleNamespace(enabled=True, token_env="RAINRAG_NOTIFY_TOKEN", poll_seconds=30)
    )
    with (
        patch("src.rainrag.api.config", cfg),
        patch("src.rainrag.api.notify_worker", worker),
    ):
        posted = client.post("/ingest/notify", json={"video_hash": HASH})
        status = client.get(f"/ingest/notify/{HASH}")
    assert posted.status_code == 503
    assert posted.headers.get("retry-after") == "30"
    assert status.status_code == 503


def test_stop_hands_the_queue_to_the_next_process(test_config, tmp_path):
    """After stop() the owner lock is free, so a waiting process can claim the queue."""
    config = _worker_config(test_config, tmp_path)
    fakes, clock = _Fakes(), _Clock()
    worker = _make_worker(config, fakes, clock)
    worker.start()
    worker.notify(HASH)
    worker.stop(timeout=5.0)
    assert not worker.is_running()
    successor = NotifyQueue(worker.queue.path)
    assert successor.owned
    assert successor.get(HASH) is not None
    successor.release()
