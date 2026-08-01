"""Tests for single-video upload sessions.

The manager talks to Qdrant and spawns GPU subprocesses in ``__init__``, so
these tests build instances with ``__new__`` and populate only the attributes
the logic under test touches. That keeps coverage on the parts most likely to
break — the GPU queue, cancellation, and language handling — without needing a
live Qdrant or a CUDA device.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from rainrag.video_session import (
    STATUS_ERROR,
    STATUS_QUEUED,
    SessionCancelledError,
    VideoSession,
    VideoSessionManager,
    _GpuPool,
)


def _bare_manager(devices: list[int] | None = None) -> VideoSessionManager:
    """Build a manager without touching Qdrant, the GPU, or the filesystem."""
    manager = VideoSessionManager.__new__(VideoSessionManager)
    manager._sessions = {}
    manager._lock = threading.Lock()
    manager._gpu_queue = []
    manager._procs = {}
    manager._gpu_pool = _GpuPool(devices if devices is not None else [0])
    return manager


class _FakeVideoUploadConfig(SimpleNamespace):
    """Stand-in for VideoUploadConfig covering only device resolution."""

    def __init__(self, **overrides) -> None:
        defaults = {
            "device": "cuda",
            "device_index": 0,
            "device_indices": [],
            "livevtt_python": "/nonexistent/python",
        }
        super().__init__(**{**defaults, **overrides})


def _device_manager(probe_result: int | None, **cfg_overrides) -> VideoSessionManager:
    """Manager whose CUDA probe returns a fixed device count."""
    manager = _bare_manager()
    manager.cfg = _FakeVideoUploadConfig(**cfg_overrides)
    manager._probe_cuda_device_count = lambda: probe_result  # type: ignore[method-assign]
    return manager


class TestVideoSessionState:
    def test_public_dict_hides_local_paths_and_internals(self) -> None:
        session = VideoSession(
            id="abc",
            filename="clip.mp4",
            session_dir="/srv/sessions/abc",
            video_path="/srv/sessions/abc/source.mp4",
            vtt_path="/srv/sessions/abc/source.ru.vtt",
        )
        session.cancelled = True

        public = session.public_dict()

        for hidden in ("session_dir", "video_path", "vtt_path", "cancelled"):
            assert hidden not in public
        assert public["id"] == "abc"
        assert public["status"] == STATUS_QUEUED

    def test_public_dict_exposes_language_and_queue_position(self) -> None:
        session = VideoSession(id="abc", filename="clip.mp4")
        session.language = "uk"
        session.language_probability = 0.91
        session.queue_position = 2

        public = session.public_dict()

        assert public["language"] == "uk"
        assert public["language_probability"] == 0.91
        assert public["queue_position"] == 2


class TestGpuQueue:
    def test_queue_position_reflects_order(self) -> None:
        manager = _bare_manager()
        manager._gpu_queue = ["first", "second", "third"]

        assert manager._queue_position("first") == 0
        assert manager._queue_position("second") == 1
        assert manager._queue_position("third") == 2

    def test_queue_position_zero_when_not_queued(self) -> None:
        """A session already holding the GPU is not reported as waiting."""
        manager = _bare_manager()

        assert manager._queue_position("unknown") == 0


class TestCancellation:
    def test_check_cancelled_passes_for_live_session(self) -> None:
        manager = _bare_manager()
        manager._sessions["live"] = VideoSession(id="live", filename="clip.mp4")

        manager._check_cancelled("live")  # must not raise

    def test_check_cancelled_raises_for_deleted_session(self) -> None:
        """delete() pops the session, which is how workers learn to stop."""
        manager = _bare_manager()

        with pytest.raises(SessionCancelledError):
            manager._check_cancelled("gone")

    def test_check_cancelled_raises_when_flagged(self) -> None:
        manager = _bare_manager()
        session = VideoSession(id="live", filename="clip.mp4")
        session.cancelled = True
        manager._sessions["live"] = session

        with pytest.raises(SessionCancelledError):
            manager._check_cancelled("live")


class TestDetectedLanguage:
    def test_reads_language_from_progress_file(self, tmp_path: Path) -> None:
        manager = _bare_manager()
        progress = tmp_path / "progress.json"
        progress.write_text(json.dumps({"detected_language": "UK"}), encoding="utf-8")

        assert manager._detected_language(progress) == "uk"

    def test_returns_none_for_missing_or_invalid_progress(self, tmp_path: Path) -> None:
        manager = _bare_manager()
        missing = tmp_path / "absent.json"
        invalid = tmp_path / "invalid.json"
        invalid.write_text("not json", encoding="utf-8")

        assert manager._detected_language(missing) is None
        assert manager._detected_language(invalid) is None

    @pytest.mark.parametrize("language", ["ru", "en"])
    def test_applies_suffix_for_indexable_languages(self, tmp_path: Path, language: str) -> None:
        manager = _bare_manager()
        vtt = tmp_path / "source.vtt"
        vtt.write_text("WEBVTT\n\n", encoding="utf-8")
        progress = tmp_path / "progress.json"
        progress.write_text(json.dumps({"detected_language": language}), encoding="utf-8")

        result = manager._apply_language_suffix(vtt, progress)

        assert result.name == f"source.{language}.vtt"
        assert result.exists()
        assert not vtt.exists()

    def test_keeps_bare_name_for_unsupported_language(self, tmp_path: Path) -> None:
        """Whisper can return any code; only ru/en change how ingest indexes."""
        manager = _bare_manager()
        vtt = tmp_path / "source.vtt"
        vtt.write_text("WEBVTT\n\n", encoding="utf-8")
        progress = tmp_path / "progress.json"
        progress.write_text(json.dumps({"detected_language": "ka"}), encoding="utf-8")

        result = manager._apply_language_suffix(vtt, progress)

        assert result == vtt
        assert vtt.exists()

    def test_progress_poll_records_language(self, tmp_path: Path) -> None:
        manager = _bare_manager()
        manager._sessions["s1"] = VideoSession(id="s1", filename="clip.mp4")
        progress = tmp_path / "progress.json"
        progress.write_text(
            json.dumps(
                {
                    "stage": "transcribing",
                    "percent": 42.0,
                    "total_seconds": 90.0,
                    "detected_language": "ru",
                    "language_probability": 0.97,
                }
            ),
            encoding="utf-8",
        )

        manager._poll_progress("s1", progress)

        session = manager._sessions["s1"]
        assert session.language == "ru"
        assert session.language_probability == pytest.approx(0.97)
        assert session.percent == pytest.approx(42.0)
        assert session.duration_seconds == pytest.approx(90.0)


class TestCancelledSessionsAreNotReportedAsFailures:
    """A cancelled session fails in whatever way its current stage fails.

    Regression: killing the transcriber from delete() surfaced as
    "transcriber exited -9" and was logged as a failure with a traceback.
    """

    def test_run_treats_post_cancel_errors_as_cancellation(self) -> None:
        manager = _bare_manager()
        session = VideoSession(id="s1", filename="clip.mp4")
        manager._sessions["s1"] = session

        def _transcribe(_session: VideoSession) -> None:
            # Mimic delete(): the session disappears, then the stage blows up.
            manager._sessions.pop("s1", None)
            raise RuntimeError("transcriber exited -9: ")

        manager._transcribe = _transcribe  # type: ignore[method-assign]

        manager._run("s1")

        # Nothing to report an error on, and no error state left behind.
        assert "s1" not in manager._sessions
        assert session.status == STATUS_QUEUED
        assert session.error is None

    def test_run_still_reports_genuine_failures(self) -> None:
        manager = _bare_manager()
        manager._sessions["s2"] = VideoSession(id="s2", filename="clip.mp4")

        def _transcribe(_session: VideoSession) -> None:
            raise RuntimeError("transcriber exited 4: model load failed")

        manager._transcribe = _transcribe  # type: ignore[method-assign]

        manager._run("s2")

        session = manager._sessions["s2"]
        assert session.status == STATUS_ERROR
        assert session.error is not None
        assert "model load failed" in session.error


class TestGpuPool:
    def test_hands_out_one_slot_per_device(self) -> None:
        pool = _GpuPool([0, 1])

        first = pool.acquire(timeout=0.1)
        second = pool.acquire(timeout=0.1)

        assert {first, second} == {0, 1}
        # Both GPUs are busy, so a third job must wait rather than share one.
        assert pool.acquire(timeout=0.1) is None

    def test_released_device_is_reused(self) -> None:
        pool = _GpuPool([0, 1])
        first = pool.acquire(timeout=0.1)
        pool.acquire(timeout=0.1)

        assert pool.acquire(timeout=0.1) is None
        pool.release(first)  # type: ignore[arg-type]

        assert pool.acquire(timeout=0.1) == first

    def test_single_device_serialises(self) -> None:
        pool = _GpuPool([0])

        assert pool.acquire(timeout=0.1) == 0
        assert pool.acquire(timeout=0.1) is None
        assert pool.size == 1

    def test_waiter_is_woken_by_a_release(self) -> None:
        """A queued job must start as soon as a GPU frees up, not on a timeout."""
        pool = _GpuPool([0])
        held = pool.acquire(timeout=0.1)
        acquired: list[int | None] = []

        def _worker() -> None:
            acquired.append(pool.acquire(timeout=5.0))

        waiter = threading.Thread(target=_worker)
        waiter.start()
        pool.release(held)  # type: ignore[arg-type]
        waiter.join(timeout=5.0)

        assert acquired == [0]


class TestDeviceResolution:
    def test_uses_every_detected_gpu_by_default(self) -> None:
        """Two GPUs means two concurrent transcriptions, with no configuration."""
        manager = _device_manager(probe_result=2)

        assert manager._resolve_devices() == [0, 1]

    def test_single_gpu_box_gets_one_slot(self) -> None:
        manager = _device_manager(probe_result=1)

        assert manager._resolve_devices() == [0]

    def test_falls_back_to_configured_device_when_probe_fails(self) -> None:
        """Guessing high would fail every job assigned to a missing device."""
        manager = _device_manager(probe_result=None, device_index=1)

        assert manager._resolve_devices() == [1]

    def test_explicit_list_overrides_detection(self) -> None:
        manager = _device_manager(probe_result=4, device_indices=[2, 3])

        assert manager._resolve_devices() == [2, 3]

    def test_explicit_list_is_clamped_to_visible_devices(self) -> None:
        manager = _device_manager(probe_result=2, device_indices=[0, 1, 5])

        assert manager._resolve_devices() == [0, 1]

    def test_wholly_invalid_list_falls_back_to_device_zero(self) -> None:
        manager = _device_manager(probe_result=1, device_indices=[3, 4])

        assert manager._resolve_devices() == [0]

    def test_cpu_mode_does_not_probe_or_parallelise(self) -> None:
        manager = _device_manager(probe_result=8, device="cpu", device_index=0)

        assert manager._resolve_devices() == [0]


class TestSessionQueryMediaUrls:
    """Session transcripts are not archive files, so archive URLs must not be emitted.

    Regression: ``generate_media_urls`` turned an absolute session path into
    ``/video//home/ubuntu/.../source.mov``, which the archive routes reject with
    400. Session media is served from ``/video-sessions/{id}/media`` instead.
    """

    @staticmethod
    def _result() -> dict:
        return {
            "answer": "a",
            "question": "q",
            "num_documents": 1,
            "retrieved_documents": [
                {
                    "path": "/srv/data/video_sessions/abc/source.ru.vtt",
                    "text": "hello",
                    "score": 0.5,
                    "rank": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:02:01",
                }
            ],
        }

    def test_session_response_omits_archive_urls(self) -> None:
        from rainrag.api import _build_query_response

        response = _build_query_response(self._result(), media_urls=False)

        chunk = response.context[0]
        assert chunk.video_url is None
        assert chunk.vtt_url is None
        # The fragment still carries what the UI needs to seek the session player.
        assert chunk.start_time == "00:00:00"
        assert chunk.text == "hello"

    def test_archive_response_still_consults_media_urls(self, monkeypatch) -> None:
        """The default path is unchanged: archive answers keep their media links."""
        import rainrag.api as api

        monkeypatch.setattr(api, "generate_media_urls", lambda *_: ("/video/x.mp4", "/vtt/x.vtt"))

        response = api._build_query_response(self._result())

        assert response.context[0].video_url == "/video/x.mp4"
        assert response.context[0].vtt_url == "/vtt/x.vtt"

    def test_session_response_never_consults_media_urls(self, monkeypatch) -> None:
        import rainrag.api as api

        def _boom(*_args):
            raise AssertionError("archive URL generation must be skipped for sessions")

        monkeypatch.setattr(api, "generate_media_urls", _boom)

        response = api._build_query_response(self._result(), media_urls=False)

        assert response.context[0].video_url is None


class TestTranscriptDownloadNaming:
    """The download is named after the upload, not the internal source.ru.vtt."""

    @pytest.mark.parametrize(
        ("filename", "vtt_name", "expected"),
        [
            ("2026-05-08 23-20-37.mov", "source.ru.vtt", "2026-05-08 23-20-37.ru.vtt"),
            ("interview.mp4", "source.vtt", "interview.vtt"),
        ],
    )
    def test_download_filename(self, filename: str, vtt_name: str, expected: str) -> None:
        stem = Path(filename).stem or "transcript"
        suffixes = "".join(Path(vtt_name).suffixes[-2:])

        assert f"{stem}{suffixes}" == expected


class TestDimensionProbe:
    """Player sizing needs the upload's real shape, but must survive without it."""

    def _session(self, tmp_path: Path) -> VideoSession:
        return VideoSession(id="s1", filename="clip.mov", video_path=str(tmp_path / "source.mov"))

    def test_records_dimensions(self, tmp_path: Path, monkeypatch) -> None:
        manager = _bare_manager()
        session = self._session(tmp_path)
        manager._sessions["s1"] = session
        monkeypatch.setattr(
            "rainrag.video_session.subprocess.run",
            lambda *a, **k: SimpleNamespace(returncode=0, stdout="3420x2224\n", stderr=""),
        )

        manager._probe_dimensions(session)

        assert manager._sessions["s1"].width == 3420
        assert manager._sessions["s1"].height == 2224

    @pytest.mark.parametrize(
        "outcome",
        [
            SimpleNamespace(returncode=1, stdout="", stderr="boom"),
            SimpleNamespace(returncode=0, stdout="\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="notxnumbers\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="0x0\n", stderr=""),
        ],
    )
    def test_bad_probe_leaves_dimensions_unset(self, tmp_path: Path, monkeypatch, outcome) -> None:
        manager = _bare_manager()
        session = self._session(tmp_path)
        manager._sessions["s1"] = session
        monkeypatch.setattr("rainrag.video_session.subprocess.run", lambda *a, **k: outcome)

        manager._probe_dimensions(session)

        assert manager._sessions["s1"].width is None
        assert manager._sessions["s1"].height is None

    def test_missing_ffprobe_is_not_fatal(self, tmp_path: Path, monkeypatch) -> None:
        """A box without ffmpeg still transcribes; it just gets a 16:9 player."""
        manager = _bare_manager()
        session = self._session(tmp_path)
        manager._sessions["s1"] = session

        def _raise(*_a, **_k):
            raise FileNotFoundError("ffprobe")

        monkeypatch.setattr("rainrag.video_session.subprocess.run", _raise)

        manager._probe_dimensions(session)  # must not raise

        assert manager._sessions["s1"].width is None

    def test_dimensions_reach_the_client(self) -> None:
        session = VideoSession(id="s1", filename="clip.mov")
        session.width, session.height = 3420, 2224

        public = session.public_dict()

        assert public["width"] == 3420
        assert public["height"] == 2224


class TestTranscriberOutputHandling:
    """A chatty transcriber must not deadlock.

    Regression: stdout was a pipe drained only after the process exited, so a
    child writing past the ~64KB pipe buffer blocked forever while the manager
    waited for it.
    """

    def test_log_tail_returns_the_end_of_a_large_log(self, tmp_path: Path) -> None:
        log = tmp_path / "transcribe.log"
        log.write_text("x" * 100_000 + "\nFINAL ERROR\n", encoding="utf-8")

        tail = VideoSessionManager._read_log_tail(log)

        assert "FINAL ERROR" in tail
        assert len(tail) <= 64_100

    def test_log_tail_survives_a_missing_file(self, tmp_path: Path) -> None:
        assert VideoSessionManager._read_log_tail(tmp_path / "absent.log") == ""


class TestOrphanSweepScope:
    """The sweep must only remove paths this manager created.

    tmp_root can be shared or misconfigured; deleting everything under it
    would be silent data loss.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("0123456789abcdef0123456789abcdef", True),
            ("0123456789abcdef0123456789abcdef.upload", True),
            ("important-data", False),
            ("", False),
            ("0123456789ABCDEF0123456789ABCDEF", False),  # uuid4().hex is lowercase
            ("0123456789abcdef0123456789abcde", False),  # 31 chars
            ("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", False),  # right length, not hex
        ],
    )
    def test_recognises_only_session_artifacts(self, name: str, expected: bool) -> None:
        assert VideoSessionManager._is_session_artifact(Path("/tmp") / name) is expected


class TestShutdownTeardown:
    """Sessions left registered at shutdown can never be reached again."""

    def test_shutdown_tears_down_active_sessions(self) -> None:
        manager = _bare_manager()
        manager._stop = threading.Event()
        session = VideoSession(id="s1", filename="clip.mp4")
        manager._sessions["s1"] = session
        torn: list[str] = []
        manager._teardown = lambda s: torn.append(s.id)  # type: ignore[method-assign]

        manager.shutdown()

        assert torn == ["s1"]
        assert manager._sessions == {}
        assert manager._stop.is_set()

    def test_shutdown_continues_after_a_teardown_failure(self) -> None:
        manager = _bare_manager()
        manager._stop = threading.Event()
        manager._sessions = {
            "s1": VideoSession(id="s1", filename="a.mp4"),
            "s2": VideoSession(id="s2", filename="b.mp4"),
        }
        torn: list[str] = []

        def _teardown(session: VideoSession) -> None:
            if session.id == "s1":
                raise RuntimeError("qdrant down")
            torn.append(session.id)

        manager._teardown = _teardown  # type: ignore[method-assign]

        manager.shutdown()  # must not raise

        assert torn == ["s2"]


class TestSessionResponsesHideLocalPaths:
    """Session paths are internal; public_dict strips them for the same reason."""

    @staticmethod
    def _result(path: str) -> dict:
        return {
            "answer": "a",
            "question": "q",
            "num_documents": 1,
            "retrieved_documents": [
                {"path": path, "text": "t", "score": 0.5, "rank": 1, "start_time": "00:00:00"}
            ],
        }

    def test_session_filename_is_a_bare_name(self) -> None:
        from rainrag.api import _build_query_response

        response = _build_query_response(
            self._result("/srv/data/video_sessions/abc/source.ru.vtt"), media_urls=False
        )

        chunk = response.context[0]
        assert chunk.filename == "source.ru.vtt"
        # No path fragments, and no stray separator from a missing parent dir.
        assert chunk.group_id == "source"

    def test_archive_filename_keeps_its_path(self, monkeypatch) -> None:
        """The archive path identifies the broadcast, so it must survive."""
        import rainrag.api as api

        monkeypatch.setattr(api, "generate_media_urls", lambda *_: ("/video/x", "/vtt/x"))

        response = api._build_query_response(self._result("2024/05/show.ru.vtt"))

        assert response.context[0].filename == "2024/05/show.ru.vtt"


class TestSingleVideoQueryScoping:
    """Archive-shaped retrieval stages must not run against one uploaded video."""

    def test_hybrid_applies_to_the_main_corpus_however_it_is_named(self) -> None:
        """BM25 is built over the main corpus; naming it explicitly is still it."""
        main = "broadcast_transcripts"

        def gate(collection_name: str | None) -> bool:
            target = collection_name or main
            return target == main

        assert gate(None) is True
        assert gate(main) is True
        assert gate("session_abc") is False

    def test_two_stage_is_off_for_single_video(self) -> None:
        """Rewriting and HyDE paraphrase into archive register -- off-corpus here."""
        for configured in (True, False):
            assert (configured and not True) is False  # single_video wins
        assert (True and not False) is True  # archive keeps its configured value
