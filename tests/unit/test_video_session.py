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
