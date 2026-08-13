import subprocess
from pathlib import Path

import pytest

from rainrag import openai_subtitles as subtitles


_SAMPLE_VTT = """WEBVTT

cue-id
00:00:01.000 --> 00:00:02.500 align:start
First line

00:00:03.000 --> 00:00:04.000
Second line
"""


def test_run_command_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr(subtitles.subprocess, "run", raise_timeout)

    with pytest.raises(RuntimeError, match="audio extraction timed out"):
        subtitles._run_command(["ffmpeg"], description="audio extraction")


def test_choose_chunk_boundaries_prefers_nearby_silence() -> None:
    boundaries = subtitles._choose_chunk_boundaries(
        duration_seconds=3700,
        silence_centers=[1792, 3605],
        chunk_seconds=1800,
        silence_window_seconds=30,
    )

    assert boundaries == [1792, 3605]


def test_parse_and_render_vtt_offsets_chunk_timestamps() -> None:
    cues = subtitles._parse_vtt(_SAMPLE_VTT, offset_seconds=1800)
    rendered = subtitles._render_vtt(cues)

    assert "00:30:01.000 --> 00:30:02.500 align:start" in rendered
    assert "00:30:03.000 --> 00:30:04.000" in rendered
    assert rendered.startswith("WEBVTT\n\n1\n")


def test_render_vtt_clamps_overlapping_cues() -> None:
    rendered = subtitles._render_vtt(
        [
            subtitles._VttCue(0.0, 5.0, "First"),
            subtitles._VttCue(4.0, 6.0, "Second"),
        ]
    )

    assert "00:00:00.000 --> 00:00:04.000" in rendered
    assert "00:00:04.000 --> 00:00:06.000" in rendered


def test_split_audio_reuses_small_single_chunk(tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"small")

    chunks = subtitles._split_audio(
        audio_path=audio,
        output_dir=tmp_path,
        duration_seconds=30,
        boundaries=[],
    )

    assert chunks == [subtitles._AudioChunk(path=audio, offset_seconds=0.0)]


def test_translate_job_merges_chunks_on_original_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "output" / "hash.en.vtt"
    job = subtitles.OpenAISubtitleJob(
        video_hash="hash",
        video_path=video,
        output_path=output,
    )

    def fake_extract(_video_path: Path, audio_path: Path) -> None:
        audio_path.write_bytes(b"audio")

    def fake_split(**kwargs: object) -> list[subtitles._AudioChunk]:
        output_dir = Path(str(kwargs["output_dir"]))
        first = output_dir / "one.mp3"
        second = output_dir / "two.mp3"
        first.touch()
        second.touch()
        return [
            subtitles._AudioChunk(first, 0),
            subtitles._AudioChunk(second, 1800),
        ]

    monkeypatch.setattr(subtitles, "_extract_audio", fake_extract)
    monkeypatch.setattr(subtitles, "_probe_duration", lambda _path: 3600.0)
    monkeypatch.setattr(subtitles, "_detect_silence_centers", lambda _path: [1800.0])
    monkeypatch.setattr(subtitles, "_split_audio", fake_split)
    monkeypatch.setattr(subtitles, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        subtitles,
        "_translate_chunk",
        lambda *, client, chunk, model: subtitles._parse_vtt(
            _SAMPLE_VTT, offset_seconds=chunk.offset_seconds
        ),
    )

    subtitles._translate_job(
        job,
        api_key="test-key",
        model="whisper-1",
        chunk_seconds=1800,
        silence_window_seconds=30,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "00:00:01.000 --> 00:00:02.500" in rendered
    assert "00:30:01.000 --> 00:30:02.500" in rendered
    starts = [line for line in rendered.splitlines() if " --> " in line]
    assert len(starts) == 4
    assert starts == sorted(starts)
    assert [line for line in rendered.splitlines() if line.isdigit()] == ["1", "2", "3", "4"]


def test_translate_job_tolerates_a_silent_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "hash.en.vtt"
    job = subtitles.OpenAISubtitleJob("hash", video, output)

    def fake_extract(_video_path: Path, audio_path: Path) -> None:
        audio_path.write_bytes(b"audio")

    def fake_split(**kwargs: object) -> list[subtitles._AudioChunk]:
        output_dir = Path(str(kwargs["output_dir"]))
        chunks = [output_dir / "silent.mp3", output_dir / "speech.mp3"]
        for chunk in chunks:
            chunk.touch()
        return [subtitles._AudioChunk(chunks[0], 0), subtitles._AudioChunk(chunks[1], 1800)]

    monkeypatch.setattr(subtitles, "_extract_audio", fake_extract)
    monkeypatch.setattr(subtitles, "_probe_duration", lambda _path: 3600.0)
    monkeypatch.setattr(subtitles, "_detect_silence_centers", lambda _path: [1800.0])
    monkeypatch.setattr(subtitles, "_split_audio", fake_split)
    monkeypatch.setattr(subtitles, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(
        subtitles,
        "_translate_chunk",
        lambda *, client, chunk, model: (
            []
            if chunk.offset_seconds == 0
            else subtitles._parse_vtt(_SAMPLE_VTT, offset_seconds=chunk.offset_seconds)
        ),
    )

    subtitles._translate_job(
        job,
        api_key="test-key",
        model="whisper-1",
        chunk_seconds=1800,
        silence_window_seconds=30,
    )

    assert "00:30:01.000 --> 00:30:02.500" in output.read_text(encoding="utf-8")


def test_translate_job_rejects_an_entirely_silent_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "video.mp4"
    video.touch()
    output = tmp_path / "hash.en.vtt"
    job = subtitles.OpenAISubtitleJob("hash", video, output)

    monkeypatch.setattr(subtitles, "_extract_audio", lambda _source, audio: audio.touch())
    monkeypatch.setattr(subtitles, "_probe_duration", lambda _path: 30.0)
    monkeypatch.setattr(
        subtitles, "_split_audio", lambda **_kwargs: [subtitles._AudioChunk(video, 0)]
    )
    monkeypatch.setattr(subtitles, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(subtitles, "_translate_chunk", lambda **_kwargs: [])

    with pytest.raises(RuntimeError, match="no WebVTT cues for hash"):
        subtitles._translate_job(
            job,
            api_key="test-key",
            model="whisper-1",
            chunk_seconds=1800,
            silence_window_seconds=30,
        )


def test_translate_jobs_keeps_per_video_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [
        subtitles.OpenAISubtitleJob("ok", tmp_path / "ok.mp4", tmp_path / "ok.vtt"),
        subtitles.OpenAISubtitleJob("bad", tmp_path / "bad.mp4", tmp_path / "bad.vtt"),
    ]

    def fake_translate(job: subtitles.OpenAISubtitleJob, **_kwargs: object) -> None:
        if job.video_hash == "bad":
            raise RuntimeError("API unavailable")

    monkeypatch.setattr(subtitles, "_translate_job", fake_translate)

    result = subtitles.translate_jobs(jobs, api_key="test-key", workers=2)

    assert result.translated_hashes == ("ok",)
    assert result.failures == {"bad": "API unavailable"}


def test_split_audio_creates_every_requested_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    def fake_run(command: list[str], *, description: str) -> object:
        Path(command[-1]).write_bytes(b"chunk")
        return object()

    monkeypatch.setattr(subtitles, "_run_command", fake_run)

    chunks = subtitles._split_audio(
        audio_path=audio,
        output_dir=tmp_path,
        duration_seconds=20,
        boundaries=[10],
    )

    assert [chunk.offset_seconds for chunk in chunks] == [0, 10]
    assert all(chunk.path.exists() for chunk in chunks)


def test_split_audio_adds_size_boundaries_for_oversized_single_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"x" * 25)
    monkeypatch.setattr(subtitles, "MAX_OPENAI_UPLOAD_BYTES", 10)

    def fake_run(command: list[str], *, description: str) -> object:
        Path(command[-1]).write_bytes(b"chunk")
        return object()

    monkeypatch.setattr(subtitles, "_run_command", fake_run)

    chunks = subtitles._split_audio(
        audio_path=audio,
        output_dir=tmp_path,
        duration_seconds=30,
        boundaries=[],
    )

    assert [chunk.offset_seconds for chunk in chunks] == [0, 10, 20]
