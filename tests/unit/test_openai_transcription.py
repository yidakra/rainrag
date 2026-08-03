import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rainrag import openai_transcription as transcription


def test_run_command_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("ffmpeg", 1)

    monkeypatch.setattr(transcription.subprocess, "run", raise_timeout)

    with pytest.raises(RuntimeError, match="audio extraction timed out"):
        transcription._run_command(["ffmpeg"], description="audio extraction")


def test_choose_chunk_boundaries_prefers_nearby_silence() -> None:
    boundaries = transcription._choose_chunk_boundaries(
        duration_seconds=3700,
        silence_centers=[1792, 3605],
        chunk_seconds=1800,
        silence_window_seconds=30,
    )

    assert boundaries == [1792, 3605]


def test_parse_response_offsets_segments_and_normalizes_language() -> None:
    response = SimpleNamespace(
        language="russian",
        segments=[
            SimpleNamespace(start=1.0, end=2.5, text=" Первый фрагмент "),
            SimpleNamespace(start=3.0, end=4.0, text="Второй фрагмент"),
        ],
    )

    cues, language = transcription._parse_response(response, offset_seconds=1800)

    assert language == "ru"
    assert cues[0] == transcription.TranscriptionCue(1801.0, 1802.5, "Первый фрагмент")
    assert cues[1].start_seconds == 1803.0


def test_render_vtt_preserves_original_timeline() -> None:
    rendered = transcription._render_vtt([transcription.TranscriptionCue(1801.0, 1802.5, "Text")])

    assert rendered == "WEBVTT\n\n00:30:01.000 --> 00:30:02.500\nText\n"


def test_split_audio_creates_every_requested_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio")

    def fake_run(command: list[str], *, description: str) -> object:
        Path(command[-1]).write_bytes(b"chunk")
        return object()

    monkeypatch.setattr(transcription, "_run_command", fake_run)

    chunks = transcription._split_audio(
        audio_path=audio,
        output_dir=tmp_path,
        duration_seconds=20,
        boundaries=[10],
    )

    assert [chunk.offset_seconds for chunk in chunks] == [0, 10]
    assert all(chunk.path.exists() for chunk in chunks)


def test_transcribe_media_merges_timestamped_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media = tmp_path / "source.mov"
    media.touch()
    output = tmp_path / "source.vtt"
    calls: list[dict] = []
    progress: list[tuple[int, int]] = []

    def fake_extract(_media_path: Path, audio_path: Path) -> None:
        audio_path.write_bytes(b"audio")

    def fake_split(**kwargs: object) -> list[transcription._AudioChunk]:
        output_dir = Path(str(kwargs["output_dir"]))
        first = output_dir / "one.mp3"
        second = output_dir / "two.mp3"
        first.touch()
        second.touch()
        return [
            transcription._AudioChunk(first, 0),
            transcription._AudioChunk(second, 1800),
        ]

    class FakeTranscriptions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(
                language="english",
                segments=[SimpleNamespace(start=1.0, end=2.5, text="A cue")],
            )

    class FakeClient:
        audio = SimpleNamespace(transcriptions=FakeTranscriptions())

    monkeypatch.setattr(transcription, "_extract_audio", fake_extract)
    monkeypatch.setattr(transcription, "_probe_duration", lambda _path: 3600.0)
    monkeypatch.setattr(transcription, "_detect_silence_centers", lambda _path: [1800.0])
    monkeypatch.setattr(transcription, "_split_audio", fake_split)
    monkeypatch.setattr(transcription, "OpenAI", lambda **_kwargs: FakeClient())

    result = transcription.transcribe_media(
        media,
        output,
        api_key="test-key",
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    rendered = output.read_text(encoding="utf-8")
    assert "00:00:01.000 --> 00:00:02.500" in rendered
    assert "00:30:01.000 --> 00:30:02.500" in rendered
    assert result.language == "en"
    assert result.cue_count == 2
    assert result.chunk_count == 2
    assert progress == [(1, 2), (2, 2)]
    assert calls[0]["response_format"] == "verbose_json"
    assert calls[0]["timestamp_granularities"] == ["segment"]
    assert "prompt" not in calls[0]
    assert calls[1]["prompt"] == "A cue"


def test_timestamped_transcription_rejects_unsupported_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires whisper-1"):
        transcription.transcribe_media(
            tmp_path / "source.mp4",
            tmp_path / "source.vtt",
            api_key="test-key",
            model="gpt-transcribe",
        )
