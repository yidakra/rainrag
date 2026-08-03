"""Timestamped OpenAI transcription for uploaded video sessions."""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_CHUNK_SECONDS = 30 * 60
DEFAULT_SILENCE_WINDOW_SECONDS = 30.0
FFMPEG_TIMEOUT_SECONDS = 30 * 60
MAX_OPENAI_UPLOAD_BYTES = 24 * 1024 * 1024
_SILENCE_RE = re.compile(r"silence_(?P<kind>start|end):\s*(?P<seconds>-?\d+(?:\.\d+)?)")
_LANGUAGE_CODES = {
    "english": "en",
    "russian": "ru",
    "ukrainian": "uk",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "italian": "it",
    "portuguese": "pt",
    "polish": "pl",
    "belarusian": "be",
    "georgian": "ka",
}


@dataclass(frozen=True)
class TranscriptionCue:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class OpenAITranscriptionResult:
    language: str | None
    duration_seconds: float
    cue_count: int
    chunk_count: int


@dataclass(frozen=True)
class _AudioChunk:
    path: Path
    offset_seconds: float


def _run_command(command: list[str], *, description: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{command[0]} is required for {description}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{description} timed out") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{description} failed{suffix}") from exc


def _extract_audio(media_path: Path, output_path: Path) -> None:
    _run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "32k",
            str(output_path),
        ],
        description=f"audio extraction for {media_path.name}",
    )


def _probe_duration(audio_path: Path) -> float:
    proc = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        description=f"duration probe for {audio_path.name}",
    )
    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned an invalid duration for {audio_path.name}") from exc
    if duration <= 0:
        raise RuntimeError(f"Audio duration must be positive for {audio_path.name}")
    return duration


def _detect_silence_centers(audio_path: Path) -> list[float]:
    proc = _run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(audio_path),
            "-af",
            "silencedetect=noise=-35dB:d=0.5",
            "-f",
            "null",
            "-",
        ],
        description=f"silence detection for {audio_path.name}",
    )
    centers: list[float] = []
    silence_start: float | None = None
    for match in _SILENCE_RE.finditer(proc.stderr):
        seconds = max(0.0, float(match.group("seconds")))
        if match.group("kind") == "start":
            silence_start = seconds
        elif silence_start is not None and seconds >= silence_start:
            centers.append((silence_start + seconds) / 2)
            silence_start = None
    return centers


def _choose_chunk_boundaries(
    *,
    duration_seconds: float,
    silence_centers: list[float],
    chunk_seconds: float,
    silence_window_seconds: float,
) -> list[float]:
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if silence_window_seconds < 0:
        raise ValueError("silence_window_seconds cannot be negative")

    boundaries: list[float] = []
    cursor = 0.0
    minimum_chunk = min(60.0, chunk_seconds / 4)
    while duration_seconds - cursor > chunk_seconds:
        target = cursor + chunk_seconds
        lower = max(cursor + minimum_chunk, target - silence_window_seconds)
        upper = min(duration_seconds - minimum_chunk, target + silence_window_seconds)
        candidates = [center for center in silence_centers if lower <= center <= upper]
        boundary = (
            min(candidates, key=lambda center: abs(center - target)) if candidates else target
        )
        if boundary <= cursor or boundary >= duration_seconds:
            break
        boundaries.append(boundary)
        cursor = boundary
    return boundaries


def _split_audio(
    *,
    audio_path: Path,
    output_dir: Path,
    duration_seconds: float,
    boundaries: list[float],
) -> list[_AudioChunk]:
    size_bytes = audio_path.stat().st_size
    if not boundaries and size_bytes <= MAX_OPENAI_UPLOAD_BYTES:
        return [_AudioChunk(path=audio_path, offset_seconds=0.0)]

    points = [0.0, *boundaries, duration_seconds]
    if size_bytes > MAX_OPENAI_UPLOAD_BYTES:
        size_safe_points = [points[0]]
        for start, end in zip(points, points[1:], strict=False):
            estimated_bytes = size_bytes * (end - start) / duration_seconds
            part_count = int(estimated_bytes // MAX_OPENAI_UPLOAD_BYTES) + 1
            size_safe_points.extend(
                start + (end - start) * part / part_count for part in range(1, part_count + 1)
            )
        points = size_safe_points

    chunks: list[_AudioChunk] = []
    for index, (start, end) in enumerate(zip(points, points[1:], strict=False), start=1):
        chunk_path = output_dir / f"chunk-{index:04}.mp3"
        _run_command(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{end - start:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "32k",
                str(chunk_path),
            ],
            description=f"audio chunk {index} for {audio_path.name}",
        )
        if chunk_path.stat().st_size > MAX_OPENAI_UPLOAD_BYTES:
            raise RuntimeError(
                f"Prepared audio chunk exceeds the OpenAI upload limit: {chunk_path}"
            )
        chunks.append(_AudioChunk(path=chunk_path, offset_seconds=start))
    return chunks


def _response_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _normalize_language(language: str | None) -> str | None:
    value = (language or "").strip().lower().replace("_", "-")
    if not value:
        return None
    if value in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[value]
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)?", value):
        return value
    return None


def _parse_response(
    response: Any, *, offset_seconds: float
) -> tuple[list[TranscriptionCue], str | None]:
    cues: list[TranscriptionCue] = []
    for segment in _response_field(response, "segments", []) or []:
        text = str(_response_field(segment, "text", "") or "").strip()
        if not text:
            continue
        relative_start = float(_response_field(segment, "start", 0.0) or 0.0)
        relative_end = float(_response_field(segment, "end", relative_start) or relative_start)
        start = relative_start + offset_seconds
        end = relative_end + offset_seconds
        if end <= start:
            continue
        cues.append(TranscriptionCue(start_seconds=start, end_seconds=end, text=text))
    return cues, _normalize_language(_response_field(response, "language"))


def _format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def _render_vtt(cues: list[TranscriptionCue]) -> str:
    lines = ["WEBVTT", ""]
    ordered = sorted(cues, key=lambda item: item.start_seconds)
    for position, cue in enumerate(ordered):
        end_seconds = cue.end_seconds
        if position + 1 < len(ordered):
            end_seconds = min(
                end_seconds,
                max(cue.start_seconds, ordered[position + 1].start_seconds),
            )
        lines.extend(
            [
                f"{_format_timestamp(cue.start_seconds)} --> {_format_timestamp(end_seconds)}",
                cue.text,
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def transcribe_media(
    media_path: Path,
    output_path: Path,
    *,
    api_key: str,
    model: str = "whisper-1",
    language: str | None = None,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    silence_window_seconds: float = DEFAULT_SILENCE_WINDOW_SECONDS,
    progress_callback: Callable[[int, int], None] | None = None,
) -> OpenAITranscriptionResult:
    """Transcribe media to timestamped VTT in its original language."""
    if not api_key.strip():
        raise ValueError("OpenAI API key cannot be empty")
    if model != "whisper-1":
        raise ValueError("Timestamped VTT transcription currently requires whisper-1")

    with tempfile.TemporaryDirectory(prefix="rainrag-openai-upload-") as tmp:
        temp_dir = Path(tmp)
        audio_path = temp_dir / "audio.mp3"
        _extract_audio(media_path, audio_path)
        duration = _probe_duration(audio_path)
        silence_centers = _detect_silence_centers(audio_path) if duration > chunk_seconds else []
        boundaries = _choose_chunk_boundaries(
            duration_seconds=duration,
            silence_centers=silence_centers,
            chunk_seconds=chunk_seconds,
            silence_window_seconds=silence_window_seconds,
        )
        chunks = _split_audio(
            audio_path=audio_path,
            output_dir=temp_dir,
            duration_seconds=duration,
            boundaries=boundaries,
        )

        client = OpenAI(api_key=api_key, timeout=600.0, max_retries=2)
        all_cues: list[TranscriptionCue] = []
        language_durations: defaultdict[str, float] = defaultdict(float)
        prompt: str | None = None
        language_hint = _normalize_language(language)

        for index, chunk in enumerate(chunks, start=1):
            with chunk.path.open("rb") as audio_file:
                kwargs: dict[str, Any] = {
                    "file": audio_file,
                    "model": model,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                    "temperature": 0,
                }
                if language_hint:
                    kwargs["language"] = language_hint
                if prompt:
                    kwargs["prompt"] = prompt
                response = client.audio.transcriptions.create(**kwargs)

            cues, detected_language = _parse_response(
                response,
                offset_seconds=chunk.offset_seconds,
            )
            if cues:
                all_cues.extend(cues)
                if detected_language:
                    chunk_duration = max(cue.end_seconds for cue in cues) - chunk.offset_seconds
                    language_durations[detected_language] += chunk_duration
                prompt = " ".join(cue.text for cue in cues)[-800:] or None
            if progress_callback:
                progress_callback(index, len(chunks))

        if not all_cues:
            raise RuntimeError("OpenAI returned no timestamped cues for the uploaded video")

        detected = language_hint
        if not detected and language_durations:
            detected = max(language_durations, key=lambda code: language_durations[code])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output_path.with_suffix(f"{output_path.suffix}.tmp")
        temporary_output.write_text(_render_vtt(all_cues), encoding="utf-8")
        temporary_output.replace(output_path)

    return OpenAITranscriptionResult(
        language=detected,
        duration_seconds=duration,
        cue_count=len(all_cues),
        chunk_count=len(chunks),
    )
