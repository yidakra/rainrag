"""Generate timestamp-aligned English WebVTT subtitles with OpenAI Whisper."""

from __future__ import annotations

import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


DEFAULT_CHUNK_SECONDS = 30 * 60
DEFAULT_SILENCE_WINDOW_SECONDS = 30.0
MAX_OPENAI_UPLOAD_BYTES = 24 * 1024 * 1024
_TIMING_RE = re.compile(
    r"^(?P<start>(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3})\s+-->\s+"
    r"(?P<end>(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3})(?P<settings>.*)$"
)
_SILENCE_RE = re.compile(r"silence_(?P<kind>start|end):\s*(?P<seconds>\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class OpenAISubtitleJob:
    video_hash: str
    video_path: Path
    output_path: Path


@dataclass(frozen=True)
class TranslationBatchResult:
    translated_hashes: tuple[str, ...]
    failures: dict[str, str]


@dataclass(frozen=True)
class _AudioChunk:
    path: Path
    offset_seconds: float


@dataclass(frozen=True)
class _VttCue:
    start_seconds: float
    end_seconds: float
    text: str
    settings: str = ""


def _run_command(command: list[str], *, description: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{command[0]} is required for {description}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{description} failed{suffix}") from exc


def _extract_audio(video_path: Path, output_path: Path) -> None:
    _run_command(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
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
        description=f"audio extraction for {video_path.name}",
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
        seconds = float(match.group("seconds"))
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


def _timestamp_to_seconds(value: str) -> float:
    parts = value.replace(",", ".").split(":")
    if len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"Invalid WebVTT timestamp: {value!r}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_vtt(content: str, *, offset_seconds: float = 0.0) -> list[_VttCue]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    cues: list[_VttCue] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = block.strip().splitlines()
        timing_index = next(
            (index for index, line in enumerate(lines) if _TIMING_RE.match(line.strip())),
            None,
        )
        if timing_index is None:
            continue
        match = _TIMING_RE.match(lines[timing_index].strip())
        assert match is not None
        text = "\n".join(lines[timing_index + 1 :]).strip()
        if not text:
            continue
        cues.append(
            _VttCue(
                start_seconds=_timestamp_to_seconds(match.group("start")) + offset_seconds,
                end_seconds=_timestamp_to_seconds(match.group("end")) + offset_seconds,
                text=text,
                settings=match.group("settings").strip(),
            )
        )
    return cues


def _format_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def _render_vtt(cues: list[_VttCue]) -> str:
    lines = ["WEBVTT", ""]
    ordered = sorted(cues, key=lambda item: item.start_seconds)
    for position, cue in enumerate(ordered):
        end_seconds = cue.end_seconds
        if position + 1 < len(ordered):
            end_seconds = min(
                end_seconds,
                max(cue.start_seconds, ordered[position + 1].start_seconds),
            )
        timing = f"{_format_timestamp(cue.start_seconds)} --> {_format_timestamp(end_seconds)}"
        if cue.settings:
            timing = f"{timing} {cue.settings}"
        lines.extend([str(position + 1), timing, cue.text, ""])
    return "\n".join(lines).rstrip() + "\n"


def _translate_chunk(*, client: OpenAI, chunk: _AudioChunk, model: str) -> list[_VttCue]:
    with chunk.path.open("rb") as audio_file:
        response = client.audio.translations.create(
            file=audio_file,
            model=model,
            response_format="vtt",
            temperature=0,
        )
    content = response if isinstance(response, str) else str(getattr(response, "text", ""))
    return _parse_vtt(content, offset_seconds=chunk.offset_seconds)


def _translate_job(
    job: OpenAISubtitleJob,
    *,
    api_key: str,
    model: str,
    chunk_seconds: float,
    silence_window_seconds: float,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"rainrag-openai-{job.video_hash[:12]}-") as tmp:
        temp_dir = Path(tmp)
        audio_path = temp_dir / "audio.mp3"
        _extract_audio(job.video_path, audio_path)
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
        cues: list[_VttCue] = []
        for chunk in chunks:
            cues.extend(_translate_chunk(client=client, chunk=chunk, model=model))
        if not cues:
            raise RuntimeError(f"OpenAI returned no WebVTT cues for {job.video_hash}")

        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = job.output_path.with_suffix(f"{job.output_path.suffix}.tmp")
        temporary_output.write_text(_render_vtt(cues), encoding="utf-8")
        temporary_output.replace(job.output_path)


def translate_jobs(
    jobs: list[OpenAISubtitleJob],
    *,
    api_key: str,
    workers: int = 2,
    model: str = "whisper-1",
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    silence_window_seconds: float = DEFAULT_SILENCE_WINDOW_SECONDS,
) -> TranslationBatchResult:
    """Translate videos concurrently and keep per-video failures isolated."""
    if not api_key.strip():
        raise ValueError("OpenAI API key cannot be empty")

    translated: list[str] = []
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _translate_job,
                job,
                api_key=api_key,
                model=model,
                chunk_seconds=chunk_seconds,
                silence_window_seconds=silence_window_seconds,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures[job.video_hash] = str(exc)
            else:
                translated.append(job.video_hash)

    return TranslationBatchResult(
        translated_hashes=tuple(sorted(translated)),
        failures=failures,
    )
