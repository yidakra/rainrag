#!/usr/bin/env python3
"""Transcribe a single video/audio file to WebVTT using faster-whisper.

Standalone by design: it depends only on the standard library plus
``faster_whisper`` and must be run with an interpreter that has that package
installed (the livevtt virtualenv on this box:
``/home/ubuntu/livevtt/.venv/bin/python``). It deliberately does NOT import
``rainrag`` — the RainRAG environment does not ship the CUDA/ctranslate2 stack.

Transcript-only (no translation) to keep latency ~halved for the upload MVP.
Progress is written as JSON to ``--progress-file`` (atomically) so the
orchestrating process can poll it while transcription runs.

Exit code 0 on success; non-zero on failure (details on stderr and, if a
progress file was requested, in its ``stage``/``error`` fields).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


def _format_timestamp(seconds: float) -> str:
    """Format seconds as a WebVTT timestamp: HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _write_progress(progress_file: Path | None, payload: dict) -> None:
    """Atomically write a JSON progress snapshot (best-effort)."""
    if progress_file is None:
        return
    try:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(progress_file.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        Path(tmp).replace(progress_file)
    except Exception:  # progress is advisory; never fail the run over it
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe one media file to WebVTT")
    p.add_argument("input", type=Path, help="Path to the input video/audio file")
    p.add_argument("--output", type=Path, required=True, help="Output .vtt path")
    p.add_argument("--model", default="large-v3-turbo", help="faster-whisper model")
    p.add_argument(
        "--compute-type",
        default="int8_float16",
        help="ctranslate2 compute type (int8_float16 avoids OOM on large models)",
    )
    p.add_argument("--language", default="ru", help="Source language code")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    p.add_argument("--device-index", type=int, default=0, help="CUDA device index")
    p.add_argument("--beam-size", type=int, default=5, help="Decoding beam size")
    p.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Silero VAD filtering (on by default)",
    )
    p.add_argument("--progress-file", type=Path, help="JSON progress output path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress = args.progress_file

    if not args.input.exists():
        _write_progress(progress, {"stage": "error", "error": f"input not found: {args.input}"})
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    _write_progress(progress, {"stage": "loading_model", "percent": 0.0})

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - environment guard
        _write_progress(progress, {"stage": "error", "error": f"faster_whisper import failed: {exc}"})
        print(f"faster_whisper import failed: {exc}", file=sys.stderr)
        return 3

    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            device_index=args.device_index,
            compute_type=args.compute_type,
        )
    except Exception as exc:
        _write_progress(progress, {"stage": "error", "error": f"model load failed: {exc}"})
        traceback.print_exc()
        return 4

    _write_progress(progress, {"stage": "transcribing", "percent": 0.0})

    try:
        segments, info = model.transcribe(
            str(args.input),
            language=args.language,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
        )
        total = float(getattr(info, "duration", 0.0) or 0.0)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        cue_count = 0
        # faster-whisper yields segments lazily; iteration drives the actual work.
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write("WEBVTT\n\n")
            for seg in segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                fh.write(f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}\n")
                fh.write(f"{text}\n\n")
                fh.flush()
                cue_count += 1
                if total > 0:
                    pct = max(0.0, min(100.0, (float(seg.end) / total) * 100.0))
                    _write_progress(
                        progress,
                        {
                            "stage": "transcribing",
                            "percent": round(pct, 1),
                            "processed_seconds": round(float(seg.end), 1),
                            "total_seconds": round(total, 1),
                            "cues": cue_count,
                        },
                    )
    except Exception as exc:
        _write_progress(progress, {"stage": "error", "error": f"transcription failed: {exc}"})
        traceback.print_exc()
        return 5

    _write_progress(
        progress,
        {
            "stage": "done",
            "percent": 100.0,
            "total_seconds": round(total, 1),
            "cues": cue_count,
            "output": str(args.output),
        },
    )
    print(f"wrote {cue_count} cues to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
