"""Subtitle format helpers.

The archive holds WebVTT (livevtt output). A client that wants SRT gets the
same cues renumbered, with comma decimal marks and hour fields, and without
the WebVTT header, NOTE and STYLE blocks, or cue settings.
"""

from __future__ import annotations

import re


_TIMESTAMP = re.compile(r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})")
_ARROW = re.compile(r"^(\S+)\s+-->\s+(\S+)(.*)$")


def _srt_time(ts: str) -> str | None:
    m = _TIMESTAMP.fullmatch(ts.strip())
    if not m:
        return None
    hours = int(m.group(1) or 0)
    return f"{hours:02d}:{m.group(2)}:{m.group(3)},{m.group(4)}"


def vtt_to_srt(text: str) -> str:
    """Convert WebVTT text to SRT. Cues without a readable timing line are dropped."""
    blocks = re.split(r"\r?\n\s*\r?\n", text.strip().replace("﻿", ""))
    out: list[str] = []
    n = 0
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        head = lines[0].strip()
        if head.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        timing_index = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if timing_index is None:
            continue
        m = _ARROW.match(lines[timing_index].strip())
        if not m:
            continue
        start, end = _srt_time(m.group(1)), _srt_time(m.group(2))
        if start is None or end is None:
            continue
        cue_text = lines[timing_index + 1 :]
        if not cue_text:
            continue
        n += 1
        out.append(f"{n}\n{start} --> {end}\n" + "\n".join(cue_text))
    return "\n\n".join(out) + ("\n" if out else "")
