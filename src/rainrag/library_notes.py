"""Where a topic is spoken about in an episode, with timecodes.

Varya's ask (86cbhemwp): an editor picking archive material has to watch it or
read the whole transcript to find out that a guest spends four minutes on
Ukraine in the middle of an hour. She wants that up front, as a note on the
card, «с 2:05 по 6:57 ... и с фрагментом расшифровки».

This is the deterministic half: which topics are spoken about and when. The
other half she asked for, "начало выпуска затянуто, рекомендую сократить", is
a judgement about pacing that needs a model to read the transcript, which is
a paid batch over the pool and is not run without asking.

The topic list is editorial and lives in a file she can edit. Nothing here
decides what is worth flagging; it only finds what it is told to look for.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path


# Two cues about the same topic four seconds apart are one passage, not two
# notes. Wider than a cue gap and narrower than a change of subject.
MERGE_GAP_SECONDS = 45.0

# A passage shorter than this is a mention in passing rather than something an
# editor needs to plan around.
MIN_SPAN_SECONDS = 5.0

_TIMESTAMP = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)


@dataclass(frozen=True)
class Cue:
    """One subtitle line: when it is said and what is said."""

    start: float
    end: float
    text: str


@dataclass
class Span:
    """A stretch of one episode spent on one topic."""

    topic: str
    start: float
    end: float
    quote: str = ""
    hits: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "topic": self.topic,
            "start": round(self.start, 1),
            "end": round(self.end, 1),
            "quote": self.quote,
            "hits": self.hits,
        }


@dataclass
class Topic:
    """A thing to look for, and the spellings that count as it."""

    name: str
    patterns: tuple[str, ...] = ()
    _regex: re.Pattern[str] | None = field(default=None, repr=False, compare=False)

    def matches(self, text: str) -> bool:
        if self._regex is None:
            # Word-initial rather than whole-word: Russian inflects, so
            # «Украина» has to catch «Украине» and «Украину» while «Украинец»
            # is a different word an editor did not ask about. Matching a bare
            # substring would find «Крым» inside «Крымск» too.
            joined = "|".join(re.escape(p) for p in (self.patterns or (self.name,)))
            object.__setattr__(self, "_regex", re.compile(rf"\b(?:{joined})\w{{0,3}}\b", re.I))
        assert self._regex is not None
        return bool(self._regex.search(text))


def parse_timestamp(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return (
        int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis.ljust(3, "0")) / 1000
    )


def parse_vtt(text: str) -> list[Cue]:
    """Cues from a WebVTT body, ignoring everything that is not a cue.

    Written against the archive's own files rather than the spec: they carry
    a numeric identifier line before each timing line and no styling blocks.
    Anything unrecognised is skipped rather than raising, because one odd
    file must not take out the note for every other episode.
    """
    cues: list[Cue] = []
    pending: tuple[float, float] | None = None
    said: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = _TIMESTAMP.search(stripped)
        if match:
            if pending and said:
                cues.append(Cue(pending[0], pending[1], " ".join(said).strip()))
            groups = match.groups()
            pending = (parse_timestamp(*groups[:4]), parse_timestamp(*groups[4:]))
            said = []
        elif pending is not None and stripped and not stripped.isdigit():
            said.append(stripped)
    if pending and said:
        cues.append(Cue(pending[0], pending[1], " ".join(said).strip()))
    return cues


def load_topics(path: Path) -> list[Topic]:
    """The editorial list of what is worth flagging.

    A missing file means nobody has asked for anything, which is not an
    error: no topics, no notes, and the card renders as it did before.
    """
    if not path.exists():
        return []
    topics: list[Topic] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
        for row in reader:
            name = (row.get("topic") or "").strip()
            if not name:
                continue
            spellings = [p.strip() for p in (row.get("patterns") or "").split(";") if p.strip()]
            topics.append(Topic(name=name, patterns=tuple(spellings) or (name,)))
    return topics


def find_spans(
    cues: list[Cue],
    topics: list[Topic],
    merge_gap: float = MERGE_GAP_SECONDS,
    min_span: float = MIN_SPAN_SECONDS,
) -> list[Span]:
    """Where each topic is spoken about, as merged passages.

    Merged because a subject that comes up is talked about for a while: an
    editor wants "с 19:04 по 24:01", not forty timecodes a few seconds apart.
    """
    spans: list[Span] = []
    for topic in topics:
        current: Span | None = None
        for cue in cues:
            if not topic.matches(cue.text):
                continue
            if current is not None and cue.start - current.end <= merge_gap:
                current.end = max(current.end, cue.end)
                current.hits += 1
                continue
            if current is not None:
                spans.append(current)
            current = Span(topic=topic.name, start=cue.start, end=cue.end, quote=cue.text[:300])
        if current is not None:
            spans.append(current)
    kept = [s for s in spans if s.end - s.start >= min_span or s.hits > 1]
    kept.sort(key=lambda s: (s.start, s.topic))
    return kept


def format_timecode(seconds: float) -> str:
    """m:ss, or h:mm:ss past the hour, the way the editor writes them."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
