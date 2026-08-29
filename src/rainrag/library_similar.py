"""Rank archive episodes by similarity to a seed, the way an editor would.

This answers Varya's first query — «найди похожие по темам и спикерам видео
длиной от 30 минут в жанре лекции или интервью» — and its shape is taken from
the answers she wrote by hand rather than invented:

    454556, 454501, 77645   лекции Хакамады          same speaker
    449964                  интервью с Хакамадой     same speaker, other genre
    484740                  интервью с Шульман       «пересекаются темы политики
                                                      и женского лидерства»
    431298                  интервью с Полозковой    «политики, саморазвития,
                                                      женского лидерства,
                                                      писательства»

Two signals, then, and in that order: the same person speaking, and overlapping
subjects. Note what the second pair shows — she accepts an episode with a
different speaker when several *distinctive* subjects line up. «женщины-лидеры»
is doing the work there, not «политика», which half the archive is about. So
subject overlap is weighted by rarity: a tag shared by two episodes out of five
hundred says far more than one shared by three hundred.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


# A speaker in common is the strongest signal an editor uses -- four of the six
# expected results are simply "the same person again" -- but it must not become
# the only signal, or the two theme-matched interviews could never surface.
SPEAKER_WEIGHT = 3.0


def normalise_tag(tag: str) -> str:
    """Fold a tag to a comparable form: lowercase, no punctuation, no ё."""
    return re.sub(r"[^\w\s-]", "", str(tag).lower().replace("ё", "е")).strip()


def normalise_person(name: str) -> str:
    """Fold a person's name for comparison.

    Surname-last is the convention in both the CMS and the titles, and the
    surname is the part that stays constant across «Ирина Хакамада» and
    «Хакамада» -- so names compare on their longest token.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", str(name).lower().replace("ё", "е"))
    tokens = [t for t in cleaned.split() if len(t) > 2]
    return max(tokens, key=len) if tokens else ""


@dataclass
class Episode:
    """One tagged episode, as the ranker sees it."""

    video_hash: str
    content_id: str | None = None
    title: str | None = None
    program: str | None = None
    date: str | None = None
    duration_seconds: float | None = None
    genre: list[str] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    url: str | None = None

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Episode:
        """Build from a tagging-run JSONL row.

        Speakers merge the CMS presenter with the model's `guest`: for a
        lecture the CMS field is the lecturer, for an interview the guest is
        the one the search is actually about, and neither alone is enough.
        """
        speakers = list(record.get("presenter_cms") or []) + list(record.get("guest") or [])
        return cls(
            video_hash=record["video_hash"],
            content_id=record.get("content_id"),
            title=record.get("title"),
            program=record.get("program"),
            date=record.get("date"),
            duration_seconds=record.get("duration_seconds"),
            genre=list(record.get("genre") or []),
            subject=list(record.get("subject") or []),
            speakers=speakers,
            url=record.get("url"),
        )


@dataclass
class Scored:
    """An episode with its score and the reason for it, for display."""

    episode: Episode
    score: float
    shared_speakers: list[str]
    shared_subjects: list[str]

    def explain(self) -> str:
        bits = []
        if self.shared_speakers:
            bits.append("тот же спикер: " + ", ".join(self.shared_speakers))
        if self.shared_subjects:
            bits.append("общие темы: " + ", ".join(self.shared_subjects[:6]))
        return "; ".join(bits) or "нет пересечений"


def subject_idf(episodes: Iterable[Episode]) -> dict[str, float]:
    """Inverse document frequency over subject tags.

    «политика» appears on a large share of a news archive and separates
    nothing; «теория пустоты» appears twice and separates everything. Without
    this weighting the ranking is dominated by whichever broad tags the model
    happens to emit most often.
    """
    episodes = list(episodes)
    counts: Counter[str] = Counter()
    for ep in episodes:
        for tag in {normalise_tag(t) for t in ep.subject if normalise_tag(t)}:
            counts[tag] += 1
    total = max(len(episodes), 1)
    return {tag: math.log(total / count) for tag, count in counts.items()}


def score_pair(
    seed: Episode, candidate: Episode, idf: dict[str, float]
) -> tuple[float, list[str], list[str]]:
    """Score one candidate against the seed, returning the reasons too.

    The reasons matter as much as the number: an editor deciding whether to
    spend an hour watching a tape wants to know *why* it was suggested.
    """
    seed_speakers = {normalise_person(s) for s in seed.speakers if normalise_person(s)}
    cand_speakers = {normalise_person(s) for s in candidate.speakers if normalise_person(s)}
    shared_speaker_keys = seed_speakers & cand_speakers
    shared_speakers = [s for s in candidate.speakers if normalise_person(s) in shared_speaker_keys]

    seed_subjects = {normalise_tag(t) for t in seed.subject if normalise_tag(t)}
    cand_subjects = {normalise_tag(t) for t in candidate.subject if normalise_tag(t)}
    shared_keys = seed_subjects & cand_subjects

    # Normalise by the seed's own weight so the score reads as "how much of
    # what makes this episode distinctive is also here", not "how many tags
    # does the candidate happen to have".
    seed_weight = sum(idf.get(t, 0.0) for t in seed_subjects) or 1.0
    subject_score = sum(idf.get(t, 0.0) for t in shared_keys) / seed_weight

    score = subject_score + SPEAKER_WEIGHT * len(shared_speaker_keys)

    # Show the rarest shared subjects first: those are the ones that explain
    # the match, and the ones Varya cites in her own rationale.
    shared_subjects = sorted(
        (t for t in candidate.subject if normalise_tag(t) in shared_keys),
        key=lambda t: -idf.get(normalise_tag(t), 0.0),
    )
    return score, shared_speakers, shared_subjects


def find_similar(
    seed: Episode,
    candidates: Iterable[Episode],
    *,
    min_duration_minutes: float | None = None,
    genres: Iterable[str] | None = None,
    limit: int = 10,
    idf: dict[str, float] | None = None,
) -> list[Scored]:
    """Rank candidates against a seed, applying the editor's hard filters.

    Duration and genre are filters rather than score terms because that is how
    they are asked for: "видео длиной от 30 минут в жанре лекции или интервью"
    excludes, it does not merely prefer. The seed itself is never returned.
    """
    pool = [c for c in candidates if c.video_hash != seed.video_hash]
    if idf is None:
        idf = subject_idf([seed, *pool])

    wanted_genres = {normalise_tag(g) for g in genres} if genres else None
    results: list[Scored] = []
    for candidate in pool:
        too_short = (
            min_duration_minutes is not None
            and (candidate.duration_seconds or 0) < min_duration_minutes * 60
        )
        wrong_genre = bool(wanted_genres) and not (
            {normalise_tag(g) for g in candidate.genre} & wanted_genres
        )
        if too_short or wrong_genre:
            continue
        score, speakers, subjects = score_pair(seed, candidate, idf)
        # Drop only what has nothing in common. Scoring zero is not the same
        # thing: a candidate sharing just one archive-wide tag like «политика»
        # has an IDF-weighted score of exactly 0, and silently discarding it
        # would hide real -- if weak -- overlap from the editor.
        if not speakers and not subjects:
            continue
        results.append(Scored(candidate, score, speakers, subjects))

    results.sort(key=lambda r: (-r.score, r.episode.date or ""))
    return results[:limit]
