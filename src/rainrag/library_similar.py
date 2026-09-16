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

from rainrag.library_programs import Programme, programme_for, resolve_speakers


# A speaker in common is the strongest signal an editor uses -- four of the six
# expected results are simply "the same person again" -- but it must not become
# the only signal, or the two theme-matched interviews could never surface.
SPEAKER_WEIGHT = 3.0


def normalise_tag(tag: str) -> str:
    """Fold a tag to a comparable form: lowercase, no punctuation, no ё."""
    return re.sub(r"[^\w\s-]", "", str(tag).lower().replace("ё", "е")).strip()


def person_key(name: str) -> tuple[str, str]:
    """Split a name into (surname, given name); given name is "" when absent.

    Surname-last is the convention in both the CMS and the titles, and the
    surname is the part that stays constant across «Ирина Хакамада» and
    «Хакамада». The surname took the *longest* token until review caught it,
    which silently broke matching for anyone whose given name is longer than
    their surname: «Екатерина Шульман» folded to «екатерина». Шульман is one
    of the six results the ranking is measured against.

    The surname is the last token longer than two letters, so a latin suffix
    («Noize MC») cannot become one. The given name is whatever stands before
    it, which may be an initial: «И. Хакамада» is (хакамада, и), not
    (хакамада, ""). Keeping the initial matters, because "no given name at
    all" is what lets two spellings match, and an initial is a real signal
    that the surname is shared with someone else. A patronymic sits in the
    middle and is ignored: «Владимир Вольфович Жириновский» is
    (жириновский, владимир).
    """
    tokens = re.sub(r"[^\w\s-]", " ", str(name).lower().replace("ё", "е")).split()
    long_tokens = [t for t in tokens if len(t) > 2]
    if not long_tokens:
        return "", ""
    surname = long_tokens[-1]
    before = tokens[: len(tokens) - 1 - tokens[::-1].index(surname)]
    return surname, before[0] if before else ""


def normalise_person(name: str) -> str:
    """The surname alone, used to *group* spellings of one name.

    Not a test of whether two names denote the same person: «Дмитрий Быков»
    and «Юрий Быков» share this key. Use `people_match` for that.
    """
    return person_key(name)[0]


def given_names_agree(left: str, right: str) -> bool:
    """Can these two given names belong to one person?

    An absent given name agrees with anything: the model credits a bare
    «Хакамада» where the CMS has «Ирина Хакамада». An *initial* is not
    absent, and treating it as such reopened the bug this guard exists for,
    because «Д. Быков» then matched «Юрий Быков» (Tenki on #87). An initial
    agrees with a name that starts with it and with nothing else.
    """
    if not left or not right:
        return True
    if len(left) == 1 or len(right) == 1:
        return left[0] == right[0]
    return left == right


def people_match(left: str, right: str) -> bool:
    """Do two spellings denote the same person?

    Surnames must agree. Given names must agree too, unless one side has none
    at all: the model routinely returns a bare «Хакамада» where the CMS has
    «Ирина Хакамада», and refusing that match would lose most real overlap.

    Matching on the surname alone was the original rule and it suggested Юрий
    Быков for Дмитрий Быков, and Алиса Ахеджакова for Лия Ахеджакова, in the
    «тот же спикер» column, where being the same person is the entire claim
    (reported by Varya, 2026-09-15).
    """
    left_surname, left_given = person_key(left)
    right_surname, right_given = person_key(right)
    if not left_surname or left_surname != right_surname:
        return False
    return given_names_agree(left_given, right_given)


def _identify(names: Iterable[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str] | None]]:
    """(the distinct people named, the person each spelling refers to).

    Spellings of one person collapse: «Хакамада», «И. Хакамада» and «Ирина
    Хакамада» are one identity, because a bare surname or an initial is
    absorbed by a full name it agrees with. Two people who merely share a
    surname stay two, which is the whole point -- counting them as one let
    `speaker_axis` report a full match when half the seed's speakers were
    someone else (CodeRabbit on #87).
    """
    names = list(names)
    keys = [person_key(n) for n in names]
    by_surname: dict[str, list[str]] = {}
    for surname, given in keys:
        if not surname:
            continue
        givens = by_surname.setdefault(surname, [])
        if not given:
            continue
        match = next((g for g in givens if given_names_agree(given, g)), None)
        if match is None:
            givens.append(given)
        elif len(given) > len(match):
            # A full name supersedes the initial it was first seen as.
            givens[givens.index(match)] = given
    identities = [
        (surname, given) for surname, givens in by_surname.items() for given in (givens or [""])
    ]

    def resolve(key: tuple[str, str]) -> tuple[str, str] | None:
        surname, given = key
        if not surname:
            return None
        fits = [i for i in identities if i[0] == surname and given_names_agree(given, i[1])]
        if not fits:
            return None
        # Prefer the identity whose given name this spelling actually carries.
        return next((i for i in fits if given and i[1] == given), fits[0])

    return identities, [resolve(k) for k in keys]


def person_identities(names: Iterable[str]) -> list[tuple[str, str]]:
    """The distinct people a list of spellings refers to."""
    return _identify(names)[0]


def shared_people(
    seed_names: Iterable[str], candidate_names: Iterable[str]
) -> tuple[list[str], list[tuple[str, str]]]:
    """(candidate spellings the seed also has, the seed identities they matched).

    The names are for the reason line and keep the candidate's own spelling.
    The identities are the *seed's*, and each side is spent once: a candidate
    crediting both «Хакамада» and «Ирина Хакамада» matches one person, and a
    seed crediting two different Быковы is only half matched by a candidate
    with one of them.

    One spelling cannot be two people. A bare «Быков» agrees with every
    Быков in the seed, so without the spend-once rule it earned full credit
    against a seed naming two of them (Tenki on #87). Unambiguous pairs are
    taken first, so a candidate naming Дмитрий outright is never consumed by
    a bare surname that could have gone elsewhere.
    """
    candidate_names = list(candidate_names)
    seed_ids = person_identities(seed_names)
    cand_ids, cand_of_name = _identify(candidate_names)

    pairs = sorted(
        # Specific pairs (both sides name a person) sort before ambiguous ones.
        (0 if seed_given and cand_given else 1, s_index, c_index)
        for s_index, (seed_surname, seed_given) in enumerate(seed_ids)
        for c_index, (cand_surname, cand_given) in enumerate(cand_ids)
        if seed_surname == cand_surname and given_names_agree(seed_given, cand_given)
    )

    used_seed: set[int] = set()
    used_cand: set[int] = set()
    for _, s_index, c_index in pairs:
        if s_index in used_seed or c_index in used_cand:
            continue
        used_seed.add(s_index)
        used_cand.add(c_index)

    matched = [seed_ids[i] for i in sorted(used_seed)]
    matched_cand = {cand_ids[i] for i in used_cand}
    names = [
        name
        for name, identity in zip(candidate_names, cand_of_name, strict=True)
        if identity in matched_cand
    ]
    return names, matched


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
    presenter_demoted: bool = False
    # Genre as Varya's Programs tab defines it for this episode's programme,
    # empty when the programme is absent from the table or she left it blank.
    # `genre` above is the model's per-episode guess and they are different
    # things: see `filter_genres`.
    programme_genres: list[str] = field(default_factory=list)

    @classmethod
    def from_record(
        cls, record: dict[str, Any], programmes: dict[str, Programme] | None = None
    ) -> Episode:
        """Build from a tagging-run JSONL row.

        Speakers merge the CMS presenter with the model's `guest`: for a
        lecture the CMS field is the lecturer, for an interview the guest is
        the one the search is actually about, and neither alone is enough.

        With a programme table the editorial rule applies on top and drops the
        presenter for the genres where they interview rather than speak. See
        `library_programs.resolve_speakers`. Passing nothing keeps the old
        behaviour, so existing callers rank exactly as before.
        """
        resolution = resolve_speakers(record, programmes)
        speakers = resolution.speakers
        programme = programme_for(record.get("program"), programmes or {})
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
            presenter_demoted=resolution.presenter_demoted,
            programme_genres=list(programme.genres) if programme else [],
        )


def episode_identity(episode: Episode) -> str:
    """The key under which two records are the same episode.

    ``video_hash``, not ``content_id``: eight content_ids in the tagged pool
    have two *different* hashes with identical titles but runtimes differing by
    up to 33%, and whether those are duplicate ingests or genuinely different
    cuts is an editorial question. Collapsing them here would hide archive
    content from a shortlist, which is the opposite of what the Library needs.
    """
    return episode.video_hash


def dedupe_latest(episodes: Iterable[Episode]) -> list[Episode]:
    """Collapse repeated records of one episode, keeping the last.

    ``library_tags.jsonl`` is opened in append mode, so tagging an episode
    again adds a row rather than replacing one -- three episodes in the current
    pool carry two successful rows each. Ranking over the raw file returns the
    same episode twice: it burns a slot on a shortlist an editor reads by eye,
    and the two rows disagree on exactly the fields used as hard filters
    (``genre`` and ``duration_seconds``), so which one wins must not be left to
    scoring order.

    Last wins, because rows are appended in the order they were produced and
    the later one reflects the newer run. That also repairs bad metadata: the
    superseded record for 484740 claims a duration of 195180 seconds -- 54
    hours -- against the 3253 seconds actually computed from its transcript.
    """
    latest: dict[str, Episode] = {}
    for episode in episodes:
        latest[episode_identity(episode)] = episode
    return list(latest.values())


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
    shared_speakers, shared_speaker_keys = shared_people(seed.speakers, candidate.speakers)

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


def filter_genres(episode: Episode) -> set[str]:
    """The genres an episode is filtered on: the programme's, else the model's.

    Varya's Programs tab is the editorial authority on genre -- it is already
    what decides whether a presenter counts as a speaker -- and the model's
    per-episode labels are a different, noisier thing. The model gives an
    episode several genres, so «Утро на Дожде» tagged «новости, интервью»
    survived a filter of «интервью» even though the programme is news: 114
    such episodes in that programme alone, and 119 in «Здесь и сейчас»
    (reported by Varya, 2026-09-15).

    Falling back to the model's labels rather than dropping the episode keeps
    the 123 tagged episodes whose programme has no reviewed genre, and
    everything with no programme at all, reachable through the filter.
    """
    source = episode.programme_genres or episode.genre
    return {normalise_tag(g) for g in source if normalise_tag(g)}


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
    # Identity, not object equality: a re-tagged seed would otherwise rank
    # first against itself as a perfect match.
    seed_id = episode_identity(seed)
    pool = [c for c in dedupe_latest(candidates) if episode_identity(c) != seed_id]
    if idf is None:
        idf = subject_idf([seed, *pool])

    wanted_genres = {normalise_tag(g) for g in genres} if genres else None
    results: list[Scored] = []
    for candidate in pool:
        too_short = (
            min_duration_minutes is not None
            and (candidate.duration_seconds or 0) < min_duration_minutes * 60
        )
        wrong_genre = bool(wanted_genres) and not (filter_genres(candidate) & wanted_genres)
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
