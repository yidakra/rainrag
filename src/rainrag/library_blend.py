"""One blended shortlist above the two columns, weighted the way Varya asked.

ClickUp 86cbgymc5, in her words: «сначала показывать топ-5, где должны быть
самые сильные кандидаты по обеим категориям», with the weights «спикер 40,
темы 30, аудитория 20, глубина смотрения 5, cpm 5».

The two existing columns stay. They answer "who else speaks about this" and
"what else is about this", which are different questions an editor asks
separately. This adds the one list that answers "what should I watch first".

**Why the weights are renormalised rather than applied flat.** Analytics exist
for the couple of hundred episodes already published on the Library and for
nothing else in a 13,808-episode archive. Scoring a missing axis as zero would
put every unpublished episode below every published one on 30 points it had no
way to earn, which inverts the whole point: the archive is where the unwatched
material is. So an axis counts only when *both* episodes carry the data, and
the score is divided by the weight actually in play. A pair sharing only
speaker and theme is scored out of 70, not out of 100.

The trade that buys: a candidate with poor audience overlap is penalised while
one with no audience data is not. That is the right way round for this job,
where a missing measurement must never read as a bad one, but it does mean the
top five will tilt towards analytics-bearing episodes as coverage grows. Revisit
when the metrics file stops being nearly empty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rainrag.library_similar import Episode, normalise_person, normalise_tag


# Varya's opening weights. She expects to move them: «Потом, возможно,
# подкорректируем.»
WEIGHTS = {
    "speaker": 40.0,
    "theme": 30.0,
    "audience": 20.0,
    "depth": 5.0,
    "cpm": 5.0,
}

AXIS_LABELS = {
    "ru": {"audience": "аудитория", "depth": "глубина смотрения", "cpm": "CPM"},
    "en": {"audience": "audience", "depth": "watch depth", "cpm": "CPM"},
}
_LEAD = {
    "ru": {"speaker": "тот же спикер: ", "theme": "общие темы: ", "none": "нет пересечений"},
    "en": {"speaker": "same speaker: ", "theme": "shared subjects: ", "none": "no overlap"},
}


@dataclass(frozen=True)
class Audience:
    """What YouTube Analytics knows about one upload.

    Every field is optional and independently so: a channel without
    monetisation returns retention but no CPM, and the demographics report has
    to be fetched one video at a time, so it lags the rest.
    """

    age_gender: dict[str, float] | None = None
    average_view_duration: float | None = None
    playback_based_cpm: float | None = None


@dataclass
class Blended:
    """A candidate in the shortlist, with the reason split by axis."""

    episode: Episode
    score: float
    axes: dict[str, float] = field(default_factory=dict)
    shared_speakers: list[str] = field(default_factory=list)
    shared_subjects: list[str] = field(default_factory=list)

    def explain(self, lang: str = "ru") -> str:
        """Name only the axes that actually contributed.

        A row can reach the shortlist on theme weight alone. Saying "тот же
        спикер" then would be false, and this feature has already produced
        four variants of exactly that mistake.
        """
        lead = _LEAD.get(lang, _LEAD["ru"])
        labels = AXIS_LABELS.get(lang, AXIS_LABELS["ru"])
        parts: list[str] = []
        if self.axes.get("speaker") and self.shared_speakers:
            parts.append(lead["speaker"] + ", ".join(self.shared_speakers))
        if self.axes.get("theme") and self.shared_subjects:
            parts.append(lead["theme"] + ", ".join(self.shared_subjects[:5]))
        for axis in ("audience", "depth", "cpm"):
            if self.axes.get(axis):
                parts.append(f"{labels[axis]}: {self.axes[axis]:.0%}")
        return "; ".join(parts) or lead["none"]


def speaker_axis(seed: Episode, candidate: Episode) -> tuple[float | None, list[str]]:
    """Share of the seed's speakers who also speak in the candidate.

    None when the seed has nobody credited: there is no question to answer,
    which is not the same as answering it with a zero. 832 episodes are in
    that state and they must not be pushed down the list for it.
    """
    seed_keys = {normalise_person(s) for s in seed.speakers if normalise_person(s)}
    if not seed_keys:
        return None, []
    cand_keys = {normalise_person(s) for s in candidate.speakers if normalise_person(s)}
    shared = seed_keys & cand_keys
    names = [s for s in candidate.speakers if normalise_person(s) in shared]
    return len(shared) / len(seed_keys), names


def theme_axis(
    seed: Episode, candidate: Episode, idf: dict[str, float]
) -> tuple[float | None, list[str]]:
    """Share of the seed's distinctive subject weight present in the candidate.

    Same normalisation the two columns already use, so the shortlist and the
    «Похожие темы» column cannot disagree about what counts as a theme match.
    """
    seed_keys = {normalise_tag(t) for t in seed.subject if normalise_tag(t)}
    if not seed_keys:
        return None, []
    cand_keys = {normalise_tag(t) for t in candidate.subject if normalise_tag(t)}
    shared = seed_keys & cand_keys
    total = sum(idf.get(t, 0.0) for t in seed_keys)
    if total <= 0:
        return None, []
    score = sum(idf.get(t, 0.0) for t in shared) / total
    names = sorted(
        (t for t in candidate.subject if normalise_tag(t) in shared),
        key=lambda t: -idf.get(normalise_tag(t), 0.0),
    )
    return score, names


def _cosine(left: dict[str, float], right: dict[str, float]) -> float | None:
    """Cosine between two distributions.

    Disjoint keys give 0.0, not None. Two audiences that share no age band at
    all are maximally *unlike*, which is a measurement; returning None would
    file it as "not measured" and hand the candidate the same free pass as an
    episode with no analytics. That distinction is the whole basis of the
    renormalisation, so it has to hold here too.

    None is reserved for a vector that carries no signal, an all-zero mix,
    where the cosine is undefined rather than zero.
    """
    ln = math.sqrt(sum(v * v for v in left.values()))
    rn = math.sqrt(sum(v * v for v in right.values()))
    if ln <= 0 or rn <= 0:
        return None
    dot = sum(left[k] * right[k] for k in set(left) & set(right))
    return max(0.0, min(1.0, dot / (ln * rn)))


def audience_axis(seed: Audience | None, candidate: Audience | None) -> float | None:
    """How alike the two age-and-gender mixes are."""
    if seed is None or candidate is None or not seed.age_gender or not candidate.age_gender:
        return None
    return _cosine(seed.age_gender, candidate.age_gender)


def _ratio(left: float | None, right: float | None) -> float | None:
    """Scale-free closeness of two positive measurements, 1.0 when equal.

    A ratio rather than a difference because the quantities have no natural
    scale in common: 40 seconds apart is a lot for a short and nothing for a
    lecture, while "within 10%" means the same thing for both.
    """
    if left is None or right is None:
        return None
    if left <= 0 or right <= 0:
        return None
    return min(left, right) / max(left, right)


def blend_pair(
    seed: Episode,
    candidate: Episode,
    idf: dict[str, float],
    seed_audience: Audience | None = None,
    candidate_audience: Audience | None = None,
    weights: dict[str, float] | None = None,
) -> Blended:
    """Score one pair over every axis that both episodes can answer."""
    weights = weights or WEIGHTS
    speaker, speaker_names = speaker_axis(seed, candidate)
    theme, theme_names = theme_axis(seed, candidate, idf)
    scores: dict[str, float | None] = {
        "speaker": speaker,
        "theme": theme,
        "audience": audience_axis(seed_audience, candidate_audience),
        "depth": _ratio(
            seed_audience.average_view_duration if seed_audience else None,
            candidate_audience.average_view_duration if candidate_audience else None,
        ),
        "cpm": _ratio(
            seed_audience.playback_based_cpm if seed_audience else None,
            candidate_audience.playback_based_cpm if candidate_audience else None,
        ),
    }
    present = {axis: value for axis, value in scores.items() if value is not None}
    available = sum(weights.get(axis, 0.0) for axis in present)
    total = (
        sum(weights.get(axis, 0.0) * value for axis, value in present.items()) / available
        if available > 0
        else 0.0
    )
    return Blended(
        episode=candidate,
        score=total,
        axes=present,
        shared_speakers=speaker_names,
        shared_subjects=theme_names,
    )


def blended_top(
    seed: Episode,
    candidates: list[Episode],
    idf: dict[str, float],
    audiences: dict[str, Audience] | None = None,
    limit: int = 5,
    weights: dict[str, float] | None = None,
) -> list[Blended]:
    """The shortlist: best candidates across all axes, strongest first.

    ``audiences`` is keyed on ``video_hash``. Candidates that overlap the seed
    on nothing at all are dropped rather than padding the list to ``limit``:
    five rows an editor cannot use are worse than two they can.
    """
    audiences = audiences or {}
    seed_audience = audiences.get(seed.video_hash)
    scored = [
        blend_pair(
            seed,
            candidate,
            idf,
            seed_audience,
            audiences.get(candidate.video_hash),
            weights,
        )
        for candidate in candidates
    ]
    usable = [b for b in scored if b.score > 0]
    # Ties broken by video_hash so two runs over the same pool agree; set
    # iteration order otherwise decides, and it varies with PYTHONHASHSEED.
    usable.sort(key=lambda b: (-b.score, b.episode.video_hash))
    return usable[:limit]
