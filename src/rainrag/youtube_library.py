"""Link «Библиотека Дождя» YouTube uploads back to the archive episodes they came from.

Every query Varya wrote starts from a published video — «вот пример видео
[RohuZGgpC_k]», «топ-10 по playbackBasedCpm», «исходя из популярности спикеров
на YouTube» — and none of them can be answered without knowing which archive
episode each upload *is*. Her sheet has a `youtube_id` column with three rows
filled; the other 232 are not written down anywhere.

**They cannot be recovered reliably, and that is the finding.** Held out against
the three pairs Varya linked by hand, automated matching gets 1 of 3 -- and the
two failures are confidently wrong, one scoring 0.90 for the wrong episode. The
channel rewrites titles for search («Дмитрий Быков о романе Владимира Сорокина
"Сердца Четырех"» for an archive record called «Владимир Сорокин "Сердца
Четырех", 1991 год»), and runtime cannot separate two lectures by the same
speaker that both run 42 minutes. Title alone linked 8 of 235 uploads; adding
runtime reached 27, but precision is what matters here and it is not there.

So this module does not pretend to produce the mapping. It produces a **review
queue**: every upload, its best candidate and a confidence, written to CSV so an
editor confirms or corrects rather than building the list from nothing. A wrong
link would silently corrupt every recommendation built on top of it, which costs
more than the hours saved by guessing.

Public data only — the Data API with an API key. Views and CPM per video need
OAuth as the channel owner and are a separate concern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import httpx


YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
LIBRARY_CHANNEL_ID = "UCj0rOlAiR8FWos5YIB6gfow"

# Above this, one archive episode is the obvious source. Below, a human looks.
# Set from the observed score distribution rather than taste: real matches
# cluster high because the Library reuses archive titles nearly verbatim, and
# the gap to the next-best candidate is what separates a match from a coincidence.
STRONG_MATCH = 0.82
MARGIN_OVER_RUNNER_UP = 0.06

# Duration is the load-bearing signal, not the title. The Library republishes
# the episode unedited and rewrites the title for search -- «Дмитрий Быков о
# романе Владимира Сорокина "Сердца Четырех"» for an archive record called
# «Владимир Сорокин "Сердца Четырех", 1991 год». On the three pairs Varya
# linked by hand, runtimes agree to within 12 seconds.
DURATION_TOLERANCE_SECONDS = 45.0

# Boilerplate the Library adds when republishing. Left in, it drags every
# similarity score toward the mean and flattens the distinctions that matter.
_TITLE_NOISE = re.compile(
    r"\b(дождь|телеканал дождь|библиотека дождя|полная версия|полный выпуск"
    r"|архив|выпуск от \d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalise_title(title: str) -> str:
    """Reduce a title to comparable words.

    Case, punctuation, ё/е and the channel's own boilerplate all differ between
    the CMS and the YouTube upload while meaning nothing.
    """
    text = str(title or "").lower().replace("ё", "е")
    text = _TITLE_NOISE.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def title_similarity(a: str, b: str) -> float:
    """0..1 similarity between two titles.

    Combines sequence similarity with word overlap: the first catches small
    rewordings, the second survives reordering and truncation, which is how
    the Library's titles usually differ from the archive's.
    """
    na, nb = normalise_title(a), normalise_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    seq = SequenceMatcher(None, na, nb).ratio()
    wa, wb = set(na.split()), set(nb.split())
    overlap = len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
    return max(seq, overlap, (seq + overlap) / 2)


@dataclass
class YouTubeVideo:
    """One upload on the Library channel, as the public API sees it."""

    video_id: str
    title: str
    published_at: str | None = None
    duration_seconds: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    description: str = ""


@dataclass
class Match:
    """A proposed link between an upload and an archive episode."""

    video_id: str
    youtube_title: str
    content_id: str | None
    archive_title: str | None
    score: float
    runner_up: float
    confidence: str = field(default="none")

    @property
    def is_confident(self) -> bool:
        return self.confidence in ("exact", "strong")


def parse_iso8601_duration(value: str) -> float | None:
    """Seconds from YouTube's PT#H#M#S duration format."""
    match = re.fullmatch(
        r"P(?:(?P<d>\d+)D)?T?(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?", value or ""
    )
    if not match:
        return None
    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    total = parts["d"] * 86400 + parts["h"] * 3600 + parts["m"] * 60 + parts["s"]
    return float(total) or None


def fetch_channel_videos(
    api_key: str,
    channel_id: str = LIBRARY_CHANNEL_ID,
    *,
    client: Any | None = None,
    max_pages: int = 40,
) -> list[YouTubeVideo]:
    """Every upload on a channel, with public statistics.

    Walks the uploads playlist rather than using search, which caps out and
    silently omits older videos. Two calls per page: the playlist gives ids,
    the videos endpoint gives duration and view counts.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=30)
    try:
        info = http.get(
            f"{YOUTUBE_API}/channels",
            params={"part": "contentDetails", "id": channel_id, "key": api_key},
        ).json()
        items = info.get("items") or []
        if not items:
            raise RuntimeError(f"channel {channel_id} not found or not public")
        uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        video_ids: list[str] = []
        page_token = None
        for _ in range(max_pages):
            params = {
                "part": "contentDetails",
                "playlistId": uploads,
                "maxResults": 50,
                "key": api_key,
            }
            if page_token:
                params["pageToken"] = page_token
            page = http.get(f"{YOUTUBE_API}/playlistItems", params=params).json()
            video_ids += [
                i["contentDetails"]["videoId"]
                for i in page.get("items", [])
                if i.get("contentDetails", {}).get("videoId")
            ]
            page_token = page.get("nextPageToken")
            if not page_token:
                break

        videos: list[YouTubeVideo] = []
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start : start + 50]
            details = http.get(
                f"{YOUTUBE_API}/videos",
                params={
                    "part": "snippet,contentDetails,statistics",
                    "id": ",".join(batch),
                    "key": api_key,
                },
            ).json()
            for item in details.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                videos.append(
                    YouTubeVideo(
                        video_id=item["id"],
                        title=snippet.get("title", ""),
                        published_at=(snippet.get("publishedAt") or "")[:10] or None,
                        duration_seconds=parse_iso8601_duration(
                            item.get("contentDetails", {}).get("duration", "")
                        ),
                        view_count=int(stats["viewCount"]) if stats.get("viewCount") else None,
                        like_count=int(stats["likeCount"]) if stats.get("likeCount") else None,
                        description=snippet.get("description", "")[:2000],
                    )
                )
        return videos
    finally:
        if owns_client:
            http.close()


def build_title_index(archive: list[tuple[str, str]]) -> dict[str, set[int]]:
    """Word -> positions in ``archive``, for cheap candidate lookup.

    Scoring every upload against all 103k archive titles is 24M sequence
    comparisons and takes tens of minutes. Real matches always share several
    words, so a word index narrows each upload to a few dozen candidates first
    and the expensive comparison only runs on those.
    """
    index: dict[str, set[int]] = {}
    for position, (_content_id, title) in enumerate(archive):
        for word in set(normalise_title(title).split()):
            if len(word) > 3:  # short words match everything and narrow nothing
                index.setdefault(word, set()).add(position)
    return index


def candidate_positions(
    title: str, index: dict[str, set[int]], *, max_candidates: int = 400
) -> list[int]:
    """Archive positions worth scoring against this title.

    Ranked by how many words they share, so the cap keeps the most promising
    candidates rather than an arbitrary slice.
    """
    words = [w for w in set(normalise_title(title).split()) if len(w) > 3]
    if not words:
        return []
    counts: dict[int, int] = {}
    for word in words:
        for position in index.get(word, ()):  # noqa: SIM118 - set membership iteration
            counts[position] = counts.get(position, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [position for position, _shared in ranked[:max_candidates]]


def duration_agreement(a: float | None, b: float | None) -> float:
    """1.0 for runtimes that agree, falling to 0 past the tolerance.

    Returns 0.5 when either side is unknown -- absent evidence must neither
    confirm a match nor veto one.
    """
    if not a or not b:
        return 0.5
    delta = abs(a - b)
    if delta <= DURATION_TOLERANCE_SECONDS:
        return 1.0
    return max(0.0, 1.0 - (delta - DURATION_TOLERANCE_SECONDS) / 300.0)


def match_video(
    video: YouTubeVideo,
    archive: list[tuple[str, str]],
    *,
    known: dict[str, str] | None = None,
    index: dict[str, set[int]] | None = None,
    durations: dict[str, float] | None = None,
) -> Match:
    """Find the archive episode an upload came from.

    ``archive`` is (content_id, title). ``known`` is the mapping Varya has
    already written by hand, which always wins: an editor's link is a fact and
    must never be second-guessed by string similarity.

    Confidence is not just the top score. A title that scores 0.85 against two
    different episodes is ambiguous, not strong -- so the margin over the
    runner-up decides too, and near-ties are demoted for review.
    """
    if known and video.video_id in known:
        content_id = known[video.video_id]
        title = next((t for c, t in archive if c == content_id), None)
        return Match(video.video_id, video.title, content_id, title, 1.0, 0.0, "editor")

    pool = (
        [archive[i] for i in candidate_positions(video.title, index)]
        if index is not None
        else archive
    )

    def combined(content_id: str, title: str) -> float:
        title_score = title_similarity(video.title, title)
        if durations is None:
            return title_score
        agreement = duration_agreement(video.duration_seconds, durations.get(content_id))
        # A runtime that disagrees vetoes; one that agrees lifts a retitled
        # episode over a coincidental word overlap. Titles alone linked 8 of
        # 235 uploads, because the channel rewrites them.
        if agreement == 0.0:
            return 0.0
        return 0.45 * title_score + 0.55 * agreement

    scored = sorted(((combined(c, t), c, t) for c, t in pool), key=lambda x: -x[0])
    if not scored:
        return Match(video.video_id, video.title, None, None, 0.0, 0.0, "none")

    best, content_id, archive_title = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0

    if best >= 0.995:
        confidence = "exact"
    elif best >= STRONG_MATCH and (best - runner_up) >= MARGIN_OVER_RUNNER_UP:
        confidence = "strong"
    elif best >= 0.6:
        confidence = "review"
    else:
        confidence = "none"
        content_id, archive_title = None, None
    return Match(
        video.video_id, video.title, content_id, archive_title, best, runner_up, confidence
    )
