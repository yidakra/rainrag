"""What performs on «Библиотека Дождя»: uploads joined to archive episodes.

Varya's second and third queries («топ-10 по playbackBasedCpm», «исходя из
популярности спикеров на YouTube») need one table: each Library upload, the
archive episode it came from, that episode's programme and speakers, and how
the upload did. The mapping now exists (211 editor pairs plus the matcher's
confident tier), the archive side comes from the videos cache, and public
view counts come with the map. CPM does not: it is owner-private YouTube
Analytics data, arriving as a Studio CSV export or via OAuth. So the metrics
join is schema-driven and optional -- views work today, revenue columns
appear when a metrics file exists.

Everything here is plain functions over dicts so it can be tested without
Streamlit and reused by a script.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


# Column names follow Varya's «YT metrics» sheet exactly; a Studio export is
# renamed to these once, not adapted to on every read.
METRIC_COLUMNS = (
    "views",
    "engagedViews",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
    "estimatedRevenue",
    "cpm",
    "playbackBasedCpm",
)


def normalise_person(name: str) -> str:
    """Fold a name for aggregation: the CMS writes «Ирина Хакамада», the
    tagger «ирина хакамада», and both are one speaker."""
    return re.sub(r"\s+", " ", str(name).strip().lower().replace("ё", "е"))


def display_name(variants: list[str]) -> str:
    """Prefer the capitalised spelling when the same person appears in both."""
    return sorted(variants, key=lambda v: (v == v.lower(), v))[0]


@dataclass
class Upload:
    youtube_id: str
    youtube_title: str
    published_at: str | None
    view_count: int | None
    duration_seconds: float | None
    content_id: str | None
    archive_title: str | None
    archive_url: str | None
    archive_date: str | None
    program: str | None = None
    speakers: list[str] = field(default_factory=list)
    tagged: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


def load_metrics(path: Path) -> dict[str, dict[str, float]]:
    """youtube_id -> latest-snapshot metrics, from a CSV in the sheet's schema.

    Snapshots accumulate (one row per video per date); the newest wins.
    Missing file means "no analytics yet", which is the normal state today.
    """
    if not path.exists():
        return {}
    # Per metric, not per row: a newer snapshot with a blank cell must not
    # erase a value an older snapshot did have.
    latest: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            yt = (row.get("youtube_id") or "").strip()
            if not yt:
                continue
            snap = (row.get("snapshot_date") or "").strip()
            for col in METRIC_COLUMNS:
                raw = (row.get(col) or "").replace(",", ".").strip()
                try:
                    val = float(raw)
                except ValueError:
                    continue
                prev = latest[yt].get(col)
                if prev is None or snap >= prev[0]:
                    latest[yt][col] = (snap, val)
    return {yt: {col: v for col, (_, v) in cols.items()} for yt, cols in latest.items()}


def build_uploads(
    map_rows: list[dict[str, Any]],
    videos_by_hash: dict[str, Any],
    tags_by_content: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, float]] | None = None,
) -> list[Upload]:
    """Join the three sources into one row per *linked* upload.

    Only editor/exact/strong links are used: the review band was right about
    40% of the time against the editor's pairs, and a performance table built
    on wrong links would credit views to the wrong speakers.
    """
    metrics = metrics or {}
    out: list[Upload] = []
    for m in map_rows:
        if m.get("confidence") not in ("editor", "exact", "strong") or not m.get("content_id"):
            continue
        cid = str(m["content_id"])
        video = videos_by_hash.get(m.get("archive_video_hash") or "")
        tag = tags_by_content.get(cid)
        speakers: list[str] = []
        if video is not None:
            speakers += list(getattr(video, "presenters", None) or [])
        if tag:
            speakers += list(tag.get("presenter_cms") or []) + list(tag.get("guest") or [])
        # dedupe by folded form, keep the nicest spelling
        by_key: dict[str, list[str]] = defaultdict(list)
        for s in speakers:
            if normalise_person(s):
                by_key[normalise_person(s)].append(s)
        out.append(
            Upload(
                youtube_id=m["youtube_id"],
                youtube_title=m.get("youtube_title") or "",
                published_at=m.get("published_at"),
                view_count=m.get("view_count"),
                duration_seconds=m.get("duration_seconds"),
                content_id=cid,
                archive_title=m.get("archive_title"),
                archive_url=m.get("archive_url"),
                archive_date=m.get("archive_date"),
                program=(getattr(video, "program", None) if video is not None else None)
                or (tag.get("program") if tag else None),
                speakers=[display_name(v) for v in by_key.values()],
                tagged=tag is not None,
                metrics=metrics.get(m["youtube_id"], {}),
            )
        )
    return out


def aggregate(uploads: list[Upload], key: str, metric: str = "views") -> list[dict[str, Any]]:
    """Group uploads by "speaker" or "program"; sum and median of a metric.

    Median is reported alongside the total because one viral upload should
    not make a speaker look reliably strong: Varya's success criterion is
    *reliably* hitting 15-20k views, which is a median question.
    """
    # Group speakers by folded name, not display string: one upload may only
    # know the tagger's lowercase spelling while another has the CMS one, and
    # they are the same person in the table.
    groups: dict[str, list[Upload]] = defaultdict(list)
    spellings: dict[str, list[str]] = defaultdict(list)
    for u in uploads:
        if key == "speaker":
            for s in u.speakers:
                k = normalise_person(s)
                groups[k].append(u)
                spellings[k].append(s)
        else:
            groups[u.program or "(без программы)"].append(u)
    rows: list[dict[str, Any]] = []
    for gkey, ups in groups.items():
        name = display_name(spellings[gkey]) if key == "speaker" else gkey
        vals = [_metric(u, metric) for u in ups]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        # "uploads" is the sample the stats describe, not the whole group:
        # CPM may exist for 3 of a speaker's 10 uploads, and a median over 3
        # labelled as 10 would misrepresent the evidence.
        row: dict[str, Any] = {
            key: name,
            "uploads": len(vals),
            "total": sum(vals),
            "median": median(vals),
            "best": max(vals),
        }
        rows.append(row)
    rows.sort(key=lambda r: -float(r["total"]))
    return rows


def _metric(u: Upload, metric: str) -> float | None:
    if metric == "views":
        return float(u.view_count) if u.view_count is not None else None
    v = u.metrics.get(metric)
    return float(v) if v is not None else None
