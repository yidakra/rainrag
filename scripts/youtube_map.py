#!/usr/bin/env python3
"""Build the YouTube ID ↔ archive episode mapping for «Библиотека Дождя».

Varya's queries all begin from a published video, and nothing today links a
YouTube upload to the archive record it came from. Her sheet has three rows
filled by hand; this recovers the rest by matching titles, and is explicit
about which links are certain and which need an editor.

    scripts/youtube_map.py                 # build the mapping
    scripts/youtube_map.py --review        # only the uncertain ones

Writes JSON for the next tool and CSV for a human to correct, because the
uncertain rows are the point: a wrong link would silently corrupt every
recommendation built on top of it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class ShrunkMapError(RuntimeError):
    """The new map has fewer uploads than the one on disk."""


def _replace(path: Path, text: str) -> None:
    """Write beside the target and rename into place; a reader never sees half a file.

    Synced before and after the rename. Without it a power loss could leave
    the renamed file empty, and an empty map counts as zero entries, which
    would let the next run's shorter listing straight past the shrink
    guard (CodeRabbit on #113).
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_outputs(
    payload: list[dict[str, Any]],
    by_content_id: dict[str, dict[str, Any]],
    out: Path,
    csv_out: Path,
    index_out: Path,
    *,
    allow_shrink: bool = False,
) -> None:
    """Write the map, its CSV, and the archive index, each atomically.

    The Library reads the map while this runs and shows nothing for a torn
    file, so each output is written beside itself and renamed into place.

    And it refuses to shrink. This runs unattended every morning now, and a
    channel listing that came back short, after a quota error or a page
    that never answered, would replace a complete map with a partial one;
    the review tab would quietly lose uploads and the metrics pull would
    stop asking about them. A channel that really lost a video is the
    operator's call, with --allow-shrink.

    The same rule covers the archive index: a metadata directory read while
    half synced would otherwise shrink it, and every typed content id for
    the missing half would read as "not in the archive" until the next run
    (CodeRabbit on #113).

    The map is written last. It is the output the unit's contract is about,
    "a failed rebuild keeps the previous map", and a failure writing the CSV
    or the index after it had already been replaced would have broken that
    while the unit went on to the snapshot (CodeRabbit on #113).
    """
    if not allow_shrink:
        for path, now, what in (
            (out, len(payload), "uploads"),
            (index_out, len(by_content_id), "archive episodes"),
        ):
            before = _count_on_disk(path)
            if now < before:
                raise ShrunkMapError(f"{now} {what} now, {before} in {path.name}")

    out.parent.mkdir(parents=True, exist_ok=True)
    _replace(index_out, json.dumps(by_content_id, ensure_ascii=False))

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(
        [
            "youtube_id",
            "youtube_title",
            "content_id",
            "archive_title",
            "score",
            "confidence",
            "published_at",
            "view_count",
        ]
    )
    for r in sorted(payload, key=lambda r: (r["confidence"] != "review", -r["score"])):
        w.writerow(
            [
                r["youtube_id"],
                r["youtube_title"],
                r["content_id"] or "",
                r["archive_title"] or "",
                r["score"],
                r["confidence"],
                r["published_at"] or "",
                r["view_count"] or "",
            ]
        )
    _replace(csv_out, buf.getvalue())
    _replace(out, json.dumps(payload, ensure_ascii=False, indent=2))


def _count_on_disk(path: Path) -> int:
    """How many entries the file holds now; zero when absent or torn, so a
    torn file never blocks the rebuild that repairs it."""
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return 0


def archive_index(
    archive: list[tuple[str, str]],
    archive_info: dict[str, dict[str, Any]],
    durations: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """content_id -> title, date, url, hash, runtime for every archive episode."""
    return {
        cid: {
            "title": name,
            "date": archive_info.get(cid, {}).get("archive_date"),
            "url": archive_info.get(cid, {}).get("archive_url"),
            "video_hash": archive_info.get(cid, {}).get("archive_video_hash"),
            "duration_seconds": durations.get(cid),
        }
        for cid, name in archive
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos-cache",
        default=str(REPO_ROOT / "data" / "library_videos.jsonl"),
        help="folded-video cache from library_tag_batch; falls back to --raw-cache",
    )
    parser.add_argument(
        "--raw-cache", default=str(REPO_ROOT / "data" / "library_catalogue.raw.json")
    )
    parser.add_argument("--gold", default=str(REPO_ROOT / "data" / "library_gold.json"))
    parser.add_argument(
        "--gold-only",
        action="store_true",
        help="allow regenerating without the editor mapping CSV",
    )
    parser.add_argument(
        "--known-csv",
        default=str(REPO_ROOT / "data" / "varya_published.csv"),
        help="editor-confirmed youtube_id,content_id pairs; these always win",
    )
    parser.add_argument("--metadata-dir", default="/home/ubuntu/rainrag/web_metadata")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "youtube_map.json"))
    parser.add_argument("--csv-out", default=str(REPO_ROOT / "data" / "youtube_map.csv"))
    parser.add_argument("--review", action="store_true", help="print the uncertain matches")
    parser.add_argument(
        "--index-out",
        default=str(REPO_ROOT / "data" / "archive_by_content_id.json"),
        help="content_id -> title/date/url/hash for the whole archive; the review tab's manual link",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="write a map with fewer uploads than the one on disk (the channel lost videos)",
    )
    args = parser.parse_args(argv)

    # Validated before anything else is read: a silent fallback here would
    # regenerate the map without the editor's 211 overrides while still
    # reporting success -- the worst kind of wrong. Absence must be an
    # explicit choice, and the error must not hide behind whichever other
    # file happens to be missing too.
    known_csv = Path(args.known_csv)
    if not known_csv.exists() and not args.gold_only:
        print(
            f"editor mapping not found: {known_csv}\n"
            "pass --gold-only to regenerate without editor overrides",
            file=sys.stderr,
        )
        return 1

    from rainrag.youtube_library import build_title_index, fetch_channel_videos, match_video

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        print("YOUTUBE_API_KEY is not set")
        return 1

    # Archive titles, keyed by CMS content id.
    archive: list[tuple[str, str]] = []
    for path in Path(args.metadata_dir).glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("id") and d.get("name"):
            archive.append((str(d["id"]), str(d["name"])))
    print(f"Archive titles: {len(archive):,}".replace(",", " "))

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    known = {
        e["youtube_id"]: e["content_id"]
        for e in gold.get("episodes", [])
        if e.get("youtube_id") and e.get("content_id")
    }
    # The editor's own mapping sheet (Варя's Content tab) outranks everything:
    # audited against it, the matcher's confident tier was 23/23 correct but
    # its review tier was right ~40% of the time. Facts beat similarity.
    if known_csv.exists():
        with open(known_csv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                yt = (row.get("youtube_id") or "").strip()
                cid = (row.get("content_id") or "").strip()
                if yt and cid:
                    known[yt] = cid
    print(f"Hand-written links (gold + editor sheet): {len(known)}")

    videos = fetch_channel_videos(api_key)
    print(f"Channel uploads: {len(videos)}")

    # Archive runtimes, keyed by content id: the strongest matching signal.
    # The folded cache keeps this at ~0.3 GB; folding the raw scroll costs
    # 8.8 GB, which is why the tagger stopped doing it (#65). Fall back to the
    # scroll only when nothing has built the cache yet.
    from library_tag_batch import load_videos_cache

    videos_cache = Path(args.videos_cache)
    if videos_cache.exists():
        folded = load_videos_cache(videos_cache)
    else:
        from library_catalogue import fold_chunks_to_videos

        folded = fold_chunks_to_videos(json.loads(Path(args.raw_cache).read_text(encoding="utf-8")))
    hash_to_duration = {
        v.video_key: v.duration_seconds for v in folded.values() if v.duration_seconds
    }
    durations: dict[str, float] = {}
    # Date and URL ride along for free: the reviewer deciding whether a
    # YouTube upload is this archive episode compares dates and runtimes and
    # wants to click through -- without these, every card outside the tagged
    # pool showed a bare title and nothing to check it against.
    archive_info: dict[str, dict[str, Any]] = {}
    for path in Path(args.metadata_dir).glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cid = str(d.get("id") or "")
        if not cid:
            continue
        seconds = hash_to_duration.get(str(d.get("video_hash")))
        if seconds:
            durations[cid] = seconds
        archive_info[cid] = {
            "archive_date": (d.get("date_active_start") or "")[:10] or None,
            "archive_url": d.get("url"),
            # lets the UI join programme/presenters from the videos cache for
            # episodes outside the tagged pool (most Library uploads are <30 min)
            "archive_video_hash": str(d.get("video_hash") or "") or None,
        }
    print(f"Archive runtimes known for {len(durations):,} episodes".replace(",", " "))

    index = build_title_index(archive)
    matches = [
        match_video(v, archive, known=known, index=index, durations=durations) for v in videos
    ]
    by_confidence: dict[str, int] = {}
    for m in matches:
        by_confidence[m.confidence] = by_confidence.get(m.confidence, 0) + 1

    linked = [m for m in matches if m.is_confident or m.confidence == "editor"]
    print("\nConfidence:")
    for level in ("editor", "exact", "strong", "review", "none"):
        if by_confidence.get(level):
            print(f"  {level:>7}: {by_confidence[level]}")
    print(
        f"\n  usable links: {len(linked)}/{len(videos)} ({100 * len(linked) // max(len(videos), 1)}%)"
    )

    views = {v.video_id: v for v in videos}
    payload = [
        {
            "youtube_id": m.video_id,
            "youtube_title": m.youtube_title,
            "content_id": m.content_id,
            "archive_title": m.archive_title,
            "score": round(m.score, 3),
            "confidence": m.confidence,
            "published_at": views[m.video_id].published_at,
            "view_count": views[m.video_id].view_count,
            "duration_seconds": views[m.video_id].duration_seconds,
            "archive_duration_seconds": durations.get(m.content_id or ""),
            **archive_info.get(
                m.content_id or "",
                {"archive_date": None, "archive_url": None, "archive_video_hash": None},
            ),
        }
        for m in matches
    ]
    # One row per archive episode, so a content id typed into the review tab
    # can be shown as a title before it is linked. The videos cache is keyed
    # by hash and the tagged pool covers a tenth of the archive; nothing else
    # answers "which episode is 412707?" without reading 105k files.
    by_content_id = archive_index(archive, archive_info, durations)
    try:
        write_outputs(
            payload,
            by_content_id,
            Path(args.out),
            Path(args.csv_out),
            Path(args.index_out),
            allow_shrink=args.allow_shrink,
        )
    except ShrunkMapError as exc:
        print(
            f"REFUSING to write {args.out}: {exc}\n"
            "  the previous map is kept; pass --allow-shrink if the channel really lost videos",
            file=sys.stderr,
        )
        return 1

    if args.review:
        print("\nNeeds a human (top 15):")
        for r in [p for p in payload if p["confidence"] == "review"][:15]:
            print(f"  [{r['score']:.2f}] YT: {r['youtube_title'][:56]}")
            print(f"         ?= {(r['archive_title'] or '')[:56]}")

    print(f"\n  JSON: {args.out}\n  CSV:  {args.csv_out}\n  index: {args.index_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
