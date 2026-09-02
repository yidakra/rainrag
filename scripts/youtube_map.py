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
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


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
    args = parser.parse_args(argv)

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
    known_csv = Path(args.known_csv)
    if not known_csv.exists():
        if not args.gold_only:
            # A silent fallback here would regenerate the map without the
            # editor's 211 overrides while still reporting success -- the
            # worst kind of wrong. Absence must be an explicit choice.
            print(
                f"editor mapping not found: {known_csv}\n"
                "pass --gold-only to regenerate without editor overrides",
                file=sys.stderr,
            )
            return 1
    else:
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
            **archive_info.get(m.content_id or "", {"archive_date": None, "archive_url": None}),
        }
        for m in matches
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(args.csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
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

    if args.review:
        print("\nNeeds a human (top 15):")
        for r in [p for p in payload if p["confidence"] == "review"][:15]:
            print(f"  [{r['score']:.2f}] YT: {r['youtube_title'][:56]}")
            print(f"         ?= {(r['archive_title'] or '')[:56]}")

    print(f"\n  JSON: {args.out}\n  CSV:  {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
