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


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-cache", default=str(REPO_ROOT / "data" / "library_catalogue.raw.json")
    )
    parser.add_argument("--gold", default=str(REPO_ROOT / "data" / "library_gold.json"))
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
    print(f"Hand-written links from the sheet: {len(known)}")

    videos = fetch_channel_videos(api_key)
    print(f"Channel uploads: {len(videos)}")

    # Archive runtimes, keyed by content id: the strongest matching signal.
    durations: dict[str, float] = {}
    payloads = json.loads(Path(args.raw_cache).read_text(encoding="utf-8"))
    from library_catalogue import fold_chunks_to_videos

    hash_to_duration = {
        v.video_key: v.duration_seconds
        for v in fold_chunks_to_videos(payloads).values()
        if v.duration_seconds
    }
    for path in Path(args.metadata_dir).glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        seconds = hash_to_duration.get(str(d.get("video_hash")))
        if d.get("id") and seconds:
            durations[str(d["id"])] = seconds
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
