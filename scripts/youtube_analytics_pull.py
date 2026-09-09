#!/usr/bin/env python3
"""Append today's YouTube Analytics snapshot for every Library upload.

    scripts/youtube_analytics_pull.py --auth              # print consent URL (one-time)
    scripts/youtube_analytics_pull.py --auth-code CODE    # finish consent (one-time)
    scripts/youtube_analytics_pull.py                     # pull + append snapshot

Requires the optional dependency group:  uv sync --extra analytics
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=str(REPO_ROOT / "data" / "youtube_map.json"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "youtube_metrics.csv"))
    parser.add_argument("--client", default=str(REPO_ROOT / "data" / "google_oauth_client.json"))
    parser.add_argument("--token", default=str(REPO_ROOT / "data" / "google_oauth_token.json"))
    parser.add_argument("--start", default="2015-01-01", help="lifetime start for totals")
    parser.add_argument("--auth", action="store_true", help="print the consent URL and exit")
    parser.add_argument("--auth-code", default=None)
    args = parser.parse_args(argv)

    from rainrag.youtube_analytics import (
        append_snapshot,
        fetch_video_metrics,
        load_credentials,
        rows_to_snapshot,
    )

    creds = load_credentials(Path(args.client), Path(args.token), auth_code=args.auth_code)
    if args.auth or args.auth_code:
        print("credentials stored; run again without flags to pull")
        return 0

    video_ids = sorted(
        {m["youtube_id"] for m in json.loads(Path(args.map).read_text(encoding="utf-8"))}
    )
    if not video_ids:
        print(f"no youtube ids in {args.map}; nothing to pull")
        return 0
    today = dt.date.today().isoformat()
    records = fetch_video_metrics(creds, video_ids, args.start, today)
    if not records:
        print(f"the API returned no rows for {len(video_ids)} videos; nothing written")
        return 0
    n = append_snapshot(Path(args.out), rows_to_snapshot(records, today))
    metrics_seen = sorted({k for r in records for k in r if k != "video"})
    print(f"snapshot {today}: {n} videos, metrics {metrics_seen} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
