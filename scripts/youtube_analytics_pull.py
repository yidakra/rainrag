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
import socket
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
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help="skip the per-video age/gender requests (one API call per upload)",
    )
    parser.add_argument(
        "--slack-channel", default="", help="post the outcome here; empty means stdout only"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel requests for the per-video demographics"
    )
    args = parser.parse_args(argv)
    # The demographics report can take 16 s per answer; a request that never
    # answers must not pin a worker for the rest of the night.
    socket.setdefaulttimeout(90)
    notify = _notifier(args.slack_channel)

    from rainrag.youtube_analytics import (
        ConsentRequired,
        append_snapshot,
        fetch_video_demographics,
        fetch_video_metrics,
        load_credentials,
        merge_demographics,
        rows_to_snapshot,
    )

    try:
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
        # UTC explicitly: the timer fires at 05:20 UTC and the snapshot date
        # must not depend on the host's zone. dt.date.today() is process-local.
        today = dt.datetime.now(dt.timezone.utc).date().isoformat()
        records = fetch_video_metrics(creds, video_ids, args.start, today)
        if not records:
            notify(f"YouTube pull {today}: the API returned no rows for {len(video_ids)} videos")
            return 1
        demo_note = "demographics skipped"
        all_demographics_failed = False
        if not args.no_demographics:
            demographics, failed = fetch_video_demographics(
                creds, video_ids, args.start, today, workers=args.workers
            )
            merge_demographics(records, demographics)
            demo_note = f"age/gender on {len(demographics)}"
            if failed:
                demo_note += f", {len(failed)} request(s) failed"
            all_demographics_failed = bool(failed) and len(failed) == len(video_ids)
        n = append_snapshot(Path(args.out), rows_to_snapshot(records, today))
        metrics_seen = sorted({k for r in records for k in r if k != "video"})
        print(f"snapshot {today}: {n} videos, metrics {metrics_seen} -> {args.out}")
        if all_demographics_failed:
            # The metrics are safe on disk, but every demographics request
            # failing is a systemic problem, not noise, and the timer should
            # show it rather than stay green.
            notify(
                f"YouTube pull {today}: {n} videos written, but EVERY demographics request failed"
            )
            return 2
        notify(f"YouTube pull {today}: {n} videos, {demo_note}")
        return 0
    except ConsentRequired:
        # The --auth run stopping to print a consent URL is the expected
        # outcome, not a failure; nothing to alert on.
        raise
    except SystemExit as exc:
        # Everything else load_credentials refuses (expired consent, missing
        # client, missing PKCE verifier) is worth surfacing in Slack, not only
        # in the journal, whichever flags the run had.
        if exc.code not in (0, None):
            notify(f"YouTube pull failed: {exc}")
        raise
    except Exception as exc:  # noqa: BLE001 - reported, then re-raised for the journal
        notify(f"YouTube pull failed: {type(exc).__name__}: {exc}")
        raise


def _notifier(channel: str):
    """Print always; post to Slack too when a channel is given and a token exists."""
    import os

    token = os.getenv("SLACK_BOT_TOKEN", "").strip()

    def notify(text: str) -> None:
        print(text)
        if channel and token:
            sys.path.insert(0, str(REPO_ROOT))
            from scripts.health_check import post_slack_alert

            post_slack_alert(f":bar_chart: {text}", token=token, channel=channel)

    return notify


if __name__ == "__main__":
    raise SystemExit(main())
