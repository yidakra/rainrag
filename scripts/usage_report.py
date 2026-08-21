#!/usr/bin/env python3
"""Summarise how the video-import feature is actually being used.

Reads the ``[usage]`` lines the API writes to the journal and prints counts by
platform, outcome and day. Intended for the question "is anyone using this, and
what breaks when they do?" without anyone having to learn journalctl.

    scripts/usage_report.py                # last 7 days
    scripts/usage_report.py --days 30
    scripts/usage_report.py --failures     # only the attempts that failed

Outcomes are the HTTP status the user got, so they read directly:
    ok         the video downloaded and a session was created
    http_422   nothing downloadable at that link (dead post, no video in it)
    http_413   over the size limit
    http_503   Telegram is enabled but not configured
    error      an unexpected fault -- these are the ones worth chasing
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter, defaultdict


USAGE_RE = re.compile(r"\[usage\] (?P<fields>event=\S+(?: \S+=\S*)*)")
# journalctl short-iso prefixes each line with a timestamp we use for the daily split.
DATE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})")


def read_journal(days: int, unit: str) -> list[str]:
    """Return journal lines for the unit over the last N days."""
    try:
        proc = subprocess.run(
            [
                "journalctl",
                "-u",
                unit,
                "--since",
                f"{days} days ago",
                "--no-pager",
                "-o",
                "short-iso",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("journalctl not found; this script expects a systemd host.")
    except subprocess.TimeoutExpired:
        sys.exit("journalctl timed out.")
    if proc.returncode != 0 and not proc.stdout:
        sys.exit(f"journalctl failed: {proc.stderr.strip()[:200]}")
    return proc.stdout.splitlines()


def parse(lines: list[str]) -> list[dict[str, str]]:
    """Extract usage events, newest last."""
    events = []
    for line in lines:
        m = USAGE_RE.search(line)
        if not m:
            continue
        fields = dict(part.split("=", 1) for part in m.group("fields").split(" ") if "=" in part)
        d = DATE_RE.match(line)
        if d:
            fields["date"] = d.group("date")
        events.append(fields)
    return events


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit != "GB" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--days", type=int, default=7, help="how far back to look (default 7)")
    ap.add_argument("--unit", default="rainrag-api", help="systemd unit to read")
    ap.add_argument("--failures", action="store_true", help="list failed attempts only")
    args = ap.parse_args()

    events = parse(read_journal(args.days, args.unit))
    imports = [e for e in events if e.get("event") == "video_import"]

    print(f"Video imports, last {args.days} day(s): {len(imports)} attempt(s)")
    if not imports:
        print("\nNo attempts recorded. Either nobody used it, or the API predates usage logging.")
        return 0

    ok = sum(1 for e in imports if e.get("outcome") == "ok")
    print(
        f"  succeeded: {ok}    failed: {len(imports) - ok}    "
        f"success rate: {100 * ok / len(imports):.0f}%"
    )

    by_source: dict[str, Counter] = defaultdict(Counter)
    for e in imports:
        by_source[e.get("source", "?")][e.get("outcome", "?")] += 1

    print("\nBy platform:")
    for source, outcomes in sorted(by_source.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(outcomes.values())
        good = outcomes.get("ok", 0)
        detail = ", ".join(f"{k} {v}" for k, v in outcomes.most_common() if k != "ok")
        print(
            f"  {source:<12} {total:>3} attempt(s), {good} ok" + (f"  ({detail})" if detail else "")
        )

    print("\nBy day:")
    per_day: dict[str, Counter] = defaultdict(Counter)
    for e in imports:
        per_day[e.get("date", "?")][e.get("outcome", "?")] += 1
    for day in sorted(per_day):
        c = per_day[day]
        print(f"  {day}  {sum(c.values()):>3} attempt(s), {c.get('ok', 0)} ok")

    sizes = [int(e["bytes"]) for e in imports if e.get("bytes", "").isdigit()]
    if sizes:
        print(
            f"\nDownloaded: {len(sizes)} file(s), {human_bytes(sum(sizes))} total, "
            f"{human_bytes(sum(sizes) / len(sizes))} average"
        )

    vias = Counter(e["via"] for e in imports if e.get("via"))
    if vias:
        print("Backend: " + ", ".join(f"{k} {v}" for k, v in vias.most_common()))

    # Why things failed matters more than how often: "blocked" and "geo" are the
    # platform refusing us, not users pasting bad links, and they call for
    # different action.
    reasons = Counter(e["reason"] for e in imports if e.get("reason"))
    if reasons:
        labels = {
            "blocked": "platform throttled or refused us (retryable)",
            "geo": "region-locked (will not work from this server)",
            "no_media": "nothing downloadable at the link",
            "failed": "download failed for another reason",
        }
        print("\nWhy failures happened:")
        for reason, count in reasons.most_common():
            print(f"  {reason:<10} {count:>3}  {labels.get(reason, '')}")

    if args.failures:
        failed = [e for e in imports if e.get("outcome") != "ok"]
        print(f"\nFailed attempts ({len(failed)}):")
        for e in failed:
            print(
                f"  {e.get('date', '?')}  {e.get('source', '?'):<10} {e.get('outcome', '?'):<10} "
                f"{e.get('reason', '-'):<9} {e.get('seconds', '?')}s  via={e.get('via', '-')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
