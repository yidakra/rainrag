#!/usr/bin/env python3
"""Summarise how RainRAG is actually being used.

Reads the ``[usage]`` lines the API writes to the journal and prints counts by
platform, outcome and day, for both questions asked of the archive and videos
imported into it. Intended for the question "is anyone using this, and what
breaks when they do?" without anyone having to learn journalctl.

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


class JournalError(RuntimeError):
    """The journal could not be read."""


class JournalUnavailableError(JournalError):
    """There is no journalctl on this host at all (e.g. a laptop, a container).

    Separate from JournalError so callers can tell "no systemd here" (nothing is
    wrong, there is just nothing to read) from "systemd is here and refused us"
    (something is wrong).
    """


def journal_lines(since: str, unit: str, timeout: float = 120) -> list[str]:
    """Return journal lines for the unit since a journalctl time expression.

    Raises JournalError instead of exiting, so callers that have other work to do
    -- scripts/health_check.py runs four more checks -- can carry on.
    """
    try:
        proc = subprocess.run(
            [
                "journalctl",
                "-u",
                unit,
                "--since",
                since,
                "--no-pager",
                "-o",
                "short-iso",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise JournalUnavailableError("journalctl not found; this expects a systemd host.") from exc
    except subprocess.TimeoutExpired as exc:
        raise JournalError("journalctl timed out.") from exc
    if proc.returncode != 0 and not proc.stdout:
        raise JournalError(f"journalctl failed: {proc.stderr.strip()[:200]}")
    return proc.stdout.splitlines()


def read_journal(days: int, unit: str) -> list[str]:
    """Return journal lines for the unit over the last N days, or exit on failure."""
    try:
        return journal_lines(f"{days} days ago", unit)
    except JournalError as exc:
        sys.exit(str(exc))


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
    report_queries([e for e in events if e.get("event") == "query"], args)
    return report_imports([e for e in events if e.get("event") == "video_import"], args)


def report_queries(queries: list[dict[str, str]], args: argparse.Namespace) -> None:
    """Print the question-answering side: volume, failures, latency, tokens."""
    print(f"Questions asked, last {args.days} day(s): {len(queries)} attempt(s)")
    if not queries:
        print("  none recorded (nobody asked, or the API predates query usage logging)\n")
        return

    ok = sum(1 for e in queries if e.get("outcome") == "ok")
    print(
        f"  succeeded: {ok}    failed: {len(queries) - ok}    "
        f"success rate: {100 * ok / len(queries):.0f}%"
    )

    modes = Counter(e.get("mode", "?") for e in queries)
    print("  " + ", ".join(f"{k}: {v}" for k, v in modes.most_common()))

    bad = Counter(e["outcome"] for e in queries if e.get("outcome") != "ok")
    if bad:
        labels = {
            "http_429": "server busy, too many at once",
            "http_504": "timed out generating the answer",
            "http_500": "unexpected fault -- worth chasing",
            "http_503": "query engine not initialised",
            "http_404": "session gone",
        }
        print("  failures:")
        for outcome, count in bad.most_common():
            print(f"    {outcome:<10} {count:>3}  {labels.get(outcome, '')}")

    # Latency is the complaint users actually voice ("it hangs"), so show the
    # slow tail rather than just an average that hides it.
    times = sorted(float(e["seconds"]) for e in queries if _is_number(e.get("seconds")))
    if times:
        p50 = times[len(times) // 2]
        p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
        print(f"  time: median {p50:.1f}s, p95 {p95:.1f}s, slowest {times[-1]:.1f}s")

    docs = [int(e["docs"]) for e in queries if e.get("docs", "").isdigit()]
    if docs:
        empty = sum(1 for d in docs if d == 0)
        note = f", {empty} retrieved nothing" if empty else ""
        print(f"  retrieval: {sum(docs) / len(docs):.1f} chunks average{note}")

    tokens_in = sum(int(e["tokens_in"]) for e in queries if e.get("tokens_in", "").isdigit())
    tokens_out = sum(int(e["tokens_out"]) for e in queries if e.get("tokens_out", "").isdigit())
    if tokens_in or tokens_out:
        measured = sum(1 for e in queries if e.get("tokens_in", "").isdigit())
        print(
            f"  tokens: {tokens_in:,} in, {tokens_out:,} out "
            f"(measured on {measured}/{len(queries)} attempts)"
        )
    print()


def _is_number(value: str | None) -> bool:
    if not value:
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def report_imports(imports: list[dict[str, str]], args: argparse.Namespace) -> int:
    """Print the video-import side."""
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
