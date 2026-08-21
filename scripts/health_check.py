#!/usr/bin/env python3
"""Check that the deployed rainrag stack is actually working, and shut up if it is.

Designed to run from a systemd timer every 30 minutes. When everything is fine it
prints nothing and exits 0, so the journal stays empty and the unit stays
"inactive (dead)". When something is wrong it prints a short report and exits 1,
which puts rainrag-health.service into a failed state -- visible in
`systemctl list-units --failed` and in `journalctl -u rainrag-health`.

That silence is the whole design. A monitor that emits a paragraph every half hour
gets filtered into a folder nobody opens, and then the one run that mattered is
filtered too. Output means "a human should look".

Checks, and what each one is protecting against:

  api        the FastAPI process answering on 127.0.0.1:8001. If this is down the
             UI shows errors on every search.
  streamlit  both UIs (7860, 7861). One can die while the other lives -- they are
             separate units -- so both are probed.
  imports    the share of failed video imports over a window. Guards against the
             silent breakage: yt-dlp getting blocked, Telegram creds expiring.
             Deliberately tolerant, see --min-attempts / --failure-threshold.
  disk       transcoded HLS segments and downloaded video land on disk and are not
             always cleaned up. A full disk breaks imports in a confusing way.
  session    data/telegram.session is a credential (it authenticates as a real
             Telegram user). It must exist when Telegram imports are enabled, and
             must not be readable by anyone else.

Run by hand with --verbose to see every check, including the passing ones.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml


# Allow running as `python scripts/health_check.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.usage_report import (  # noqa: E402
    JournalError,
    JournalUnavailableError,
    journal_lines,
    parse,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_API_URL = "http://localhost:8001/openapi.json"
DEFAULT_STREAMLIT_PORTS = (7860, 7861)
DEFAULT_UNIT = "rainrag-api"
DEFAULT_DISK_PATHS = ("/tmp/rainrag_hls_cache", str(REPO_ROOT / "data"))
DEFAULT_SESSION_FILE = str(REPO_ROOT / "data" / "telegram.session")
DEFAULT_CONFIG = str(REPO_ROOT / "config.yaml")

OK = "ok"
FAIL = "FAIL"
SKIP = "skip"

# HTTP probes get one retry: a service restart is a few seconds of nothing
# answering, and a deploy should not put this unit into a failed state.
HTTP_ATTEMPTS = 2
HTTP_RETRY_DELAY = 3.0

# Labels for the `reason=` field the API logs on a failed import, so the report
# says what to do rather than just what happened.
REASON_LABELS = {
    "blocked": "platform refused us",
    "geo": "region-locked",
    "no_media": "no video at the link",
    "failed": "download failed",
    "unspecified": "no reason logged",
}


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict. `detail` is one line, written for a human at 3am."""

    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


# --------------------------------------------------------------------------- #
# Pure logic (unit-tested; no network, no systemd, no clock)
# --------------------------------------------------------------------------- #


def parse_share(raw: str) -> float:
    """Parse a threshold written any of the ways a human would write it.

    "50", "50%" and "0.5" all mean half. The rule: a trailing % always means
    percent, otherwise anything above 1 is read as a percentage and anything at or
    below 1 as a fraction. So `--disk-threshold 90` and `--failure-threshold 0.5`
    both do what they look like they do.
    """
    text = raw.strip()
    explicit_percent = text.endswith("%")
    if explicit_percent:
        text = text[:-1].strip()
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {raw!r}") from None
    if explicit_percent or value > 1:
        value /= 100
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"threshold out of range: {raw!r}")
    return value


def used_share(total: int, free: int) -> float:
    """Fraction of a filesystem in use, counting root-reserved blocks as used.

    This is `(total - free) / total`, which reads a little higher than `df`'s Use%
    (df divides by total-minus-reserved). We want the pessimistic number: the
    reserve is not available to the ubuntu user either.
    """
    if total <= 0:
        return 0.0
    return (total - free) / total


def evaluate_failure_rate(
    events: list[dict[str, str]],
    *,
    min_attempts: int,
    threshold: float,
) -> CheckResult:
    """Decide whether the import failure rate is worth waking someone for.

    Two guards against false alarms, because failed imports are a normal part of
    the job: a journalist pastes a dead link, or a link to a region-locked video,
    and gets an error. That is the feature working.

      * fewer than `min_attempts` attempts in the window -> never alarm. One
        failure out of one attempt is 100% and means nothing.
      * failure share must strictly exceed `threshold` -> a run of bad luck at
        exactly the line is not an incident.
    """
    imports = [e for e in events if e.get("event") == "video_import"]
    attempts = len(imports)
    if attempts == 0:
        return CheckResult("imports", OK, "no import attempts in the window")

    failures = [e for e in imports if e.get("outcome") != "ok"]
    share = len(failures) / attempts
    summary = f"{len(failures)}/{attempts} attempt(s) failed ({share:.0%})"

    if attempts < min_attempts:
        return CheckResult("imports", OK, f"{summary}; below --min-attempts={min_attempts}")
    if share <= threshold:
        return CheckResult("imports", OK, f"{summary}; at or under {threshold:.0%}")

    reasons = Counter(e.get("reason") or "unspecified" for e in failures)
    detail = ", ".join(
        f"{count}x {reason} ({REASON_LABELS.get(reason, 'unknown reason')})"
        for reason, count in reasons.most_common()
    )
    return CheckResult("imports", FAIL, f"{summary}, over {threshold:.0%}: {detail}")


def evaluate_session_file(path: str, *, mode: int | None, telegram_enabled: bool) -> CheckResult:
    """Judge the Telegram session file. `mode` is None when the file is missing.

    Looser than 0600 means any group or other bit is set. Stricter (0400) is fine
    and must not alarm.
    """
    if mode is None:
        if telegram_enabled:
            return CheckResult(
                "session",
                FAIL,
                f"telegram_enabled is true but {path} is missing; "
                "t.me imports will fail (scripts/telegram_login.py)",
            )
        return CheckResult("session", SKIP, f"{path} absent, telegram_enabled is false")

    perms = mode & 0o777
    if perms & 0o077:
        return CheckResult(
            "session",
            FAIL,
            f"{path} is mode {perms:04o}, expected 0600 -- it is a live Telegram "
            "credential readable by other users; chmod 600 it",
        )
    return CheckResult("session", OK, f"{path} present, mode {perms:04o}")


def exit_code(results: list[CheckResult]) -> int:
    return 1 if any(r.failed for r in results) else 0


def render(results: list[CheckResult], *, verbose: bool) -> str:
    """Return what to print: nothing at all when healthy and not verbose."""
    problems = [r for r in results if r.failed]
    if not problems and not verbose:
        return ""

    shown = results if verbose else problems
    if problems:
        header = f"rainrag health: {len(problems)} problem(s) of {len(results)} check(s)"
    else:
        skipped = sum(1 for r in results if r.status == SKIP)
        header = f"rainrag health: all {len(results) - skipped} check(s) fine"
        if skipped:
            header += f", {skipped} skipped"
    lines = [header]
    lines += [f"  [{r.status}] {r.name}: {r.detail}" for r in shown]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Checks that touch the world
# --------------------------------------------------------------------------- #


def probe_http(name: str, url: str, timeout: float) -> CheckResult:
    """GET a URL once and insist on 200."""
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed localhost URLs
            status = response.status
    except HTTPError as exc:
        return CheckResult(name, FAIL, f"{url} returned HTTP {exc.code}")
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return CheckResult(name, FAIL, f"{url} unreachable ({reason})")
    if status != 200:
        return CheckResult(name, FAIL, f"{url} returned HTTP {status}")
    return CheckResult(name, OK, f"{url} -> 200")


def check_http(
    name: str,
    url: str,
    timeout: float,
    *,
    attempts: int = HTTP_ATTEMPTS,
    delay: float = HTTP_RETRY_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> CheckResult:
    """Probe a URL, retrying briefly before calling it down.

    A restart of rainrag-api leaves a window of a few seconds where nothing
    answers on 8001, and a deploy is not an outage worth a failed unit. One retry
    turns that into silence while still catching anything that is actually down --
    the point is to be believed, and a monitor that cries during every restart
    stops being believed.
    """
    result = probe_http(name, url, timeout)
    for _ in range(max(attempts - 1, 0)):
        if not result.failed:
            return result
        sleep(delay)
        result = probe_http(name, url, timeout)
    if result.failed and attempts > 1:
        return CheckResult(name, FAIL, f"{result.detail}, {attempts} attempts")
    return result


def check_streamlit(ports: list[int], timeout: float) -> CheckResult:
    """Probe every Streamlit port; report the ones that are down."""
    bad = []
    for port in ports:
        result = check_http("streamlit", f"http://localhost:{port}/", timeout)
        if result.failed:
            bad.append(result.detail)
    if bad:
        return CheckResult("streamlit", FAIL, "; ".join(bad))
    return CheckResult("streamlit", OK, f"ports {', '.join(str(p) for p in ports)} -> 200")


def read_usage_events(
    *, window_hours: int, unit: str, journal_file: str | None
) -> tuple[list[dict[str, str]], CheckResult | None]:
    """Collect [usage] events. Returns (events, problem) -- problem is set instead
    of events when the journal could not be read."""
    if journal_file:
        try:
            lines = Path(journal_file).read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return [], CheckResult("imports", FAIL, f"cannot read {journal_file}: {exc}")
        return parse(lines), None

    try:
        lines = journal_lines(f"{window_hours} hours ago", unit)
    except JournalUnavailableError as exc:
        # No systemd on this host: there is nothing to read and nothing is wrong.
        return [], CheckResult("imports", SKIP, str(exc))
    except JournalError as exc:
        # systemd is here and would not talk to us. That is itself a problem.
        return [], CheckResult("imports", FAIL, str(exc))
    return parse(lines), None


def check_disk(paths: list[str], threshold: float) -> CheckResult:
    """Check every distinct filesystem behind the given paths.

    Paths that do not exist yet (the HLS cache is created on first use) are walked
    up to their nearest existing parent, and paths sharing a filesystem are
    reported once.
    """
    seen: set[int] = set()
    full: list[str] = []
    fine: list[str] = []
    for raw in paths:
        base = _nearest_existing(Path(raw))
        if base is None:
            continue
        try:
            device = base.stat().st_dev
            usage = shutil.disk_usage(base)
        except OSError as exc:
            full.append(f"{raw} unreadable ({exc})")
            continue
        if device in seen:
            continue
        seen.add(device)
        share = used_share(usage.total, usage.free)
        label = f"{base} {share:.0%} used"
        if share > threshold:
            gb_free = usage.free / 1024**3
            full.append(f"{label}, over {threshold:.0%} ({gb_free:.1f} GB free)")
        else:
            fine.append(label)
    if full:
        return CheckResult("disk", FAIL, "; ".join(full))
    if not fine:
        return CheckResult("disk", SKIP, "no checkable paths")
    return CheckResult("disk", OK, "; ".join(fine))


def _nearest_existing(path: Path) -> Path | None:
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return None


def telegram_enabled(config_path: str) -> bool:
    """Read video_upload.telegram_enabled straight out of the YAML.

    Deliberately not rainrag.config.load_config: that calls load_dotenv() and
    builds the whole pydantic model. A health check should have no side effects and
    should not fall over because some unrelated config section is invalid. This
    flag has no environment override, so the plain read is faithful.
    """
    try:
        with Path(config_path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return False
    section = data.get("video_upload") or {}
    return bool(section.get("telegram_enabled", False))


def check_session(path: str, config_path: str) -> CheckResult:
    try:
        mode: int | None = Path(path).stat().st_mode
    except OSError:
        mode = None
    return evaluate_session_file(path, mode=mode, telegram_enabled=telegram_enabled(config_path))


# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--window-hours", type=int, default=24, help="import history to consider (default 24)"
    )
    ap.add_argument(
        "--min-attempts",
        type=int,
        default=3,
        help="never alarm on fewer import attempts than this (default 3)",
    )
    ap.add_argument(
        "--failure-threshold",
        type=parse_share,
        default=0.5,
        help="alarm above this share of failed imports (default 50%%)",
    )
    ap.add_argument(
        "--disk-threshold",
        type=parse_share,
        default=0.9,
        help="alarm above this share of a filesystem in use (default 90%%)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="print every check, not just the failures (for running by hand)",
    )

    targets = ap.add_argument_group(
        "targets", "override what gets probed; mostly useful for testing this script"
    )
    targets.add_argument(
        "--api-url", default=DEFAULT_API_URL, help="API URL expected to return 200"
    )
    targets.add_argument(
        "--streamlit-port",
        type=int,
        action="append",
        dest="streamlit_ports",
        metavar="PORT",
        help=f"repeatable; default {' and '.join(str(p) for p in DEFAULT_STREAMLIT_PORTS)}",
    )
    targets.add_argument("--unit", default=DEFAULT_UNIT, help="systemd unit holding [usage] lines")
    targets.add_argument(
        "--journal-file",
        help="read [usage] lines from a file instead of journalctl (dry runs, testing)",
    )
    targets.add_argument(
        "--disk-path",
        action="append",
        dest="disk_paths",
        metavar="PATH",
        help=f"repeatable; default {' and '.join(DEFAULT_DISK_PATHS)}",
    )
    targets.add_argument("--session-file", default=DEFAULT_SESSION_FILE)
    targets.add_argument("--config", default=DEFAULT_CONFIG, help="config.yaml to read (read-only)")
    targets.add_argument(
        "--timeout", type=float, default=5.0, help="HTTP timeout in seconds (default 5)"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ports = args.streamlit_ports or list(DEFAULT_STREAMLIT_PORTS)
    disk_paths = args.disk_paths or list(DEFAULT_DISK_PATHS)

    results = [
        check_http("api", args.api_url, args.timeout),
        check_streamlit(ports, args.timeout),
    ]

    events, problem = read_usage_events(
        window_hours=args.window_hours, unit=args.unit, journal_file=args.journal_file
    )
    results.append(
        problem
        or evaluate_failure_rate(
            events, min_attempts=args.min_attempts, threshold=args.failure_threshold
        )
    )

    results.append(check_disk(disk_paths, args.disk_threshold))
    results.append(check_session(args.session_file, args.config))

    report = render(results, verbose=args.verbose)
    if report:
        print(report)
    return exit_code(results)


if __name__ == "__main__":
    raise SystemExit(main())
