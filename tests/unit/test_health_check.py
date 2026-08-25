"""Unit tests for the pure decision logic in scripts/health_check.py.

Nothing here touches the network, journalctl or the real deployment: the point of
factoring the decisions out of the probes is that the judgement calls -- when to
alarm, when to stay quiet -- can be pinned down in tests.
"""

from __future__ import annotations

import argparse
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import health_check as hc


def _import_event(outcome: str, reason: str | None = None) -> dict[str, str]:
    event = {"event": "video_import", "outcome": outcome, "source": "youtube"}
    if reason is not None:
        event["reason"] = reason
    return event


# --------------------------------------------------------------------------- #
# Threshold parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("50", 0.5),
        ("50%", 0.5),
        ("0.5", 0.5),
        (" 90 ", 0.9),
        ("90%", 0.9),
        ("0.9", 0.9),
        ("1", 1.0),  # bare 1 is a fraction, i.e. 100%
        ("1%", 0.01),  # explicit % is always percent
        ("0", 0.0),
        ("100", 1.0),
    ],
)
def test_parse_share_accepts_percent_and_fraction(raw: str, expected: float) -> None:
    assert hc.parse_share(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "half", "-1", "101", "-5%", "abc%"])
def test_parse_share_rejects_nonsense(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        hc.parse_share(raw)


# --------------------------------------------------------------------------- #
# Import failure rate
# --------------------------------------------------------------------------- #


def test_no_attempts_is_quiet() -> None:
    result = hc.evaluate_failure_rate([], min_attempts=3, threshold=0.5)
    assert result.status == hc.OK


def test_query_events_do_not_count_towards_the_import_rate() -> None:
    """Queries and imports share the [usage] channel but not this check.

    Questions are asked far more often than videos are imported, so letting
    query events into this calculation would swamp the import signal it exists
    to watch -- and a burst of query timeouts would read as import breakage.
    """
    events = [
        {"event": "query", "outcome": "http_504"},
        {"event": "query", "outcome": "http_500"},
        {"event": "query", "outcome": "http_500"},
        {"event": "query", "outcome": "http_429"},
    ]
    result = hc.evaluate_failure_rate(events, min_attempts=3, threshold=0.5)
    assert result.status == hc.OK
    assert "no import attempts" in result.detail


def test_single_failure_does_not_alarm() -> None:
    """One journalist pasting one dead link is 100% failure and is not an incident."""
    events = [_import_event("http_422", "no_media")]
    result = hc.evaluate_failure_rate(events, min_attempts=3, threshold=0.5)
    assert result.status == hc.OK
    assert "min-attempts" in result.detail


def test_two_of_four_is_quiet_at_the_default_threshold() -> None:
    """Exactly at the line is not over it -- this is production's current state."""
    events = [
        _import_event("ok"),
        _import_event("ok"),
        _import_event("http_422", "no_media"),
        _import_event("http_503", "blocked"),
    ]
    result = hc.evaluate_failure_rate(events, min_attempts=3, threshold=0.5)
    assert result.status == hc.OK
    assert "2/4" in result.detail


def test_three_of_four_alarms_and_lists_reasons() -> None:
    events = [
        _import_event("ok"),
        _import_event("http_503", "blocked"),
        _import_event("http_503", "blocked"),
        _import_event("error", "failed"),
    ]
    result = hc.evaluate_failure_rate(events, min_attempts=3, threshold=0.5)
    assert result.failed
    assert "3/4" in result.detail
    assert "2x blocked" in result.detail
    assert "1x failed" in result.detail


def test_failure_without_a_reason_field_is_labelled() -> None:
    """The API omits reason= on some failures (e.g. http_422 from MTProto)."""
    events = [
        _import_event("http_422"),
        _import_event("http_422"),
        _import_event("http_503", "blocked"),
    ]
    result = hc.evaluate_failure_rate(events, min_attempts=3, threshold=0.5)
    assert result.failed
    assert "2x unspecified" in result.detail


def test_non_import_events_are_ignored() -> None:
    events = [
        {"event": "search", "outcome": "error"},
        _import_event("ok"),
    ]
    result = hc.evaluate_failure_rate(events, min_attempts=1, threshold=0.5)
    assert result.status == hc.OK
    assert "1 attempt" in result.detail


def test_min_attempts_one_alarms_on_a_lone_failure() -> None:
    """The tolerance is a knob, not a hard-coded opinion."""
    result = hc.evaluate_failure_rate(
        [_import_event("error", "failed")], min_attempts=1, threshold=0.5
    )
    assert result.failed


def test_parsed_journal_lines_feed_the_decision() -> None:
    """End to end from real journal text through the shared parser."""
    lines = [
        "2026-08-21T15:47:15+00:00 host env[1]: [usage] event=video_import outcome=ok "
        "seconds=14.8 source=telegram via=mtproto bytes=31971860",
        "2026-08-21T15:47:15+00:00 host env[1]: [usage] event=video_import outcome=http_503 "
        "seconds=2.0 source=youtube via=yt-dlp reason=blocked",
        "2026-08-21T15:48:15+00:00 host env[1]: [usage] event=video_import outcome=http_503 "
        "seconds=2.0 source=youtube via=yt-dlp reason=blocked",
        "2026-08-21T15:49:15+00:00 host env[1]: unrelated log line",
    ]
    from scripts.usage_report import parse

    result = hc.evaluate_failure_rate(parse(lines), min_attempts=3, threshold=0.5)
    assert result.failed
    assert "2/3" in result.detail
    assert "2x blocked" in result.detail


# --------------------------------------------------------------------------- #
# Telegram session file permissions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("perms", [0o600, 0o400, 0o200])
def test_tight_session_permissions_are_fine(perms: int) -> None:
    # st_mode, not bare permission bits: that is what Path.stat() hands back.
    result = hc.evaluate_session_file(
        "/x/telegram.session", mode=stat.S_IFREG | perms, telegram_enabled=True
    )
    assert result.status == hc.OK


@pytest.mark.parametrize("perms", [0o644, 0o640, 0o604, 0o660, 0o666, 0o700 | 0o001])
def test_loose_session_permissions_alarm(perms: int) -> None:
    result = hc.evaluate_session_file(
        "/x/telegram.session", mode=stat.S_IFREG | perms, telegram_enabled=True
    )
    assert result.failed
    assert "0600" in result.detail


def test_loose_permissions_alarm_even_when_telegram_is_disabled() -> None:
    """The file is a credential whether or not the feature is switched on."""
    result = hc.evaluate_session_file(
        "/x/telegram.session", mode=stat.S_IFREG | 0o644, telegram_enabled=False
    )
    assert result.failed


def test_missing_session_alarms_only_when_telegram_is_enabled() -> None:
    enabled = hc.evaluate_session_file("/x/telegram.session", mode=None, telegram_enabled=True)
    assert enabled.failed
    assert "missing" in enabled.detail

    disabled = hc.evaluate_session_file("/x/telegram.session", mode=None, telegram_enabled=False)
    assert disabled.status == hc.SKIP


def test_file_type_bits_do_not_confuse_the_permission_check(tmp_path) -> None:
    """st_mode carries S_IFREG as well as the permission bits."""
    path = tmp_path / "telegram.session"
    path.write_text("not-a-real-session")
    path.chmod(0o600)
    result = hc.evaluate_session_file(str(path), mode=path.stat().st_mode, telegram_enabled=True)
    assert result.status == hc.OK
    assert stat.S_ISREG(path.stat().st_mode)


def test_check_session_reads_the_filesystem(tmp_path) -> None:
    session = tmp_path / "telegram.session"
    session.write_text("x")
    session.chmod(0o666)
    config = tmp_path / "config.yaml"
    config.write_text("video_upload:\n  telegram_enabled: true\n")

    result = hc.check_session(str(session), str(config))
    assert result.failed


def test_telegram_enabled_reads_yaml(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("video_upload:\n  telegram_enabled: true\n")
    assert hc.telegram_enabled(str(config)) is True

    config.write_text("video_upload:\n  enabled: true\n")
    assert hc.telegram_enabled(str(config)) is False

    assert hc.telegram_enabled(str(tmp_path / "nope.yaml")) is False


# --------------------------------------------------------------------------- #
# Disk
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("total", "free", "expected"),
    [(100, 10, 0.9), (100, 100, 0.0), (100, 0, 1.0), (0, 0, 0.0)],
)
def test_used_share(total: int, free: int, expected: float) -> None:
    assert hc.used_share(total, free) == pytest.approx(expected)


def test_disk_check_reports_each_filesystem_once(tmp_path) -> None:
    """tmp_path and a child of it share a filesystem; one line, not two."""
    child = tmp_path / "cache"
    child.mkdir()
    result = hc.check_disk([str(tmp_path), str(child)], 1.0)
    assert result.status == hc.OK
    assert result.detail.count("%") == 1


def test_disk_check_walks_up_to_an_existing_parent(tmp_path) -> None:
    """The HLS cache directory may not exist yet; that is not a failure."""
    result = hc.check_disk([str(tmp_path / "not" / "created" / "yet")], 1.0)
    assert result.status == hc.OK
    assert str(tmp_path) in result.detail


def test_disk_check_alarms_over_threshold(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed numbers, so the verdict does not depend on the host's free space."""
    monkeypatch.setattr(
        hc.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, free=5 * 1024**3, used=95 * 1024**3),
    )
    result = hc.check_disk([str(tmp_path)], 0.9)
    assert result.failed
    assert "95% used" in result.detail
    assert "5.0 GB free" in result.detail


def test_disk_check_is_quiet_just_under_the_threshold(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        hc.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, free=11 * 1024**3, used=89 * 1024**3),
    )
    result = hc.check_disk([str(tmp_path)], 0.9)
    assert result.status == hc.OK


def test_nearest_existing_returns_the_path_itself_when_it_exists(tmp_path) -> None:
    assert hc._nearest_existing(tmp_path) == tmp_path


# --------------------------------------------------------------------------- #
# Reporting and exit codes
# --------------------------------------------------------------------------- #


def test_healthy_run_prints_nothing() -> None:
    results = [
        hc.CheckResult("api", hc.OK, "fine"),
        hc.CheckResult("session", hc.SKIP, "not applicable"),
    ]
    assert hc.render(results, verbose=False) == ""
    assert hc.exit_code(results) == 0


def test_verbose_prints_every_check() -> None:
    results = [
        hc.CheckResult("api", hc.OK, "fine"),
        hc.CheckResult("imports", hc.SKIP, "no journal"),
    ]
    report = hc.render(results, verbose=True)
    assert "all 1 check(s) fine, 1 skipped" in report
    assert "[ok] api: fine" in report
    assert "[skip] imports: no journal" in report


def test_failures_are_reported_and_exit_non_zero() -> None:
    results = [
        hc.CheckResult("api", hc.OK, "fine"),
        hc.CheckResult("disk", hc.FAIL, "/ 99% used"),
    ]
    report = hc.render(results, verbose=False)
    assert "1 problem(s) of 2 check(s)" in report
    assert "disk" in report
    assert "api" not in report  # quiet about what works, even when something breaks
    assert hc.exit_code(results) == 1


def test_journal_file_override_is_used_instead_of_journalctl(tmp_path) -> None:
    journal = tmp_path / "journal.txt"
    journal.write_text(
        "2026-08-21T15:47:15+00:00 host env[1]: [usage] event=video_import outcome=error "
        "seconds=1.0 source=youtube reason=failed\n"
    )
    events, problem = hc.read_usage_events(
        window_hours=24, unit="rainrag-api", journal_file=str(journal)
    )
    assert problem is None
    assert events == [
        {
            "event": "video_import",
            "outcome": "error",
            "seconds": "1.0",
            "source": "youtube",
            "reason": "failed",
            "date": "2026-08-21",
        }
    ]


def test_unreadable_journal_file_is_a_failure(tmp_path) -> None:
    _events, problem = hc.read_usage_events(
        window_hours=24, unit="rainrag-api", journal_file=str(tmp_path / "missing.txt")
    )
    assert problem is not None
    assert problem.failed


def test_missing_journalctl_is_skipped_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a laptop there is no journal to read, and nothing is wrong."""

    def _unavailable(*_args: object, **_kwargs: object) -> list[str]:
        raise hc.JournalUnavailableError("journalctl not found")

    monkeypatch.setattr(hc, "journal_lines", _unavailable)
    _events, problem = hc.read_usage_events(window_hours=24, unit="u", journal_file=None)
    assert problem is not None
    assert problem.status == hc.SKIP


def test_journal_that_refuses_us_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken(*_args: object, **_kwargs: object) -> list[str]:
        raise hc.JournalError("journalctl failed: permission denied")

    monkeypatch.setattr(hc, "journal_lines", _broken)
    _events, problem = hc.read_usage_events(window_hours=24, unit="u", journal_file=None)
    assert problem is not None
    assert problem.failed


def test_defaults_do_not_collide_with_the_live_ports() -> None:
    """Guard rail: the checks must probe production, not some other port."""
    assert hc.DEFAULT_STREAMLIT_PORTS == (7860, 7861)
    assert "8001" in hc.DEFAULT_API_URL
    assert Path(hc.DEFAULT_SESSION_FILE).name == "telegram.session"


# --------------------------------------------------------------------------- #
# HTTP retry
# --------------------------------------------------------------------------- #


def test_http_check_retries_before_declaring_a_service_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restart window is a few seconds of nothing answering, not an outage."""
    verdicts = [
        hc.CheckResult("api", hc.FAIL, "http://x unreachable (Connection refused)"),
        hc.CheckResult("api", hc.OK, "http://x -> 200"),
    ]
    monkeypatch.setattr(hc, "probe_http", lambda *_args: verdicts.pop(0))
    slept: list[float] = []
    result = hc.check_http("api", "http://x", 5.0, sleep=slept.append)
    assert result.status == hc.OK
    assert slept == [hc.HTTP_RETRY_DELAY]
    assert verdicts == []


def test_http_check_reports_when_every_attempt_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _down(name: str, url: str, _timeout: float) -> hc.CheckResult:
        calls.append(url)
        return hc.CheckResult(name, hc.FAIL, f"{url} unreachable (Connection refused)")

    monkeypatch.setattr(hc, "probe_http", _down)
    result = hc.check_http("api", "http://x", 5.0, sleep=lambda _d: None)
    assert result.failed
    assert "2 attempts" in result.detail
    assert len(calls) == 2


def test_http_check_does_not_sleep_when_the_first_probe_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hc, "probe_http", lambda name, url, _t: hc.CheckResult(name, hc.OK, url))

    def _fail_if_called(_delay: float) -> None:
        pytest.fail("a healthy service must not cost the check any wall clock")

    assert hc.check_http("api", "http://x", 5.0, sleep=_fail_if_called).status == hc.OK


def test_verbose_header_counts_skips_separately() -> None:
    results = [
        hc.CheckResult("api", hc.OK, "fine"),
        hc.CheckResult("imports", hc.SKIP, "no journal"),
    ]
    assert "all 1 check(s) fine, 1 skipped" in hc.render(results, verbose=True)


class TestReadinessNotJustStatusCode:
    """A 200 is not enough: /health answers 200 while reporting "degraded"."""

    def test_healthy_body_passes(self):
        r = hc.evaluate_readiness("api", "/health", 200, b'{"status":"healthy"}')
        assert not r.failed

    @pytest.mark.parametrize(
        "body",
        [
            b'{"status":"degraded","qdrant_connected":false,"model_loaded":true}',
            b'{"status":"degraded","qdrant_connected":true,"model_loaded":false}',
            b'{"status":"starting"}',
        ],
    )
    def test_non_healthy_status_fails(self, body: bytes):
        assert hc.evaluate_readiness("api", "/health", 200, body).failed

    def test_names_the_unavailable_component(self):
        r = hc.evaluate_readiness(
            "api", "/health", 200, b'{"status":"degraded","qdrant_connected":false}'
        )
        assert "qdrant_connected" in r.detail

    @pytest.mark.parametrize("body", [b"", b"<html>not json", b"[]", b'{"no_status":1}'])
    def test_bodies_without_a_status_field_are_judged_on_the_code(self, body: bytes):
        """Streamlit serves HTML; it must not be required to report readiness."""
        assert not hc.evaluate_readiness("streamlit", "/", 200, body).failed

    def test_non_200_still_fails(self):
        assert hc.evaluate_readiness("api", "/health", 503, b"").failed


class TestSessionMustBeARegularFile:
    def test_directory_at_the_session_path_fails(self):
        r = hc.evaluate_session_file("/x/s", mode=stat.S_IFDIR | 0o700, telegram_enabled=True)
        assert r.failed and "regular file" in r.detail

    def test_fifo_at_the_session_path_fails(self):
        r = hc.evaluate_session_file("/x/s", mode=stat.S_IFIFO | 0o600, telegram_enabled=True)
        assert r.failed

    def test_regular_file_at_0600_still_passes(self):
        r = hc.evaluate_session_file("/x/s", mode=stat.S_IFREG | 0o600, telegram_enabled=True)
        assert not r.failed


class TestConfigThatIsNotAMapping:
    """A hand-edited config must not abort every other check with a traceback."""

    @pytest.mark.parametrize("text", ["just a string\n", "- a\n- b\n", "42\n", ""])
    def test_non_mapping_document_is_false_not_a_crash(self, tmp_path, text: str):
        cfg = tmp_path / "c.yaml"
        cfg.write_text(text)
        assert hc.telegram_enabled(str(cfg)) is False

    def test_non_mapping_video_upload_section(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("video_upload: not-a-mapping\n")
        assert hc.telegram_enabled(str(cfg)) is False

    def test_a_real_mapping_still_reads_true(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("video_upload:\n  telegram_enabled: true\n")
        assert hc.telegram_enabled(str(cfg)) is True


# --------------------------------------------------------------------------- #
# Slack connector check
# --------------------------------------------------------------------------- #


def test_ok_status_counts_as_healthy() -> None:
    """The Slack connector's /health says "ok" where the API says "healthy"."""
    result = hc.evaluate_readiness(
        "slack", "http://x/health", 200, b'{"status": "ok", "bot_token_configured": true}'
    )
    assert result.status == hc.OK


def test_degraded_status_still_fails() -> None:
    result = hc.evaluate_readiness("api", "http://x/health", 200, b'{"status": "degraded"}')
    assert result.status == hc.FAIL


def test_slack_check_is_skipped_by_default() -> None:
    """Deployments without the connector must stay green: empty default = no check."""
    args = hc.build_parser().parse_args([])
    assert args.slack_url == ""
