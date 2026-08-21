"""Tests for the query section of scripts/usage_report.py.

The report is read by people deciding whether something is wrong, so the parts
worth pinning down are the ones that would quietly mislead: a percentage
computed off the wrong denominator, or token totals presented as though every
attempt had been measured when only some were.
"""

from __future__ import annotations

import argparse

import pytest

from scripts import usage_report as ur


def _query(outcome: str = "ok", **fields: str) -> dict[str, str]:
    event = {"event": "query", "outcome": outcome, "mode": "corpus"}
    event.update(fields)
    return event


def _report(events: list[dict[str, str]], capsys, days: int = 7) -> str:
    ur.report_queries(events, argparse.Namespace(days=days))
    return capsys.readouterr().out


class TestEmptyState:
    def test_no_queries_says_so_without_dividing_by_zero(self, capsys):
        out = _report([], capsys)
        assert "0 attempt(s)" in out
        assert "none recorded" in out


class TestCounts:
    def test_success_rate(self, capsys):
        out = _report([_query(), _query(), _query("http_500"), _query("http_504")], capsys)
        assert "succeeded: 2" in out
        assert "failed: 2" in out
        assert "success rate: 50%" in out

    def test_modes_are_broken_out(self, capsys):
        out = _report(
            [_query(mode="corpus"), _query(mode="session"), _query(mode="session")], capsys
        )
        assert "session: 2" in out
        assert "corpus: 1" in out

    def test_failures_are_labelled(self, capsys):
        out = _report([_query("http_504"), _query("http_429")], capsys)
        assert "timed out" in out
        assert "server busy" in out

    def test_no_failure_section_when_everything_worked(self, capsys):
        out = _report([_query(), _query()], capsys)
        assert "failures:" not in out


class TestLatency:
    def test_reports_median_and_tail(self, capsys):
        events = [_query(seconds=str(s)) for s in (1, 2, 3, 4, 100)]
        out = _report(events, capsys)
        assert "median 3.0s" in out
        assert "slowest 100.0s" in out

    def test_unparseable_seconds_are_skipped(self, capsys):
        out = _report([_query(seconds="?"), _query(seconds="2.0")], capsys)
        assert "median 2.0s" in out


class TestRetrieval:
    def test_average_chunks(self, capsys):
        out = _report([_query(docs="4"), _query(docs="6")], capsys)
        assert "5.0 chunks average" in out

    def test_empty_retrievals_are_called_out(self, capsys):
        """A query that retrieved nothing answered from nothing -- worth seeing."""
        out = _report([_query(docs="0"), _query(docs="4")], capsys)
        assert "1 retrieved nothing" in out

    def test_no_note_when_everything_retrieved_something(self, capsys):
        out = _report([_query(docs="3")], capsys)
        assert "retrieved nothing" not in out


class TestTokens:
    def test_totals_and_measured_denominator(self, capsys):
        """Only some providers report usage; the report must not imply otherwise."""
        events = [
            _query(tokens_in="1000", tokens_out="100"),
            _query(tokens_in="500", tokens_out="50"),
            _query(),  # no usage reported
        ]
        out = _report(events, capsys)
        assert "1,500 in" in out
        assert "150 out" in out
        assert "measured on 2/3 attempts" in out

    def test_token_line_omitted_when_nothing_measured(self, capsys):
        out = _report([_query(), _query()], capsys)
        assert "tokens:" not in out


@pytest.mark.parametrize(
    "value,expected", [("1.5", True), ("0", True), ("", False), ("?", False), (None, False)]
)
def test_is_number(value, expected):
    assert ur._is_number(value) is expected
