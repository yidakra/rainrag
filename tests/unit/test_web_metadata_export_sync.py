"""Recovering CMS articles the per-hash lookup cannot see.

Verified on 2026-09-29: for some videos `GET /video/{hash}/article` answers 404
while the article exists and its page returns 200 on tvrain.tv. The bulk export
does not go through the same hash mapping and has them.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from web_metadata_export_sync import (  # noqa: E402
    articles_in_period,
    drop_from_misses,
    windows,
    write_article,
)


def test_windows_cover_the_period_without_gaps_or_overlap():
    since, until = dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 1)
    spans = windows(since, until, 60)
    assert spans[0][0] == since
    assert spans[-1][1] == until
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == next_start


def test_a_period_shorter_than_one_window_is_a_single_span():
    since, until = dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 5)
    assert windows(since, until, 179) == [(since, until)]


class _Client:
    """Fails whole windows, succeeds on the narrower retry slices."""

    def __init__(self, fail_wide: bool = False, fail_all: bool = False):
        self.fail_wide = fail_wide
        self.fail_all = fail_all
        self.calls: list[int] = []

    def export_batch(self, start_time: int, end_time: int):
        span_days = (end_time - start_time) / 86400
        self.calls.append(int(span_days))
        if self.fail_all or (self.fail_wide and span_days > 100):
            raise RuntimeError("500 Internal Server Error")
        # A real 40-hex hash: the writer refuses anything else.
        return [{"video_hash": f"{start_time:040x}", "url": "https://tvrain.tv/x", "name": "x"}]


def test_a_failed_window_is_retried_in_slices_rather_than_lost():
    client = _Client(fail_wide=True)
    found, failed = articles_in_period(client, dt.datetime(2024, 1, 1), dt.datetime(2024, 7, 1))
    assert found, "the narrower retry should have recovered articles"
    assert failed == []
    assert max(client.calls) > 100 and min(client.calls) <= 59


def test_windows_that_fail_even_when_sliced_are_reported_not_swallowed():
    """The export 500s on parts of 2019-2022; claiming full coverage would lie."""
    client = _Client(fail_all=True)
    found, failed = articles_in_period(client, dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1))
    assert found == {}
    assert failed, "the caller must be able to say which periods went unchecked"


def test_articles_without_a_hash_are_skipped():
    class Nameless:
        def export_batch(self, start_time: int, end_time: int):
            return [{"url": "https://tvrain.tv/x"}, {"video_hash": "", "url": "y"}]

    found, _ = articles_in_period(Nameless(), dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 1))
    assert found == {}


def test_write_article_publishes_by_rename(tmp_path):
    video_hash = "a" * 40
    write_article(tmp_path, video_hash, {"video_hash": video_hash, "name": "Заголовок"})
    written = json.loads((tmp_path / f"{video_hash}.json").read_text(encoding="utf-8"))
    assert written["name"] == "Заголовок"
    # No temporary files left behind.
    assert [p.name for p in tmp_path.iterdir()] == [f"{video_hash}.json"]


def test_recovered_hashes_leave_the_misses_file(tmp_path):
    """Otherwise the backfill keeps skipping the videos we just found."""
    misses = tmp_path / "misses.txt"
    misses.write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    assert drop_from_misses(misses, {"bbb"}) == 1
    assert misses.read_text(encoding="utf-8").split() == ["aaa", "ccc"]


def test_dropping_nothing_leaves_the_file_untouched(tmp_path):
    misses = tmp_path / "misses.txt"
    misses.write_text("aaa\nbbb\n", encoding="utf-8")
    before = misses.read_text(encoding="utf-8")
    assert drop_from_misses(misses, {"zzz"}) == 0
    assert drop_from_misses(misses, set()) == 0
    assert misses.read_text(encoding="utf-8") == before


def test_a_missing_misses_file_is_not_an_error(tmp_path):
    assert drop_from_misses(tmp_path / "nope.txt", {"aaa"}) == 0


@pytest.mark.parametrize("content", ["", "\n\n"])
def test_an_empty_misses_file_survives(tmp_path, content):
    misses = tmp_path / "misses.txt"
    misses.write_text(content, encoding="utf-8")
    assert drop_from_misses(misses, {"aaa"}) == 0


class TestHashSafety:
    """Tenki on #93: the export is remote input and the hash becomes a filename."""

    def test_a_valid_hash_is_lowercased(self):
        from web_metadata_export_sync import safe_hash

        assert safe_hash("A" * 40) == "a" * 40

    @pytest.mark.parametrize(
        "bad",
        [
            "../../etc/passwd",
            "/absolute/path",
            "sub/dir",
            "a" * 39,
            "a" * 41,
            "z" * 40,
            "",
            "   ",
            None,
            12345,
        ],
    )
    def test_anything_that_is_not_a_bare_40_hex_string_is_refused(self, bad):
        from web_metadata_export_sync import safe_hash

        assert safe_hash(bad) is None

    def test_write_article_refuses_an_unsafe_hash_rather_than_writing_it(self, tmp_path):
        from web_metadata_export_sync import write_article

        with pytest.raises(ValueError):
            write_article(tmp_path, "../escape", {"name": "x"})
        assert list(tmp_path.iterdir()) == []

    def test_articles_with_an_unusable_hash_never_reach_the_cache(self):
        class Hostile:
            def export_batch(self, start_time: int, end_time: int):
                return [
                    {"video_hash": "../../etc/passwd", "name": "bad"},
                    {"video_hash": "b" * 40, "name": "good"},
                ]

        found, _ = articles_in_period(Hostile(), dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 1))
        assert list(found) == ["b" * 40]


def test_an_auth_failure_is_raised_not_counted_as_a_flaky_window():
    """A stale token would otherwise look exactly like the API's known 500s."""
    import httpx

    class Unauthorised:
        def export_batch(self, start_time: int, end_time: int):
            request = httpx.Request("GET", "https://library.tvrain.tv/article/export")
            raise httpx.HTTPStatusError(
                "401", request=request, response=httpx.Response(401, request=request)
            )

    with pytest.raises(httpx.HTTPStatusError):
        articles_in_period(Unauthorised(), dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 1))


def test_a_server_error_is_still_treated_as_a_flaky_window():
    import httpx

    class Broken:
        def export_batch(self, start_time: int, end_time: int):
            request = httpx.Request("GET", "https://library.tvrain.tv/article/export")
            raise httpx.HTTPStatusError(
                "500", request=request, response=httpx.Response(500, request=request)
            )

    found, failed = articles_in_period(Broken(), dt.datetime(2024, 1, 1), dt.datetime(2024, 2, 1))
    assert found == {} and failed


def test_naive_dates_are_read_as_utc_not_as_the_hosts_timezone():
    """Local-time boundaries shift every window and can drop edge articles."""
    from web_metadata_export_sync import _as_utc

    naive = dt.datetime(2024, 1, 1)
    assert _as_utc(naive).tzinfo is dt.timezone.utc
    assert _as_utc(naive).timestamp() == dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    aware = dt.datetime(2024, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=3)))
    assert _as_utc(aware) == aware
