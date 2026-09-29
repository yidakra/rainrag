"""Recovering CMS articles the per-hash lookup cannot see.

Verified on 2026-09-29: for some videos `GET /video/{hash}/article` answers 404
while the article exists and its page returns 200 on tvrain.tv. The bulk export
does not go through the same hash mapping and has them.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from web_metadata_export_sync import (  # noqa: E402
    articles_in_period,
    drop_from_misses,
    retry_delay,
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


def test_a_hash_recovered_into_an_existing_file_still_leaves_the_misses_file(tmp_path, monkeypatch):
    """A run that crashed between writing the article and clearing the misses.

    The hash then sits in both places, and the backfill consults the misses
    file before the cache, so that video would be skipped forever with its
    article already on disk. Clearing only the newly written hashes leaves it
    stuck (Tenki on #93).
    """
    from web_metadata_export_sync import main

    import rainrag.web_metadata_api as api

    video_hash = "b" * 40
    directory = tmp_path / "web_metadata"
    directory.mkdir()
    (directory / f"{video_hash}.json").write_text(
        json.dumps({"video_hash": video_hash, "name": "уже записано"}), encoding="utf-8"
    )
    misses = tmp_path / "misses.txt"
    misses.write_text(f"{video_hash}\n{'c' * 40}\n", encoding="utf-8")

    class _Export:
        def export_batch(self, start_time: int, end_time: int):
            return [{"video_hash": video_hash, "url": "https://tvrain.tv/x", "name": "x"}]

    monkeypatch.setattr("rainrag.config.load_config", lambda p: _FakeConfig(directory))
    monkeypatch.setattr(
        api.WebMetadataAPIClient, "from_env", classmethod(lambda cls, **kw: _Export())
    )
    assert (
        main(
            [
                "--since",
                "2024-01-01",
                "--until",
                "2024-01-10",
                "--metadata-dir",
                str(directory),
                "--misses-file",
                str(misses),
            ]
        )
        == 0
    )

    assert misses.read_text(encoding="utf-8").split() == ["c" * 40]


class _FakeConfig:
    """Just the `web_metadata` section the script reads."""

    def __init__(self, path):
        self.web_metadata = type(
            "_Section",
            (),
            {
                "path": str(path),
                "api_url": "https://cms.test",
                "api_token_env": "DEPLOY_TOKEN",
            },
        )()


class _Response:
    def __init__(self, status_code: int, retry_after: str | None = None):
        self.status_code = status_code
        self.headers = {"Retry-After": retry_after} if retry_after else {}


def _throttler(monkeypatch, *, always: bool):
    """A client that answers 429, and a record of every sleep it caused."""
    import httpx
    import web_metadata_export_sync as sync

    slept: list[float] = []
    monkeypatch.setattr(sync.time, "sleep", slept.append)

    class _Throttled:
        def __init__(self):
            self.calls: list[int] = []

        def export_batch(self, start_time: int, end_time: int):
            self.calls.append(int((end_time - start_time) / 86400))
            if always or len(self.calls) == 1:
                raise httpx.HTTPStatusError(
                    "429", request=httpx.Request("GET", "https://x"), response=_Response(429, "7")
                )
            return [{"video_hash": f"{start_time:040x}", "url": "u", "name": "n"}]

    return _Throttled(), slept


def test_a_throttled_window_is_waited_out_not_sliced_into_more_requests(monkeypatch):
    """Slicing a 429 turns one refused call into four while the server sheds
    load, which is how a soft limit becomes a blocked token (Tenki on #93)."""
    client, slept = _throttler(monkeypatch, always=True)
    found, failed = articles_in_period(client, dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1))

    assert found == {}
    assert failed == [(dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1))], "the window, not slices"
    assert len(client.calls) == 4, "the first call plus the retry budget, and no slice retries"
    assert slept == [7.0, 7.0, 7.0], "the server's own Retry-After, honoured each time"


def test_a_window_that_recovers_after_waiting_is_kept(monkeypatch):
    client, slept = _throttler(monkeypatch, always=False)
    found, failed = articles_in_period(client, dt.datetime(2024, 1, 1), dt.datetime(2024, 3, 1))

    assert found and failed == []
    assert slept == [7.0]


def test_the_wait_falls_back_to_backoff_when_the_server_names_no_period():
    """And is capped: an hour-long Retry-After would hang a hand-run script."""
    assert retry_delay(_Response(429), 0) < retry_delay(_Response(429), 3)
    assert retry_delay(_Response(429), 99) == retry_delay(_Response(429, "99999"), 0)
    assert retry_delay(_Response(429, "12"), 0) == 12.0
    assert retry_delay(_Response(429, "not a number"), 0) == retry_delay(_Response(429), 0)


def test_a_published_article_survives_a_crash_right_after_the_rename(tmp_path, monkeypatch):
    """The rename is only durable once the directory entry is flushed too."""
    import web_metadata_export_sync as sync

    synced: list[bool] = []
    real = sync._fsync_dir
    monkeypatch.setattr(sync, "_fsync_dir", lambda d: (synced.append(True), real(d))[1])
    sync.write_article(tmp_path, "d" * 40, {"name": "x"})
    misses = tmp_path / "m.txt"
    misses.write_text("d" * 40 + "\n", encoding="utf-8")
    sync.drop_from_misses(misses, {"d" * 40})

    assert len(synced) == 2, "both the cache write and the misses rewrite"


def test_a_failed_misses_rewrite_leaves_no_temporary_file_behind(tmp_path, monkeypatch):
    """A full disk would otherwise litter data/ with a dot-file per run."""
    import web_metadata_export_sync as sync

    misses = tmp_path / "misses.txt"
    misses.write_text("e" * 40 + "\n", encoding="utf-8")

    def _boom(*a, **kw):
        raise OSError("No space left on device")

    monkeypatch.setattr(sync.os, "fsync", _boom)
    with pytest.raises(OSError):
        sync.drop_from_misses(misses, {"e" * 40})

    assert [p.name for p in tmp_path.iterdir()] == ["misses.txt"]
    assert misses.read_text(encoding="utf-8").split() == ["e" * 40], "untouched"


def test_a_published_article_is_readable_like_the_rest_of_the_cache(tmp_path):
    """mkstemp creates 0600, and a bare rename would publish that: a file in
    the middle of the shared cache that no other account can open."""
    import web_metadata_export_sync as sync

    misses = tmp_path / "misses.txt"
    misses.write_text("f" * 40 + "\n", encoding="utf-8")
    sync.write_article(tmp_path, "f" * 40, {"name": "x"})
    sync.drop_from_misses(misses, {"f" * 40})

    mask = os.umask(0)
    os.umask(mask)
    expected = 0o666 & ~mask
    assert (tmp_path / f"{'f' * 40}.json").stat().st_mode & 0o777 == expected
    assert misses.stat().st_mode & 0o777 == expected


def test_a_unicode_digit_in_retry_after_falls_back_instead_of_crashing():
    """``str.isdigit`` is true for superscripts that ``float`` then refuses."""
    assert retry_delay(_Response(429, "²"), 0) == retry_delay(_Response(429), 0)


def test_the_defaults_resolve_where_the_backfill_looks(monkeypatch, tmp_path):
    """Anchoring these to the script's own repository would let a deployment
    root run the two against different files, and the recovered article and
    the cleared miss would both be invisible to the backfill."""
    import web_metadata_export_sync as sync

    import rainrag.web_metadata_api as api

    cache = tmp_path / "from-the-config"
    (tmp_path / "data").mkdir()
    (tmp_path / "config.yaml").write_text("unused", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr("rainrag.config.load_config", lambda p: _FakeConfig(cache))

    video_hash = "1" * 40

    class _Export:
        def export_batch(self, start_time: int, end_time: int):
            return [{"video_hash": video_hash, "url": "u", "name": "n"}]

    seen: dict = {}

    def _from_env(cls, **kw):
        seen.update(kw)
        return _Export()

    monkeypatch.setattr(api.WebMetadataAPIClient, "from_env", classmethod(_from_env))
    assert sync.main(["--since", "2024-01-01", "--until", "2024-01-10"]) == 0

    assert seen == {"base_url": "https://cms.test", "token_env": "DEPLOY_TOKEN"}, (
        "the endpoint the backfill uses, not from_env's hardcoded default"
    )

    assert (cache / f"{video_hash}.json").exists(), "the cache directory from the config"


def test_one_failed_write_does_not_abandon_the_rest_of_the_run(tmp_path, monkeypatch, capsys):
    """Raising here jumped past the miss-clearing entirely, leaving every
    article already in the cache with its hash still in the misses file: the
    skip-forever state, produced by an ordinary disk error."""
    import web_metadata_export_sync as sync

    import rainrag.web_metadata_api as api

    good, bad = "2" * 40, "3" * 40
    cache = tmp_path / "web_metadata"
    cache.mkdir()
    misses = tmp_path / "misses.txt"
    misses.write_text(f"{good}\n{bad}\n", encoding="utf-8")

    class _Export:
        def export_batch(self, start_time: int, end_time: int):
            return [{"video_hash": h, "url": "u", "name": "n"} for h in (good, bad)]

    real = sync.write_article

    def _flaky(directory, video_hash, article):
        if video_hash == bad:
            raise OSError("No space left on device")
        return real(directory, video_hash, article)

    monkeypatch.setattr(sync, "write_article", _flaky)
    monkeypatch.setattr("rainrag.config.load_config", lambda p: _FakeConfig(cache))
    monkeypatch.setattr(
        api.WebMetadataAPIClient, "from_env", classmethod(lambda cls, **kw: _Export())
    )
    code = sync.main(
        ["--since", "2024-01-01", "--until", "2024-01-10", "--misses-file", str(misses)]
    )

    assert code == 1, "the run finished, but not everything it found was recovered"
    assert (cache / f"{good}.json").exists(), "the run carried on past the failure"
    assert misses.read_text(encoding="utf-8").split() == [bad], "only the one still missing"
    assert "write failed" in capsys.readouterr().out


def _run(monkeypatch, tmp_path, client, extra=()):
    import web_metadata_export_sync as sync

    import rainrag.web_metadata_api as api

    cache = tmp_path / "web_metadata"
    cache.mkdir(exist_ok=True)
    monkeypatch.setattr("rainrag.config.load_config", lambda p: _FakeConfig(cache))
    monkeypatch.setattr(api.WebMetadataAPIClient, "from_env", classmethod(lambda cls, **kw: client))
    return sync.main(
        [
            "--since",
            "2024-01-01",
            "--until",
            "2024-03-01",
            "--misses-file",
            str(tmp_path / "misses.txt"),
            *extra,
        ]
    )


def test_a_run_that_could_not_check_the_whole_period_exits_non_zero(monkeypatch, tmp_path):
    """Otherwise a run where every window 500s looks like full coverage, and
    the caller concludes those videos genuinely have no article."""
    assert _run(monkeypatch, tmp_path, _Client(fail_all=True)) == 1
    assert _run(monkeypatch, tmp_path, _Client(fail_all=True), ("--dry-run",)) == 1


def test_a_complete_run_exits_zero(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path, _Client()) == 0
