"""Name search reads the metadata folder once, then only what changed (issue #122).

Production holds about 107,000 ``{hash}.json`` files (1 GB). The old search
parsed every one of them on every request and took more than 60 s. These tests
count the parses instead of timing them, so they are deterministic.
"""

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rainrag import ingest
from rainrag.ingest import WebMetadataLoader, name_index_for, warm_name_index


@pytest.fixture(autouse=True)
def fresh_process_caches():
    """Clear the process-wide index and API caches around each test."""
    for cache in (
        ingest._NAME_INDEXES,
        ingest._API_BATCH_CACHE,
        ingest._API_BATCH_FETCH_LOCKS,
        ingest._API_BATCH_FAILED_AT,
    ):
        cache.clear()
    yield
    for cache in (
        ingest._NAME_INDEXES,
        ingest._API_BATCH_CACHE,
        ingest._API_BATCH_FETCH_LOCKS,
        ingest._API_BATCH_FAILED_AT,
    ):
        cache.clear()


def _hash(i: int) -> str:
    """Return a 40-hex video hash for a test number."""
    return f"{i:040x}"


def _write(folder: Path, i: int, name: str, date: str = "2026-01-01T00:00:00", **extra) -> Path:
    """Write one metadata record and return its path."""
    path = folder / f"{_hash(i)}.json"
    record = {
        "video_hash": _hash(i),
        "name": name,
        "date_active_start": date,
        "url": f"https://tvrain.tv/v/{i}/",
        "teleshow": {"name": "Show"},
        "description": "x" * 200,
        **extra,
    }
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


def _touch_dir(folder: Path) -> None:
    """Move the folder mtime forward, as a rename by a writer would."""
    st = folder.stat()
    os.utime(folder, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def parses(monkeypatch):
    """Count the json.load calls that the index makes."""
    calls = {"n": 0}
    real = json.load

    def counting(*args, **kwargs):
        """Count one parse, then parse as json.load does."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ingest.json, "load", counting)
    return calls


def test_search_matches_and_ranks_like_before(tmp_path):
    """Matching and ranking are the same as before the index."""
    _write(tmp_path, 1, "Навальный: последнее интервью", "2024-01-01")
    _write(tmp_path, 2, "навальный", "2023-01-01")
    _write(tmp_path, 3, "Интервью о Навальном", "2025-01-01")
    _write(tmp_path, 4, "Про Навального и Навальный штаб", "2022-01-01")
    _write(tmp_path, 5, "Тимченко", "2025-06-01")
    results = WebMetadataLoader(tmp_path).search_by_name("Навальный")
    # exact name first, then prefix, then substring; newest first inside a tier
    assert [r["video_hash"] for r in results] == [_hash(2), _hash(1), _hash(4)]


def test_results_carry_only_the_fields_the_api_returns(tmp_path):
    """A result holds only the five fields the route returns."""
    _write(tmp_path, 1, "Новости")
    (result,) = WebMetadataLoader(tmp_path).search_by_name("новости")
    assert set(result) == {"video_hash", "name", "date_active_start", "url", "teleshow"}
    assert result["teleshow"] == {"name": "Show"}


def test_a_second_search_parses_nothing(tmp_path, parses):
    """A second search, also from a new loader, parses no file."""
    for i in range(50):
        _write(tmp_path, i, f"Выпуск {i}")
    loader = WebMetadataLoader(tmp_path)
    assert len(loader.search_by_name("выпуск")) == 50
    assert parses["n"] == 50
    # A new loader per request, as the API does: the index is process-wide.
    assert len(WebMetadataLoader(tmp_path).search_by_name("выпуск 1")) == 11
    assert parses["n"] == 50


def test_only_a_new_or_changed_file_is_parsed(tmp_path, parses):
    """After a change, only the new and the changed file are parsed."""
    for i in range(10):
        _write(tmp_path, i, f"Выпуск {i}")
    loader = WebMetadataLoader(tmp_path)
    loader.search_by_name("выпуск")
    assert parses["n"] == 10

    _write(tmp_path, 10, "Новый выпуск")
    changed = _write(tmp_path, 3, "Выпуск 3, исправлен")
    st = changed.stat()
    os.utime(changed, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _touch_dir(tmp_path)

    names = {r["name"] for r in loader.search_by_name("выпуск")}
    assert parses["n"] == 12
    assert "Новый выпуск" in names
    assert "Выпуск 3, исправлен" in names
    assert "Выпуск 3" not in names


def test_a_deleted_file_leaves_the_index(tmp_path):
    """A deleted file drops out of the results."""
    _write(tmp_path, 1, "Выпуск 1")
    gone = _write(tmp_path, 2, "Выпуск 2")
    loader = WebMetadataLoader(tmp_path)
    assert len(loader.search_by_name("выпуск")) == 2
    gone.unlink()
    _touch_dir(tmp_path)
    assert [r["video_hash"] for r in loader.search_by_name("выпуск")] == [_hash(1)]


def test_broken_and_temp_files_are_skipped(tmp_path):
    """Broken, non-object and temp files give no result."""
    _write(tmp_path, 1, "Выпуск")
    (tmp_path / f"{_hash(2)}.json").write_text("{not json", encoding="utf-8")
    (tmp_path / f"{_hash(3)}.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / f"{_hash(4)}.json.abc.tmp").write_text('{"name": "Выпуск"}', encoding="utf-8")
    assert [r["video_hash"] for r in WebMetadataLoader(tmp_path).search_by_name("выпуск")] == [
        _hash(1)
    ]


def test_an_in_place_edit_is_seen_after_max_age(tmp_path):
    """An edit that keeps the folder mtime is seen after max_age."""
    path = _write(tmp_path, 1, "Старое имя")
    loader = WebMetadataLoader(tmp_path)
    assert loader.search_by_name("старое")
    # An in-place write does not change the folder mtime; max_age covers it.
    record = json.loads(path.read_text(encoding="utf-8"))
    record["name"] = "Новое имя"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    name_index_for(tmp_path)._max_age = 0.0
    time.sleep(0.01)
    assert loader.search_by_name("новое")
    assert not loader.search_by_name("старое")


def test_a_missing_folder_gives_no_results(tmp_path):
    """A folder that does not exist gives no results."""
    assert WebMetadataLoader(tmp_path / "absent").search_by_name("выпуск") == []


def test_warm_builds_the_index_before_the_first_search(tmp_path, parses):
    """Warm-up parses the files, so the first search parses none."""
    for i in range(5):
        _write(tmp_path, i, f"Выпуск {i}")
    warm_name_index(tmp_path)
    assert parses["n"] == 5
    assert len(WebMetadataLoader(tmp_path).search_by_name("выпуск")) == 5
    assert parses["n"] == 5


def test_api_mode_does_not_read_the_local_folder(tmp_path, parses):
    """In api mode no local file is parsed."""
    _write(tmp_path, 1, "Локальный выпуск")
    client = MagicMock(base_url="https://library.example")
    client.export_batch.return_value = [{"video_hash": _hash(9), "name": "Выпуск из API"}]
    results = WebMetadataLoader(tmp_path, source="api", api_client=client).search_by_name("выпуск")
    assert [r["video_hash"] for r in results] == [_hash(9)]
    assert parses["n"] == 0


def test_the_api_batch_is_shared_by_loaders_until_the_ttl(tmp_path, monkeypatch):
    """New loaders reuse one API batch until the TTL ends."""
    client = MagicMock(base_url="https://library.example")
    client.export_batch.return_value = [{"video_hash": _hash(7), "name": "Выпуск из API"}]
    for _ in range(3):
        loader = WebMetadataLoader(tmp_path, source="hybrid", api_client=client)
        assert [r["video_hash"] for r in loader.search_by_name("выпуск")] == [_hash(7)]
    assert client.export_batch.call_count == 1

    monkeypatch.setenv("RAINRAG_WEB_METADATA_BATCH_TTL_SECONDS", "0")
    WebMetadataLoader(tmp_path, source="hybrid", api_client=client).search_by_name("выпуск")
    assert client.export_batch.call_count == 2


def test_an_unreadable_folder_gives_no_results_not_an_error(tmp_path, monkeypatch):
    """A folder that cannot be listed gives no results, not an error."""
    _write(tmp_path, 1, "Выпуск")
    loader = WebMetadataLoader(tmp_path)
    assert loader.search_by_name("выпуск")

    def fail(_path):
        """Raise as an unreadable folder does."""
        raise PermissionError("denied")

    monkeypatch.setattr(ingest.os, "scandir", fail)
    _touch_dir(tmp_path)
    assert loader.search_by_name("выпуск") == []


def test_a_failed_warm_up_is_logged_not_raised(tmp_path, monkeypatch):
    """A failing warm-up logs the error and does not raise."""

    def boom(_path):
        """Fail the way a broken index lookup would."""
        raise RuntimeError("broken")

    monkeypatch.setattr(ingest, "name_index_for", boom)
    warm_name_index(tmp_path)  # must not raise


def test_a_search_during_a_refresh_does_not_wait_for_it(tmp_path, monkeypatch):
    """While a scan runs, another search gets the current entries at once."""
    _write(tmp_path, 1, "Выпуск 1")
    index = name_index_for(tmp_path)
    assert len(index.entries(tmp_path)) == 1

    started, release = threading.Event(), threading.Event()
    real_scan = ingest._NameIndex._scan

    def slow_scan(path, known):
        """Signal the start, wait for the test, then scan."""
        started.set()
        release.wait(5)
        return real_scan(path, known)

    monkeypatch.setattr(ingest._NameIndex, "_scan", staticmethod(slow_scan))
    _write(tmp_path, 2, "Выпуск 2")
    _touch_dir(tmp_path)
    refresher = threading.Thread(target=index.entries, args=(tmp_path,))
    refresher.start()
    assert started.wait(5)

    t0 = time.monotonic()
    current = index.entries(tmp_path)  # the scan is still blocked
    assert time.monotonic() - t0 < 1.0
    assert [e["video_hash"] for e in current] == [_hash(1)]

    release.set()
    refresher.join(5)
    assert sorted(e["video_hash"] for e in index.entries(tmp_path)) == [_hash(1), _hash(2)]


def test_an_empty_api_batch_is_cached_too(tmp_path):
    """An empty API batch is fetched once within the TTL."""
    client = MagicMock(base_url="https://library.example")
    client.export_batch.return_value = []
    for _ in range(3):
        WebMetadataLoader(tmp_path, source="api", api_client=client).search_by_name("выпуск")
    assert client.export_batch.call_count == 1


def test_concurrent_requests_fetch_the_api_batch_once(tmp_path):
    """Four concurrent requests make one API fetch."""
    client = MagicMock(base_url="https://library.example")

    def slow_export():
        """Return one record after a short delay."""
        time.sleep(0.3)
        return [{"video_hash": _hash(7), "name": "Выпуск из API"}]

    client.export_batch.side_effect = slow_export
    results: list[list] = []

    def search():
        """Run one name search from a new loader and keep its results."""
        loader = WebMetadataLoader(tmp_path, source="api", api_client=client)
        results.append(loader.search_by_name("выпуск"))

    threads = [threading.Thread(target=search) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(5)
    assert client.export_batch.call_count == 1
    assert all([r["video_hash"] for r in res] == [_hash(7)] for res in results)
    assert len(results) == 4


def test_a_failed_fetch_is_not_retried_by_every_waiting_request(tmp_path, monkeypatch):
    """A failed fetch is not retried by each waiting request."""
    _write(tmp_path, 1, "Локальный выпуск")
    client = MagicMock(base_url="https://library.example")

    def failing_export():
        """Fail after a short delay, as an unreachable library would."""
        time.sleep(0.2)
        raise ConnectionError("library down")

    client.export_batch.side_effect = failing_export
    results: list[list] = []

    def search():
        """Run one name search from a new loader and keep its results."""
        loader = WebMetadataLoader(tmp_path, source="hybrid", api_client=client)
        results.append(loader.search_by_name("выпуск"))

    threads = [threading.Thread(target=search) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(5)
    assert client.export_batch.call_count == 1
    # Every search still answers from the local folder.
    assert len(results) == 4
    assert all([r["video_hash"] for r in res] == [_hash(1)] for res in results)

    # After the retry window the next request asks the API again.
    monkeypatch.setattr(ingest, "_API_BATCH_RETRY_SECONDS", 0.0)
    client.export_batch.side_effect = None
    client.export_batch.return_value = []
    WebMetadataLoader(tmp_path, source="hybrid", api_client=client).search_by_name("выпуск")
    assert client.export_batch.call_count == 2


def test_waiting_requests_do_not_wait_for_the_disk_writes(tmp_path, monkeypatch):
    """A second request gets the batch while the first still writes files."""
    client = MagicMock(base_url="https://library.example")
    client.export_batch.return_value = [{"video_hash": _hash(7), "name": "Выпуск из API"}]
    persisting, release = threading.Event(), threading.Event()
    real_write = WebMetadataLoader._write_cache_record

    def slow_write(cache_file, data):
        """Signal the write, wait for the test, then write."""
        persisting.set()
        release.wait(5)
        real_write(cache_file, data)

    monkeypatch.setattr(WebMetadataLoader, "_write_cache_record", staticmethod(slow_write))
    first = threading.Thread(
        target=lambda: WebMetadataLoader(tmp_path, source="api", api_client=client).search_by_name(
            "выпуск"
        )
    )
    first.start()
    assert persisting.wait(5)

    t0 = time.monotonic()
    loader = WebMetadataLoader(tmp_path, source="api", api_client=client)
    assert [r["video_hash"] for r in loader.search_by_name("выпуск")] == [_hash(7)]
    assert time.monotonic() - t0 < 1.0
    assert client.export_batch.call_count == 1

    release.set()
    first.join(5)
