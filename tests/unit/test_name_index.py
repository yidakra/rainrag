"""Name search reads the metadata folder once, then only what changed (issue #122).

Production holds about 107,000 ``{hash}.json`` files (1 GB). The old search
parsed every one of them on every request and took more than 60 s. These tests
count the parses instead of timing them, so they are deterministic.
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rainrag import ingest
from rainrag.ingest import WebMetadataLoader, name_index_for, warm_name_index


@pytest.fixture(autouse=True)
def fresh_process_caches():
    ingest._NAME_INDEXES.clear()
    ingest._API_BATCH_CACHE.clear()
    yield
    ingest._NAME_INDEXES.clear()
    ingest._API_BATCH_CACHE.clear()


def _hash(i: int) -> str:
    return f"{i:040x}"


def _write(folder: Path, i: int, name: str, date: str = "2026-01-01T00:00:00", **extra) -> Path:
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
    calls = {"n": 0}
    real = json.load

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ingest.json, "load", counting)
    return calls


def test_search_matches_and_ranks_like_before(tmp_path):
    _write(tmp_path, 1, "Навальный: последнее интервью", "2024-01-01")
    _write(tmp_path, 2, "навальный", "2023-01-01")
    _write(tmp_path, 3, "Интервью о Навальном", "2025-01-01")
    _write(tmp_path, 4, "Про Навального и Навальный штаб", "2022-01-01")
    _write(tmp_path, 5, "Тимченко", "2025-06-01")
    results = WebMetadataLoader(tmp_path).search_by_name("Навальный")
    # exact name first, then prefix, then substring; newest first inside a tier
    assert [r["video_hash"] for r in results] == [_hash(2), _hash(1), _hash(4)]


def test_results_carry_only_the_fields_the_api_returns(tmp_path):
    _write(tmp_path, 1, "Новости")
    (result,) = WebMetadataLoader(tmp_path).search_by_name("новости")
    assert set(result) == {"video_hash", "name", "date_active_start", "url", "teleshow"}
    assert result["teleshow"] == {"name": "Show"}


def test_a_second_search_parses_nothing(tmp_path, parses):
    for i in range(50):
        _write(tmp_path, i, f"Выпуск {i}")
    loader = WebMetadataLoader(tmp_path)
    assert len(loader.search_by_name("выпуск")) == 50
    assert parses["n"] == 50
    # A new loader per request, as the API does: the index is process-wide.
    assert len(WebMetadataLoader(tmp_path).search_by_name("выпуск 1")) == 11
    assert parses["n"] == 50


def test_only_a_new_or_changed_file_is_parsed(tmp_path, parses):
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
    _write(tmp_path, 1, "Выпуск 1")
    gone = _write(tmp_path, 2, "Выпуск 2")
    loader = WebMetadataLoader(tmp_path)
    assert len(loader.search_by_name("выпуск")) == 2
    gone.unlink()
    _touch_dir(tmp_path)
    assert [r["video_hash"] for r in loader.search_by_name("выпуск")] == [_hash(1)]


def test_broken_and_temp_files_are_skipped(tmp_path):
    _write(tmp_path, 1, "Выпуск")
    (tmp_path / f"{_hash(2)}.json").write_text("{not json", encoding="utf-8")
    (tmp_path / f"{_hash(3)}.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / f"{_hash(4)}.json.abc.tmp").write_text('{"name": "Выпуск"}', encoding="utf-8")
    assert [r["video_hash"] for r in WebMetadataLoader(tmp_path).search_by_name("выпуск")] == [
        _hash(1)
    ]


def test_an_in_place_edit_is_seen_after_max_age(tmp_path):
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
    assert WebMetadataLoader(tmp_path / "absent").search_by_name("выпуск") == []


def test_warm_builds_the_index_before_the_first_search(tmp_path, parses):
    for i in range(5):
        _write(tmp_path, i, f"Выпуск {i}")
    warm_name_index(tmp_path)
    assert parses["n"] == 5
    assert len(WebMetadataLoader(tmp_path).search_by_name("выпуск")) == 5
    assert parses["n"] == 5


def test_api_mode_does_not_read_the_local_folder(tmp_path, parses):
    _write(tmp_path, 1, "Локальный выпуск")
    client = MagicMock(base_url="https://library.example")
    client.export_batch.return_value = [{"video_hash": _hash(9), "name": "Выпуск из API"}]
    results = WebMetadataLoader(tmp_path, source="api", api_client=client).search_by_name("выпуск")
    assert [r["video_hash"] for r in results] == [_hash(9)]
    assert parses["n"] == 0


def test_the_api_batch_is_shared_by_loaders_until_the_ttl(tmp_path, monkeypatch):
    client = MagicMock(base_url="https://library.example")
    client.export_batch.return_value = [{"video_hash": _hash(7), "name": "Выпуск из API"}]
    for _ in range(3):
        loader = WebMetadataLoader(tmp_path, source="hybrid", api_client=client)
        assert [r["video_hash"] for r in loader.search_by_name("выпуск")] == [_hash(7)]
    assert client.export_batch.call_count == 1

    monkeypatch.setenv("RAINRAG_WEB_METADATA_BATCH_TTL_SECONDS", "0")
    WebMetadataLoader(tmp_path, source="hybrid", api_client=client).search_by_name("выпуск")
    assert client.export_batch.call_count == 2
