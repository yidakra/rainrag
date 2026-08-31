"""Tests for the tagging batch's data loading.

The batch is the only thing that reads 139k CMS articles and a folded archive,
so how it loads them decides whether a tagging run can share the box with a
reindex. These cover that, not the tagging itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    d = tmp_path / "web_metadata"
    d.mkdir()
    (d / "abc123.json").write_text(
        json.dumps({"video_hash": "abc123", "id": 4711, "name": "Тест"}), encoding="utf-8"
    )
    (d / "broken.json").write_text("{not json", encoding="utf-8")
    return d


def test_load_cms_article_reads_one_file_by_hash(metadata_dir: Path):
    """The cache is named by hash, so no index is needed to find an article."""
    from library_tag_batch import load_cms_article

    assert load_cms_article(metadata_dir, "abc123")["id"] == 4711


def test_load_cms_article_returns_empty_for_missing_or_corrupt(metadata_dir: Path):
    """A tagging run covers thousands of episodes; one bad file is not fatal."""
    from library_tag_batch import load_cms_article

    assert load_cms_article(metadata_dir, "nosuchhash") == {}
    assert load_cms_article(metadata_dir, "broken") == {}


def test_videos_cache_round_trips_the_fields_the_tagger_uses(tmp_path: Path):
    from library_catalogue import Video
    from library_tag_batch import load_videos_cache, write_videos_cache

    videos = {
        "h1": Video(
            video_key="h1",
            title="Лекция",
            program="Лекции на Дожде",
            date="2019-05-01",
            duration_seconds=2400.0,
            url="https://tvrain.tv/x",
            presenters=["Ирина Хакамада"],
            themes=["политика"],
            chunks=42,
        )
    }
    path = tmp_path / "videos.jsonl"
    assert write_videos_cache(path, videos) == 1

    loaded = load_videos_cache(path)
    assert set(loaded) == {"h1"}
    v = loaded["h1"]
    assert (v.title, v.program, v.date) == ("Лекция", "Лекции на Дожде", "2019-05-01")
    assert v.duration_seconds == 2400.0
    assert v.presenters == ["Ирина Хакамада"]
    # Catalogue-only fields are deliberately not carried: the tagger never
    # reads them and they are what makes the raw scroll expensive.
    assert v.themes == []
    assert v.chunks == 0


def test_videos_cache_skips_blank_lines(tmp_path: Path):
    from library_tag_batch import load_videos_cache

    path = tmp_path / "videos.jsonl"
    path.write_text(json.dumps({"video_key": "h1", "title": "A"}) + "\n\n", encoding="utf-8")
    assert set(load_videos_cache(path)) == {"h1"}


def test_stale_videos_cache_is_refolded_when_raw_cache_is_newer(tmp_path, capsys):
    """After a reindex regenerates the scroll, the folded cache must not win."""
    import os

    import library_tag_batch as batch

    raw = tmp_path / "raw.json"
    cache = tmp_path / "videos.jsonl"
    cache.write_text(
        json.dumps({"video_key": "old", "title": "устаревшее"}) + "\n", encoding="utf-8"
    )
    raw.write_text(json.dumps([{"path": "new.ru.vtt", "web_title": "свежее"}]), encoding="utf-8")
    os.utime(cache, (1_000_000, 1_000_000))
    os.utime(raw, (2_000_000, 2_000_000))

    # Exercise main's loading branch via a dry run against the tmp files.
    rc = batch.main(
        [
            "--raw-cache",
            str(raw),
            "--videos-cache",
            str(cache),
            "--gold",
            str(tmp_path / "missing_gold.json"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "refolding" in out
    # And the cache now holds the refolded archive, not the stale row.
    reloaded = batch.load_videos_cache(cache)
    assert set(reloaded) == {"new"}
