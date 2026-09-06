"""Tests for the Library performance join and aggregation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _map(yt, cid, conf="editor", views=100, vh="h1", **kw):
    return {
        "youtube_id": yt,
        "youtube_title": f"yt {yt}",
        "content_id": cid,
        "confidence": conf,
        "view_count": views,
        "archive_video_hash": vh,
        "published_at": "2025-01-01",
        **kw,
    }


def test_build_uploads_uses_only_trusted_links_and_joins_both_sources():
    from rainrag.library_performance import build_uploads

    videos = {"h1": SimpleNamespace(program="Синдеева", presenters=["Наталья Синдеева"])}
    tags = {"1": {"presenter_cms": ["Наталья Синдеева"], "guest": ["Екатерина Шульман"]}}
    rows = [
        _map("a", "1"),
        _map("b", "2", conf="review"),  # ~40% right against editor pairs: excluded
        _map("c", None, conf="exact"),  # no candidate
    ]
    ups = build_uploads(rows, videos, tags)
    assert [u.youtube_id for u in ups] == ["a"]
    u = ups[0]
    assert u.program == "Синдеева"
    assert u.tagged is True
    # CMS and tagger spellings of one person collapse to one speaker
    assert sorted(u.speakers) == ["Екатерина Шульман", "Наталья Синдеева"]


def test_speaker_spelling_variants_merge_and_prefer_capitalised():
    from rainrag.library_performance import build_uploads

    videos = {"h1": SimpleNamespace(program=None, presenters=[])}
    tags = {"1": {"presenter_cms": ["Ирина Хакамада"], "guest": ["ирина хакамада"]}}
    ups = build_uploads([_map("a", "1")], videos, tags)
    assert ups[0].speakers == ["Ирина Хакамада"]


def test_untagged_episode_still_gets_programme_and_presenters_from_cache():
    """Most Library uploads are under 30 minutes and therefore untagged."""
    from rainrag.library_performance import build_uploads

    videos = {
        "h1": SimpleNamespace(program="Сто лекций с Дмитрием Быковым", presenters=["Дмитрий Быков"])
    }
    ups = build_uploads([_map("a", "1")], videos, {})
    assert ups[0].tagged is False
    assert ups[0].program == "Сто лекций с Дмитрием Быковым"
    assert ups[0].speakers == ["Дмитрий Быков"]


def test_aggregate_by_speaker_reports_total_median_and_best():
    from rainrag.library_performance import aggregate, build_uploads

    videos = {"h1": SimpleNamespace(program="P", presenters=["A"])}
    ups = build_uploads(
        [_map("x", "1", views=1000), _map("y", "2", views=100), _map("z", "3", views=10)],
        videos,
        {},
    )
    rows = aggregate(ups, "speaker")
    assert rows == [
        {"speaker": "A", "uploads": 3, "total": 1110.0, "median": 100.0, "best": 1000.0}
    ]


def test_load_metrics_keeps_latest_snapshot_and_parses_decimal_commas(tmp_path: Path):
    from rainrag.library_performance import load_metrics

    p = tmp_path / "metrics.csv"
    p.write_text(
        "youtube_id,snapshot_date,views,playbackBasedCpm\n"
        "a,2026-08-01,100,1.5\n"
        'a,2026-09-01,250,"2,25"\n'
        "b,2026-09-01,,3.0\n",
        encoding="utf-8",
    )
    m = load_metrics(p)
    assert m["a"] == {"views": 250.0, "playbackBasedCpm": 2.25}
    # a newer snapshot with blank cells must not erase older good values
    p.write_text(
        "youtube_id,snapshot_date,views,playbackBasedCpm\nc,2026-08-01,100,1.5\nc,2026-09-01,,\n",
        encoding="utf-8",
    )
    assert load_metrics(p)["c"] == {"views": 100.0, "playbackBasedCpm": 1.5}
    assert m["b"] == {"playbackBasedCpm": 3.0}
    assert load_metrics(tmp_path / "absent.csv") == {}


def test_aggregate_by_cpm_skips_uploads_without_that_metric():
    from rainrag.library_performance import aggregate, build_uploads

    videos = {"h1": SimpleNamespace(program="P", presenters=["A"])}
    ups = build_uploads(
        [_map("x", "1"), _map("y", "2")], videos, {}, metrics={"x": {"playbackBasedCpm": 4.0}}
    )
    rows = aggregate(ups, "program", metric="playbackBasedCpm")
    # "uploads" is the sample size behind the stats: one of the two has CPM
    assert rows == [{"program": "P", "uploads": 1, "total": 4.0, "median": 4.0, "best": 4.0}]


def test_aggregate_merges_speaker_spellings_across_uploads():
    """One upload knows only the tagger's lowercase name, another the CMS one."""
    from rainrag.library_performance import aggregate, build_uploads

    videos = {
        "h1": SimpleNamespace(program="P", presenters=["Ирина Хакамада"]),
        "h2": SimpleNamespace(program="P", presenters=[]),
    }
    tags = {"2": {"presenter_cms": [], "guest": ["ирина хакамада"]}}
    ups = build_uploads(
        [_map("x", "1", views=10, vh="h1"), _map("y", "2", views=30, vh="h2")], videos, tags
    )
    rows = aggregate(ups, "speaker")
    assert rows == [
        {"speaker": "Ирина Хакамада", "uploads": 2, "total": 40.0, "median": 20.0, "best": 30.0}
    ]
