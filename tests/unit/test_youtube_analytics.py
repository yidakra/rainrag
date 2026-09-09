"""Tests for the Analytics transform and CSV writing (no Google calls)."""

from __future__ import annotations

from pathlib import Path


def test_rows_to_snapshot_maps_positional_rows_into_sheet_schema():
    from rainrag.youtube_analytics import CSV_COLUMNS, rows_to_snapshot

    headers = [{"name": "video"}, {"name": "views"}, {"name": "playbackBasedCpm"}]
    rows = [["abcDEF123-_", 1200, 2.5], ["xyzXYZ987-_", 5, None]]
    out = rows_to_snapshot(headers, rows, "2026-09-09")
    assert [r["youtube_id"] for r in out] == ["abcDEF123-_", "xyzXYZ987-_"]
    assert out[0]["views"] == 1200 and out[0]["playbackBasedCpm"] == 2.5
    assert out[0]["snapshot_date"] == "2026-09-09"
    # metrics the API did not return stay blank, and every sheet column exists
    assert out[1]["playbackBasedCpm"] == "" and out[1]["cpm"] == ""
    assert set(out[0]) == set(CSV_COLUMNS)


def test_rows_to_snapshot_requires_the_video_dimension():
    import pytest

    from rainrag.youtube_analytics import rows_to_snapshot

    with pytest.raises(ValueError):
        rows_to_snapshot([{"name": "views"}], [[1]], "2026-09-09")


def test_append_snapshot_writes_header_once_and_is_readable_by_load_metrics(tmp_path: Path):
    from rainrag.library_performance import load_metrics
    from rainrag.youtube_analytics import append_snapshot, rows_to_snapshot

    p = tmp_path / "metrics.csv"
    h = [{"name": "video"}, {"name": "views"}, {"name": "playbackBasedCpm"}]
    append_snapshot(p, rows_to_snapshot(h, [["a", 10, 1.0]], "2026-09-01"))
    append_snapshot(p, rows_to_snapshot(h, [["a", 20, 1.5]], "2026-09-09"))
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("youtube_id,snapshot_date,views") and len(lines) == 3
    # the Library UI's loader sees the newest snapshot
    assert load_metrics(p)["a"] == {"views": 20.0, "playbackBasedCpm": 1.5}


def test_chunked_and_rejected_metric():
    from rainrag.youtube_analytics import _rejected_metric, chunked

    ids = [str(i) for i in range(1203)]
    assert [len(c) for c in chunked(ids)] == [500, 500, 203]
    assert (
        _rejected_metric(
            "Unknown identifier (engagedViews) given in field parameters.metrics",
            ["views", "engagedViews"],
        )
        == "engagedViews"
    )
    assert _rejected_metric("quota exceeded", ["views"]) is None
