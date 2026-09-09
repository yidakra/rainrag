"""Tests for the Analytics transform and CSV writing (no Google calls)."""

from __future__ import annotations

import stat
from pathlib import Path


def test_rows_to_dicts_zips_each_batch_with_its_own_headers():
    """A metric rejected on a later batch must not shift earlier rows."""
    from rainrag.youtube_analytics import rows_to_dicts

    wide = rows_to_dicts([{"name": "video"}, {"name": "views"}, {"name": "cpm"}], [["a", 10, 1.5]])
    narrow = rows_to_dicts([{"name": "video"}, {"name": "views"}], [["b", 20]])
    assert wide == [{"video": "a", "views": 10, "cpm": 1.5}]
    assert narrow == [{"video": "b", "views": 20}]


def test_rows_to_dicts_requires_the_video_dimension_only_when_rows_exist():
    import pytest

    from rainrag.youtube_analytics import rows_to_dicts

    assert rows_to_dicts([{"name": "views"}], []) == []
    with pytest.raises(ValueError):
        rows_to_dicts([{"name": "views"}], [[1]])


def test_rows_to_snapshot_maps_records_into_sheet_schema_with_blanks():
    from rainrag.youtube_analytics import CSV_COLUMNS, rows_to_snapshot

    out = rows_to_snapshot(
        [{"video": "a", "views": 1200, "playbackBasedCpm": 2.5}, {"video": "b", "views": 5}],
        "2026-09-09",
    )
    assert out[0]["views"] == 1200 and out[0]["playbackBasedCpm"] == 2.5
    assert out[0]["snapshot_date"] == "2026-09-09"
    assert out[1]["playbackBasedCpm"] == "" and out[1]["cpm"] == ""
    assert set(out[0]) == set(CSV_COLUMNS)


def test_append_snapshot_header_once_and_readable_by_load_metrics(tmp_path: Path):
    from rainrag.library_performance import load_metrics
    from rainrag.youtube_analytics import append_snapshot, rows_to_snapshot

    p = tmp_path / "metrics.csv"
    append_snapshot(
        p, rows_to_snapshot([{"video": "a", "views": 10, "playbackBasedCpm": 1.0}], "2026-09-01")
    )
    append_snapshot(
        p, rows_to_snapshot([{"video": "a", "views": 20, "playbackBasedCpm": 1.5}], "2026-09-09")
    )
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("youtube_id,snapshot_date,views") and len(lines) == 3
    assert load_metrics(p)["a"] == {"views": 20.0, "playbackBasedCpm": 1.5}


def test_token_file_is_owner_only(tmp_path: Path):
    from rainrag.youtube_analytics import _write_token

    t = tmp_path / "token.json"
    t.write_text("old", encoding="utf-8")
    t.chmod(0o644)
    _write_token(t, "{}")
    assert stat.S_IMODE(t.stat().st_mode) == 0o600
    assert t.read_text(encoding="utf-8") == "{}"


def test_chunked_and_rejected_metric_exact():
    from rainrag.youtube_analytics import _rejected_metric, chunked

    assert [len(c) for c in chunked([str(i) for i in range(1203)])] == [500, 500, 203]
    msg = "Unknown identifier (engagedViews) given in field parameters.metrics"
    assert _rejected_metric(msg, ["views", "engagedViews"]) == "engagedViews"
    # a shorter metric name must not match inside a longer identifier
    assert _rejected_metric("bad metric 'playbackBasedCpm'", ["cpm", "playbackBasedCpm"]) == (
        "playbackBasedCpm"
    )
    assert _rejected_metric("quota exceeded", ["views"]) is None
