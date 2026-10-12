"""Tests for the map builder's write step: atomic, and it will not shrink."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from youtube_map import ShrunkMapError, write_outputs  # noqa: E402


def _row(youtube_id: str, content_id: str | None = "1", confidence: str = "review") -> dict:
    return {
        "youtube_id": youtube_id,
        "youtube_title": "t",
        "content_id": content_id,
        "archive_title": "a",
        "score": 0.5,
        "confidence": confidence,
        "published_at": "2026-10-01T00:00:00Z",
        "view_count": 1,
    }


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return tmp_path / "map.json", tmp_path / "map.csv", tmp_path / "index.json"


def test_outputs_land_atomically_with_the_archive_index(tmp_path):
    out, csv_out, idx = _paths(tmp_path)
    episode = {"title": "Выпуск", "date": "2020-01-01", "url": "u", "video_hash": "h"}
    write_outputs([_row("a"), _row("b")], {"1": episode}, out, csv_out, idx)

    assert [r["youtube_id"] for r in json.loads(out.read_text("utf-8"))] == ["a", "b"]
    assert json.loads(idx.read_text("utf-8")) == {"1": episode}
    assert csv_out.read_text("utf-8").splitlines()[0].startswith("youtube_id,")
    # Written beside and renamed into place: no scratch file is left behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_a_map_that_came_back_short_is_refused_unless_told(tmp_path):
    """Unattended every morning now. A listing cut short by a quota error or
    a page that never answered must not replace a complete map with a
    partial one: the review tab would quietly lose uploads and the metrics
    pull would stop asking about them."""
    out, csv_out, idx = _paths(tmp_path)
    write_outputs([_row("a"), _row("b")], {}, out, csv_out, idx)

    with pytest.raises(ShrunkMapError):
        write_outputs([_row("a")], {}, out, csv_out, idx)
    assert len(json.loads(out.read_text("utf-8"))) == 2, "the previous map must survive"

    write_outputs([_row("a")], {}, out, csv_out, idx, allow_shrink=True)
    assert len(json.loads(out.read_text("utf-8"))) == 1


def test_the_archive_index_is_guarded_the_same_way(tmp_path):
    """A metadata directory read half synced would shrink the index, and
    every typed content id for the missing half would read as "not in the
    archive" until the next run (CodeRabbit on #113)."""
    out, csv_out, idx = _paths(tmp_path)
    write_outputs([_row("a")], {"1": {"title": "x"}, "2": {"title": "y"}}, out, csv_out, idx)

    with pytest.raises(ShrunkMapError):
        write_outputs([_row("a")], {"1": {"title": "x"}}, out, csv_out, idx)
    assert len(json.loads(idx.read_text("utf-8"))) == 2
    write_outputs([_row("a")], {"1": {"title": "x"}}, out, csv_out, idx, allow_shrink=True)
    assert len(json.loads(idx.read_text("utf-8"))) == 1


def test_the_map_is_replaced_only_after_the_other_outputs_succeed(tmp_path):
    """ "A failed rebuild keeps the previous map" is the unit's contract, and
    the map used to be replaced first, so a CSV or index write failing after
    it left a new map behind a failed run (CodeRabbit on #113)."""
    out, csv_out, idx = _paths(tmp_path)
    write_outputs([_row("a")], {}, out, csv_out, idx)

    csv_out.unlink()
    csv_out.mkdir()  # the CSV write now fails: a directory stands in its way
    with pytest.raises(OSError):
        write_outputs([_row("a"), _row("b")], {}, out, csv_out, idx)
    assert len(json.loads(out.read_text("utf-8"))) == 1, "the previous map must survive"


def test_a_torn_previous_map_does_not_block_the_rebuild(tmp_path):
    out, csv_out, idx = _paths(tmp_path)
    out.write_text('[{"youtube_id": "a"', encoding="utf-8")
    write_outputs([_row("a")], {}, out, csv_out, idx)
    assert len(json.loads(out.read_text("utf-8"))) == 1
