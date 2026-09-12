"""Tests for the programme-table sync script's coverage report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _tags(tmp_path: Path, *records: dict) -> Path:
    path = tmp_path / "library_tags.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return path


def test_a_failed_retag_does_not_discard_the_earlier_good_row(tmp_path: Path):
    """The ranker keeps the last good row; the report must count the same episodes."""
    from library_programs_sync import latest_tag_rows

    path = _tags(
        tmp_path,
        {"video_hash": "h1", "program": "Синдеева", "error": None},
        {"video_hash": "h1", "program": "Синдеева", "error": "RuntimeError: boom"},
    )
    rows = latest_tag_rows(path)
    assert [r["video_hash"] for r in rows] == ["h1"]


def test_the_last_good_row_wins_over_an_earlier_one(tmp_path: Path):
    from library_programs_sync import latest_tag_rows

    path = _tags(
        tmp_path,
        {"video_hash": "h1", "program": "Старое", "error": None},
        {"video_hash": "h1", "program": "Новое", "error": None},
    )
    assert [r["program"] for r in latest_tag_rows(path)] == ["Новое"]


def test_an_episode_that_only_ever_failed_is_not_counted(tmp_path: Path):
    from library_programs_sync import latest_tag_rows

    path = _tags(tmp_path, {"video_hash": "h1", "error": "boom"})
    assert latest_tag_rows(path) == []


def test_a_missing_or_malformed_file_is_survivable(tmp_path: Path):
    from library_programs_sync import latest_tag_rows

    assert latest_tag_rows(tmp_path / "absent.jsonl") == []
    broken = tmp_path / "broken.jsonl"
    broken.write_text('not json\n{"video_hash": "h1"}\n[]\n', encoding="utf-8")
    assert [r["video_hash"] for r in latest_tag_rows(broken)] == ["h1"]


def test_the_export_must_be_the_programs_tab(tmp_path: Path):
    """A whole-workbook export would otherwise write a table with no titles."""
    from library_programs_sync import validate_export

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("tag_id,tag,status\n1,Путин,active\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Programs tab"):
        validate_export(wrong)


def test_an_empty_export_is_rejected(tmp_path: Path):
    from library_programs_sync import validate_export

    empty = tmp_path / "empty.csv"
    empty.write_text("title,genre\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no data rows"):
        validate_export(empty)


def test_a_padded_header_is_still_counted_as_a_genre(tmp_path: Path):
    """The report said "0 with a genre" for a table the loader reads fine."""
    from library_programs_sync import validate_export

    export = tmp_path / "export.csv"
    export.write_text(" title , genre \nСиндеева,интервью\n", encoding="utf-8")
    rows = validate_export(export)
    assert rows[0]["genre"] == "интервью"
    assert rows[0]["title"] == "Синдеева"
