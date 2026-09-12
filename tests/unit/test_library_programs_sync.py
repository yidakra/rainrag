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


def test_load_excluded_normalises_and_survives_a_missing_or_broken_file(tmp_path: Path):
    from library_programs_sync import load_excluded

    assert load_excluded(tmp_path / "absent.json") == set()
    broken = tmp_path / "broken.json"
    broken.write_text("not json", encoding="utf-8")
    assert load_excluded(broken) == set()
    good = tmp_path / "excluded.json"
    good.write_text('["Hard Day\\u2019s Night", "  ", "Архив"]', encoding="utf-8")
    loaded = load_excluded(good)
    assert len(loaded) == 2
    from rainrag.library_programs import normalise_title

    assert normalise_title("Hard Day's Night") in loaded


def test_coverage_leaves_non_programmes_out_of_the_denominator(tmp_path: Path, capsys):
    """Counting them as gaps reported 83% for a table that reaches 98%."""
    from library_programs_sync import report_coverage

    table = tmp_path / "programs.csv"
    table.write_text("title,genre\nСиндеева,интервью\nАрхив,\n", encoding="utf-8")
    tags = _tags(
        tmp_path,
        {"video_hash": "a", "program": "Синдеева"},
        {"video_hash": "b", "program": "Архив"},
        {"video_hash": "c", "program": "Архив"},
    )
    excluded = tmp_path / "excluded.json"
    excluded.write_text('["Архив"]', encoding="utf-8")
    report_coverage(table, tags, excluded)
    out = capsys.readouterr().out
    assert "out of scope (not a programme): 2" in out
    assert "in scope: 1" in out
    assert "has a genre: 1 (100%)" in out


def test_episodes_with_no_programme_can_be_excluded_by_their_bucket_name(tmp_path: Path, capsys):
    """The catalogue leaves `program` empty; the sheet calls it «(без программы)»."""
    from library_programs_sync import report_coverage

    table = tmp_path / "programs.csv"
    table.write_text("title,genre\nСиндеева,интервью\n", encoding="utf-8")
    tags = _tags(tmp_path, {"video_hash": "a", "program": None}, {"video_hash": "b", "program": ""})
    excluded = tmp_path / "excluded.json"
    excluded.write_text('["(без программы)"]', encoding="utf-8")
    report_coverage(table, tags, excluded)
    assert "out of scope (not a programme): 2" in capsys.readouterr().out


def test_a_coloured_row_with_a_genre_stays_in_scope(tmp_path: Path):
    """«ONLINE» is highlighted and has a genre; only unusable rows drop out."""
    openpyxl = pytest.importorskip("openpyxl")
    from library_programs_sync import excluded_from_workbook

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Programs"
    ws.append(["content_id", "title", "genre"])
    ws.append(["1", "Синдеева", "интервью"])
    ws.append(["2", "Архив", None])
    ws.append(["3", "ONLINE", "разговор со зрителями"])
    red = openpyxl.styles.PatternFill(start_color="FFF4CCCC", fill_type="solid")
    yellow = openpyxl.styles.PatternFill(start_color="FFFFF2CC", fill_type="solid")
    for cell in ws[3]:
        cell.fill = red
    for cell in ws[4]:
        cell.fill = yellow
    path = tmp_path / "book.xlsx"
    wb.save(path)
    assert excluded_from_workbook(path) == ["Архив"]


def test_a_workbook_without_the_programs_sheet_is_rejected(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    from library_programs_sync import excluded_from_workbook

    wb = openpyxl.Workbook()
    wb.active.title = "Tags"
    path = tmp_path / "wrong.xlsx"
    wb.save(path)
    with pytest.raises(SystemExit, match="Programs"):
        excluded_from_workbook(path)


def test_a_blank_column_in_the_header_does_not_truncate_the_row(tmp_path: Path):
    """Counting named columns instead of locating the last one dropped genre."""
    openpyxl = pytest.importorskip("openpyxl")
    from library_programs_sync import _csv_from_workbook

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Programs"
    ws.append(["content_id", None, "title", "genre"])
    ws.append(["1", None, "Синдеева", "интервью"])
    book = tmp_path / "book.xlsx"
    wb.save(book)
    table = tmp_path / "programs.csv"
    _csv_from_workbook(book, table)

    from rainrag.library_programs import load_programmes, programme_for

    sindeeva = programme_for("Синдеева", load_programmes(table))
    assert sindeeva is not None
    assert sindeeva.genres == ("интервью",)


def test_an_exclusion_file_that_is_not_a_list_is_ignored(tmp_path: Path):
    """A bare JSON string is iterable and would exclude single letters."""
    from library_programs_sync import load_excluded

    for payload in ('"Архив"', "null", "42", '{"a": 1}'):
        path = tmp_path / "excluded.json"
        path.write_text(payload, encoding="utf-8")
        assert load_excluded(path) == set(), payload


def test_the_exclusion_file_directory_is_created_if_missing(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    import library_programs_sync as sync

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Programs"
    ws.append(["content_id", "title", "genre"])
    ws.append(["1", "Синдеева", "интервью"])
    book = tmp_path / "book.xlsx"
    wb.save(book)
    nested = tmp_path / "reports" / "current" / "excluded.json"
    sync.main(
        [
            str(book),
            "--table",
            str(tmp_path / "programs.csv"),
            "--tags",
            str(tmp_path / "absent.jsonl"),
            "--excluded",
            str(nested),
        ]
    )
    assert nested.exists()
