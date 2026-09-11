"""Tests for the programme table and the editorial speaker rule."""

from __future__ import annotations

from pathlib import Path

from rainrag.library_programs import (
    Programme,
    load_programmes,
    normalise_title,
    parse_genres,
    programme_for,
    resolve_speakers,
)


HEADER = "content_id,asset_type,title,parent_id,released_date,presenter,genre,comment\n"


def _table(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "library_programs.csv"
    path.write_text(HEADER + "".join(rows), encoding="utf-8")
    return path


def test_curly_apostrophe_in_the_sheet_still_matches_the_catalogue():
    """250 episodes of this programme were missing their genre over one character."""
    assert normalise_title("Hard Day’s Night") == normalise_title("Hard Day's Night")


def test_title_matching_ignores_case_and_repeated_spaces():
    assert normalise_title("  Сто  ЛЕКЦИЙ с Быковым ") == normalise_title("Сто лекций с Быковым")


def test_parse_genres_splits_the_multi_value_cell_and_folds_case():
    assert parse_genres("Аналитика, ток-шоу") == ("аналитика", "ток-шоу")
    assert parse_genres("") == ()


def test_presenter_is_a_speaker_for_a_lecture_but_not_for_an_interview():
    assert Programme("Сто лекций", ("лекция",)).presenter_is_speaker
    assert not Programme("Синдеева", ("интервью",)).presenter_is_speaker


def test_any_demoting_genre_is_enough_not_all_of_them():
    """ "аналитика, ток-шоу" is still a show where the guest is the draw."""
    assert not Programme("Что-то", ("аналитика", "ток-шоу")).presenter_is_speaker


def test_load_programmes_reads_titles_genres_and_presenters(tmp_path: Path):
    path = _table(
        tmp_path,
        ",program,Синдеева,,25.04.2014,Наталья Синдеева,интервью,\n",
        ",program,Сто лекций,,12.09.2015,Дмитрий Быков,лекция,\n",
    )
    programmes = load_programmes(path)
    assert set(programmes) == {normalise_title("Синдеева"), normalise_title("Сто лекций")}
    sindeeva = programme_for("Синдеева", programmes)
    assert sindeeva is not None
    assert sindeeva.presenter == "Наталья Синдеева"
    assert sindeeva.genres == ("интервью",)


def test_a_missing_table_is_not_an_error(tmp_path: Path):
    """The table is hand-maintained in a sheet and can lag a fresh checkout."""
    assert load_programmes(tmp_path / "absent.csv") == {}


def test_programme_lookup_of_an_unknown_or_empty_name_returns_none(tmp_path: Path):
    programmes = load_programmes(_table(tmp_path, ",program,Синдеева,,,,интервью,\n"))
    assert programme_for("Не программа", programmes) is None
    assert programme_for(None, programmes) is None


def test_interview_drops_the_presenter_and_keeps_the_guest(tmp_path: Path):
    """Varya's point: «Синдеева» must surface the guest's other episodes."""
    programmes = load_programmes(
        _table(tmp_path, ",program,Синдеева,,,Наталья Синдеева,интервью,\n")
    )
    record = {
        "program": "Синдеева",
        "presenter_cms": ["Наталья Синдеева"],
        "guest": ["Михаил Ходорковский"],
    }
    resolution = resolve_speakers(record, programmes)
    assert resolution.speakers == ["Михаил Ходорковский"]
    assert resolution.presenter_demoted


def test_lecture_keeps_the_presenter_who_is_the_lecturer(tmp_path: Path):
    programmes = load_programmes(_table(tmp_path, ",program,Сто лекций,,,Дмитрий Быков,лекция,\n"))
    record = {"program": "Сто лекций", "presenter_cms": ["Дмитрий Быков"], "guest": []}
    resolution = resolve_speakers(record, programmes)
    assert resolution.speakers == ["Дмитрий Быков"]
    assert not resolution.presenter_demoted


def test_interview_without_a_guest_is_left_with_no_speaker(tmp_path: Path):
    """270 episodes land here: better none than the wrong one, and the UI says so."""
    programmes = load_programmes(_table(tmp_path, ",program,Синдеева,,,,интервью,\n"))
    record = {"program": "Синдеева", "presenter_cms": ["Наталья Синдеева"], "guest": []}
    resolution = resolve_speakers(record, programmes)
    assert resolution.speakers == []
    assert resolution.presenter_demoted


def test_demotion_is_not_reported_when_there_was_no_presenter_to_demote(tmp_path: Path):
    programmes = load_programmes(_table(tmp_path, ",program,Синдеева,,,,интервью,\n"))
    record = {"program": "Синдеева", "presenter_cms": [], "guest": ["Гость"]}
    assert not resolve_speakers(record, programmes).presenter_demoted


def test_without_a_programme_table_the_old_order_is_kept_exactly():
    """An out-of-date checkout must degrade to the previous ranking, not drop presenters."""
    record = {"program": "Синдеева", "presenter_cms": ["Ведущая"], "guest": ["Гость"]}
    assert resolve_speakers(record, None).speakers == ["Ведущая", "Гость"]
    assert resolve_speakers(record, {}).speakers == ["Ведущая", "Гость"]


def test_a_programme_absent_from_the_table_keeps_the_presenter(tmp_path: Path):
    programmes = load_programmes(_table(tmp_path, ",program,Синдеева,,,,интервью,\n"))
    record = {"program": "Другая", "presenter_cms": ["Ведущий"], "guest": ["Гость"]}
    assert resolve_speakers(record, programmes).speakers == ["Ведущий", "Гость"]


def test_blank_names_are_dropped_rather_than_ranked_as_a_speaker():
    record = {"program": None, "presenter_cms": ["", None], "guest": ["Гость"]}
    assert resolve_speakers(record, None).speakers == ["Гость"]
