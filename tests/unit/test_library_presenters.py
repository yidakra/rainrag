"""Tests for the hand-filled presenter table and how it reaches the ranker."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rainrag.library_presenters import load_presenters, split_presenters  # noqa: E402
from rainrag.library_programs import Programme, resolve_speakers  # noqa: E402


def _table(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "presenters.csv"
    path.write_text("content_id,presenter\n" + body, encoding="utf-8")
    return path


def test_a_panel_in_one_cell_becomes_several_speakers():
    """«Лев Гудков, Анна Качкаева, Наталья Тихонова, Эмиль Паин» is four
    people; kept whole the ranker looks for a person of that name."""
    assert split_presenters("Лев Гудков, Анна Качкаева, Эмиль Паин") == [
        "Лев Гудков",
        "Анна Качкаева",
        "Эмиль Паин",
    ]
    assert split_presenters("  Дмитрий Быков  ") == ["Дмитрий Быков"]
    assert split_presenters("") == []


def test_rows_without_a_content_id_or_a_name_are_skipped(tmp_path):
    table = _table(tmp_path, "379869,Дмитрий Быков\n,Никто\n390494,\n")
    assert load_presenters(table) == {"379869": ["Дмитрий Быков"]}


def test_a_missing_table_is_not_an_error(tmp_path):
    """It is maintained by hand and can lag a fresh checkout."""
    assert load_presenters(tmp_path / "absent.csv") == {}


def test_a_sheets_export_with_a_byte_order_mark_still_loads(tmp_path):
    """Read as plain utf-8 the mark glues to the first header name, the column
    is unreachable, and the table silently loads empty."""
    path = tmp_path / "bom.csv"
    path.write_text("﻿content_id,presenter\n379869,Дмитрий Быков\n", encoding="utf-8")
    assert load_presenters(path) == {"379869": ["Дмитрий Быков"]}


def test_an_episode_with_no_cms_presenter_gains_one():
    """The gap this exists for: 49 lectures carried no speaker at all, so they
    sat out the heaviest axis in the ranking (86cbhq9q8)."""
    record = {"content_id": "344036", "program": "Лекции на Дожде"}
    overrides = {"344036": ["Андрей Максимов"]}
    assert resolve_speakers(record, {}, overrides).speakers == ["Андрей Максимов"]


def test_the_cms_wins_where_it_knows_a_presenter():
    """The table is a patch for the lecture programmes, not a second source of
    truth for the whole archive."""
    record = {"content_id": "344036", "presenter_cms": ["Из CMS"]}
    overrides = {"344036": ["Из таблицы"]}
    assert resolve_speakers(record, {}, overrides).speakers == ["Из CMS"]


def test_an_override_is_still_subject_to_the_genre_rule():
    """It goes in through the presenter path on purpose: a programme where the
    host interviews rather than speaks must not gain a speaker this way."""
    programmes = {"ток-шоу": Programme(title="Ток-шоу", genres=("ток-шоу",))}
    record = {"content_id": "1", "program": "Ток-шоу", "guest": ["Гость"]}
    resolution = resolve_speakers(record, programmes, {"1": ["Ведущий"]})
    assert resolution.speakers == ["Гость"]
    assert resolution.presenter_demoted


def test_without_a_table_nothing_changes():
    record = {"content_id": "344036", "guest": ["Гость"]}
    assert resolve_speakers(record, {}, None).speakers == ["Гость"]
    assert resolve_speakers(record, {}, {}).speakers == ["Гость"]


def test_coverage_counts_only_the_episodes_that_really_gain_a_speaker(tmp_path):
    """Counting rows without CMS people overstated it twice: the tag file is
    append-only, so an older row still counted after a newer one gained a
    presenter, and an override on a programme whose genre demotes presenters
    was counted though the episode ends up with no speaker (CodeRabbit)."""
    import json
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from library_presenters_sync import coverage

    table = _table(tmp_path, "1,Лектор\n2,Ведущий\n3,Кто-то\n")
    programs = tmp_path / "programs.csv"
    programs.write_text("title,genre\nТок-шоу,ток-шоу\nЛекции,лекция\n", encoding="utf-8")
    tags = tmp_path / "tags.jsonl"
    tags.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in (
                # Gains a speaker: no CMS people, a programme that keeps them.
                {"video_hash": "a", "content_id": "1", "program": "Лекции"},
                # An older row with nothing, then a newer one with a presenter:
                # the pool keeps the newer, so this gains nothing.
                {"video_hash": "b", "content_id": "2", "program": "Лекции"},
                {
                    "video_hash": "b",
                    "content_id": "2",
                    "program": "Лекции",
                    "presenter_cms": ["Из CMS"],
                },
                # The genre demotes presenters, so the override buys nothing.
                {"video_hash": "c", "content_id": "3", "program": "Ток-шоу"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert coverage(table, tags, programs) == (3, 3, 1)
