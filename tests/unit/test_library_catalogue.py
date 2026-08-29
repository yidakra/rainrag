"""Tests for the Библиотека Дождя catalogue and content model.

The catalogue exists to answer "what линейки are in this archive": the editor's
complaint is that it is a black box. The load-bearing behaviours are folding
chunks up to videos without double-counting language variants, and never
silently discarding an editor's genre decisions when the draft is regenerated.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from library_catalogue import (  # noqa: E402
    episodes_for_program,
    fold_chunks_to_videos,
    summarise_programs,
    video_key_from_path,
)

from rainrag.library import (  # noqa: E402
    CONFIRMED_PROGRAM_GENRES,
    draft_genre,
    load_program_genres,
    write_program_genre_draft,
)


def _chunk(hash_: str, lang: str = "ru", **over):
    payload = {
        "path": f"/archive/ab/cd/{hash_}.{lang}.vtt",
        "language": lang,
        "web_program": "Лекции на Дожде",
        "web_title": "Лекция",
        "web_date": "2018-01-08",
        "duration_seconds": 2535.0,
        "web_presenters": ["Ирина Хакамада"],
        "web_tags_theme": ["психология"],
        "web_tags_person": ["Владимир Путин"],
    }
    payload.update(over)
    return payload


class TestFolding:
    def test_language_variants_are_one_video(self) -> None:
        """<hash>.ru.vtt and <hash>.en.vtt are one broadcast, not two episodes."""
        videos = fold_chunks_to_videos([_chunk("aaa", "ru"), _chunk("aaa", "en")])
        assert len(videos) == 1
        assert videos["aaa"].languages == {"ru", "en"}

    def test_chunks_accumulate_metadata_does_not_duplicate(self) -> None:
        videos = fold_chunks_to_videos([_chunk("aaa") for _ in range(5)])
        video = videos["aaa"]
        assert video.chunks == 5
        assert video.presenters == ["Ирина Хакамада"]

    def test_missing_path_is_skipped(self) -> None:
        assert fold_chunks_to_videos([{"web_title": "no path"}]) == {}

    def test_first_non_empty_value_wins(self) -> None:
        """Metadata repeats per chunk; an empty first chunk must not win."""
        videos = fold_chunks_to_videos(
            [_chunk("aaa", web_title=None), _chunk("aaa", web_title="Настоящее название")]
        )
        assert videos["aaa"].title == "Настоящее название"

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/archive/ab/cd/deadbeef.ru.vtt", "deadbeef"),
            ("/archive/ab/cd/deadbeef.en.vtt", "deadbeef"),
            ("deadbeef.vtt", "deadbeef"),
        ],
    )
    def test_video_key(self, path: str, expected: str) -> None:
        assert video_key_from_path(path) == expected


class TestProgrammeSummary:
    def test_groups_and_counts_episodes(self) -> None:
        videos = fold_chunks_to_videos(
            [_chunk("aaa"), _chunk("bbb"), _chunk("ccc", web_program="Синдеева")]
        )
        programs = {p["program"]: p for p in summarise_programs(videos)}
        assert programs["Лекции на Дожде"]["episodes"] == 2
        assert programs["Синдеева"]["episodes"] == 1

    def test_untagged_videos_are_kept_not_dropped(self) -> None:
        """How much of the archive has no programme is itself a finding."""
        videos = fold_chunks_to_videos([_chunk("aaa", web_program=None)])
        assert summarise_programs(videos)[0]["program"] == "(без программы)"

    def test_date_range_reported(self) -> None:
        videos = fold_chunks_to_videos(
            [_chunk("aaa", web_date="2011-04-06"), _chunk("bbb", web_date="2022-01-07")]
        )
        p = summarise_programs(videos)[0]
        assert (p["first_date"], p["last_date"]) == ("2011-04-06", "2022-01-07")

    def test_episodes_listing_is_newest_first(self) -> None:
        videos = fold_chunks_to_videos(
            [_chunk("aaa", web_date="2015-01-01"), _chunk("bbb", web_date="2020-01-01")]
        )
        dates = [e["date"] for e in episodes_for_program(videos, "Лекции на Дожде")]
        assert dates == ["2020-01-01", "2015-01-01"]


class TestGenreTable:
    def test_confirmed_genres_come_from_the_sheet(self) -> None:
        assert draft_genre("Лекции на Дожде") == ("лекция", "мастер-класс")
        assert draft_genre("Синдеева") == ("интервью",)

    def test_name_hints_produce_a_draft(self) -> None:
        assert draft_genre("Сто лекций с кем-то") == ("лекция",)

    def test_uninformative_name_yields_nothing(self) -> None:
        """Better an empty cell for review than a confident wrong guess."""
        assert draft_genre("Здесь и сейчас") == ()

    def test_draft_marks_provenance(self, tmp_path: Path) -> None:
        path = tmp_path / "program_genres.csv"
        write_program_genre_draft(
            path,
            [
                {"program": "Лекции на Дожде", "episodes": 661, "median_duration_minutes": 18},
                {"program": "Здесь и сейчас", "episodes": 47942, "median_duration_minutes": 6},
            ],
            reviewed={},
        )
        with open(path, encoding="utf-8") as f:
            rows = {r["program"]: r for r in csv.DictReader(f)}
        assert rows["Лекции на Дожде"]["source"] == "confirmed"
        assert rows["Здесь и сейчас"]["source"] == "unknown"

    def test_regenerating_preserves_editorial_decisions(self, tmp_path: Path) -> None:
        """A reindex must not quietly wipe someone's genre calls."""
        path = tmp_path / "program_genres.csv"
        programs = [{"program": "Здесь и сейчас", "episodes": 47942, "median_duration_minutes": 6}]
        write_program_genre_draft(path, programs, reviewed={"Здесь и сейчас": ("ток-шоу",)})
        again = load_program_genres(path)
        assert again["Здесь и сейчас"] == ("ток-шоу",)
        write_program_genre_draft(path, programs, reviewed=again)
        assert load_program_genres(path)["Здесь и сейчас"] == ("ток-шоу",)

    def test_missing_table_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_program_genres(tmp_path / "absent.csv") == {}

    def test_sheet_genres_are_in_the_vocabulary(self) -> None:
        from rainrag.library import GENRES

        for genres in CONFIRMED_PROGRAM_GENRES.values():
            for g in genres:
                assert g in GENRES


class TestReviewFindings:
    """Regressions for what review caught on the first version."""

    def test_median_is_the_true_median_for_even_counts(self) -> None:
        """sorted()[n//2] is the upper middle: [60s, 600s] reported 10.0 min."""
        videos = fold_chunks_to_videos(
            [_chunk("aaa", duration_seconds=60.0), _chunk("bbb", duration_seconds=600.0)]
        )
        assert summarise_programs(videos)[0]["median_duration_minutes"] == pytest.approx(5.5)

    def test_odd_counts_still_take_the_middle(self) -> None:
        videos = fold_chunks_to_videos(
            [
                _chunk("a", duration_seconds=60.0),
                _chunk("b", duration_seconds=120.0),
                _chunk("c", duration_seconds=600.0),
            ]
        )
        assert summarise_programs(videos)[0]["median_duration_minutes"] == pytest.approx(2.0)

    def test_genre_table_keeps_programmes_below_the_episode_filter(self, tmp_path: Path) -> None:
        """--min-episodes narrows the catalogue, never the genre table.

        Rewriting the table from a filtered list would delete the reviewed rows
        of every smaller programme -- the exact loss the function exists to
        prevent.
        """
        path = tmp_path / "program_genres.csv"
        write_program_genre_draft(
            path,
            [{"program": "Маленькая линейка", "episodes": 3, "median_duration_minutes": 40}],
            reviewed={"Маленькая линейка": ("интервью",)},
        )
        assert load_program_genres(path)["Маленькая линейка"] == ("интервью",)
