"""Tests for the YouTube ↔ archive bridge.

The bridge deliberately does not auto-link: held out against the editor's three
hand-made links it scores 1/3, with confident errors. What is tested here is
that it stays honest — an editor's link always wins, disagreeing runtimes veto,
and uncertain matches are labelled for review rather than accepted.
"""

from __future__ import annotations

import pytest

from rainrag.youtube_library import (
    YouTubeVideo,
    build_title_index,
    duration_agreement,
    match_video,
    normalise_title,
    parse_iso8601_duration,
    title_similarity,
)


ARCHIVE = [
    (
        "454595",
        "Как разбудить интуицию и использовать ее в своих целях? Мастер-класс Ирины Хакамады",
    ),
    ("428416", "Михаил Булгаков «Мастер и Маргарита», 1966 год"),
    ("999999", "Прогноз погоды на выходные"),
]


def video(title: str, seconds: float | None = 2535.0, vid: str = "abc") -> YouTubeVideo:
    return YouTubeVideo(video_id=vid, title=title, duration_seconds=seconds)


class TestNormalisation:
    def test_channel_boilerplate_is_stripped(self) -> None:
        assert "дождь" not in normalise_title("Телеканал Дождь: Лекция")

    def test_case_and_punctuation_fold(self) -> None:
        assert normalise_title("«Мастер и Маргарита», 1966!") == normalise_title(
            "мастер и маргарита 1966"
        )

    def test_identical_titles_score_one(self) -> None:
        assert title_similarity("Лекция о Булгакове", "Лекция о Булгакове") == 1.0

    def test_unrelated_titles_score_low(self) -> None:
        assert title_similarity("Прогноз погоды", "Лекция о Булгакове") < 0.4


class TestDuration:
    @pytest.mark.parametrize(
        "iso,seconds", [("PT42M15S", 2535.0), ("PT1H2M", 3720.0), ("PT30S", 30.0)]
    )
    def test_iso_parsing(self, iso: str, seconds: float) -> None:
        assert parse_iso8601_duration(iso) == seconds

    def test_garbage_duration_is_none(self) -> None:
        assert parse_iso8601_duration("nonsense") is None

    def test_close_runtimes_agree(self) -> None:
        """The three known pairs agreed to within 12 seconds."""
        assert duration_agreement(2535.0, 2547.0) == 1.0

    def test_distant_runtimes_disagree(self) -> None:
        assert duration_agreement(600.0, 3600.0) == 0.0

    def test_unknown_runtime_neither_confirms_nor_vetoes(self) -> None:
        assert duration_agreement(None, 2535.0) == 0.5


class TestMatching:
    def test_editor_link_always_wins(self) -> None:
        """A hand-made link is a fact; string similarity must not override it."""
        m = match_video(video("Совершенно другое название"), ARCHIVE, known={"abc": "428416"})
        assert m.content_id == "428416"
        assert m.confidence == "editor"

    def test_exact_title_and_runtime_is_confident(self) -> None:
        m = match_video(video(ARCHIVE[0][1]), ARCHIVE, durations={"454595": 2535.0})
        assert m.content_id == "454595"
        assert m.is_confident

    def test_runtime_disagreement_vetoes(self) -> None:
        """A ten-minute clip is not the 42-minute lecture it is named after."""
        m = match_video(video(ARCHIVE[0][1], seconds=600.0), ARCHIVE, durations={"454595": 2535.0})
        assert m.content_id != "454595" or not m.is_confident

    def test_weak_match_is_flagged_for_review_not_accepted(self) -> None:
        m = match_video(video("Что-то про интуицию", seconds=2535.0), ARCHIVE)
        assert not m.is_confident

    def test_no_plausible_candidate_yields_nothing(self) -> None:
        m = match_video(video("Футбольный матч Спартак Зенит", seconds=5400.0), ARCHIVE)
        assert m.content_id is None
        assert m.confidence == "none"

    def test_index_narrows_candidates_without_changing_the_winner(self) -> None:
        index = build_title_index(ARCHIVE)
        with_index = match_video(video(ARCHIVE[1][1]), ARCHIVE, index=index)
        without = match_video(video(ARCHIVE[1][1]), ARCHIVE)
        assert with_index.content_id == without.content_id == "428416"

    def test_empty_archive_does_not_crash(self) -> None:
        assert match_video(video("что угодно"), []).content_id is None


def test_candidate_positions_is_deterministic_at_the_cap_boundary():
    """Ties at the candidate cap must not depend on hash-seed iteration order.

    Two runs of the real matcher disagreed on 39 of 236 uploads because the
    cap sliced a tie group in set-iteration order. Lowest positions win now.
    """
    from rainrag.youtube_library import build_title_index, candidate_positions

    # 30 archive titles sharing exactly one word with the query -> all tie.
    archive = [(str(i), f"передача выпуск{i} новости") for i in range(30)]
    index = build_title_index(archive)
    picked = candidate_positions("новости недели", index, max_candidates=10)
    assert picked == sorted(picked) == list(range(10))
