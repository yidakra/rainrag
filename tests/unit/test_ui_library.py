"""Tests for the Библиотека mode's logic (not the Streamlit rendering)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


def _ep(video_hash, **kw):
    from rainrag.library_similar import Episode

    return Episode(video_hash=video_hash, **kw)


def test_load_tagged_episodes_drops_errors_and_dedupes(tmp_path: Path):
    from ui_library import load_tagged_episodes

    p = tmp_path / "tags.jsonl"
    rows = [
        {"video_hash": "a", "content_id": "1", "subject": ["x"]},
        {"video_hash": "b", "error": "boom"},
        {"video_hash": "a", "content_id": "1", "subject": ["y"]},  # re-tag: last wins
        "not json at all",
    ]
    p.write_text(
        "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows), encoding="utf-8"
    )
    eps = load_tagged_episodes(p)
    assert [e.video_hash for e in eps] == ["a"]
    assert eps[0].subject == ["y"]


def test_search_matches_title_and_program_case_and_yo_insensitively():
    from ui_library import search_episodes

    eps = [
        _ep("1", title="Лекция Ирины Хакамады", date="2020-01-01"),
        _ep("2", program="Сто лекций с Дмитрием Быковым", date="2021-01-01"),
        _ep("3", title="Про всё остальное", date="2022-01-01"),
    ]
    hits = search_episodes(eps, "ЛЕКЦИ")
    assert [e.video_hash for e in hits] == ["2", "1"]  # newest first
    # ё in the query must match е in the data and vice versa
    assert [e.video_hash for e in search_episodes(eps, "всЁ")] == ["3"]


def test_search_with_empty_needle_returns_nothing():
    from ui_library import search_episodes

    assert search_episodes([_ep("1", title="x")], "   ") == []


def test_split_by_speaker_partitions_and_preserves_order():
    from rainrag.library_similar import Scored
    from ui_library import split_by_speaker

    a = Scored(_ep("a"), 3.1, ["Ирина Хакамада"], [])
    b = Scored(_ep("b"), 0.2, [], ["политика"])
    c = Scored(_ep("c"), 3.0, ["Ирина Хакамада"], ["интуиция"])
    same, themed = split_by_speaker([a, b, c])
    assert [r.episode.video_hash for r in same] == ["a", "c"]
    assert [r.episode.video_hash for r in themed] == ["b"]


def test_decisions_round_trip_and_last_verdict_wins(tmp_path: Path):
    from ui_library import append_decision, load_decisions

    p = tmp_path / "decisions.csv"
    append_decision("yt1", "100", "match", path=p)
    append_decision("yt2", None, "skip", path=p)
    append_decision("yt1", "100", "no_match", path=p)  # editor changed their mind
    assert load_decisions(p) == {"yt1": "no_match", "yt2": "skip"}


def test_review_queue_hides_decided_and_editor_rows_and_orders_by_confidence():
    from ui_library import review_queue

    matches = [
        {"youtube_id": "r1", "confidence": "review", "score": 0.7},
        {"youtube_id": "e1", "confidence": "editor", "score": 1.0},
        {"youtube_id": "s1", "confidence": "strong", "score": 0.9},
        {"youtube_id": "d1", "confidence": "strong", "score": 0.95},
        {"youtube_id": "n1", "confidence": "none", "score": 0.0},
    ]
    queue = review_queue(matches, decisions={"d1": "match"})
    # editor rows are already ground truth; decided rows are done
    assert [m["youtube_id"] for m in queue] == ["s1", "r1", "n1"]


def test_feedback_round_trip_and_last_verdict_wins(tmp_path: Path):
    from ui_library import append_feedback, load_feedback

    p = tmp_path / "feedback.csv"
    append_feedback("454595", "484740", "theme", 7, "good", path=p)
    append_feedback("454595", "431298", "theme", 9, "bad", path=p)
    append_feedback("454595", "484740", "theme", 7, "bad", path=p)  # changed their mind
    marks = load_feedback(p)
    assert marks == {("454595", "484740"): "bad", ("454595", "431298"): "bad"}


def test_feedback_file_gets_a_header_exactly_once(tmp_path: Path):
    from ui_library import append_feedback

    p = tmp_path / "feedback.csv"
    append_feedback("1", "2", "speaker", 1, "good", path=p)
    append_feedback("1", "3", "speaker", 2, "good", path=p)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("seed_content_id,")
    assert len(lines) == 3


def test_feedback_marks_are_per_pair_not_per_column(tmp_path: Path):
    """A verdict follows the (seed, candidate) pair across columns.

    The columns partition one result list, so a pair is shown in exactly one
    of them per render; if re-tagging later moves it, the editor's judgment
    moves with it rather than presenting the pair as unjudged. The CSV still
    records the column each verdict was given in.
    """
    from ui_library import append_feedback, load_feedback

    p = tmp_path / "feedback.csv"
    append_feedback("454595", "484740", "theme", 7, "good", path=p)
    append_feedback("454595", "484740", "speaker", 2, "bad", path=p)
    assert load_feedback(p) == {("454595", "484740"): "bad"}
    rows = p.read_text(encoding="utf-8").strip().splitlines()
    assert rows[1].split(",")[2] == "theme" and rows[2].split(",")[2] == "speaker"


def test_split_by_speaker_never_puts_one_episode_in_both_columns():
    """The invariant the per-pair feedback key rests on."""
    from rainrag.library_similar import Scored
    from ui_library import split_by_speaker

    results = [
        Scored(_ep("a"), 3.1, ["Ирина Хакамада"], ["интуиция"]),
        Scored(_ep("b"), 0.2, [], ["политика"]),
    ]
    same, themed = split_by_speaker(results)
    assert {r.episode.video_hash for r in same} & {r.episode.video_hash for r in themed} == set()
    assert len(same) + len(themed) == len(results)


def test_youtube_id_extraction_from_urls_and_bare_ids():
    from ui_library import youtube_id_from_query

    assert youtube_id_from_query("https://youtu.be/RohuZGgpC_k") == "RohuZGgpC_k"
    assert youtube_id_from_query("https://www.youtube.com/watch?v=RohuZGgpC_k&t=5") == "RohuZGgpC_k"
    assert youtube_id_from_query("https://youtube.com/shorts/N8XZOHbIiA8") == "N8XZOHbIiA8"
    assert youtube_id_from_query("RohuZGgpC_k") == "RohuZGgpC_k"
    # a title fragment must never be mistaken for an id
    assert youtube_id_from_query("интуиция") is None
    assert youtube_id_from_query("Хакамада мастер-класс") is None
    assert youtube_id_from_query("management!") is None
