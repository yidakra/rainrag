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
