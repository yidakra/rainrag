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
    # an 11-char lowercase English word is a search, not an id
    assert youtube_id_from_query("managements") is None
    assert youtube_id_from_query("MANAGEMENTS") is None


def test_resolve_prefers_the_decision_files_own_content_id(tmp_path, monkeypatch):
    """A review-tab confirmation pins its content_id: a later map regeneration
    must not be able to silently repoint the link."""
    import ui_library

    dec = tmp_path / "decisions.csv"
    dec.write_text(
        "youtube_id,content_id,verdict,decided_at\nabcDEF123-_,111,match,2026\n",
        encoding="utf-8",
    )
    mp = tmp_path / "map.json"
    mp.write_text(
        json.dumps([{"youtube_id": "abcDEF123-_", "content_id": "999", "confidence": "editor"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_library, "DECISIONS_PATH", dec)
    monkeypatch.setattr(ui_library, "MAP_PATH", mp)
    from ui_library import resolve_youtube_id

    assert resolve_youtube_id("abcDEF123-_") == "111"


def test_load_map_rows_survives_a_torn_or_missing_file(tmp_path, monkeypatch):
    import ui_library

    mp = tmp_path / "map.json"
    monkeypatch.setattr(ui_library, "MAP_PATH", mp)
    from ui_library import load_map_rows

    assert load_map_rows(mp) == []
    mp.write_text('[{"youtube_id": "x"', encoding="utf-8")  # mid-regeneration
    assert load_map_rows(mp) == []
    mp.write_text('["bad row", {"youtube_id": "x"}]', encoding="utf-8")  # garbage rows
    assert load_map_rows(mp) == [{"youtube_id": "x"}]


def test_youtube_id_requires_a_youtube_host():
    from ui_library import youtube_id_from_query

    assert youtube_id_from_query("https://example.com/watch?v=abcDEF123-_") is None
    assert youtube_id_from_query("https://evil.com/shorts/abcDEF123-_") is None
    assert youtube_id_from_query("https://m.youtube.com/watch?v=abcDEF123-_") == "abcDEF123-_"
    assert youtube_id_from_query("youtube.com/watch?v=abcDEF123-_") == "abcDEF123-_"
    # share links often carry a trailing slash
    assert youtube_id_from_query("https://youtu.be/RohuZGgpC_k/") == "RohuZGgpC_k"


def test_resolve_honors_an_explicit_rejection(tmp_path, monkeypatch):
    """A review-tab «не то» must not be overridden by the map's candidate."""
    import ui_library

    dec = tmp_path / "decisions.csv"
    dec.write_text(
        "youtube_id,content_id,verdict,decided_at\nabcDEF123-_,999,no_match,2026\n",
        encoding="utf-8",
    )
    mp = tmp_path / "map.json"
    mp.write_text(
        json.dumps([{"youtube_id": "abcDEF123-_", "content_id": "999", "confidence": "exact"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui_library, "DECISIONS_PATH", dec)
    monkeypatch.setattr(ui_library, "MAP_PATH", mp)
    assert ui_library.resolve_youtube_id("abcDEF123-_") is None


def test_map_generator_fails_without_the_editor_csv(tmp_path):
    import subprocess
    import sys as _sys

    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "youtube_map.py"
    r = subprocess.run(
        [_sys.executable, str(script), "--known-csv", str(tmp_path / "absent.csv")],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "YOUTUBE_API_KEY": "x"},
    )
    assert r.returncode == 1
    assert "editor mapping not found" in r.stderr


def test_search_untagged_finds_indexed_videos_the_tagger_skipped():
    from types import SimpleNamespace

    from ui_library import search_untagged

    videos = {
        "h1": SimpleNamespace(
            title="Михаил Кузмин «Александрийские песни», 1908",
            program="Сто лекций",
            date="2019-01-01",
            duration_seconds=1620.0,
            url="u1",
        ),
        "h2": SimpleNamespace(
            title="Про всё остальное",
            program=None,
            date="2020-01-01",
            duration_seconds=600.0,
            url="u2",
        ),
    }
    hits = search_untagged(videos, "кузмин")
    assert [e.video_hash for e in hits] == ["h1"]
    assert hits[0].subject == [] and hits[0].content_id is None
    assert search_untagged(videos, "   ") == []


def test_search_untagged_excludes_tagged_hashes_before_the_cut():
    """Tagged matches must not crowd untagged ones out of the limit."""
    from types import SimpleNamespace

    from ui_library import search_untagged

    videos = {
        f"t{i}": SimpleNamespace(
            title="Лекция", program=None, date=f"2020-01-{i:02d}", duration_seconds=1.0, url=None
        )
        for i in range(1, 6)
    }
    videos["u1"] = SimpleNamespace(
        title="Лекция", program=None, date="2019-01-01", duration_seconds=1.0, url=None
    )
    hits = search_untagged(videos, "лекция", limit=3, exclude={f"t{i}" for i in range(1, 6)})
    assert [e.video_hash for e in hits] == ["u1"]


def test_display_title_prefers_cms_then_transcript_then_placeholder():
    from ui_library import display_title

    cms = _ep("a", title="Лекция")
    no_cms = _ep("b")
    synthetic = {"b": "Глава СБУ Малюк уходит в отставку."}
    assert display_title(cms, "ru", synthetic) == ("Лекция", False)
    assert display_title(no_cms, "ru", synthetic) == ("Глава СБУ Малюк уходит в отставку.", True)
    assert display_title(no_cms, "ru", {}) == ("(без названия)", True)


def test_search_matches_synthetic_titles_for_cms_less_episodes():
    from ui_library import search_episodes

    eps = [_ep("b", date="2026-01-05"), _ep("c", title="Другое", date="2025-01-01")]
    hits = search_episodes(eps, "малюк", synthetic={"b": "Глава СБУ Малюк уходит в отставку."})
    assert [e.video_hash for e in hits] == ["b"]


def test_load_untitled_titles_tolerates_missing_and_bad_files(tmp_path):
    from ui_library import load_untitled_titles

    assert load_untitled_titles(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("[1,2", encoding="utf-8")
    assert load_untitled_titles(bad) == {}
    ok = tmp_path / "ok.json"
    ok.write_text('{"h1": "Заголовок", "h2": ""}', encoding="utf-8")
    assert load_untitled_titles(ok) == {"h1": "Заголовок"}


def test_snippet_takes_first_sentence_and_caps():
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "scripts"))
    from library_untitled_titles import snippet

    assert (
        snippet("Добрый вечер, знатоки, гости клуба. Это четвёртая игра.")
        == "Добрый вечер, знатоки, гости клуба."
    )
    long = "слово " * 40
    assert len(snippet(long)) <= 90 and snippet(long).endswith("…")


def test_stand_in_titles_are_markdown_escaped_but_cms_titles_are_not():
    from ui_library import display_title

    hostile = {"b": "Смотри](https://evil) `x` *y*"}
    title, stand_in = display_title(_ep("b"), "ru", hostile)
    assert stand_in and "](" not in title and "`x`" not in title
    assert title == "Смотри\\]\\(https://evil\\) \\`x\\` \\*y\\*"
    # CMS titles are trusted and pass through untouched
    assert display_title(_ep("a", title="A [b] *c*"), "ru", hostile) == ("A [b] *c*", False)


def test_load_untitled_titles_keeps_only_nonblank_strings(tmp_path):
    from ui_library import load_untitled_titles

    p = tmp_path / "t.json"
    p.write_text(
        '{"h1": "  ok  ", "h2": ["bad"], "h3": 5, "h4": "   ", "h5": ""}', encoding="utf-8"
    )
    assert load_untitled_titles(p) == {"h1": "ok"}


def test_snippet_returns_a_short_first_sentence():
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "scripts"))
    from library_untitled_titles import snippet

    assert snippet("Привет. Сегодня обсуждаем важное.") == "Привет."
    assert snippet("Конец без пробела после точки.") == "Конец без пробела после точки."


def test_untitled_hashes_use_last_row_wins_like_the_ui():
    import json as _json
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "scripts"))
    from library_untitled_titles import untitled_hashes

    lines = [
        _json.dumps({"video_hash": "a", "title": None}),
        _json.dumps({"video_hash": "a", "title": "Появилось название"}),  # re-tag gained a title
        _json.dumps({"video_hash": "b", "title": "Было"}),
        _json.dumps({"video_hash": "b", "title": None}),  # re-tag lost it
        _json.dumps({"video_hash": "c", "error": "boom"}),
        "{torn",
    ]
    assert untitled_hashes(lines) == ["b"]


def test_empty_speaker_column_names_the_actual_cause():
    """Three causes look identical to the editor unless the message differs."""
    from rainrag.library_similar import Episode
    from ui_library import empty_speaker_reason

    has_speaker = Episode(video_hash="h", speakers=["Ирина Хакамада"])
    assert empty_speaker_reason(has_speaker) == "nothing_similar"

    demoted = Episode(video_hash="h", speakers=[], presenter_demoted=True)
    assert empty_speaker_reason(demoted) == "demoted_no_guest"

    empty_card = Episode(video_hash="h", speakers=[], presenter_demoted=False)
    assert empty_speaker_reason(empty_card) == "no_speaker"


def test_every_empty_speaker_message_exists_in_both_languages():
    """A missing key would render the key itself into the editor's view."""
    from ui_library import _T

    for key in (
        "nothing_similar",
        "demoted_no_guest",
        "no_speaker",
        "presenter_demoted",
        "speakers_hidden",
    ):
        for lang in ("ru", "en"):
            assert _T[lang][key].strip()


def test_stat_key_distinguishes_two_writes_inside_one_filesystem_tick(tmp_path):
    """Seconds-resolution mtime would collide and keep serving stale genres."""
    from ui_library import _stat_key

    path = tmp_path / "library_programs.csv"
    path.write_text("title,genre\nA,лекция\n", encoding="utf-8")
    first = _stat_key(path)
    path.write_text("title,genre\nA,интервью\nB,ток-шоу\n", encoding="utf-8")
    assert _stat_key(path) != first


def test_stat_key_of_a_missing_file_is_stable_and_not_an_error(tmp_path):
    from ui_library import _stat_key

    assert _stat_key(tmp_path / "absent.csv") == (0, 0)


def _scored(video_hash, speakers):
    from rainrag.library_similar import Scored

    return Scored(_ep(video_hash, speakers=list(speakers)), 0.1, [], ["тема"])


def test_speaker_options_come_from_the_results_and_merge_name_variants():
    from ui_library import speaker_options

    rows = [
        _scored("a", ["Ирина Хакамада"]),
        _scored("b", ["Хакамада", "Екатерина Шульман"]),
        _scored("c", ["Ирина Хакамада"]),
        _scored("d", []),
    ]
    # One entry per person, the commonest spelling of the name, sorted.
    assert speaker_options(rows) == ["Екатерина Шульман", "Ирина Хакамада"]


def test_visible_results_without_a_filter_is_the_plain_cut():
    from ui_library import visible_results

    rows = [_scored(str(i), ["Ирина Хакамада"]) for i in range(5)]
    assert visible_results(rows, None, limit=3) == rows[:3]


def test_visible_results_with_every_speaker_selected_changes_nothing():
    from ui_library import speaker_options, visible_results

    rows = [
        _scored("a", ["Ирина Хакамада"]),
        _scored("b", ["Екатерина Шульман"]),
        _scored("c", []),
    ]
    assert visible_results(rows, speaker_options(rows), limit=10) == rows


def test_visible_results_keeps_an_episode_if_any_of_its_speakers_stays_selected():
    from ui_library import visible_results

    rows = [
        _scored("solo", ["Дмитрий Быков"]),
        _scored("duo", ["Дмитрий Быков", "Екатерина Шульман"]),
    ]
    kept = visible_results(rows, ["Екатерина Шульман"], limit=10)
    assert [r.episode.video_hash for r in kept] == ["duo"]


def test_visible_results_never_hides_an_episode_with_no_speakers():
    from ui_library import visible_results

    rows = [_scored("nameless", []), _scored("named", ["Дмитрий Быков"])]
    kept = visible_results(rows, ["Екатерина Шульман"], limit=10)
    assert [r.episode.video_hash for r in kept] == ["nameless"]


def test_visible_results_with_nothing_selected_hides_every_attributable_row():
    from ui_library import visible_results

    rows = [_scored("a", ["Ирина Хакамада"]), _scored("b", ["Екатерина Шульман"])]
    assert visible_results(rows, [], limit=10) == []


def test_hiding_a_speaker_backfills_the_visible_list_to_its_full_length():
    from ui_library import SIMILAR_DISPLAY_LIMIT, visible_results

    # A theme column deep enough to refill: the first four rows are Хакамада,
    # the rest Шульман, and only ten rows are ever shown.
    rows = [_scored(f"t{i}", ["Ирина Хакамада"]) for i in range(4)]
    rows += [_scored(f"t{i}", ["Екатерина Шульман"]) for i in range(4, 16)]

    shown = visible_results(
        rows, ["Ирина Хакамада", "Екатерина Шульман"], limit=SIMILAR_DISPLAY_LIMIT
    )
    assert [r.episode.video_hash for r in shown] == [f"t{i}" for i in range(10)]

    backfilled = visible_results(rows, ["Екатерина Шульман"], limit=SIMILAR_DISPLAY_LIMIT)
    # Still ten rows, not six: candidates t10 to t13 moved up into the gap.
    assert len(backfilled) == SIMILAR_DISPLAY_LIMIT
    assert [r.episode.video_hash for r in backfilled] == [f"t{i}" for i in range(4, 14)]


def test_speaker_options_are_empty_when_no_result_credits_anyone():
    from ui_library import speaker_options, visible_results

    rows = [_scored("a", []), _scored("b", [])]
    # No options means no widget and no filter, so nothing is hidden.
    assert speaker_options(rows) == []
    assert visible_results(rows, None, limit=10) == rows


def test_speaker_filter_matches_a_surname_typed_without_the_given_name():
    from ui_library import visible_results

    rows = [_scored("a", ["Ирина Хакамада"]), _scored("b", ["Екатерина Шульман"])]
    kept = visible_results(rows, ["Хакамада"], limit=10)
    assert [r.episode.video_hash for r in kept] == ["a"]


def test_a_column_emptied_by_the_filter_is_not_called_an_empty_card():
    """Four causes once the filter exists; three of them are not "no speaker"."""
    from ui_library import _T, empty_speaker_reason, visible_results

    seedless = _ep("seed", speakers=[])
    assert empty_speaker_reason(seedless) == "no_speaker"

    # The ranker did find matches; the editor hid them. Different message.
    rows = [_scored("a", ["Ирина Хакамада"])]
    assert visible_results(rows, [], limit=10) == []
    assert _T["ru"]["speakers_hidden"].strip()
    assert _T["en"]["speakers_hidden"].strip()


def test_the_filter_starts_empty_so_nothing_is_pre_selected():
    """185 pre-ticked chips on a real seed pushed the results off the screen."""
    from ui_library import carried_selection, picked_speakers

    options = ["Дмитрий Быков", "Ирина Хакамада"]
    picked = picked_speakers(None, [], [])
    assert picked == []
    assert carried_selection(options, picked) == []


def test_choosing_a_speaker_narrows_to_her_and_is_remembered():
    from ui_library import carried_selection, picked_speakers

    options = ["Дмитрий Быков", "Ирина Хакамада"]
    picked = picked_speakers(options, ["Ирина Хакамада"], [])
    assert carried_selection(options, picked) == ["Ирина Хакамада"]


def test_removing_a_chosen_speaker_forgets_her():
    from ui_library import carried_selection, picked_speakers

    options = ["Дмитрий Быков", "Ирина Хакамада"]
    picked = picked_speakers(options, [], ["Ирина Хакамада"])
    assert carried_selection(options, picked) == []


def test_a_choice_survives_the_speaker_leaving_the_pool_and_coming_back():
    """Narrowing the duration must not silently drop what the editor picked."""
    from ui_library import carried_selection, picked_speakers

    everyone = ["Дмитрий Быков", "Ирина Хакамада"]
    narrowed = ["Дмитрий Быков"]
    picked = picked_speakers(everyone, ["Ирина Хакамада"], [])
    assert carried_selection(narrowed, picked) == []
    picked = picked_speakers(narrowed, [], picked)
    assert carried_selection(everyone, picked) == ["Ирина Хакамада"]


def test_a_choice_survives_the_name_being_spelled_differently():
    """The chip's label is whichever spelling the current pool uses most."""
    from ui_library import carried_selection, picked_speakers

    picked = picked_speakers(["Дмитрий Быков", "Ирина Хакамада"], ["Ирина Хакамада"], [])
    assert carried_selection(["Дмитрий Быков", "Хакамада"], picked) == ["Хакамада"]


def test_an_empty_choice_leaves_every_result_visible():
    """Empty means no filter, not "hide everything"."""
    from ui_library import visible_results

    rows = [_scored("a", ["Ирина Хакамада"]), _scored("b", ["Дмитрий Быков"])]
    assert visible_results(rows, None, limit=10) == rows


def _speaker_state(*seeds):
    from ui_library import speaker_state_keys

    state = {"library_seed_pick": "irrelevant", "library_genres": ["лекция"]}
    for seed in seeds:
        for key in speaker_state_keys(seed):
            state[key] = ["Ирина Хакамада"]
    return state


def test_another_seeds_speaker_ticks_are_dropped_from_session_state():
    from ui_library import speaker_state_keys, stale_speaker_keys

    state = _speaker_state("old", "new")
    stale = stale_speaker_keys(state, speaker_state_keys("new"))
    # Only the previous seed's keys go, and no unrelated widget is touched.
    assert sorted(stale) == sorted(speaker_state_keys("old"))


def test_a_detour_through_a_seed_with_no_filter_drops_the_old_ticks():
    from ui_library import speaker_state_keys, stale_speaker_keys

    # An untagged seed shows no filter and so has no keys of its own. The
    # previous seed's must still go, or coming back to it would restore ticks
    # the editor set two seeds ago.
    state = _speaker_state("old")
    stale = stale_speaker_keys(state, speaker_state_keys("untagged"))
    assert sorted(stale) == sorted(speaker_state_keys("old"))


def test_the_shortlist_pool_draws_on_both_columns_not_just_the_first():
    """With a full speaker column the theme pool must still reach the top 5."""
    from ui_library import SIMILAR_POOL_LIMIT, visible_results

    same = [_scored(f"s{i}", ["Ирина Хакамада"]) for i in range(SIMILAR_POOL_LIMIT)]
    themed = [_scored(f"t{i}", ["Кто-то Другой"]) for i in range(SIMILAR_POOL_LIMIT)]
    pool = [
        r.episode
        for rows in (same, themed)
        for r in visible_results(rows, None, limit=SIMILAR_POOL_LIMIT)
    ]
    hashes = {e.video_hash for e in pool}
    assert any(h.startswith("s") for h in hashes)
    assert any(h.startswith("t") for h in hashes)
    assert len(pool) == SIMILAR_POOL_LIMIT * 2
