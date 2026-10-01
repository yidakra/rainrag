"""Tests for the flagged-passage notes: what is said about what, and when."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from rainrag.library_notes import (  # noqa: E402
    Topic,
    find_spans,
    format_timecode,
    load_topics,
    parse_vtt,
)


VTT = """WEBVTT

1
00:00:01.000 --> 00:00:05.000
Добрый вечер, сегодня говорим про Беларусь.

2
00:00:10.000 --> 00:00:20.000
На Украине в это время происходило другое.

3
00:00:25.000 --> 00:00:35.000
И в Украину тоже поехали наблюдатели.

4
01:02:03.000 --> 01:02:10.000
Крым мы обсудим отдельно.
"""


def test_cues_carry_their_timing_and_their_words():
    cues = parse_vtt(VTT)
    assert len(cues) == 4
    assert cues[0].start == 1.0 and cues[0].end == 5.0
    assert "Беларусь" in cues[0].text
    # An hour in: the hours field is optional in the archive's files.
    assert cues[3].start == 3723.0


def test_a_file_that_is_not_a_transcript_yields_nothing_rather_than_raising():
    """One odd file must not take out the note for every other episode."""
    assert parse_vtt("") == []
    assert parse_vtt("not a subtitle file at all") == []


def test_an_inflected_mention_still_counts():
    """Russian inflects: «Украине» and «Украину» are the same subject."""
    spans = find_spans(parse_vtt(VTT), [Topic("Украина", ("Украин",))])
    assert len(spans) == 1, "two cues 5 seconds apart are one passage"
    assert spans[0].hits == 2
    assert spans[0].start == 10.0 and spans[0].end == 35.0


def test_passages_far_apart_stay_apart():
    """An editor wants «с 19:04 по 24:01», not one span covering the hour."""
    text = (
        "WEBVTT\n\n"
        "1\n00:02:00.000 --> 00:02:30.000\nСначала про Крым.\n\n"
        "2\n01:02:03.000 --> 01:02:40.000\nИ снова Крым, много позже.\n"
    )
    spans = find_spans(parse_vtt(text), [Topic("Крым", ("Крым",))])
    assert len(spans) == 2
    assert format_timecode(spans[0].start) == "2:00"
    assert format_timecode(spans[1].start) == "1:02:03"


def test_a_single_passing_mention_is_dropped_but_a_repeated_one_is_kept():
    """A word said once in two seconds is not something to plan around."""
    brief = "WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nКрым.\n"
    assert find_spans(parse_vtt(brief), [Topic("Крым", ("Крым",))]) == []
    twice = brief + "\n2\n00:00:10.000 --> 00:00:11.000\nКрым снова.\n"
    assert len(find_spans(parse_vtt(twice), [Topic("Крым", ("Крым",))])) == 1


def test_a_topic_does_not_match_a_longer_unrelated_word():
    """«Крым» must not be found inside «Крымск», which is a different place."""
    text = (
        "WEBVTT\n\n1\n00:00:01.000 --> 00:00:20.000\nГород Крымскулинский, дважды Крымскулинский.\n"
    )
    assert find_spans(parse_vtt(text), [Topic("Крым", ("Крым",))]) == []


def test_the_topic_list_is_editorial_and_may_be_absent(tmp_path):
    """No topics, no notes, and the card renders as it did before."""
    assert load_topics(tmp_path / "absent.csv") == []
    path = tmp_path / "t.csv"
    path.write_text("topic,patterns\nУкраина,Украин;Киев\nПустое,\n,Ничего\n", encoding="utf-8")
    topics = load_topics(path)
    assert [t.name for t in topics] == ["Украина", "Пустое"]
    assert topics[0].patterns == ("Украин", "Киев")
    # A row with no spellings falls back to looking for its own name.
    assert topics[1].patterns == ("Пустое",)


def test_timecodes_read_the_way_an_editor_writes_them():
    assert format_timecode(0) == "0:00"
    assert format_timecode(125) == "2:05"
    assert format_timecode(3723) == "1:02:03"


def _span(topic, start, end, hits):
    return {"topic": topic, "start": start, "end": end, "hits": hits, "quote": ""}


def test_a_topic_the_episode_only_brushes_past_earns_no_note():
    """With no bar, 54% of 13 808 episodes were flagged and the median
    passage was 27 seconds: wallpaper rather than a signal."""
    from rainrag.library_notes import qualifying_spans

    assert qualifying_spans([_span("Крым", 10, 40, 2)]) == []


def test_a_topic_the_episode_dwells_on_is_kept_whole():
    """Totals across the episode, not per passage: four minutes spread over
    three passages is what an editor plans around, and each may be short."""
    from rainrag.library_notes import qualifying_spans

    spans = [
        _span("Украина", 0, 50, 1),
        _span("Украина", 300, 350, 1),
        _span("Украина", 600, 640, 1),
    ]
    assert qualifying_spans(spans) == spans


def test_each_topic_is_judged_on_its_own():
    from rainrag.library_notes import qualifying_spans

    spans = [_span("Украина", 0, 200, 5), _span("Крым", 400, 410, 1)]
    assert [s["topic"] for s in qualifying_spans(spans)] == ["Украина"]


def test_the_bar_is_a_dial_the_caller_can_turn():
    """It is an editorial judgement, so it is an argument rather than a law."""
    from rainrag.library_notes import qualifying_spans

    spans = [_span("Крым", 10, 40, 2)]
    assert qualifying_spans(spans, min_seconds=10, min_hits=1) == spans
