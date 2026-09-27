"""Stand-in titles for episodes with no CMS card.

Varya, 2026-09-25, researching the October plan: «в выдаче много выпусков без
возможности их идентифицировать», and she proposed dropping them from results.
The stand-in was the transcript's first sentence, which is the least
identifying part of a broadcast. These episodes are the freshest material in
the archive, so the answer is a better label, not hiding them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from library_untitled_titles import (  # noqa: E402
    UBIQUITOUS_SHARE,
    descriptor,
    latest_records,
    subject_frequency,
    untitled_hashes,
)


def test_frequency_is_the_share_of_episodes_carrying_a_subject():
    records = [
        {"subject": ["политика", "выборы"]},
        {"subject": ["политика"]},
        {"subject": ["политика", "балет"]},
        {"subject": []},  # no subjects: not counted in the denominator
    ]
    freq = subject_frequency(records)
    assert freq["политика"] == 1.0
    assert freq["выборы"] == freq["балет"] == 1 / 3


def test_a_subject_repeated_on_one_episode_counts_once():
    freq = subject_frequency([{"subject": ["Политика", "политика", "ПОЛИТИКА"]}])
    assert freq["политика"] == 1.0


def test_the_label_leads_with_guests_then_distinctive_subjects():
    freq = {"политика": 0.5, "курильские острова": 0.001, "санкции": 0.002}
    label = descriptor(
        {"guest": ["Давид Шарп", "Илья Шуманов"], "subject": ["политика", "курильские острова"]},
        freq,
    )
    assert label == "Давид Шарп, Илья Шуманов · курильские острова"


def test_ubiquitous_subjects_are_dropped():
    """«политика» is on half the archive and distinguishes nothing."""
    freq = {"политика": 0.49, "балет": 0.001}
    assert descriptor({"subject": ["политика", "балет"]}, freq) == "балет"
    assert UBIQUITOUS_SHARE < 0.49


def test_subjects_keep_the_taggers_order_rather_than_being_ranked_by_rarity():
    """Rarity-first promoted typos and throwaways: one episode led with «торт»."""
    freq = {"брак": 0.01, "торт": 0.00001}
    assert descriptor({"subject": ["брак", "торт"]}, freq) == "брак, торт"


def test_at_most_two_guests_and_four_subjects():
    freq = dict.fromkeys("абвгде", 0.001)
    label = descriptor(
        {"guest": ["Один", "Два", "Три"], "subject": list("абвгде")},
        freq,
    )
    assert label == "Один, Два · а, б, в, г"


def test_a_long_tag_does_not_cost_the_shorter_ones_after_it():
    """The loop used to stop at the first tag that did not fit."""
    freq = {"x" * 200: 0.001, "коротко": 0.001}
    label = descriptor({"subject": ["x" * 200, "коротко"]}, freq)
    assert label == "коротко"


def test_nothing_usable_returns_empty_so_the_caller_falls_back():
    assert descriptor({}, {}) == ""
    assert descriptor({"guest": [], "subject": ["политика"]}, {"политика": 0.9}) == ""
    assert descriptor({"guest": ["  "], "subject": [" "]}, {}) == ""


def test_a_guest_longer_than_the_cap_truncates_and_keeps_no_subjects():
    """The negative-room branch: room goes below zero and every subject is skipped."""
    label = descriptor({"guest": ["x" * 50], "subject": ["балет"]}, {"балет": 0.001}, max_chars=40)
    assert label == "x" * 39 + "…"
    assert " · " not in label


def test_truncation_cuts_cleanly_and_marks_the_cut():
    """Pins the marker and the rstrip, not just the length."""
    label = descriptor({"guest": ["Имя Фамилия"]}, {}, max_chars=5)
    assert label == "Имя…"  # cut lands on a space, which is stripped before the marker


def test_a_label_that_fits_is_left_alone():
    label = descriptor({"guest": ["Имя"], "subject": ["балет"]}, {"балет": 0.001}, max_chars=40)
    assert label == "Имя · балет"


def test_subject_frequency_survives_having_nothing_to_count():
    """The denominator guard: a plain count/total would raise here."""
    assert subject_frequency([]) == {}
    assert subject_frequency([{"subject": []}, {"subject": None}, {}]) == {}


def test_latest_row_wins_and_unusable_rows_are_ignored():
    """Must agree with the UI's dedupe_latest or the two disagree about a re-tag."""
    lines = [
        json.dumps({"video_hash": "a", "title": None, "subject": ["один"]}),
        "",
        "not json at all",
        json.dumps({"video_hash": "a", "title": None, "subject": ["два"]}),
        json.dumps({"video_hash": "b", "title": None, "error": "boom"}),
        json.dumps({"title": None, "subject": ["без хэша"]}),
    ]
    records = latest_records(lines)
    assert set(records) == {"a"}
    assert records["a"]["subject"] == ["два"]


def test_a_re_tagged_episode_that_gained_a_title_stops_getting_a_stand_in():
    lines = [
        json.dumps({"video_hash": "a", "title": None}),
        json.dumps({"video_hash": "a", "title": "Настоящее название"}),
        json.dumps({"video_hash": "b", "title": "Было название"}),
        json.dumps({"video_hash": "b", "title": None}),
    ]
    # a gained a title, b lost one: the stand-in follows the latest row both ways.
    assert untitled_hashes(lines) == ["b"]


def test_a_tag_that_normalises_to_nothing_is_dropped():
    """It misses the frequency map, so the filter used to wave it through."""
    assert descriptor({"subject": ["—", "!!!", "балет"]}, {"балет": 0.001}) == "балет"


def test_two_spellings_of_one_subject_take_one_slot():
    """«балет» and «балет!» normalise the same; they used to take two."""
    freq = {"балет": 0.001, "опера": 0.001}
    label = descriptor({"subject": ["Балет", "балет!", "опера"]}, freq)
    assert label == "Балет, опера"


def test_the_filter_and_the_frequency_map_agree_on_the_key():
    """Both sides must key on the normalised form or the filter is bypassed."""
    freq = subject_frequency([{"subject": ["Политика"]}, {"subject": ["политика!"]}])
    # One subject seen on both episodes, not two subjects on one each.
    assert freq == {"политика": 1.0}
    assert descriptor({"subject": ["политика!"]}, freq) == ""
