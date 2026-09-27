"""Stand-in titles for episodes with no CMS card.

Varya, 2026-09-25, researching the October plan: «в выдаче много выпусков без
возможности их идентифицировать», and she proposed dropping them from results.
The stand-in was the transcript's first sentence, which is the least
identifying part of a broadcast. These episodes are the freshest material in
the archive, so the answer is a better label, not hiding them.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from library_untitled_titles import (  # noqa: E402
    UBIQUITOUS_SHARE,
    descriptor,
    subject_frequency,
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


def test_the_label_is_capped():
    freq = {"a" * 80: 0.001}
    label = descriptor({"guest": ["Имя Фамилия"], "subject": ["a" * 80]}, freq, max_chars=40)
    assert len(label) <= 40
