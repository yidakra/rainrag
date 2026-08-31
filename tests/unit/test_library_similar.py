"""Tests for episode similarity ranking.

The behaviours here are taken from the answers Varya wrote by hand for her
first query: the same speaker dominates, distinctive shared subjects can carry
an episode with a different speaker, and duration/genre exclude rather than
merely demote.
"""

from __future__ import annotations

import pytest

from rainrag.library_similar import (
    Episode,
    find_similar,
    normalise_person,
    normalise_tag,
    score_pair,
    subject_idf,
)


def ep(hash_: str, *, subject=(), speakers=(), genre=("лекция",), minutes=45, **kw) -> Episode:
    return Episode(
        video_hash=hash_,
        content_id=kw.get("content_id", hash_),
        title=kw.get("title", hash_),
        program=kw.get("program", "Лекции на Дожде"),
        date=kw.get("date", "2018-01-08"),
        duration_seconds=minutes * 60,
        genre=list(genre),
        subject=list(subject),
        speakers=list(speakers),
    )


class TestNormalisation:
    def test_tags_fold_case_and_punctuation(self):
        assert normalise_tag("Женщины-Лидеры,") == normalise_tag("женщины-лидеры")

    def test_yo_folds_to_ye(self):
        assert normalise_tag("Всё") == normalise_tag("Все")

    def test_person_compares_on_surname(self):
        """«Ирина Хакамада» and «Хакамада» are the same speaker."""
        assert normalise_person("Ирина Хакамада") == normalise_person("Хакамада")

    def test_person_ignores_initials(self):
        assert normalise_person("И. Хакамада") == normalise_person("Ирина Хакамада")


class TestIdf:
    def test_common_tags_weigh_less_than_rare_ones(self):
        pool = [ep(f"v{i}", subject=["политика"]) for i in range(20)]
        pool.append(ep("rare", subject=["политика", "теория пустоты"]))
        idf = subject_idf(pool)
        assert idf["теория пустоты"] > idf["политика"]

    def test_universal_tag_carries_no_weight(self):
        pool = [ep(f"v{i}", subject=["политика"]) for i in range(10)]
        assert subject_idf(pool)["политика"] == pytest.approx(0.0)


class TestScoring:
    def test_same_speaker_outranks_topic_overlap(self):
        """Four of the six expected results are simply the same person again."""
        seed = ep("seed", subject=["политика", "лидерство"], speakers=["Ирина Хакамада"])
        same_speaker = ep("a", subject=["кулинария"], speakers=["Ирина Хакамада"])
        same_topics = ep("b", subject=["политика", "лидерство"], speakers=["Кто-то Другой"])
        idf = subject_idf([seed, same_speaker, same_topics])
        assert score_pair(seed, same_speaker, idf)[0] > score_pair(seed, same_topics, idf)[0]

    def test_distinctive_shared_topics_beat_generic_ones(self):
        """«женщины-лидеры» separates; «политика» does not."""
        pool = [ep(f"noise{i}", subject=["политика"]) for i in range(30)]
        seed = ep("seed", subject=["политика", "женщины-лидеры"])
        distinctive = ep("d", subject=["женщины-лидеры"])
        generic = ep("g", subject=["политика"])
        idf = subject_idf([seed, distinctive, generic, *pool])
        assert score_pair(seed, distinctive, idf)[0] > score_pair(seed, generic, idf)[0]

    def test_reasons_are_reported_for_the_editor(self):
        seed = ep("seed", subject=["политика"], speakers=["Ирина Хакамада"])
        cand = ep("a", subject=["политика"], speakers=["Ирина Хакамада"])
        score, speakers, subjects = score_pair(seed, cand, subject_idf([seed, cand]))
        assert speakers == ["Ирина Хакамада"]
        assert subjects == ["политика"]
        assert score > 0

    def test_nothing_in_common_scores_zero(self):
        seed = ep("seed", subject=["политика"])
        cand = ep("a", subject=["кулинария"])
        assert score_pair(seed, cand, subject_idf([seed, cand]))[0] == 0


class TestFindSimilar:
    def test_seed_is_never_returned(self):
        seed = ep("seed", subject=["политика"])
        assert all(r.episode.video_hash != "seed" for r in find_similar(seed, [seed]))

    def test_duration_is_a_filter_not_a_preference(self):
        """«длиной от 30 минут» excludes; a great 10-minute match must not appear."""
        seed = ep("seed", subject=["политика"], speakers=["Хакамада"])
        short = ep("short", subject=["политика"], speakers=["Хакамада"], minutes=10)
        out = find_similar(seed, [short], min_duration_minutes=30)
        assert out == []

    def test_genre_is_a_filter(self):
        seed = ep("seed", subject=["политика"])
        news = ep("news", subject=["политика"], genre=["новости"])
        assert find_similar(seed, [news], genres=["лекция", "интервью"]) == []

    def test_genre_filter_accepts_any_listed_genre(self):
        seed = ep("seed", subject=["политика"])
        interview = ep("i", subject=["политика"], genre=["интервью"])
        assert len(find_similar(seed, [interview], genres=["лекция", "интервью"])) == 1

    def test_results_are_ranked_by_score(self):
        seed = ep("seed", subject=["политика", "лидерство"], speakers=["Хакамада"])
        weak = ep("weak", subject=["политика"])
        strong = ep("strong", subject=["политика", "лидерство"], speakers=["Хакамада"])
        out = find_similar(seed, [weak, strong])
        assert [r.episode.video_hash for r in out] == ["strong", "weak"]

    def test_limit_is_respected(self):
        seed = ep("seed", subject=["политика"])
        pool = [ep(f"v{i}", subject=["политика"]) for i in range(10)]
        assert len(find_similar(seed, pool, limit=3)) == 3

    def test_untagged_candidates_do_not_crash(self):
        seed = ep("seed", subject=["политика"])
        assert find_similar(seed, [ep("empty", subject=[], speakers=[])]) == []


class TestFromRecord:
    def test_speakers_merge_cms_presenter_and_model_guest(self):
        """For a lecture the CMS field is the lecturer; for an interview the
        guest is who the search is about. Neither alone is enough."""
        e = Episode.from_record(
            {
                "video_hash": "h",
                "presenter_cms": ["Наталья Синдеева"],
                "guest": ["Екатерина Шульман"],
                "subject": ["политика"],
            }
        )
        assert e.speakers == ["Наталья Синдеева", "Екатерина Шульман"]

    def test_missing_fields_default_empty(self):
        e = Episode.from_record({"video_hash": "h"})
        assert (e.subject, e.speakers, e.genre) == ([], [], [])


class TestSurnameMatchingRegression:
    """Review caught that this used the longest token, not the surname.

    Anyone whose given name is longer than their surname silently failed to
    match — including Шульман, one of the six episodes the ranking is scored
    against.
    """

    @pytest.mark.parametrize(
        "full,bare",
        [
            ("Екатерина Шульман", "Шульман"),
            ("Дмитрий Быков", "Быков"),
            ("Наталья Синдеева", "Синдеева"),
            ("Ирина Хакамада", "Хакамада"),
            ("Вера Полозкова", "Полозкова"),
        ],
    )
    def test_full_name_matches_bare_surname(self, full: str, bare: str) -> None:
        assert normalise_person(full) == normalise_person(bare)

    def test_different_people_do_not_collide(self) -> None:
        assert normalise_person("Екатерина Шульман") != normalise_person("Екатерина Шаврина")

    def test_single_token_name_survives(self) -> None:
        assert normalise_person("Хакамада") == "хакамада"

    def test_speaker_bonus_now_applies_to_shulman(self) -> None:
        """The end-to-end effect: the bonus was silently never awarded."""
        seed = ep("seed", subject=["политика"], speakers=["Екатерина Шульман"])
        cand = ep("a", subject=["политика"], speakers=["Шульман"])
        _score, speakers, _subjects = score_pair(seed, cand, subject_idf([seed, cand]))
        assert speakers == ["Шульман"]


def _ep(
    video_hash, *, content_id=None, subject=(), speakers=(), genre=("лекция",), duration=3600.0
):
    from rainrag.library_similar import Episode

    return Episode(
        video_hash=video_hash,
        content_id=content_id,
        duration_seconds=duration,
        genre=list(genre),
        subject=list(subject),
        speakers=list(speakers),
    )


def test_dedupe_latest_keeps_the_last_row_for_a_repeated_hash():
    """The tag file is appended to, so the newer run must win."""
    from rainrag.library_similar import dedupe_latest

    older = _ep("aaa", content_id="1", subject=["старое"], duration=195180.0)
    newer = _ep("aaa", content_id="1", subject=["новое"], duration=3253.0)
    result = dedupe_latest([older, newer])

    assert len(result) == 1
    assert result[0].subject == ["новое"]
    # The superseded row for 484740 claimed 54 hours; keeping it would let a
    # bad runtime through the duration filter.
    assert result[0].duration_seconds == 3253.0


def test_dedupe_latest_keeps_distinct_hashes_sharing_a_content_id():
    """Two cuts of one CMS article are an editorial question, not a duplicate.

    Eight content_ids in the pool have two hashes with identical titles and
    runtimes up to 33% apart. Collapsing them would hide archive content.
    """
    from rainrag.library_similar import dedupe_latest

    result = dedupe_latest([_ep("aaa", content_id="7"), _ep("bbb", content_id="7")])
    assert {e.video_hash for e in result} == {"aaa", "bbb"}


def test_find_similar_returns_a_repeated_episode_only_once():
    from rainrag.library_similar import find_similar

    seed = _ep("seed", content_id="0", subject=["интуиция"], speakers=["Ирина Хакамада"])
    dup_a = _ep("dup", content_id="9", subject=["интуиция"], speakers=["Ирина Хакамада"])
    dup_b = _ep(
        "dup", content_id="9", subject=["интуиция", "лидерство"], speakers=["Ирина Хакамада"]
    )

    results = find_similar(seed, [dup_a, dup_b], limit=10)
    assert [r.episode.video_hash for r in results] == ["dup"]
    # Last row wins, so the shortlist shows the newer tagging.
    assert "лидерство" in results[0].episode.subject


def test_find_similar_excludes_a_re_tagged_seed_from_its_own_results():
    """A duplicate row of the seed would otherwise rank first against itself."""
    from rainrag.library_similar import find_similar

    seed = _ep("seed", content_id="0", subject=["интуиция"], speakers=["Ирина Хакамада"])
    seed_again = _ep("seed", content_id="0", subject=["интуиция"], speakers=["Ирина Хакамада"])
    other = _ep("other", content_id="1", subject=["интуиция"])

    results = find_similar(seed, [seed_again, other], limit=10)
    assert [r.episode.video_hash for r in results] == ["other"]
