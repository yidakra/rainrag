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


def test_from_record_applies_the_programme_speaker_rule():
    """An interview's presenter must not become the thing the ranking matches on."""
    from rainrag.library_programs import Programme
    from rainrag.library_similar import Episode

    programmes = {"синдеева": Programme("Синдеева", ("интервью",), "Наталья Синдеева")}
    record = {
        "video_hash": "h1",
        "program": "Синдеева",
        "presenter_cms": ["Наталья Синдеева"],
        "guest": ["Михаил Ходорковский"],
    }
    episode = Episode.from_record(record, programmes)
    assert episode.speakers == ["Михаил Ходорковский"]
    assert episode.presenter_demoted


def test_from_record_without_a_programme_table_is_unchanged():
    from rainrag.library_similar import Episode

    record = {
        "video_hash": "h1",
        "program": "Синдеева",
        "presenter_cms": ["Наталья Синдеева"],
        "guest": ["Михаил Ходорковский"],
    }
    episode = Episode.from_record(record)
    assert episode.speakers == ["Наталья Синдеева", "Михаил Ходорковский"]
    assert not episode.presenter_demoted


def test_the_guest_not_the_interviewer_drives_the_ranking():
    """End to end: two interviews by the same host must not match each other."""
    from rainrag.library_programs import Programme
    from rainrag.library_similar import Episode, find_similar

    programmes = {"синдеева": Programme("Синдеева", ("интервью",), "Наталья Синдеева")}
    make = lambda h, guest: Episode.from_record(  # noqa: E731
        {
            "video_hash": h,
            "program": "Синдеева",
            "presenter_cms": ["Наталья Синдеева"],
            "guest": [guest],
            "subject": [],
            "duration_seconds": 3600,
        },
        programmes,
    )
    seed = make("seed", "Михаил Ходорковский")
    same_guest = make("same", "Михаил Ходорковский")
    other_guest = make("other", "Юлия Навальная")
    ranked = find_similar(seed, [same_guest, other_guest], limit=5)
    assert ranked[0].episode.video_hash == "same"
    assert ranked[0].shared_speakers == ["Михаил Ходорковский"]
    assert not [r for r in ranked if r.episode.video_hash == "other" and r.shared_speakers]


# ------------------------------------------------- same surname, other person


class TestPeopleMatch:
    """Varya, 2026-09-15: Юрий Быков was offered for Дмитрий Быков."""

    def test_same_surname_different_given_name_is_not_the_same_person(self):
        from rainrag.library_similar import people_match

        assert not people_match("Дмитрий Быков", "Юрий Быков")
        assert not people_match("Лия Ахеджакова", "Алиса Ахеджакова")

    def test_a_bare_surname_still_matches_the_full_name(self):
        """The model routinely returns only the surname; refusing that would
        lose most of the real overlap."""
        from rainrag.library_similar import people_match

        assert people_match("Ирина Хакамада", "Хакамада")
        assert people_match("Хакамада", "Ирина Хакамада")
        assert people_match("Екатерина Шульман", "Шульман")

    def test_initials_and_patronymics_do_not_block_a_match(self):
        from rainrag.library_similar import people_match

        assert people_match("И. Хакамада", "Ирина Хакамада")
        assert people_match("Владимир Вольфович Жириновский", "Жириновский")

    def test_different_surnames_never_match(self):
        from rainrag.library_similar import people_match

        assert not people_match("Екатерина Шульман", "Екатерина Шаврина")
        assert not people_match("", "Хакамада")

    def test_person_key_splits_surname_and_given_name(self):
        from rainrag.library_similar import person_key

        assert person_key("Ирина Хакамада") == ("хакамада", "ирина")
        assert person_key("Хакамада") == ("хакамада", "")
        # An initial is kept, not discarded: see TestInitialsAreNotAnAbsentGivenName.
        assert person_key("И. Хакамада") == ("хакамада", "и")
        assert person_key("") == ("", "")

    def test_shared_people_counts_one_person_once_but_keeps_both_spellings(self):
        from rainrag.library_similar import shared_people

        names, matched = shared_people(
            ["Ирина Хакамада"], ["Хакамада", "Юрий Быков", "Ирина Хакамада"]
        )
        assert names == ["Хакамада", "Ирина Хакамада"]
        assert matched == [("хакамада", "ирина")]

    def test_the_wrong_namesake_no_longer_scores_as_a_shared_speaker(self):
        from rainrag.library_similar import Episode, score_pair

        seed = Episode(video_hash="a", speakers=["Дмитрий Быков"], subject=["литература"])
        namesake = Episode(video_hash="b", speakers=["Юрий Быков"], subject=["литература"])
        himself = Episode(video_hash="c", speakers=["Быков"], subject=["литература"])
        idf = {"литература": 1.0}
        assert score_pair(seed, namesake, idf)[1] == []
        assert score_pair(seed, himself, idf)[1] == ["Быков"]

    def test_normalise_person_stays_the_surname_grouping_key(self):
        """The UI groups spellings by it; it is deliberately looser than a match."""
        from rainrag.library_similar import normalise_person

        assert normalise_person("Дмитрий Быков") == normalise_person("Юрий Быков") == "быков"


# ------------------------------------------------------------- genre filtering


class TestFilterGenres:
    """Varya, 2026-09-15: «Здесь и сейчас» survived a filter that excluded новости."""

    def _episode(self, **kw):
        from rainrag.library_similar import Episode

        return Episode(video_hash=kw.pop("video_hash", "h"), **kw)

    def test_the_reviewed_programme_genre_wins_over_the_model_labels(self):
        from rainrag.library_similar import filter_genres

        episode = self._episode(genre=["новости", "интервью"], programme_genres=["новости"])
        assert filter_genres(episode) == {"новости"}

    def test_the_model_labels_are_the_fallback_when_the_programme_has_none(self):
        from rainrag.library_similar import filter_genres

        episode = self._episode(genre=["новости", "интервью"], programme_genres=[])
        assert filter_genres(episode) == {"новости", "интервью"}

    def test_a_news_programme_is_excluded_from_an_interview_filter(self):
        from rainrag.library_similar import Episode, find_similar

        seed = Episode(video_hash="seed", speakers=["Ирина Хакамада"], subject=["интуиция"])
        news = Episode(
            video_hash="news",
            speakers=["Ирина Хакамада"],
            subject=["интуиция"],
            genre=["новости", "интервью"],
            programme_genres=["новости"],
        )
        talk = Episode(
            video_hash="talk",
            speakers=["Ирина Хакамада"],
            subject=["интуиция"],
            genre=["новости"],
            programme_genres=["интервью"],
        )
        found = find_similar(seed, [news, talk], genres=["интервью"], idf={"интуиция": 1.0})
        assert [r.episode.video_hash for r in found] == ["talk"]

    def test_from_record_carries_the_reviewed_genre(self):
        from rainrag.library_programs import Programme
        from rainrag.library_similar import Episode

        programmes = {"утро на дожде": Programme(title="Утро на Дожде", genres=("новости",))}
        episode = Episode.from_record(
            {"video_hash": "h", "program": "Утро на Дожде", "genre": ["новости", "интервью"]},
            programmes,
        )
        assert episode.programme_genres == ["новости"]
        assert episode.genre == ["новости", "интервью"]

    def test_without_a_programme_table_nothing_changes(self):
        from rainrag.library_similar import Episode, filter_genres

        episode = Episode.from_record(
            {"video_hash": "h", "program": "Утро на Дожде", "genre": ["новости", "интервью"]}
        )
        assert episode.programme_genres == []
        assert filter_genres(episode) == {"новости", "интервью"}


class TestInitialsAreNotAnAbsentGivenName:
    """Tenki on #87: «Д. Быков» folded to a bare surname and matched any Быков."""

    def test_an_initial_does_not_match_a_different_given_name(self):
        from rainrag.library_similar import people_match

        assert not people_match("Д. Быков", "Юрий Быков")
        assert not people_match("Юрий Быков", "Д. Быков")
        assert not people_match("И. Хакамада", "Юрий Хакамада")

    def test_an_initial_still_matches_the_name_it_abbreviates(self):
        from rainrag.library_similar import people_match

        assert people_match("Д. Быков", "Дмитрий Быков")
        assert people_match("И. Хакамада", "Ирина Хакамада")

    def test_person_key_keeps_the_initial_rather_than_dropping_it(self):
        from rainrag.library_similar import person_key

        assert person_key("И. Хакамада") == ("хакамада", "и")
        assert person_key("Хакамада") == ("хакамада", "")
        # A latin suffix is not a surname and leaves no given name behind.
        assert person_key("Noize MC") == ("noize", "")

    def test_given_names_agree_only_where_they_can(self):
        from rainrag.library_similar import given_names_agree

        assert given_names_agree("", "ирина")
        assert given_names_agree("и", "ирина")
        assert given_names_agree("ирина", "ирина")
        assert not given_names_agree("д", "юрий")
        assert not given_names_agree("дмитрий", "юрий")


class TestPersonIdentities:
    """CodeRabbit on #87: counting people by surname mis-scored namesakes."""

    def test_spellings_of_one_person_collapse(self):
        from rainrag.library_similar import person_identities

        assert person_identities(["Хакамада", "И. Хакамада", "Ирина Хакамада"]) == [
            ("хакамада", "ирина")
        ]

    def test_two_people_sharing_a_surname_stay_two(self):
        from rainrag.library_similar import person_identities

        assert person_identities(["Дмитрий Быков", "Юрий Быков"]) == [
            ("быков", "дмитрий"),
            ("быков", "юрий"),
        ]

    def test_a_bare_surname_alone_is_one_person(self):
        from rainrag.library_similar import person_identities

        assert person_identities(["Быков"]) == [("быков", "")]
        assert person_identities([]) == []
        assert person_identities(["", "  "]) == []

    def test_a_full_name_supersedes_the_initial_regardless_of_order(self):
        from rainrag.library_similar import person_identities

        assert person_identities(["И. Хакамада", "Ирина Хакамада"]) == [("хакамада", "ирина")]
        assert person_identities(["Ирина Хакамада", "И. Хакамада"]) == [("хакамада", "ирина")]

    def test_shared_people_reports_the_seed_identities_that_matched(self):
        from rainrag.library_similar import shared_people

        names, matched = shared_people(
            ["Дмитрий Быков", "Юрий Быков"], ["Юрий Быков", "Ирина Хакамада"]
        )
        assert names == ["Юрий Быков"]
        assert matched == [("быков", "юрий")]

    def test_a_partly_matched_seed_does_not_score_as_a_whole_one(self):
        """Half the seed's speakers are someone else, so half the weight."""
        from rainrag.library_similar import Episode, score_pair

        seed = Episode(video_hash="a", speakers=["Дмитрий Быков", "Юрий Быков"])
        one = Episode(video_hash="b", speakers=["Дмитрий Быков"])
        both = Episode(video_hash="c", speakers=["Дмитрий Быков", "Юрий Быков"])
        assert score_pair(seed, one, {})[0] < score_pair(seed, both, {})[0]


class TestOneSpellingIsOnePerson:
    """Tenki on #87 (second round): a bare surname credited every namesake."""

    def test_a_bare_surname_matches_only_one_of_two_namesakes(self):
        from rainrag.library_similar import shared_people

        names, matched = shared_people(["Дмитрий Быков", "Юрий Быков"], ["Быков"])
        assert names == ["Быков"]
        assert len(matched) == 1

    def test_a_named_candidate_is_not_consumed_by_an_ambiguous_one(self):
        """Unambiguous pairs are taken first, so «Юрий» wins its own slot."""
        from rainrag.library_similar import shared_people

        _, matched = shared_people(["Дмитрий Быков", "Юрий Быков"], ["Быков", "Юрий Быков"])
        assert matched == [("быков", "юрий")]

    def test_naming_both_namesakes_still_matches_both(self):
        from rainrag.library_similar import shared_people

        _, matched = shared_people(["Дмитрий Быков", "Юрий Быков"], ["Юрий Быков", "Дмитрий Быков"])
        assert matched == [("быков", "дмитрий"), ("быков", "юрий")]

    def test_a_bare_surname_is_still_a_full_match_for_a_single_seed_speaker(self):
        from rainrag.library_similar import shared_people

        names, matched = shared_people(["Дмитрий Быков"], ["Быков"])
        assert names == ["Быков"] and matched == [("быков", "дмитрий")]
