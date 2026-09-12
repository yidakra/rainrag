"""Tests for the blended shortlist and, above all, its weight renormalisation."""

from __future__ import annotations

from rainrag.library_blend import (
    WEIGHTS,
    Audience,
    audience_axis,
    blend_pair,
    blended_top,
    speaker_axis,
    theme_axis,
)
from rainrag.library_similar import Episode, subject_idf


def _ep(video_hash: str, speakers=(), subject=()) -> Episode:
    return Episode(video_hash=video_hash, speakers=list(speakers), subject=list(subject))


def _idf(*episodes: Episode) -> dict[str, float]:
    """IDF over the episodes plus filler.

    A subject carried by every episode in the pool has an IDF of zero, which
    is correct (it separates nothing) but makes a two-episode fixture score no
    theme overlap at all. The filler keeps the shared tags rare enough to
    carry weight, as they are in a 13,808-episode archive.
    """
    filler = [Episode(video_hash=f"filler{i}", subject=["наполнитель"]) for i in range(8)]
    return subject_idf([*episodes, *filler])


def test_a_row_on_theme_weight_alone_does_not_claim_a_speaker_match():
    """The mistake this feature has already made four times, in a fifth place."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция", "лидерство"])
    other = _ep("c", speakers=["Екатерина Шульман"], subject=["интуиция", "лидерство"])
    blended = blend_pair(seed, other, _idf(seed, other))
    assert blended.axes.get("speaker") == 0.0
    assert not blended.shared_speakers
    assert "спикер" not in blended.explain()
    assert "общие темы" in blended.explain()


def test_a_missing_axis_is_not_scored_as_zero():
    """Analytics cover a couple hundred episodes; the archive has 13,808."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    twin = _ep("c", speakers=["Ирина Хакамада"], subject=["интуиция"])
    idf = _idf(seed, twin, _ep("x", subject=["другое"]))
    blended = blend_pair(seed, twin, idf)
    # Speaker and theme both perfect, no analytics for either: a full score,
    # not 70 out of 100.
    assert set(blended.axes) == {"speaker", "theme"}
    assert blended.score == 1.0


def test_an_episode_with_analytics_does_not_outrank_one_without_by_default():
    """Scoring absent axes as zero would bury the entire unpublished archive."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    published = _ep("p", speakers=["Ирина Хакамада"], subject=["интуиция"])
    unpublished = _ep("u", speakers=["Ирина Хакамада"], subject=["интуиция"])
    idf = _idf(seed, published, unpublished)
    audiences = {
        "s": Audience(age_gender={"25-34.male": 60.0}, average_view_duration=600.0),
        "p": Audience(age_gender={"25-34.male": 60.0}, average_view_duration=600.0),
    }
    ranked = blended_top(seed, [published, unpublished], idf, audiences)
    assert [b.episode.video_hash for b in ranked] == ["p", "u"]
    assert ranked[0].score == ranked[1].score == 1.0


def test_a_poor_audience_match_does_cost_the_candidate():
    """Missing must not read as bad, but bad must still read as bad."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    same = _ep("a", speakers=["Ирина Хакамада"], subject=["интуиция"])
    different = _ep("b", speakers=["Ирина Хакамада"], subject=["интуиция"])
    idf = _idf(seed, same, different)
    audiences = {
        "s": Audience(age_gender={"25-34.male": 100.0}),
        "a": Audience(age_gender={"25-34.male": 100.0}),
        "b": Audience(age_gender={"55-64.female": 100.0}),
    }
    ranked = blended_top(seed, [same, different], idf, audiences)
    assert [b.episode.video_hash for b in ranked] == ["a", "b"]
    assert ranked[0].score > ranked[1].score


def test_the_weights_are_the_ones_varya_specified():
    assert WEIGHTS == {
        "speaker": 40.0,
        "theme": 30.0,
        "audience": 20.0,
        "depth": 5.0,
        "cpm": 5.0,
    }


def test_speaker_outweighs_theme_at_the_stated_ratio():
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    by_speaker = _ep("a", speakers=["Ирина Хакамада"], subject=["другое"])
    by_theme = _ep("b", speakers=["Кто-то Другой"], subject=["интуиция"])
    idf = _idf(seed, by_speaker, by_theme)
    ranked = blended_top(seed, [by_theme, by_speaker], idf)
    assert [b.episode.video_hash for b in ranked] == ["a", "b"]


def test_a_seed_with_no_speakers_drops_the_axis_rather_than_zeroing_it():
    """832 episodes have nobody credited and must still get a shortlist."""
    seed = _ep("s", subject=["интуиция"])
    candidate = _ep("c", speakers=["Кто-то"], subject=["интуиция"])
    blended = blend_pair(seed, candidate, _idf(seed, candidate))
    assert "speaker" not in blended.axes
    assert blended.axes["theme"] > 0


def test_speaker_axis_is_the_share_of_the_seeds_speakers_not_a_raw_count():
    seed = _ep("s", speakers=["Один Первый", "Два Второй"])
    half = _ep("c", speakers=["Один Первый"])
    assert speaker_axis(seed, half)[0] == 0.5
    assert speaker_axis(seed, _ep("d", speakers=["Кто-то Третий"]))[0] == 0.0
    assert speaker_axis(_ep("e"), half)[0] is None


def test_theme_axis_is_none_when_the_seed_has_no_subjects():
    seed = _ep("s")
    assert theme_axis(seed, _ep("c", subject=["тема"]), {})[0] is None


def test_audience_axis_needs_both_sides():
    profile = Audience(age_gender={"25-34.male": 50.0})
    assert audience_axis(profile, None) is None
    assert audience_axis(None, profile) is None
    assert audience_axis(profile, Audience(age_gender={})) is None
    assert audience_axis(profile, profile) == 1.0
    # Disjoint age bands is a measurement of dissimilarity, not missing data.
    assert audience_axis(profile, Audience(age_gender={"55-64.female": 50.0})) == 0.0
    # An all-zero mix carries no signal, so the cosine is undefined.
    assert audience_axis(profile, Audience(age_gender={"25-34.male": 0.0})) is None


def test_depth_and_cpm_compare_by_ratio_so_scale_does_not_decide():
    seed = _ep("s", subject=["тема"])
    candidate = _ep("c", subject=["тема"])
    idf = _idf(seed, candidate)
    close = blend_pair(
        seed,
        candidate,
        idf,
        Audience(average_view_duration=600.0, playback_based_cpm=2.0),
        Audience(average_view_duration=540.0, playback_based_cpm=2.0),
    )
    assert close.axes["depth"] == 0.9
    assert close.axes["cpm"] == 1.0


def test_nonpositive_measurements_drop_the_axis_rather_than_dividing():
    seed = _ep("s", subject=["тема"])
    candidate = _ep("c", subject=["тема"])
    blended = blend_pair(
        seed,
        candidate,
        _idf(seed, candidate),
        Audience(average_view_duration=0.0, playback_based_cpm=0.0),
        Audience(average_view_duration=600.0, playback_based_cpm=2.0),
    )
    assert "depth" not in blended.axes
    assert "cpm" not in blended.axes


def test_candidates_overlapping_on_nothing_are_dropped_not_padded():
    """Five rows an editor cannot use are worse than one they can."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    good = _ep("g", speakers=["Ирина Хакамада"], subject=["интуиция"])
    unrelated = [_ep(f"u{i}", speakers=["Никто Другой"], subject=["другое"]) for i in range(6)]
    idf = _idf(seed, good, *unrelated)
    ranked = blended_top(seed, [good, *unrelated], idf)
    assert [b.episode.video_hash for b in ranked] == ["g"]


def test_the_shortlist_is_capped_and_ordered_deterministically():
    seed = _ep("s", subject=["тема"])
    pool = [_ep(f"c{i}", subject=["тема"]) for i in range(9)]
    idf = _idf(seed, *pool)
    ranked = blended_top(seed, pool, idf, limit=5)
    assert len(ranked) == 5
    # Equal scores, so the tie-break must be stable rather than set order.
    assert [b.episode.video_hash for b in ranked] == sorted(b.episode.video_hash for b in ranked)
    assert blended_top(seed, list(reversed(pool)), idf, limit=5) == ranked


def test_explain_names_every_contributing_axis_and_nothing_else():
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    candidate = _ep("c", speakers=["Ирина Хакамада"], subject=["интуиция"])
    blended = blend_pair(
        seed,
        candidate,
        _idf(seed, candidate),
        Audience(average_view_duration=600.0),
        Audience(average_view_duration=600.0),
    )
    text = blended.explain()
    assert "тот же спикер: Ирина Хакамада" in text
    assert "общие темы: интуиция" in text
    assert "глубина смотрения: 100%" in text
    assert "CPM" not in text
    assert "аудитория" not in text


def test_custom_weights_are_honoured_so_varya_can_retune():
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    by_speaker = _ep("a", speakers=["Ирина Хакамада"], subject=["другое"])
    by_theme = _ep("b", speakers=["Другой Человек"], subject=["интуиция"])
    idf = _idf(seed, by_speaker, by_theme)
    themes_first = {"speaker": 10.0, "theme": 90.0, "audience": 0.0, "depth": 0.0, "cpm": 0.0}
    ranked = blended_top(seed, [by_speaker, by_theme], idf, weights=themes_first)
    assert [b.episode.video_hash for b in ranked] == ["b", "a"]


def test_explain_is_available_in_english_too():
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    candidate = _ep("c", speakers=["Ирина Хакамада"], subject=["интуиция"])
    text = blend_pair(seed, candidate, _idf(seed, candidate)).explain("en")
    assert text.startswith("same speaker: Ирина Хакамада")
    assert "shared subjects: интуиция" in text


def test_an_uncredited_candidate_is_unknown_on_the_speaker_axis_not_a_mismatch():
    """1,102 episodes have nobody credited; a 40-weight zero buried them."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    uncredited = _ep("u", speakers=[], subject=["интуиция"])
    other_person = _ep("o", speakers=["Кто-то Другой"], subject=["интуиция"])
    assert speaker_axis(seed, uncredited)[0] is None
    assert speaker_axis(seed, other_person)[0] == 0.0
    idf = _idf(seed, uncredited, other_person)
    assert "speaker" not in blend_pair(seed, uncredited, idf).axes
    assert blend_pair(seed, other_person, idf).axes["speaker"] == 0.0


def test_an_uncredited_candidate_outranks_one_credited_to_someone_else():
    """Not knowing must beat knowing it is wrong, on equal themes."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    uncredited = _ep("u", speakers=[], subject=["интуиция"])
    other_person = _ep("o", speakers=["Кто-то Другой"], subject=["интуиция"])
    idf = _idf(seed, uncredited, other_person)
    ranked = blended_top(seed, [other_person, uncredited], idf)
    assert [b.episode.video_hash for b in ranked] == ["u", "o"]


def test_a_zero_weight_axis_is_not_offered_as_a_reason():
    """Retuning theme to 0 must not still print «общие темы» as the explanation."""
    seed = _ep("s", speakers=["Ирина Хакамада"], subject=["интуиция"])
    candidate = _ep("c", speakers=["Ирина Хакамада"], subject=["интуиция"])
    speaker_only = {"speaker": 100.0, "theme": 0.0, "audience": 0.0, "depth": 0.0, "cpm": 0.0}
    blended = blend_pair(seed, candidate, _idf(seed, candidate), weights=speaker_only)
    assert "theme" not in blended.axes
    assert "общие темы" not in blended.explain()
    assert "тот же спикер" in blended.explain()
