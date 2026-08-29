"""Tests for transcript-based episode tagging.

The tagger fills the four fields the CMS has no source for. What must hold: a
bad response for one episode costs that episode only, transient provider
errors are retried rather than recorded as permanent failures, and the
transcript sampling keeps the parts that carry topical signal.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rainrag.library_tagger import (
    _is_retryable,
    build_fewshot_block,
    condense_transcript,
    parse_tagging_response,
    read_vtt_text,
    tag_episode,
    tag_overlap_score,
)


VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
Добрый вечер, сегодня у нас в гостях

2
00:00:04.000 --> 00:00:08.000
Екатерина Шульман, политолог.
"""


class TestReadVtt:
    def test_strips_timestamps_and_cue_numbers(self, tmp_path: Path) -> None:
        path = tmp_path / "a.vtt"
        path.write_text(VTT, encoding="utf-8")
        text = read_vtt_text(path)
        assert "-->" not in text and "WEBVTT" not in text
        assert "Екатерина Шульман" in text

    def test_handles_broken_encoding(self, tmp_path: Path) -> None:
        path = tmp_path / "b.vtt"
        path.write_bytes(b"WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\n\xff\xfe text\n")
        assert "text" in read_vtt_text(path)


class TestCondense:
    def test_short_transcript_is_untouched(self) -> None:
        assert condense_transcript("короткий текст") == "короткий текст"

    def test_long_transcript_keeps_head_middle_and_tail(self) -> None:
        text = "A" * 10_000 + "B" * 40_000 + "C" * 10_000
        out = condense_transcript(text, head=100, middle=100, tail=100)
        assert out.startswith("A") and out.endswith("C")
        assert "B" in out, "the middle must be sampled, not skipped"
        assert len(out) < len(text)


class TestParseResponse:
    def test_extracts_json_from_prose(self) -> None:
        raw = 'Вот карточка:\n```json\n{"subject": ["политика"], "genre": ["интервью"]}\n```'
        parsed = parse_tagging_response(raw)
        assert parsed["subject"] == ["политика"]
        assert parsed["genre"] == ["интервью"]

    def test_missing_keys_become_empty_lists(self) -> None:
        assert parse_tagging_response('{"subject": ["x"]}')["guest"] == []

    def test_string_is_coerced_to_list(self) -> None:
        assert parse_tagging_response('{"genre": "лекция"}')["genre"] == ["лекция"]

    def test_duplicates_are_dropped_case_insensitively(self) -> None:
        out = parse_tagging_response('{"subject": ["Политика", "политика", "ПОЛИТИКА"]}')
        assert out["subject"] == ["Политика"]

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_tagging_response("модель отказалась отвечать")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_tagging_response("[1, 2, 3]")


class TestRetryClassification:
    @pytest.mark.parametrize(
        "message",
        ["Status 429 Rate limit exceeded", "connection timed out", "503 overloaded"],
    )
    def test_transient_errors_are_retryable(self, message: str) -> None:
        assert _is_retryable(RuntimeError(message))

    def test_a_bad_transcript_is_not_retryable(self) -> None:
        assert not _is_retryable(ValueError("no JSON object in response"))


class TestTagEpisode:
    def _engine(self, *responses):
        engine = MagicMock()
        engine.generate_answer.side_effect = responses
        return engine

    def test_happy_path(self) -> None:
        engine = self._engine('{"subject": ["политика"], "genre": ["интервью"]}')
        result = tag_episode(
            engine,
            video_hash="h",
            content_id="1",
            title="t",
            program="p",
            date="2019-04-27",
            duration_minutes=54,
            presenters=["Наталья Синдеева"],
            mentioned=[],
            transcript="текст",
        )
        assert result.error is None
        assert result.subject == ["политика"]

    def test_rate_limit_is_retried_then_succeeds(self, monkeypatch) -> None:
        monkeypatch.setattr("rainrag.library_tagger.time.sleep", lambda _s: None)
        engine = self._engine(
            RuntimeError("Status 429 Rate limit exceeded"), '{"subject": ["политика"]}'
        )
        result = tag_episode(
            engine,
            video_hash="h",
            content_id="1",
            title="t",
            program="p",
            date="d",
            duration_minutes=30,
            presenters=[],
            mentioned=[],
            transcript="текст",
        )
        assert result.error is None
        assert engine.generate_answer.call_count == 2

    def test_permanent_failure_is_recorded_not_raised(self) -> None:
        """One unusable episode must not abort a run of thousands."""
        engine = self._engine("не JSON")
        result = tag_episode(
            engine,
            video_hash="h",
            content_id="1",
            title="t",
            program="p",
            date="d",
            duration_minutes=30,
            presenters=[],
            mentioned=[],
            transcript="текст",
        )
        assert result.error is not None
        assert result.subject == []


class TestOverlapScore:
    def test_identical_tags_score_full_recall(self) -> None:
        s = tag_overlap_score(["политика", "выборы"], ["политика", "выборы"])
        assert s["recall"] == 1.0

    def test_matching_is_lenient_about_wording(self) -> None:
        """«российская политика» should count against «политика»."""
        assert tag_overlap_score(["российская политика"], ["политика"])["recall"] == 1.0

    def test_empty_gold_does_not_divide_by_zero(self) -> None:
        assert tag_overlap_score(["x"], [])["recall"] == 0.0


class TestFewshot:
    def test_block_shows_the_expected_answer_shape(self) -> None:
        block = build_fewshot_block(
            {"title": "T", "parent_program": "P", "subject": ["политика"], "genre": ["интервью"]}
        )
        assert "политика" in block and "интервью" in block and "T" in block


class TestGreedyJsonRegression:
    """A greedy `{.*}` spanned to the last brace, losing good responses."""

    def test_trailing_object_does_not_break_parsing(self) -> None:
        raw = '{"subject": ["политика"]}\n\nКомментарий: {"note": "лишнее"}'
        assert parse_tagging_response(raw)["subject"] == ["политика"]

    def test_leading_prose_with_braces_is_skipped(self) -> None:
        raw = 'Ответ ниже.\n{"subject": ["психология"], "genre": ["лекция"]}'
        assert parse_tagging_response(raw)["genre"] == ["лекция"]

    def test_nested_objects_survive(self) -> None:
        raw = '{"subject": ["x"], "meta": {"nested": true}}'
        assert parse_tagging_response(raw)["subject"] == ["x"]
