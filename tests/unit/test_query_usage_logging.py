"""Tests for query usage accounting.

Two things are being protected here:

* the ``[usage]`` line for a query is emitted exactly once per attempt, however
  the attempt ends, and never carries the question text;
* the deliberate 429/504 responses reach the client as 429/504 rather than
  being rewrapped as 500 -- which also keeps the usage line's outcome honest.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from loguru import logger

from rainrag.api import _record_query_usage, app
from rainrag.query import accumulate_token_usage


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def usage_lines():
    """Collect ``[usage]`` lines emitted during a test.

    pytest's caplog sees nothing here: the project logs through loguru, which
    does not route to the stdlib logging handlers caplog installs. Capturing
    means attaching a loguru sink.
    """
    captured: list[str] = []
    sink_id = logger.add(
        lambda message: captured.append(message.record["message"]),
        level="INFO",
        format="{message}",
    )
    try:
        yield captured
    finally:
        logger.remove(sink_id)


def _only_usage(lines: list[str]) -> list[str]:
    return [ln for ln in lines if "[usage]" in ln]


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class TestAccumulateTokenUsage:
    def test_openai_and_mistral_shape(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(
            sink, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4))
        )
        assert sink == {"tokens_in": 10, "tokens_out": 4, "llm_calls_measured": 1}

    def test_anthropic_shape(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(
            sink, SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=3))
        )
        assert sink == {"tokens_in": 7, "tokens_out": 3, "llm_calls_measured": 1}

    def test_gemini_shape(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(
            sink,
            SimpleNamespace(
                usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=2)
            ),
        )
        assert sink == {"tokens_in": 5, "tokens_out": 2, "llm_calls_measured": 1}

    def test_counts_accumulate_across_calls(self):
        """Rewriting and HyDE run before the answer; all three should be counted."""
        sink: dict[str, int] = {}
        for _ in range(3):
            accumulate_token_usage(
                sink, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4))
            )
        assert sink == {"tokens_in": 30, "tokens_out": 12, "llm_calls_measured": 3}

    def test_none_sink_is_a_no_op(self):
        accumulate_token_usage(None, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=1)))

    def test_missing_usage_records_nothing(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(sink, SimpleNamespace())
        accumulate_token_usage(sink, None)
        assert sink == {}

    def test_mock_attributes_do_not_leak_into_counts(self):
        """A MagicMock answers to every attribute; none of it is a token count."""
        sink: dict[str, int] = {}
        accumulate_token_usage(sink, MagicMock())
        assert sink == {}

    def test_negative_and_bool_values_are_ignored(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(
            sink, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=-5, completion_tokens=True))
        )
        assert sink == {}

    def test_float_counts_are_rounded_not_dropped(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(
            sink, SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12.0, completion_tokens=3.6))
        )
        assert sink == {"tokens_in": 12, "tokens_out": 4, "llm_calls_measured": 1}

    def test_string_counts_are_ignored(self):
        sink: dict[str, int] = {}
        accumulate_token_usage(
            sink, SimpleNamespace(usage=SimpleNamespace(prompt_tokens="12", completion_tokens="3"))
        )
        assert sink == {}


class TestRecordQueryUsage:
    def test_records_docs_and_tokens(self):
        usage: dict[str, object] = {}
        _record_query_usage(usage, {"cost.llm_tokens_in": 100, "cost.llm_tokens_out": 20}, docs=3)
        assert usage == {"docs": 3, "tokens_in": 100, "tokens_out": 20}

    def test_zero_tokens_are_omitted_not_logged_as_zero(self):
        usage: dict[str, object] = {}
        _record_query_usage(usage, {"cost.llm_tokens_in": 0, "cost.llm_tokens_out": 0}, docs=1)
        assert usage == {"docs": 1}

    def test_docs_derived_from_result_when_not_given(self):
        usage: dict[str, object] = {}
        _record_query_usage(usage, {"num_documents": 4})
        assert usage["docs"] == 4

    def test_mock_result_does_not_raise(self):
        usage: dict[str, object] = {}
        _record_query_usage(usage, MagicMock(), docs=2)
        assert usage["docs"] == 2


# ---------------------------------------------------------------------------
# The usage line itself
# ---------------------------------------------------------------------------


class TestQueryUsageLine:
    def test_success_emits_one_usage_line_without_the_question(self, test_client, usage_lines):
        with patch("rainrag.api.query_engine") as mock_engine, patch("rainrag.api.config"):
            mock_engine.config.llm.provider = "openai"
            mock_engine.query.return_value = {
                "question": "q",
                "answer": "a",
                "retrieved_documents": [
                    {
                        "rank": 1,
                        "score": 0.9,
                        "text": "t",
                        "path": "/test/doc1.vtt",
                        "language": "en",
                        "doc_id": "d1",
                    }
                ],
                "num_documents": 1,
                "metadata_fallback_hits": 0,
                "cost.llm_tokens_in": 120,
                "cost.llm_tokens_out": 45,
            }
            resp = test_client.post(
                "/query",
                json={"question": "sensitive question text", "language": "en"},
            )

        assert resp.status_code == 200
        lines = _only_usage(usage_lines)
        assert len(lines) == 1, lines
        line = lines[0]
        assert "event=query" in line
        assert "mode=corpus" in line
        assert "outcome=ok" in line
        assert "provider=openai" in line
        assert "docs=1" in line
        assert "tokens_in=120" in line
        assert "tokens_out=45" in line
        # The question is the one thing that must never reach the journal.
        assert "sensitive question text" not in line

    def test_engine_failure_is_recorded_as_a_single_line(self, test_client, usage_lines):
        """An engine crash is recorded as the 500 the client actually received.

        The handler turns unexpected exceptions into an HTTPException before the
        span closes over it, so the outcome tracks the response status rather
        than the internal exception type. That is the more useful of the two.
        """
        with patch("rainrag.api.query_engine") as mock_engine:
            mock_engine.config.llm.provider = "openai"
            mock_engine.query.side_effect = RuntimeError("boom")
            resp = test_client.post("/query", json={"question": "q", "language": "en"})

        assert resp.status_code == 500
        lines = _only_usage(usage_lines)
        assert len(lines) == 1, lines
        assert "outcome=http_500" in lines[0]
        assert "boom" not in lines[0]


class TestDeliberateStatusCodesSurvive:
    """The 429/504 raises were being rewrapped as 500 by the broad handler."""

    def test_timeout_surfaces_as_504(self, test_client, usage_lines, monkeypatch):
        monkeypatch.setattr("rainrag.api.QUERY_TIMEOUT_SECONDS", 0.1)

        def slow_query(**kwargs):
            # Comfortably past the 0.1s timeout without adding seconds to the
            # suite: the thread is not interruptible, so this keeps running in
            # the executor after the response is returned.
            time.sleep(1)
            return {}

        with patch("rainrag.api.query_engine") as mock_engine:
            mock_engine.config.llm.provider = "openai"
            # A real function, so the endpoint takes the threaded path.
            mock_engine.query = slow_query
            resp = test_client.post("/query", json={"question": "q", "language": "en"})

        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()
        lines = _only_usage(usage_lines)
        assert len(lines) == 1, lines
        assert "outcome=http_504" in lines[0]

    def test_busy_surfaces_as_429(self, test_client, usage_lines):
        class ExhaustedSemaphore:
            async def acquire(self):
                # asyncio.TimeoutError only became an alias of the builtin
                # TimeoutError in 3.11; on 3.10 they are distinct types and the
                # handler catches the asyncio one.
                raise asyncio.TimeoutError

            def release(self):
                pass

        with patch("rainrag.api.query_engine") as mock_engine:
            mock_engine.config.llm.provider = "openai"
            with patch("rainrag.api._get_query_semaphore", return_value=ExhaustedSemaphore()):
                resp = test_client.post("/query", json={"question": "q", "language": "en"})

        assert resp.status_code == 429
        lines = _only_usage(usage_lines)
        assert len(lines) == 1, lines
        assert "outcome=http_429" in lines[0]
