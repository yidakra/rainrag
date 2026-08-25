"""Tests for answer streaming: engine generators and the /query/stream SSE endpoint.

The load-bearing contract: joining query_stream's deltas produces exactly the
answer query() would have returned, and the "done" event is shape-identical to
/query's response — so a client can adopt streaming without changing anything
downstream of it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rainrag.api import app


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


def _sse_events(body: str) -> list[tuple[str, str]]:
    """Parse an SSE body into (event, data-json-string) pairs."""
    events = []
    for block in body.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln]
        if not lines:
            continue
        event = next(ln[len("event: ") :] for ln in lines if ln.startswith("event: "))
        data = next(ln[len("data: ") :] for ln in lines if ln.startswith("data: "))
        events.append((event, data))
    return events


# ---------------------------------------------------------------------------
# Engine: generate_answer_stream
# ---------------------------------------------------------------------------


def _chunk(text: str | None):
    """An OpenAI/Mistral-shaped streaming chunk."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))], usage=None
    )


def _engine_with_provider(provider: str):
    """A bare engine instance with just enough state for generate_answer_stream."""
    from rainrag.query import RAGQueryEngine

    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.config = SimpleNamespace(
        llm=SimpleNamespace(provider=provider),
        mistral=SimpleNamespace(model_name="m", max_tokens=100, temperature=0.0),
        openai=SimpleNamespace(model_name="o", max_tokens=100, temperature=0.0),
    )
    engine.mistral_client = MagicMock()
    engine.openai_client = MagicMock()
    return engine


class TestGenerateAnswerStream:
    def test_openai_deltas_are_yielded_in_order(self):
        engine = _engine_with_provider("openai")
        engine.openai_client.chat.completions.create.return_value = iter(
            [_chunk("Пер"), _chunk("вый"), _chunk(None), _chunk(" ответ")]
        )
        out = list(engine.generate_answer_stream([{"role": "user", "content": "q"}]))
        assert out == ["Пер", "вый", " ответ"]

    def test_openai_requests_usage_in_the_stream(self):
        engine = _engine_with_provider("openai")
        engine.openai_client.chat.completions.create.return_value = iter([_chunk("x")])
        list(engine.generate_answer_stream([{"role": "user", "content": "q"}]))
        kwargs = engine.openai_client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    def test_mistral_events_unwrap_data(self):
        engine = _engine_with_provider("mistral")
        engine.mistral_client.chat.stream.return_value = iter(
            [SimpleNamespace(data=_chunk("от")), SimpleNamespace(data=_chunk("вет"))]
        )
        out = list(engine.generate_answer_stream([{"role": "user", "content": "q"}]))
        assert out == ["от", "вет"]

    def test_unsupported_provider_falls_back_to_one_blocking_chunk(self):
        engine = _engine_with_provider("claude")
        with patch.object(engine, "generate_answer", return_value="целиком") as mock_gen:
            out = list(engine.generate_answer_stream([{"role": "user", "content": "q"}]))
        assert out == ["целиком"]
        mock_gen.assert_called_once()

    def test_error_before_first_delta_falls_back_to_blocking(self):
        """Nothing shown yet, so a complete answer beats a stack trace."""
        engine = _engine_with_provider("openai")
        engine.openai_client.chat.completions.create.side_effect = RuntimeError("no stream")
        with patch.object(engine, "generate_answer", return_value="запасной") as mock_gen:
            out = list(engine.generate_answer_stream([{"role": "user", "content": "q"}]))
        assert out == ["запасной"]
        mock_gen.assert_called_once()

    def test_error_after_first_delta_propagates(self):
        """Switching to a second, differently-worded answer mid-render is worse."""

        def chunks():
            yield _chunk("нач")
            raise RuntimeError("connection reset")

        engine = _engine_with_provider("openai")
        engine.openai_client.chat.completions.create.return_value = chunks()
        gen = engine.generate_answer_stream([{"role": "user", "content": "q"}])
        assert next(gen) == "нач"
        with pytest.raises(RuntimeError, match="connection reset"):
            list(gen)


# ---------------------------------------------------------------------------
# Engine: query_stream composition
# ---------------------------------------------------------------------------


class TestQueryStream:
    def _prepped_engine(self):
        from rainrag.query import RAGQueryEngine

        engine = RAGQueryEngine.__new__(RAGQueryEngine)
        engine.config = SimpleNamespace(
            llm=SimpleNamespace(provider="claude"),
            two_stage=SimpleNamespace(query_rewrite_enabled=True, hyde_enabled=False),
            reranker=SimpleNamespace(enabled=False),
        )
        prep = {
            "question": "q",
            "messages": [{"role": "user", "content": "q"}],
            "documents": [{"text": "t", "path": "/a.vtt", "score": 0.9, "rank": 1}],
            "query_variants": ["q"],
            "variant_retrieved_ids": {},
            "embed_calls": 1,
            "metadata_fallback_hits": 0,
            "two_stage_enabled": False,
            "token_usage": {},
        }
        return engine, prep

    def test_context_deltas_done_in_order_and_consistent(self):
        engine, prep = self._prepped_engine()
        with (
            patch.object(engine, "_prepare_query", return_value=prep),
            patch.object(engine, "generate_answer_stream", return_value=iter(["а", "б", "в"])),
        ):
            events = list(engine.query_stream("q"))

        kinds = [k for k, _ in events]
        assert kinds == ["context", "delta", "delta", "delta", "done"]
        context = events[0][1]
        done = events[-1][1]
        # The context event is the final result minus the answer.
        assert context["answer"] == ""
        assert context["retrieved_documents"] == done["retrieved_documents"]
        # Joining the deltas gives exactly the final answer.
        assert done["answer"] == "абв"

    def test_blocking_query_equals_collected_stream(self):
        """query() and query_stream() share prepare/assemble; prove the seam."""
        engine, prep = self._prepped_engine()
        with (
            patch.object(engine, "_prepare_query", return_value=prep),
            patch.object(engine, "generate_answer", return_value="абв"),
        ):
            blocking = engine.query("q")
        with (
            patch.object(engine, "_prepare_query", return_value=prep),
            patch.object(engine, "generate_answer_stream", return_value=iter(["аб", "в"])),
        ):
            streamed = next(p for k, p in engine.query_stream("q") if k == "done")
        assert blocking == streamed


# ---------------------------------------------------------------------------
# API: /query/stream
# ---------------------------------------------------------------------------


def _fake_stream(question: str = "q"):
    result = {
        "question": question,
        "answer": "",
        "retrieved_documents": [
            {
                "rank": 1,
                "score": 0.9,
                "text": "т",
                "path": "/test/doc1.vtt",
                "language": "ru",
                "doc_id": "d1",
            }
        ],
        "num_documents": 1,
        "metadata_fallback_hits": 0,
    }
    yield "context", dict(result)
    yield "delta", "от"
    yield "delta", "вет"
    yield "done", {**result, "answer": "ответ", "cost.llm_tokens_in": 10, "cost.llm_tokens_out": 2}


class TestQueryStreamEndpoint:
    def test_happy_path_event_sequence(self, test_client):
        import json as _json

        with patch("rainrag.api.query_engine") as mock_engine, patch("rainrag.api.config"):
            mock_engine.config.llm.provider = "mistral"
            mock_engine.query_stream.side_effect = lambda **kw: _fake_stream(kw["question"])
            resp = test_client.post("/query/stream", json={"question": "вопрос", "language": "ru"})

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(resp.text)
        assert [e for e, _ in events] == ["context", "delta", "delta", "done"]
        context = _json.loads(events[0][1])
        assert context["answer"] == ""
        # Full QueryResponse shape, media fields included (None under the
        # mocked config; the real values come from generate_media_urls).
        assert "video_url" in context["context"][0]
        assert context["context"][0]["text"] == "т"
        done = _json.loads(events[-1][1])
        assert done["answer"] == "ответ"
        deltas = [_json.loads(d)["text"] for e, d in events if e == "delta"]
        assert "".join(deltas) == "ответ"

    def test_pipeline_error_becomes_error_event(self, test_client):
        import json as _json

        def broken(**kw):
            yield "context", next(_fake_stream())[1]
            raise RuntimeError("engine exploded")

        with patch("rainrag.api.query_engine") as mock_engine, patch("rainrag.api.config"):
            mock_engine.config.llm.provider = "mistral"
            mock_engine.query_stream.side_effect = broken
            resp = test_client.post("/query/stream", json={"question": "q", "language": "en"})

        events = _sse_events(resp.text)
        assert events[-1][0] == "error"
        assert _json.loads(events[-1][1])["status"] == 500

    def test_requires_auth_when_configured(self, test_client, monkeypatch):
        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "secret")
        resp = test_client.post("/query/stream", json={"question": "q", "language": "en"})
        assert resp.status_code == 401

    def test_busy_is_429_before_any_streaming(self, test_client):
        class ExhaustedSemaphore:
            async def acquire(self):
                import asyncio

                raise asyncio.TimeoutError

            def release(self):
                pass

        with patch("rainrag.api.query_engine") as mock_engine:
            mock_engine.config.llm.provider = "mistral"
            with patch("rainrag.api._get_query_semaphore", return_value=ExhaustedSemaphore()):
                resp = test_client.post("/query/stream", json={"question": "q", "language": "en"})
        assert resp.status_code == 429
