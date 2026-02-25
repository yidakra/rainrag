from __future__ import annotations

import importlib
import logging
import sys
import types
from unittest.mock import MagicMock

import pytest


def test_filter_pair_accepts_on_error_and_logs(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the underlying LLM call raises, the pair should be kept and a warning logged."""

    # The create_eval_set module imports rainrag.config and rainrag.query,
    # which in turn can bring in optional dependencies. Scope stand-ins to
    # this test only so they don't leak into the global interpreter state.
    config_mod = types.ModuleType("rainrag.config")
    config_mod.load_config = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rainrag.config", config_mod)

    query_mod = types.ModuleType("rainrag.query")
    query_mod.RAGQueryEngine = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rainrag.query", query_mod)

    create_eval_set_mod = importlib.import_module("eval.datasets.create_eval_set")
    _filter_pair = create_eval_set_mod._filter_pair

    # create a dummy engine whose generate_answer will throw
    engine = MagicMock()

    def _raise(*args, **kwargs):
        # accept any positional/keyword args so the mock behaves like a real
        # generate_answer method.
        raise RuntimeError("simulated LLM failure")

    engine.generate_answer.side_effect = _raise

    pair = {"query": "dummy", "reference_answer": "ans"}
    chunk = {"text": "irrelevant"}

    caplog.set_level(logging.WARNING)
    result = _filter_pair(engine, pair, chunk)
    assert result is True

    # warning should mention the exception and the query string
    assert "LLM quality-filter call raised exception" in caplog.text
    assert "simulated LLM failure" in caplog.text
    assert "dummy" in caplog.text
