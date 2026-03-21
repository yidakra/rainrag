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

    monkeypatch.delitem(sys.modules, "eval.datasets.create_eval_set", raising=False)
    create_eval_set_mod = importlib.import_module("eval.datasets.create_eval_set")
    # Ensure the imported module is restored/removed by monkeypatch teardown.
    monkeypatch.setitem(sys.modules, "eval.datasets.create_eval_set", create_eval_set_mod)

    # NOTE: this test intentionally targets the private helper _filter_pair
    # in create_eval_set for fine-grained behavior coverage. If internal APIs
    # are refactored, update this test to use the public contract instead.
    _filter_pair = create_eval_set_mod._filter_pair

    # create a dummy engine whose generate_answer will throw
    engine = MagicMock()
    engine.generate_answer.side_effect = RuntimeError("simulated LLM failure")

    pair = {"query": "dummy", "reference_answer": "ans"}
    chunk = {"text": "irrelevant"}

    # capture only warnings from the module logger used inside _filter_pair
    caplog.set_level(logging.WARNING, logger="eval.datasets.create_eval_set")
    result = _filter_pair(engine, pair, chunk)
    assert result is True

    # there should be exactly one warning record and it should include both
    # the generic message and the simulated error text
    records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "eval.datasets.create_eval_set"
    ]
    assert len(records) == 1
    rec = records[0]
    assert "LLM quality-filter call raised exception" in rec.getMessage()
    # ensure exception info was attached
    assert rec.exc_info is not None, "Logger call should include exception info"
    # the exception text appears in the traceback portion of the log output
    assert "simulated LLM failure" in caplog.text
