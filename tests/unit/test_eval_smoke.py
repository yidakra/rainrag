"""Smoke tests: verify every eval module can be imported without errors.

These tests catch broken imports, missing ``__init__`` re-exports, and
circular-dependency regressions.  They require *no* external services, API
keys, or optional dependencies — any ``ImportError`` for an optional package
is caught and the test is skipped rather than failed.

The tests are intentionally lightweight (import-only) so they run in < 1 s
on any Python version supported by the project.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Make sure the repo root is on sys.path when running from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Modules that must always be importable (no optional deps)
# ---------------------------------------------------------------------------

CORE_MODULES = [
    "eval",
    "eval.mlflow_tracking",
    "eval.metrics.retrieval",
    "eval.metrics.answer_quality",
    "eval.metrics.cost",
    "eval.metrics.carbon",
    "eval.datasets.beir_adapter",
    "eval.experiments.base",
    "eval.experiments.ablation",
    "eval.experiments.two_stage_sweep",
]


@pytest.mark.parametrize("module_path", CORE_MODULES)
def test_import_core_module(module_path: str) -> None:
    """Each core eval module must import without raising."""
    importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Modules with optional heavy dependencies (mlflow, pandas, matplotlib …)
# ---------------------------------------------------------------------------

OPTIONAL_MODULES = [
    "eval.plot_results",
    "eval.run_eval",
    "eval.experiments.provider_comparison",
    "eval.experiments.latency",
    "eval.datasets.create_eval_set",
    "eval.datasets.review_eval_set",
]


@pytest.mark.parametrize("module_path", OPTIONAL_MODULES)
def test_import_optional_module(module_path: str) -> None:
    """Optional modules must either import cleanly or raise only ImportError."""
    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        pytest.skip(f"Optional dependency missing for {module_path}: {exc}")


# ---------------------------------------------------------------------------
# Public API surface checks
# ---------------------------------------------------------------------------


def test_ablation_conditions_list() -> None:
    """ABLATION_CONDITIONS must be a non-empty list with required keys."""
    from eval.experiments.ablation import ABLATION_CONDITIONS

    assert isinstance(ABLATION_CONDITIONS, list)
    assert len(ABLATION_CONDITIONS) >= 8
    for cond in ABLATION_CONDITIONS:
        assert "id" in cond
        assert "label" in cond
        assert "overrides" in cond


def test_two_stage_sweep_default_conditions() -> None:
    """TwoStageSweepExperiment.conditions() must return at least one condition per axis."""
    from eval.experiments.two_stage_sweep import TwoStageSweepExperiment

    exp = TwoStageSweepExperiment(dataset_path=None)
    conds = exp.conditions()
    assert len(conds) > 0

    axes = {c["tags"]["sweep_axis"] for c in conds}
    assert axes == {
        "hyde_alpha",
        "rewrite_variants",
        "pool_size",
        "merge_strategy",
        "merge_rrf_k",
        "doc_order",
    }


def test_two_stage_sweep_axis_filter() -> None:
    """Passing axes=['hyde_alpha'] must exclude other axes from conditions()."""
    from eval.experiments.two_stage_sweep import TwoStageSweepExperiment

    exp = TwoStageSweepExperiment(dataset_path=None, axes=["hyde_alpha"])
    conds = exp.conditions()
    assert all(c["tags"]["sweep_axis"] == "hyde_alpha" for c in conds)


def test_two_stage_sweep_invalid_axis_raises() -> None:
    """An unknown axis name must raise ValueError immediately."""
    from eval.experiments.two_stage_sweep import TwoStageSweepExperiment

    with pytest.raises(ValueError, match="Unknown sweep axes"):
        TwoStageSweepExperiment(dataset_path=None, axes=["nonexistent_axis"])


def test_carbon_result_as_metrics_empty_when_unavailable() -> None:
    """CarbonResult.as_metrics() must return {} when available=False."""
    from eval.metrics.carbon import CarbonResult

    result = CarbonResult(available=False, emissions_kg=9999.0)
    assert result.as_metrics() == {}


def test_carbon_result_as_metrics_full() -> None:
    """CarbonResult.as_metrics() must include all non-None fields."""
    from eval.metrics.carbon import CarbonResult

    result = CarbonResult(
        available=True,
        emissions_kg=0.001,
        energy_kwh=0.005,
        cpu_power_w=45.0,
        duration_s=30.0,
    )
    m = result.as_metrics()
    assert m["carbon.emissions_kg"] == pytest.approx(0.001)
    assert m["carbon.emissions_g"] == pytest.approx(1.0)
    assert m["carbon.energy_kwh"] == pytest.approx(0.005)
    assert m["carbon.cpu_power_w"] == pytest.approx(45.0)
    assert m["carbon.duration_s"] == pytest.approx(30.0)


def test_carbon_track_emissions_noop_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    """track_emissions must yield an unavailable CarbonResult when codecarbon is missing."""
    import builtins

    real_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "codecarbon":
            raise ImportError("codecarbon not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    orig = sys.modules.get("eval.metrics.carbon")
    try:
        # Force re-import of carbon module with patched __import__
        # clear module cache so import uses patched __import__
        sys.modules.pop("eval.metrics.carbon", None)
        carbon_mod = __import__("eval.metrics.carbon", fromlist=["*"])

        with carbon_mod.track_emissions() as result:
            pass

        assert result.available is False
        assert result.as_metrics() == {}
    finally:
        if orig is None:
            sys.modules.pop("eval.metrics.carbon", None)
        else:
            sys.modules["eval.metrics.carbon"] = orig


def _make_hit(score, doc_id, is_speech_free):
    """Helper for tests: mimic a Qdrant hit object with score and payload.

    Mirrors the inline definitions previously duplicated in two tests.
    """
    hit = MagicMock()
    hit.score = score
    hit.payload = {
        "doc_id": doc_id,
        "text": "some text",
        "path": "",
        "language": "en",
        "is_speech_free": is_speech_free,
    }
    return hit


# ---------------------------------------------------------------------------
# _scroll_chunks: is_speech_free filter (predicate logic, no heavy deps)
# ---------------------------------------------------------------------------
# These tests validate the exact filter predicate used in _scroll_chunks without
# importing the full module (which transitively requires torch and other heavy
# dependencies).  The module-level import test is covered by OPTIONAL_MODULES.
# ---------------------------------------------------------------------------


# NOTE: this helper duplicates the predicate used by
# eval.datasets.create_eval_set._scroll_chunks (lines ~100-106).
# If that function's filter changes, update this helper accordingly.
# The production predicate lives in the create_eval_set module.


def _apply_scroll_filter(payloads: list[dict], lang: str) -> list[dict]:
    """Mirror of the filter condition inside _scroll_chunks."""
    return [
        p
        for p in payloads
        if p.get("language", "en") == lang
        and p.get("text", "")
        and not p.get("is_speech_free", False)
    ]


def test_scroll_chunks_filter_excludes_speech_free() -> None:
    """is_speech_free=True payloads must be excluded by the _scroll_chunks predicate."""
    payloads = [
        {
            "language": "en",
            "text": "The president arrived.",
            "doc_id": "a",
            "is_speech_free": False,
        },
        {"language": "en", "text": "Crowd ambience.", "doc_id": "b", "is_speech_free": True},
    ]
    result = _apply_scroll_filter(payloads, "en")
    assert len(result) == 1
    assert result[0]["doc_id"] == "a"


def test_scroll_chunks_filter_includes_legacy_docs_without_flag() -> None:
    """Payloads without is_speech_free (old indexed data) must be included."""
    payloads = [
        {"language": "en", "text": "Archive footage from 2019.", "doc_id": "legacy"},
    ]
    result = _apply_scroll_filter(payloads, "en")
    assert len(result) == 1
    assert result[0]["doc_id"] == "legacy"


def test_scroll_chunks_filter_excludes_empty_text() -> None:
    """Payloads with empty text must still be excluded even if not speech-free."""
    payloads = [
        {"language": "en", "text": "", "doc_id": "no_text", "is_speech_free": False},
        {"language": "en", "text": "Has text.", "doc_id": "has_text", "is_speech_free": False},
    ]
    result = _apply_scroll_filter(payloads, "en")
    assert len(result) == 1
    assert result[0]["doc_id"] == "has_text"


def test_scroll_chunks_filter_respects_language() -> None:
    """Payloads for a different language must not be included."""
    payloads = [
        {"language": "en", "text": "English content.", "doc_id": "en_doc"},
        {"language": "ru", "text": "Русский контент.", "doc_id": "ru_doc"},
    ]
    result = _apply_scroll_filter(payloads, "en")
    assert len(result) == 1
    assert result[0]["doc_id"] == "en_doc"


def test_apply_scroll_filter_matches_create_eval_set_scroll_chunks() -> None:
    """Ensure the local predicate matches the production _scroll_chunks behavior."""
    try:
        from eval.datasets.create_eval_set import _scroll_chunks
    except ImportError as exc:
        pytest.skip(f"Skipping due to missing create_eval_set dependency: {exc}")

    class FakePoint:
        def __init__(self, payload, _id=None):
            self.payload = payload
            self.id = _id

    class FakeClient:
        def __init__(self, points):
            self._points = points

        def scroll(self, collection_name, scroll_filter, limit, offset, with_payload, with_vectors):
            return self._points, None

    class FakeEngine:
        pass

    payloads = [
        {"language": "en", "text": "Has text", "doc_id": "1", "is_speech_free": False},
        {"language": "en", "text": "", "doc_id": "2", "is_speech_free": False},
        {"language": "en", "text": "Speech-free", "doc_id": "3", "is_speech_free": True},
        {"language": "ru", "text": "Русский", "doc_id": "4", "is_speech_free": False},
        {"text": "No language field", "doc_id": "5", "is_speech_free": False},
    ]

    points = [FakePoint(p, _id=i) for i, p in enumerate(payloads, start=10)]
    engine = FakeEngine()
    engine.qdrant_client = FakeClient(points)
    engine.config = type(
        "C", (), {"qdrant": type("Q", (), {"collection_name": "test_collection"})()}
    )()

    expected = _apply_scroll_filter(payloads, "en")
    result = _scroll_chunks(engine, "en", limit=100)

    assert result == [{**p} for p in expected]


# ---------------------------------------------------------------------------
# retrieve_documents: exclude_speech_free param (requires full env with torch)
# ---------------------------------------------------------------------------


def test_retrieve_documents_exclude_speech_free_filters_results(test_config) -> None:
    """retrieve_documents(exclude_speech_free=True) must strip is_speech_free docs."""
    pytest.importorskip("qdrant_client")  # skip if not installed
    torch = pytest.importorskip("torch")  # noqa: F841 — skip if torch not installed

    from rainrag.query import RAGQueryEngine

    cfg = test_config
    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.config = cfg
    engine.bm25 = None  # no hybrid search

    fake_points = [_make_hit(0.9, "doc_a", False), _make_hit(0.8, "doc_b", True)]
    mock_result = MagicMock()
    mock_result.points = fake_points

    engine.qdrant_client = MagicMock()
    engine.qdrant_client.query_points.return_value = mock_result

    docs = engine.retrieve_documents(
        query_vector=[0.0] * 768,
        top_k=5,
        exclude_speech_free=True,
    )

    assert len(docs) == 1
    assert docs[0]["doc_id"] == "doc_a"


def test_retrieve_documents_include_speech_free_by_default(test_config) -> None:
    """retrieve_documents without exclude_speech_free must return all docs."""
    pytest.importorskip("qdrant_client")
    pytest.importorskip("torch")

    from rainrag.query import RAGQueryEngine

    cfg = test_config
    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.config = cfg
    engine.bm25 = None

    fake_points = [_make_hit(0.9, "doc_a", False), _make_hit(0.8, "doc_b", True)]
    mock_result = MagicMock()
    mock_result.points = fake_points

    engine.qdrant_client = MagicMock()
    engine.qdrant_client.query_points.return_value = mock_result

    docs = engine.retrieve_documents(
        query_vector=[0.0] * 768,
        top_k=5,
    )

    assert len(docs) == 2
