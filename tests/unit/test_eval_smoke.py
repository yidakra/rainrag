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
    assert axes == {"hyde_alpha", "rewrite_variants", "pool_size"}


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

    # Force re-import of carbon module with patched __import__
    import eval.metrics.carbon as carbon_mod

    with carbon_mod.track_emissions() as result:
        pass

    assert result.available is False
    assert result.as_metrics() == {}
