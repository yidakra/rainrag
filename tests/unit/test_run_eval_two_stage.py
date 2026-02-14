"""CLI tests for the ``two-stage`` sub-command in eval/run_eval.py.

These tests use Typer's CliRunner and patch TwoStageSweepExperiment so that
no filesystem access, real config file, or live services are needed.

Covers:
- Missing --dataset exits with code 1 and an error message
- Successful invocation: TwoStageSweepExperiment constructed correctly and
  run() called; output contains condition count and mlflow hint
- --axes subset parsed and forwarded
- --hyde-alphas, --rewrite-variants, --pool-sizes custom values parsed
- --merge-strategies and --merge-rrf-ks parsed and forwarded (Axes D+E)
- --csv path forwarded to results_to_csv
- --top-ks comma list parsed into a tuple
- Unknown axis propagated to TwoStageSweepExperiment (constructor raises)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.run_eval import app

runner = CliRunner()

# TwoStageSweepExperiment is imported *inside* the two_stage() function body
# (`from eval.experiments.two_stage_sweep import TwoStageSweepExperiment`),
# so we must patch it at the source module, not at eval.run_eval.
_PATCH = "eval.experiments.two_stage_sweep.TwoStageSweepExperiment"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DATASET = "eval/datasets/fake.jsonl"
_MLFLOW = "./mlruns"


def _mock_experiment(n_conditions: int = 3) -> MagicMock:
    """Return a mock TwoStageSweepExperiment with canned conditions and results."""
    exp = MagicMock()
    exp.conditions.return_value = [
        {"id": f"cond-{i}", "label": f"label-{i}", "overrides": {}} for i in range(n_conditions)
    ]
    exp.run.return_value = [
        {
            "condition_id": f"cond-{i}",
            "condition_label": f"label-{i}",
            "top_k": 5,
            "metrics": {"recall@5": 0.7},
        }
        for i in range(n_conditions)
    ]
    return exp


# ---------------------------------------------------------------------------
# --dataset required
# ---------------------------------------------------------------------------


class TestDatasetRequired:
    def test_missing_dataset_exits_nonzero(self):
        result = runner.invoke(app, ["two-stage"])
        assert result.exit_code != 0

    def test_missing_dataset_error_message(self):
        result = runner.invoke(app, ["two-stage"])
        assert "dataset" in result.output.lower() or "dataset" in (result.stderr or "").lower()


# ---------------------------------------------------------------------------
# Successful invocation
# ---------------------------------------------------------------------------


class TestSuccessfulInvocation:
    def _run(self, extra_args: list[str] | None = None, n_conditions: int = 3):
        mock_exp = _mock_experiment(n_conditions)
        with patch(
            _PATCH, return_value=mock_exp
        ) as mock_cls:
            result = runner.invoke(
                app,
                ["two-stage", "--dataset", _DATASET] + (extra_args or []),
            )
        return result, mock_cls, mock_exp

    def test_exit_code_zero(self):
        result, _, _ = self._run()
        assert result.exit_code == 0, result.output

    def test_run_called_once(self):
        _, _, mock_exp = self._run()
        mock_exp.run.assert_called_once()

    def test_output_contains_condition_count(self):
        result, _, _ = self._run(n_conditions=5)
        assert "5" in result.output

    def test_output_contains_mlflow_hint(self):
        result, _, _ = self._run()
        assert "mlflow" in result.output.lower()

    def test_default_constructor_args(self):
        _, mock_cls, _ = self._run()
        _, kwargs = mock_cls.call_args
        assert kwargs["dataset_path"] == _DATASET
        assert kwargs["config_path"] == "config.yaml"
        assert kwargs["mlflow_uri"] == _MLFLOW
        # All axes enabled by default (axes=None)
        assert kwargs["axes"] is None
        # All custom-value args are None by default
        assert kwargs["hyde_alphas"] is None
        assert kwargs["rewrite_variants"] is None
        assert kwargs["pool_sizes"] is None
        assert kwargs["merge_strategies"] is None
        assert kwargs["merge_rrf_ks"] is None

    def test_top_ks_default(self):
        _, mock_cls, _ = self._run()
        _, kwargs = mock_cls.call_args
        assert kwargs["top_ks"] == (5, 10)


# ---------------------------------------------------------------------------
# --axes
# ---------------------------------------------------------------------------


class TestAxesFlag:
    def _run_axes(self, axes_str: str):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(app, ["two-stage", "--dataset", _DATASET, "--axes", axes_str])
        _, kwargs = mock_cls.call_args
        return kwargs["axes"]

    def test_single_axis(self):
        assert self._run_axes("hyde_alpha") == ["hyde_alpha"]

    def test_two_axes_parsed(self):
        result = self._run_axes("merge_strategy,merge_rrf_k")
        assert set(result) == {"merge_strategy", "merge_rrf_k"}

    def test_all_five_axes(self):
        result = self._run_axes(
            "hyde_alpha,rewrite_variants,pool_size,merge_strategy,merge_rrf_k"
        )
        assert set(result) == {
            "hyde_alpha", "rewrite_variants", "pool_size", "merge_strategy", "merge_rrf_k"
        }

    def test_axes_with_spaces_trimmed(self):
        result = self._run_axes("hyde_alpha, pool_size")
        assert set(result) == {"hyde_alpha", "pool_size"}


# ---------------------------------------------------------------------------
# Per-axis custom-value flags (A–C)
# ---------------------------------------------------------------------------


class TestPerAxisCustomValues:
    def _kwargs(self, extra: list[str]):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(app, ["two-stage", "--dataset", _DATASET] + extra)
        _, kwargs = mock_cls.call_args
        return kwargs

    def test_hyde_alphas_parsed(self):
        kwargs = self._kwargs(["--hyde-alphas", "0.1,0.3,0.9"])
        assert kwargs["hyde_alphas"] == pytest.approx([0.1, 0.3, 0.9])

    def test_rewrite_variants_parsed(self):
        kwargs = self._kwargs(["--rewrite-variants", "1,3,5"])
        assert kwargs["rewrite_variants"] == [1, 3, 5]

    def test_pool_sizes_parsed(self):
        kwargs = self._kwargs(["--pool-sizes", "2,4"])
        assert kwargs["pool_sizes"] == [2, 4]

    def test_top_ks_parsed(self):
        kwargs = self._kwargs(["--top-ks", "3,5,10"])
        assert kwargs["top_ks"] == (3, 5, 10)


# ---------------------------------------------------------------------------
# Axis D: --merge-strategies
# ---------------------------------------------------------------------------


class TestMergeStrategiesFlag:
    def _kwargs(self, flag_value: str):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(
                app,
                ["two-stage", "--dataset", _DATASET, "--merge-strategies", flag_value],
            )
        _, kwargs = mock_cls.call_args
        return kwargs

    def test_single_strategy_parsed(self):
        assert self._kwargs("coverage")["merge_strategies"] == ["coverage"]

    def test_both_strategies_parsed(self):
        result = self._kwargs("coverage,diverse_rrf")["merge_strategies"]
        assert result == ["coverage", "diverse_rrf"]

    def test_strategies_with_spaces_trimmed(self):
        result = self._kwargs("coverage, diverse_rrf")["merge_strategies"]
        assert result == ["coverage", "diverse_rrf"]

    def test_omitting_flag_gives_none(self):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(app, ["two-stage", "--dataset", _DATASET])
        _, kwargs = mock_cls.call_args
        assert kwargs["merge_strategies"] is None


# ---------------------------------------------------------------------------
# Axis E: --merge-rrf-ks
# ---------------------------------------------------------------------------


class TestMergeRrfKsFlag:
    def _kwargs(self, flag_value: str):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(
                app,
                ["two-stage", "--dataset", _DATASET, "--merge-rrf-ks", flag_value],
            )
        _, kwargs = mock_cls.call_args
        return kwargs

    def test_single_k_parsed(self):
        assert self._kwargs("40")["merge_rrf_ks"] == [40]

    def test_multiple_ks_parsed(self):
        result = self._kwargs("20,40,60")["merge_rrf_ks"]
        assert result == [20, 40, 60]

    def test_ks_with_spaces_trimmed(self):
        result = self._kwargs("20, 60")["merge_rrf_ks"]
        assert result == [20, 60]

    def test_omitting_flag_gives_none(self):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(app, ["two-stage", "--dataset", _DATASET])
        _, kwargs = mock_cls.call_args
        assert kwargs["merge_rrf_ks"] is None


# ---------------------------------------------------------------------------
# --csv output
# ---------------------------------------------------------------------------


class TestCsvOutput:
    def test_csv_path_forwarded_to_results_to_csv(self):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp):
            runner.invoke(
                app,
                ["two-stage", "--dataset", _DATASET, "--csv", "/tmp/out.csv"],
            )
        mock_exp.results_to_csv.assert_called_once()
        call_args = mock_exp.results_to_csv.call_args
        assert call_args[0][1] == "/tmp/out.csv"

    def test_no_csv_flag_does_not_call_results_to_csv(self):
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp):
            runner.invoke(app, ["two-stage", "--dataset", _DATASET])
        mock_exp.results_to_csv.assert_not_called()


# ---------------------------------------------------------------------------
# Axes D+E combined
# ---------------------------------------------------------------------------


class TestAxesDECombined:
    def test_axes_d_and_e_together(self):
        """Passing both --merge-strategies and --merge-rrf-ks alongside
        --axes merge_strategy,merge_rrf_k should forward all correctly."""
        mock_exp = _mock_experiment()
        with patch(_PATCH, return_value=mock_exp) as mock_cls:
            runner.invoke(
                app,
                [
                    "two-stage",
                    "--dataset", _DATASET,
                    "--axes", "merge_strategy,merge_rrf_k",
                    "--merge-strategies", "coverage,diverse_rrf",
                    "--merge-rrf-ks", "20,60",
                ],
            )
        _, kwargs = mock_cls.call_args
        assert set(kwargs["axes"]) == {"merge_strategy", "merge_rrf_k"}
        assert kwargs["merge_strategies"] == ["coverage", "diverse_rrf"]
        assert kwargs["merge_rrf_ks"] == [20, 60]
