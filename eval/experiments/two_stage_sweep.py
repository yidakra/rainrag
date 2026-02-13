"""Two-stage retrieval parameter sweep experiment.

Sweeps three axes independently over a fixed hybrid-RRF base config,
allowing sensitivity analysis of the two-stage retrieval components:

  Axis A – HyDE alpha      : blend weight for HyDE vs raw query vector
                             [0.1, 0.3, 0.5, 0.7, 0.9]
                             (query_rewrite held at True, 2 variants)

  Axis B – Rewrite variants: number of LLM rewrite alternatives generated
                             [1, 2, 3, 5]
                             (hyde_alpha held at 0.5)

  Axis C – Pool size        : top_k_multiplier before reranking
                             [2, 3, 5]
                             (full two-stage, best defaults from A+B)

Each axis is swept with the others held at sensible defaults.  The total
number of conditions is len(HYDE_ALPHAS) + len(REWRITE_VARIANTS) + len(POOL_SIZES)
rather than the full cross-product, keeping wall-clock time manageable.

Usage
-----
    from eval.experiments.two_stage_sweep import TwoStageSweepExperiment

    exp = TwoStageSweepExperiment(
        config_path="config.yaml",
        dataset_path="eval/datasets/eval_set_en.jsonl",
    )
    results = exp.run()
    exp.results_to_csv(results, "two_stage_sweep.csv")

To run only specific axes::

    exp = TwoStageSweepExperiment(..., axes=["hyde_alpha", "rewrite_variants"])

To override the default sweep values::

    exp = TwoStageSweepExperiment(..., hyde_alphas=[0.2, 0.5, 0.8])
"""
from __future__ import annotations

from typing import Any

from eval.experiments.base import BaseExperiment


# ---------------------------------------------------------------------------
# Default sweep values
# ---------------------------------------------------------------------------

HYDE_ALPHAS: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
REWRITE_VARIANTS: list[int] = [1, 2, 3, 5]
POOL_SIZES: list[int] = [2, 3, 5]

# Defaults held constant while a different axis is swept
_DEFAULT_HYDE_ALPHA: float = 0.5
_DEFAULT_REWRITE_VARIANTS: int = 2
_DEFAULT_POOL_SIZE: int = 3

# Base config shared by all conditions: hybrid-RRF, full two-stage, no reranker
_BASE_OVERRIDES: dict[str, Any] = {
    "hybrid_search.enabled": True,
    "hybrid_search.fusion_method": "rrf",
    "two_stage.enabled": True,
    "two_stage.query_rewrite_enabled": True,
    "two_stage.hyde_enabled": True,
    "reranker.enabled": False,
}


def _hyde_alpha_conditions(alphas: list[float]) -> list[dict[str, Any]]:
    """Conditions that sweep HyDE alpha with rewrite_variants and pool_size fixed."""
    return [
        {
            "id": f"hyde-{alpha:.2f}".replace(".", ""),
            "label": f"hyde_alpha={alpha}",
            "overrides": {
                **_BASE_OVERRIDES,
                "two_stage.hyde_alpha": alpha,
                "two_stage.query_rewrite_variants": _DEFAULT_REWRITE_VARIANTS,
                "hybrid_search.top_k_multiplier": _DEFAULT_POOL_SIZE,
            },
            "tags": {
                "sweep_axis": "hyde_alpha",
                "hyde_alpha": str(alpha),
            },
        }
        for alpha in alphas
    ]


def _rewrite_variant_conditions(variants: list[int]) -> list[dict[str, Any]]:
    """Conditions that sweep query_rewrite_variants with alpha and pool_size fixed."""
    return [
        {
            "id": f"rw-{n}",
            "label": f"rewrite_variants={n}",
            "overrides": {
                **_BASE_OVERRIDES,
                "two_stage.hyde_alpha": _DEFAULT_HYDE_ALPHA,
                "two_stage.query_rewrite_variants": n,
                "hybrid_search.top_k_multiplier": _DEFAULT_POOL_SIZE,
            },
            "tags": {
                "sweep_axis": "rewrite_variants",
                "rewrite_variants": str(n),
            },
        }
        for n in variants
    ]


def _pool_size_conditions(pool_sizes: list[int]) -> list[dict[str, Any]]:
    """Conditions that sweep top_k_multiplier with alpha and variants fixed."""
    return [
        {
            "id": f"pool-{m}x",
            "label": f"pool_size={m}x",
            "overrides": {
                **_BASE_OVERRIDES,
                "two_stage.hyde_alpha": _DEFAULT_HYDE_ALPHA,
                "two_stage.query_rewrite_variants": _DEFAULT_REWRITE_VARIANTS,
                "hybrid_search.top_k_multiplier": m,
            },
            "tags": {
                "sweep_axis": "pool_size",
                "top_k_multiplier": str(m),
            },
        }
        for m in pool_sizes
    ]


class TwoStageSweepExperiment(BaseExperiment):
    """Sensitivity sweep of two-stage retrieval hyper-parameters.

    Args:
        axes: Which sweep axes to include.  Subset of
            ``{"hyde_alpha", "rewrite_variants", "pool_size"}``.
            Defaults to all three.
        hyde_alphas: HyDE alpha values to sweep (Axis A).
        rewrite_variants: Query-rewrite variant counts to sweep (Axis B).
        pool_sizes: top_k_multiplier values to sweep (Axis C).
    """

    _ALL_AXES = frozenset({"hyde_alpha", "rewrite_variants", "pool_size"})

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_path: str | None = None,
        mlflow_uri: str = "./mlruns",
        top_ks: tuple[int, ...] = (5, 10),
        axes: list[str] | None = None,
        hyde_alphas: list[float] | None = None,
        rewrite_variants: list[int] | None = None,
        pool_sizes: list[int] | None = None,
    ) -> None:
        super().__init__(
            config_path=config_path,
            dataset_path=dataset_path,
            mlflow_uri=mlflow_uri,
            experiment_name="two_stage_sweep",
            top_ks=top_ks,
        )
        self._axes = set(axes) if axes is not None else self._ALL_AXES
        unknown = self._axes - self._ALL_AXES
        if unknown:
            raise ValueError(f"Unknown sweep axes: {unknown!r}. Valid: {self._ALL_AXES}")

        self._hyde_alphas = hyde_alphas if hyde_alphas is not None else HYDE_ALPHAS
        self._rewrite_variants = rewrite_variants if rewrite_variants is not None else REWRITE_VARIANTS
        self._pool_sizes = pool_sizes if pool_sizes is not None else POOL_SIZES

    def conditions(self) -> list[dict[str, Any]]:
        """Build and return all sweep conditions for the selected axes."""
        conds: list[dict[str, Any]] = []
        if "hyde_alpha" in self._axes:
            conds.extend(_hyde_alpha_conditions(self._hyde_alphas))
        if "rewrite_variants" in self._axes:
            conds.extend(_rewrite_variant_conditions(self._rewrite_variants))
        if "pool_size" in self._axes:
            conds.extend(_pool_size_conditions(self._pool_sizes))
        return conds
