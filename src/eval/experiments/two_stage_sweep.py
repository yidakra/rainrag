"""Two-stage retrieval parameter sweep experiment.

Sweeps six axes independently over a fixed hybrid-RRF base config,
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

  Axis D – Merge strategy  : algorithm for combining per-variant result sets
                             ["coverage", "diverse_rrf"]
                             Tests VRisker greedy coverage vs diversity-weighted RRF.
                             (2 variants, pool_size=3)

  Axis E – Merge RRF k     : RRF constant k for the diverse_rrf strategy
                             [20, 40, 60]
                             (merge_strategy="diverse_rrf", 2 variants)

  Axis F – Doc order        : order of documents in the LLM prompt
                             ["rank", "reversed", "book_end"]
                             Tests 'lost in the middle' positional effects
                             (Cuconasu et al., SIGIR 2024; Liu et al., 2023).

Each axis is swept with the others held at sensible defaults.  The total
number of conditions is the sum of each axis length rather than the full
cross-product, keeping wall-clock time manageable.

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
MERGE_STRATEGIES: list[str] = ["coverage", "diverse_rrf"]
MERGE_RRF_KS: list[int] = [20, 40, 60]
DOC_ORDERS: list[str] = ["rank", "reversed", "book_end"]

# Defaults held constant while a different axis is swept
_DEFAULT_HYDE_ALPHA: float = 0.5
_DEFAULT_REWRITE_VARIANTS: int = 2
_DEFAULT_POOL_SIZE: int = 3
_DEFAULT_MERGE_STRATEGY: str = "coverage"
_DEFAULT_MERGE_RRF_K: int = 60
_DEFAULT_DOC_ORDER: str = "rank"

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
            "id": f"hyde-{alpha}".replace(".", "p"),
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


def _merge_strategy_conditions(strategies: list[str]) -> list[dict[str, Any]]:
    """Conditions sweeping the multi-variant merge strategy (Axis D).

    Compares greedy coverage-maximising merge (VRisker-style) with
    diversity-weighted multi-source RRF.  RRF k is held at the default for
    this axis; use Axis E to tune it independently.
    """
    return [
        {
            "id": f"merge-{strategy.replace('_', '')}",
            "label": f"merge={strategy}",
            "overrides": {
                **_BASE_OVERRIDES,
                "two_stage.hyde_alpha": _DEFAULT_HYDE_ALPHA,
                "two_stage.query_rewrite_variants": _DEFAULT_REWRITE_VARIANTS,
                "hybrid_search.top_k_multiplier": _DEFAULT_POOL_SIZE,
                "two_stage.merge_strategy": strategy,
                "two_stage.merge_rrf_k": _DEFAULT_MERGE_RRF_K,
            },
            "tags": {
                "sweep_axis": "merge_strategy",
                "merge_strategy": strategy,
            },
        }
        for strategy in strategies
    ]


def _merge_rrf_k_conditions(rrf_ks: list[int]) -> list[dict[str, Any]]:
    """Conditions sweeping the RRF constant k for diverse_rrf strategy (Axis E).

    Smaller k (e.g. 20) amplifies rank differences more aggressively; larger k
    (e.g. 60) is the standard literature default.  diverse_rrf is fixed so
    that the k effect is isolated.
    """
    return [
        {
            "id": f"rrf-k{k}",
            "label": f"rrf_k={k}",
            "overrides": {
                **_BASE_OVERRIDES,
                "two_stage.hyde_alpha": _DEFAULT_HYDE_ALPHA,
                "two_stage.query_rewrite_variants": _DEFAULT_REWRITE_VARIANTS,
                "hybrid_search.top_k_multiplier": _DEFAULT_POOL_SIZE,
                "two_stage.merge_strategy": "diverse_rrf",
                "two_stage.merge_rrf_k": k,
            },
            "tags": {
                "sweep_axis": "merge_rrf_k",
                "merge_rrf_k": str(k),
            },
        }
        for k in rrf_ks
    ]


def _doc_order_conditions(orders: list[str]) -> list[dict[str, Any]]:
    """Conditions sweeping the prompt document ordering strategy (Axis F).

    Tests the positional sensitivity of the LLM to document order, motivated
    by the 'lost in the middle' effect (Liu et al., 2023) and the finding by
    Cuconasu et al. (SIGIR 2024) that document position in the prompt
    significantly affects answer quality.  All other axes are held at defaults.
    """
    return [
        {
            "id": f"order-{order}",
            "label": f"doc_order={order}",
            "overrides": {
                **_BASE_OVERRIDES,
                "two_stage.hyde_alpha": _DEFAULT_HYDE_ALPHA,
                "two_stage.query_rewrite_variants": _DEFAULT_REWRITE_VARIANTS,
                "hybrid_search.top_k_multiplier": _DEFAULT_POOL_SIZE,
                "two_stage.merge_strategy": _DEFAULT_MERGE_STRATEGY,
                "two_stage.merge_rrf_k": _DEFAULT_MERGE_RRF_K,
                "two_stage.prompt_doc_order": order,
            },
            "tags": {
                "sweep_axis": "doc_order",
                "prompt_doc_order": order,
            },
        }
        for order in orders
    ]


class TwoStageSweepExperiment(BaseExperiment):
    """Sensitivity sweep of two-stage retrieval hyper-parameters.

    Args:
        axes: Which sweep axes to include.  Subset of
            ``{"hyde_alpha", "rewrite_variants", "pool_size",
               "merge_strategy", "merge_rrf_k", "doc_order"}``.
            Defaults to all six.
        hyde_alphas: HyDE alpha values to sweep (Axis A).
        rewrite_variants: Query-rewrite variant counts to sweep (Axis B).
        pool_sizes: top_k_multiplier values to sweep (Axis C).
        merge_strategies: Merge-strategy values to sweep (Axis D).
        merge_rrf_ks: RRF k values to sweep for diverse_rrf strategy (Axis E).
        doc_orders: Prompt document ordering strategies to sweep (Axis F).
    """

    _ALL_AXES = frozenset(
        {
            "hyde_alpha",
            "rewrite_variants",
            "pool_size",
            "merge_strategy",
            "merge_rrf_k",
            "doc_order",
        }
    )

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_path: str | None = None,
        mlflow_uri: str | None = None,
        top_ks: tuple[int, ...] = (5, 10),
        axes: list[str] | None = None,
        hyde_alphas: list[float] | None = None,
        rewrite_variants: list[int] | None = None,
        pool_sizes: list[int] | None = None,
        merge_strategies: list[str] | None = None,
        merge_rrf_ks: list[int] | None = None,
        doc_orders: list[str] | None = None,
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

        if not self._axes:
            raise ValueError(
                "At least one axis must be selected for sweep; your axes list is empty. See conditions() usage."
            )

        self._hyde_alphas = hyde_alphas if hyde_alphas is not None else HYDE_ALPHAS
        self._rewrite_variants = (
            rewrite_variants if rewrite_variants is not None else REWRITE_VARIANTS
        )
        self._pool_sizes = pool_sizes if pool_sizes is not None else POOL_SIZES
        self._merge_strategies = (
            merge_strategies if merge_strategies is not None else MERGE_STRATEGIES
        )
        self._merge_rrf_ks = merge_rrf_ks if merge_rrf_ks is not None else MERGE_RRF_KS
        self._doc_orders = doc_orders if doc_orders is not None else DOC_ORDERS

        # Require non-empty axis lists so conditions() has at least one value.
        if not self._hyde_alphas:
            raise ValueError("hyde_alphas must be a non-empty list")
        if not self._rewrite_variants:
            raise ValueError("rewrite_variants must be a non-empty list")
        if not self._pool_sizes:
            raise ValueError("pool_sizes must be a non-empty list")
        if not self._merge_strategies:
            raise ValueError("merge_strategies must be a non-empty list")
        if not self._merge_rrf_ks:
            raise ValueError("merge_rrf_ks must be a non-empty list")
        if not self._doc_orders:
            raise ValueError("doc_orders must be a non-empty list")

        # Validate provided sweep values
        for alpha in self._hyde_alphas:
            if isinstance(alpha, bool):
                raise ValueError(
                    f"hyde_alphas must be numeric (int/float); got {alpha!r} ({type(alpha).__name__})"
                )
            if not 0 <= float(alpha) <= 1:
                raise ValueError(f"hyde_alphas must be between 0 and 1 (inclusive); got {alpha!r}")

        for n in self._rewrite_variants:
            if isinstance(n, bool):
                raise ValueError(
                    f"rewrite_variants must be integers > 0; got {n!r} ({type(n).__name__})"
                )
            if n <= 0:
                raise ValueError(f"rewrite_variants must be > 0; got {n}")

        for m in self._pool_sizes:
            if isinstance(m, bool):
                raise ValueError(f"pool_sizes must be integers > 0; got {m!r} ({type(m).__name__})")
            if m <= 0:
                raise ValueError(f"pool_sizes must be > 0; got {m}")

        for strategy in self._merge_strategies:
            if strategy not in MERGE_STRATEGIES:
                raise ValueError(
                    f"merge_strategies must be one of {MERGE_STRATEGIES}; got {strategy!r}"
                )

        for k in self._merge_rrf_ks:
            if isinstance(k, bool):
                raise ValueError(
                    f"merge_rrf_ks must be integers > 0; got {k!r} ({type(k).__name__})"
                )
            if k <= 0:
                raise ValueError(f"merge_rrf_ks must be > 0; got {k}")

        for order in self._doc_orders:
            if order not in DOC_ORDERS:
                raise ValueError(f"doc_orders must be one of {DOC_ORDERS}; got {order!r}")

    def conditions(self) -> list[dict[str, Any]]:
        """Build and return all sweep conditions for the selected axes."""
        conds: list[dict[str, Any]] = []
        if "hyde_alpha" in self._axes:
            conds.extend(_hyde_alpha_conditions(self._hyde_alphas))
        if "rewrite_variants" in self._axes:
            conds.extend(_rewrite_variant_conditions(self._rewrite_variants))
        if "pool_size" in self._axes:
            conds.extend(_pool_size_conditions(self._pool_sizes))
        if "merge_strategy" in self._axes:
            conds.extend(_merge_strategy_conditions(self._merge_strategies))
        if "merge_rrf_k" in self._axes:
            conds.extend(_merge_rrf_k_conditions(self._merge_rrf_ks))
        if "doc_order" in self._axes:
            conds.extend(_doc_order_conditions(self._doc_orders))
        return conds
