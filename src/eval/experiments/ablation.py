"""Feature ablation experiment.

Evaluates 8 conditions that progressively add retrieval features on top of a
plain vector search baseline:

  01  vector_only                 – plain cosine similarity, no extras
  02  hybrid_rrf                  – vector + BM25 fused via RRF
  03  hybrid_weighted             – vector + BM25 fused via weighted sum
  04  hybrid_rrf + rewrite        – adds LLM query rewriting
  05  hybrid_rrf + hyde           – adds HyDE embedding blending
  06  hybrid_rrf + rewrite + hyde – query rewriting AND HyDE together
  07  hybrid_rrf + reranker       – adds Cohere reranking
  08  full_pipeline               – all features enabled

Each condition × top_k ∈ {5, 10} × language in dataset = N MLflow runs.
Results are also written to a CSV file for offline analysis.

Usage
-----
    from eval.experiments.ablation import AblationExperiment

    exp = AblationExperiment(
        config_path="config.yaml",
        dataset_path="eval/datasets/eval_set_en.jsonl",
    )
    results = exp.run()
    exp.results_to_csv(results, "ablation_results.csv")
"""

from __future__ import annotations

from typing import Any

from eval.experiments.base import BaseExperiment


#: The 8 ablation conditions. ``overrides`` uses dot-notation keys that map
#: to nested Config fields via ``apply_overrides()``.
ABLATION_CONDITIONS: list[dict[str, Any]] = [
    {
        "id": "01",
        "label": "vector_only",
        "overrides": {
            "hybrid_search.enabled": False,
            "two_stage.enabled": False,
            "reranker.enabled": False,
        },
    },
    {
        "id": "02",
        "label": "hybrid_rrf",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "rrf",
            "two_stage.enabled": False,
            "reranker.enabled": False,
        },
    },
    {
        "id": "03",
        "label": "hybrid_weighted",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "weighted",
            "two_stage.enabled": False,
            "reranker.enabled": False,
        },
    },
    {
        "id": "04",
        "label": "hybrid_rrf+rewrite",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "rrf",
            "two_stage.enabled": True,
            "two_stage.query_rewrite_enabled": True,
            "two_stage.hyde_enabled": False,
            "reranker.enabled": False,
        },
    },
    {
        "id": "05",
        "label": "hybrid_rrf+hyde",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "rrf",
            "two_stage.enabled": True,
            "two_stage.query_rewrite_enabled": False,
            "two_stage.hyde_enabled": True,
            "reranker.enabled": False,
        },
    },
    {
        "id": "06",
        "label": "hybrid_rrf+rewrite+hyde",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "rrf",
            "two_stage.enabled": True,
            "two_stage.query_rewrite_enabled": True,
            "two_stage.hyde_enabled": True,
            "reranker.enabled": False,
        },
    },
    {
        "id": "07",
        "label": "hybrid_rrf+reranker",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "rrf",
            "two_stage.enabled": False,
            "reranker.enabled": True,
        },
    },
    {
        "id": "08",
        "label": "full_pipeline",
        "overrides": {
            "hybrid_search.enabled": True,
            "hybrid_search.fusion_method": "rrf",
            "two_stage.enabled": True,
            "two_stage.query_rewrite_enabled": True,
            "two_stage.hyde_enabled": True,
            "reranker.enabled": True,
        },
    },
]


class AblationExperiment(BaseExperiment):
    """Run all 8 ablation conditions against the eval dataset.

    Args:
        condition_ids: Optional subset of condition IDs to run (e.g. ["01", "08"]).
            Runs all 8 when None.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_path: str | None = None,
        mlflow_uri: str | None = None,
        top_ks: tuple[int, ...] = (5, 10),
        condition_ids: list[str] | None = None,
    ) -> None:
        super().__init__(
            config_path=config_path,
            dataset_path=dataset_path,
            mlflow_uri=mlflow_uri,
            experiment_name="ablation",
            top_ks=top_ks,
        )
        self._condition_ids = condition_ids

    def conditions(self) -> list[dict[str, Any]]:
        if self._condition_ids is None:
            return ABLATION_CONDITIONS
        valid_ids = {c["id"] for c in ABLATION_CONDITIONS}
        missing = set(self._condition_ids) - valid_ids
        if missing:
            missing_ids = ", ".join(sorted(missing))
            raise ValueError(f"Unknown ablation condition id(s): {missing_ids}")
        return [c for c in ABLATION_CONDITIONS if c["id"] in self._condition_ids]
