"""Abstract base class for RainRAG evaluation experiments.

Subclasses define the list of *conditions* (config override dicts) and call
``self.run_condition()`` for each one.  All common logic — loading the dataset,
running queries, computing metrics, logging to MLflow — lives here.
"""

from __future__ import annotations

import copy
import math
import statistics
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import eval.mlflow_tracking as mlflow_tracking
from eval.datasets.create_eval_set import load_eval_set
from eval.metrics.answer_quality import answer_length, rouge_l
from eval.metrics.carbon import track_emissions
from eval.metrics.cost import aggregate_costs, estimate_query_cost
from eval.metrics.retrieval import aggregate_metrics, compute_all_metrics, intent_coverage_at_k
from rainrag.config import Config, load_config
from rainrag.query import RAGQueryEngine


def apply_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """Return a deep copy of *config* with dot-notation overrides applied.

    Example::

        apply_overrides(cfg, {"hybrid_search.enabled": False, "reranker.enabled": True})
    """
    cfg = copy.deepcopy(config)
    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    return cfg


class BaseExperiment(ABC):
    """Base class that all eval experiments inherit from.

    Args:
        config_path: Path to the config.yaml used as the baseline.
        dataset_path: Path to an eval JSONL file produced by create_eval_set.py.
        mlflow_uri: MLflow tracking URI (local path or remote URL).
        experiment_name: MLflow experiment name.
        top_ks: Retrieval cut-offs to evaluate at.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_path: str | None = None,
        mlflow_uri: str | None = None,
        experiment_name: str = "rainrag_eval",
        top_ks: tuple[int, ...] = (5, 10),
    ) -> None:
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.mlflow_uri = mlflow_uri or mlflow_tracking.default_tracking_uri()
        self.experiment_name = experiment_name
        self.top_ks = top_ks

        self._base_config: Config | None = None
        self._dataset: list[dict] | None = None

    @property
    def base_config(self) -> Config:
        if self._base_config is None:
            self._base_config = load_config(self.config_path)
        return self._base_config

    @property
    def dataset(self) -> list[dict]:
        if self._dataset is None:
            if self.dataset_path is None:
                raise ValueError("dataset_path must be set before running experiments")
            self._dataset = load_eval_set(self.dataset_path)
        return self._dataset

    @abstractmethod
    def conditions(self) -> list[dict[str, Any]]:
        """Return a list of condition dicts, each with keys:
        - ``id`` (str): short identifier logged to MLflow
        - ``label`` (str): human-readable name
        - ``overrides`` (dict): dot-notation config overrides
        - ``tags`` (dict, optional): extra MLflow tags
        """

    def run(self) -> list[dict[str, Any]]:
        """Execute all conditions and return a list of result summaries."""
        mlflow_tracking.setup(self.mlflow_uri, self.experiment_name)
        results = []
        for condition in self.conditions():
            for top_k in self.top_ks:
                result = self.run_condition(condition, top_k)
                results.append(result)
                self._print_result(result)
        return results

    def run_condition(
        self,
        condition: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        """Run a single condition against the full dataset and log to MLflow.

        Returns a summary dict with condition metadata and aggregated metrics.
        """
        cfg = apply_overrides(self.base_config, condition.get("overrides", {}))
        engine = RAGQueryEngine(cfg)
        engine.initialize()

        # Determine language from dataset (mixed datasets run per-language)
        langs = sorted({r.get("language", "en") for r in self.dataset})

        all_results: list[dict] = []
        with track_emissions(project_name=self.experiment_name) as carbon:
            for lang in langs:
                lang_records = [r for r in self.dataset if r.get("language", "en") == lang]
                lang_results = self._run_dataset(engine, lang_records, top_k, lang)
                all_results.extend(lang_results)

        summary = self._build_summary(condition, cfg, top_k, all_results, carbon)

        run_name = f"{condition['label']}_k{top_k}"
        tags = {"condition_id": condition["id"], "top_k": str(top_k)}
        tags.update(condition.get("tags", {}))

        with mlflow_tracking.start_run(run_name=run_name, tags=tags):
            mlflow_tracking.log_params(summary["params"])
            mlflow_tracking.log_metrics(summary["metrics"])
            mlflow_tracking.log_config_snapshot(cfg)
            mlflow_tracking.log_jsonl_as_artifact(all_results, "per_query_results.jsonl")

        return summary

    def _run_dataset(
        self,
        engine: RAGQueryEngine,
        records: list[dict],
        top_k: int,
        lang: str,
    ) -> list[dict]:
        """Query the engine for every record and return per-query result dicts."""
        results: list[dict] = []
        for record in records:
            t_start = time.perf_counter()
            try:
                response = engine.query(
                    question=record["query"],
                    top_k=top_k,
                    language=lang,
                )
                elapsed_ms = (time.perf_counter() - t_start) * 1000

                retrieved_ids = [
                    d.get("doc_id", "") for d in response.get("retrieved_documents", [])
                ]
                relevant_ids = record.get("relevant_doc_ids", [])
                relevant_set = set(relevant_ids)

                # Per-variant retrieval IDs (populated when two-stage rewriting is active)
                raw_variant_ids = response.get("variant_retrieved_ids", [retrieved_ids])
                variant_retrieved_ids: list[Sequence[str]] = [tuple(ids) for ids in raw_variant_ids]

                retrieval_metrics = (
                    compute_all_metrics(retrieved_ids, relevant_ids) if relevant_ids else {}
                )

                # Intent coverage: fraction of query variants that have ≥1 relevant
                # doc in their top-k.  Only meaningful when >1 variant was used.
                if relevant_set and len(variant_retrieved_ids) > 1:
                    for cov_k in (3, 5, 10):
                        retrieval_metrics[f"intent_coverage@{cov_k}"] = intent_coverage_at_k(
                            variant_retrieved_ids, relevant_set, cov_k
                        )

                answer = response.get("answer", "")
                ref = record.get("reference_answer", "")

                contexts = [d.get("text", "") for d in response.get("retrieved_documents", [])]
                cost_metrics = estimate_query_cost(
                    query=record["query"],
                    contexts=contexts,
                    answer=answer,
                    llm_provider=engine.config.llm.provider,
                    embed_provider=engine.config.embedding.provider,
                )
                results.append(
                    {
                        "query_id": record.get("query_id", ""),
                        "query": record["query"],
                        "language": lang,
                        "category": record.get("category", "factual"),
                        "answer": answer,
                        "reference_answer": ref,
                        "retrieved_ids": retrieved_ids,
                        "relevant_ids": relevant_ids,
                        "query_variants": response.get("query_variants", [record["query"]]),
                        "variant_retrieved_ids": variant_retrieved_ids,
                        "contexts": contexts,
                        "elapsed_ms": elapsed_ms,
                        "answer_length": answer_length(answer),
                        "rouge_l": rouge_l(answer, ref) if ref else None,
                        **retrieval_metrics,
                        **cost_metrics,
                    }
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t_start) * 1000
                results.append(
                    {
                        "query_id": record.get("query_id", ""),
                        "query": record["query"],
                        "language": lang,
                        "error": str(exc),
                        "elapsed_ms": elapsed_ms,
                    }
                )
        return results

    def _build_summary(
        self,
        condition: dict,
        cfg: Config,
        top_k: int,
        all_results: list[dict],
        carbon=None,
    ) -> dict[str, Any]:
        """Aggregate per-query results into a summary with params + metrics."""
        valid = [r for r in all_results if "error" not in r]

        # Collect all per-query retrieval + intent-coverage metrics dynamically so
        # new keys (e.g. intent_coverage@k added by _run_dataset) flow through
        # without manual maintenance of a hard-coded key list.
        _retrieval_prefixes = ("recall@", "precision@", "ndcg@", "mrr", "map", "intent_coverage@")
        per_query_retrieval = [
            {k: v for k, v in r.items() if any(k.startswith(p) for p in _retrieval_prefixes)}
            for r in valid
            if any(k.startswith("recall@") for k in r)
        ]
        agg_retrieval = aggregate_metrics(per_query_retrieval) if per_query_retrieval else {}

        latencies = [r["elapsed_ms"] for r in valid if "elapsed_ms" in r]
        latency_metrics: dict[str, float] = {}
        if latencies:
            latencies_sorted = sorted(latencies)
            latency_metrics["latency_p50_ms"] = statistics.median(latencies_sorted)
            # Use a clamped index for p95 to avoid out-of-bounds access on small lists.
            idx = max(
                0, min(len(latencies_sorted) - 1, math.ceil(len(latencies_sorted) * 0.95) - 1)
            )
            latency_metrics["latency_p95_ms"] = latencies_sorted[idx]
            latency_metrics["latency_mean_ms"] = sum(latencies) / len(latencies)

        rouge_scores = [r["rouge_l"] for r in valid if r.get("rouge_l") is not None]
        rouge_mean = sum(rouge_scores) / len(rouge_scores) if rouge_scores else None

        per_query_costs = [
            {k: v for k, v in r.items() if k.startswith("cost.")}
            for r in valid
            if any(k.startswith("cost.") for k in r)
        ]
        agg_cost = aggregate_costs(per_query_costs)

        carbon_metrics = carbon.as_metrics() if carbon is not None else {}

        metrics: dict[str, float] = {
            **agg_retrieval,
            **latency_metrics,
            **agg_cost,
            **carbon_metrics,
        }
        if rouge_mean is not None:
            metrics["rouge_l"] = rouge_mean
        metrics["num_queries"] = float(len(valid))
        metrics["num_errors"] = float(len(all_results) - len(valid))

        # Guard against missing two_stage fields (config may be partial/malformed)
        # by falling back to sensible defaults.
        two_stage_merge_strategy = getattr(cfg.two_stage, "merge_strategy", "coverage")
        two_stage_merge_rrf_k = getattr(cfg.two_stage, "merge_rrf_k", 60)
        two_stage_prompt_doc_order = getattr(cfg.two_stage, "prompt_doc_order", "rank")

        params = {
            "condition_id": condition["id"],
            "condition_label": condition["label"],
            "top_k": top_k,
            "hybrid_search.enabled": cfg.hybrid_search.enabled,
            "hybrid_search.fusion_method": cfg.hybrid_search.fusion_method,
            "hybrid_search.bm25_weight": cfg.hybrid_search.bm25_weight,
            "two_stage.enabled": cfg.two_stage.enabled,
            "two_stage.query_rewrite_enabled": cfg.two_stage.query_rewrite_enabled,
            "two_stage.query_rewrite_variants": cfg.two_stage.query_rewrite_variants,
            "two_stage.hyde_enabled": cfg.two_stage.hyde_enabled,
            "two_stage.hyde_alpha": cfg.two_stage.hyde_alpha,
            "two_stage.merge_strategy": two_stage_merge_strategy,
            "two_stage.merge_rrf_k": two_stage_merge_rrf_k,
            "two_stage.prompt_doc_order": two_stage_prompt_doc_order,
            "reranker.enabled": cfg.reranker.enabled,
            "reranker.top_n": cfg.reranker.top_n,
            "reranker.min_retrieval_score": cfg.reranker.min_retrieval_score,
            "embedding.provider": cfg.embedding.provider,
            "llm.provider": cfg.llm.provider,
            "dataset_path": self.dataset_path or "",
        }

        return {
            "condition_id": condition["id"],
            "condition_label": condition["label"],
            "top_k": top_k,
            "params": params,
            "metrics": metrics,
            "per_query": all_results,
        }

    @staticmethod
    def _print_result(result: dict) -> None:
        import math as _math

        m = result["metrics"]
        cost = m.get("cost.total_usd_est_per_query", float("nan"))
        cost_str = f"${cost:.4f}/q" if not _math.isnan(cost) else "cost=n/a"

        ndcg5 = m.get("ndcg@5", float("nan"))
        ndcg5_p10 = m.get("ndcg@5_p10", float("nan"))
        ndcg5_str = (
            f"ndcg@5={ndcg5:.3f}(p10={ndcg5_p10:.3f})"
            if not _math.isnan(ndcg5_p10)
            else f"ndcg@5={ndcg5:.3f}"
        )

        intent_cov = m.get("intent_coverage@5", float("nan"))
        intent_str = f" ic@5={intent_cov:.3f}" if not _math.isnan(intent_cov) else ""

        summary = (
            f"[{result['condition_id']}] {result['condition_label']} k={result['top_k']} | "
            f"recall@5={m.get('recall@5', float('nan')):.3f} "
            f"{ndcg5_str} "
            f"mrr={m.get('mrr', float('nan')):.3f}"
            f"{intent_str} "
            f"rouge_l={m.get('rouge_l', float('nan')):.3f} "
            f"p50={m.get('latency_p50_ms', float('nan')):.0f}ms "
            f"{cost_str}"
        )
        print(summary)

    def results_to_csv(self, results: list[dict], output: str) -> None:
        """Write a CSV summary of all conditions to *output*."""
        if not results:
            return

        try:
            import pandas as pd

            rows = []
            for r in results:
                row = {
                    "condition_id": r["condition_id"],
                    "label": r["condition_label"],
                    "top_k": r["top_k"],
                }
                row.update(r["metrics"])
                rows.append(row)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"Results written to {output}")
        except ImportError:
            # Fallback: plain CSV
            all_keys = list(results[0]["metrics"].keys())
            header = ["condition_id", "label", "top_k"] + all_keys
            with open(output, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for r in results:
                    csv_row = [str(r["condition_id"]), r["condition_label"], str(r["top_k"])]
                    csv_row += [str(r["metrics"].get(k, "")) for k in all_keys]
                    f.write(",".join(csv_row) + "\n")
            print(f"Results written to {output}")
