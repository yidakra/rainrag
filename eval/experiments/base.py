"""Abstract base class for RainRAG evaluation experiments.

Subclasses define the list of *conditions* (config override dicts) and call
``self.run_condition()`` for each one.  All common logic — loading the dataset,
running queries, computing metrics, logging to MLflow — lives here.
"""
from __future__ import annotations

import copy
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from rainrag.config import Config, load_config
from rainrag.query import RAGQueryEngine

from eval.datasets.create_eval_set import load_eval_set
from eval.metrics.retrieval import aggregate_metrics, compute_all_metrics
from eval.metrics.answer_quality import answer_length, rouge_l
import eval.mlflow_tracking as mlflow_tracking


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
        mlflow_uri: str = "./mlruns",
        experiment_name: str = "rainrag_eval",
        top_ks: tuple[int, ...] = (5, 10),
    ) -> None:
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.mlflow_uri = mlflow_uri
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
        for lang in langs:
            lang_records = [r for r in self.dataset if r.get("language", "en") == lang]
            lang_results = self._run_dataset(engine, lang_records, top_k, lang)
            all_results.extend(lang_results)

        summary = self._build_summary(condition, cfg, top_k, all_results)

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

                retrieval_metrics = (
                    compute_all_metrics(retrieved_ids, relevant_ids)
                    if relevant_ids
                    else {}
                )

                answer = response.get("answer", "")
                ref = record.get("reference_answer", "")

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
                        "contexts": [
                            d.get("text", "") for d in response.get("retrieved_documents", [])
                        ],
                        "elapsed_ms": elapsed_ms,
                        "answer_length": answer_length(answer),
                        "rouge_l": rouge_l(answer, ref) if ref else None,
                        **retrieval_metrics,
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
    ) -> dict[str, Any]:
        """Aggregate per-query results into a summary with params + metrics."""
        valid = [r for r in all_results if "error" not in r]
        retrieval_keys = ["recall@3", "recall@5", "recall@10", "precision@3",
                          "precision@5", "ndcg@5", "ndcg@10", "mrr", "map"]

        per_query_retrieval = [
            {k: r[k] for k in retrieval_keys if k in r} for r in valid if "recall@3" in r
        ]
        agg_retrieval = aggregate_metrics(per_query_retrieval) if per_query_retrieval else {}

        latencies = [r["elapsed_ms"] for r in valid if "elapsed_ms" in r]
        latency_metrics: dict[str, float] = {}
        if latencies:
            latencies_sorted = sorted(latencies)
            latency_metrics["latency_p50_ms"] = latencies_sorted[len(latencies_sorted) // 2]
            latency_metrics["latency_p95_ms"] = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            latency_metrics["latency_mean_ms"] = sum(latencies) / len(latencies)

        rouge_scores = [r["rouge_l"] for r in valid if r.get("rouge_l") is not None]
        rouge_mean = sum(rouge_scores) / len(rouge_scores) if rouge_scores else None

        metrics: dict[str, float] = {**agg_retrieval, **latency_metrics}
        if rouge_mean is not None:
            metrics["rouge_l"] = rouge_mean
        metrics["num_queries"] = float(len(valid))
        metrics["num_errors"] = float(len(all_results) - len(valid))

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
            "reranker.enabled": cfg.reranker.enabled,
            "reranker.top_n": cfg.reranker.top_n,
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
        m = result["metrics"]
        print(
            f"[{result['condition_id']}] {result['condition_label']} k={result['top_k']} | "
            f"recall@5={m.get('recall@5', float('nan')):.3f} "
            f"ndcg@5={m.get('ndcg@5', float('nan')):.3f} "
            f"mrr={m.get('mrr', float('nan')):.3f} "
            f"rouge_l={m.get('rouge_l', float('nan')):.3f} "
            f"p50={m.get('latency_p50_ms', float('nan')):.0f}ms"
        )

    def results_to_csv(self, results: list[dict], output: str) -> None:
        """Write a CSV summary of all conditions to *output*."""
        try:
            import pandas as pd

            rows = []
            for r in results:
                row = {"condition_id": r["condition_id"], "label": r["condition_label"],
                       "top_k": r["top_k"]}
                row.update(r["metrics"])
                rows.append(row)
            pd.DataFrame(rows).to_csv(output, index=False)
            print(f"Results written to {output}")
        except ImportError:
            # Fallback: plain CSV
            if not results:
                return
            all_keys = list(results[0]["metrics"].keys())
            header = ["condition_id", "label", "top_k"] + all_keys
            with open(output, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for r in results:
                    row = [str(r["condition_id"]), r["condition_label"], str(r["top_k"])]
                    row += [str(r["metrics"].get(k, "")) for k in all_keys]
                    f.write(",".join(row) + "\n")
            print(f"Results written to {output}")
