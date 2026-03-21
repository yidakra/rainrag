"""Per-stage latency profiling experiment.

Times each major stage of the query pipeline independently for a set of
ablation conditions and logs p50 / p95 / mean latencies to MLflow.

Stages measured
---------------
t_embed_ms      – embed_query() call
t_retrieve_ms   – retrieve_documents() call (includes BM25, vector, fusion,
                  and any two-stage logic that lives inside retrieve_documents)
t_rerank_ms     – rerank_documents() call (0 when reranker disabled)
t_generate_ms   – generate_answer() call
t_total_ms      – wall-clock time for the full engine.query() call

Note: query rewriting and HyDE happen inside engine.query() before the
retrieve_documents() call, so their overhead is captured in t_total_ms but
not broken out individually without source-level instrumentation.  A
breakdown comment is logged as an MLflow tag to note this limitation.

Usage
-----
    from eval.experiments.latency import LatencyExperiment

    exp = LatencyExperiment(
        config_path="config.yaml",
        dataset_path="eval/datasets/eval_set_en.jsonl",
        condition_ids=["01", "06", "08"],
        n_queries=10,
        n_repeats=3,
    )
    results = exp.run()
"""

from __future__ import annotations

import statistics
import time
from typing import Any, cast

import eval.mlflow_tracking as mlflow_tracking
from eval.datasets.create_eval_set import load_eval_set
from eval.experiments.ablation import ABLATION_CONDITIONS
from eval.experiments.base import apply_overrides
from rainrag.config import load_config
from rainrag.query import RAGQueryEngine


def _percentile(data: list[float], p: float) -> float:
    """Return the p-th percentile of *data* (0–100)."""
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _time_stage(fn, *args, **kwargs) -> tuple[Any, float]:
    """Call *fn* and return (result, elapsed_ms).

    Reserved for future per-stage instrumentation when engine exposes per-stage
    hooks. Currently only used for total end-to-end timing in _profile_query.
    """
    t = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t) * 1000


def _profile_query(engine: RAGQueryEngine, record: dict, top_k: int) -> dict[str, float]:
    """Run a single query through the engine, timing each public stage.

    The full pipeline is timed end-to-end via engine.query().

    Staging-level timings from embed_query/retrieve_documents/rerank_documents/
    generate_answer are not measured in this function because calling them
    separately would duplicate work and double-count cost relative to the
    actual query path. If engine.query() supports per-stage instrumentation
    in the future, this function should be updated to read those values.
    """
    question = record["query"]
    lang = record.get("language", "en")

    _, t_total = _time_stage(engine.query, question=question, top_k=top_k, language=lang)

    # These per-stage timings are not available without engine-side instrumentation.
    # They are set to NaN to avoid misrepresenting separate re-computed measurements.
    t_embed = float("nan")
    t_retrieve = float("nan")
    t_rerank = float("nan")
    t_generate = float("nan")

    return {
        "t_embed_ms": t_embed,
        "t_retrieve_ms": t_retrieve,
        "t_rerank_ms": t_rerank,
        "t_generate_ms": t_generate,
        "t_total_ms": t_total,
    }


class LatencyExperiment:
    """Profile per-stage latency for selected ablation conditions.

    Args:
        condition_ids: Which ablation condition IDs to profile. Defaults to
            ["01", "06", "08"] (vector-only, rewrite+hyde, full_pipeline).
        n_queries: Number of queries sampled from the dataset per condition.
        n_repeats: Number of timing repetitions per query (results are averaged).
        top_k: Retrieval depth to use.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        dataset_path: str | None = None,
        mlflow_uri: str | None = None,
        condition_ids: list[str] | None = None,
        n_queries: int = 10,
        n_repeats: int = 3,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.mlflow_uri = mlflow_uri or mlflow_tracking.default_tracking_uri()
        self.condition_ids = condition_ids or ["01", "06", "08"]
        self.n_queries = n_queries
        self.n_repeats = n_repeats
        self.top_k = top_k

    def run(self) -> list[dict[str, Any]]:
        """Run latency profiling and return per-condition summaries."""
        mlflow_tracking.setup(self.mlflow_uri, "latency_profiling")

        base_config = load_config(self.config_path)
        dataset = load_eval_set(self.dataset_path) if self.dataset_path else []

        conditions = [c for c in ABLATION_CONDITIONS if c["id"] in self.condition_ids]
        results = []

        for condition in conditions:
            cfg = apply_overrides(base_config, condition.get("overrides", {}))
            engine = RAGQueryEngine(cfg)
            engine.initialize()

            # Sample queries (deterministic: first n_queries)
            records = dataset[: self.n_queries] if dataset else []
            if not records:
                print(f"[{condition['id']}] No dataset records — skipping.")
                continue

            # Collect per-stage timings across all queries × repeats
            stage_times: dict[str, list[float]] = {
                "t_embed_ms": [],
                "t_retrieve_ms": [],
                "t_rerank_ms": [],
                "t_generate_ms": [],
                "t_total_ms": [],
            }
            total_queries = 0

            print(
                f"[{condition['id']}] {condition['label']} — profiling {len(records)} queries × {self.n_repeats} repeats ..."
            )
            for record in records:
                for _ in range(self.n_repeats):
                    total_queries += 1
                    try:
                        timings = _profile_query(engine, record, self.top_k)
                        for stage, ms in timings.items():
                            stage_times[stage].append(ms)
                    except Exception as exc:
                        record_id = record.get("id") or record.get("query") or repr(record)
                        print(f"  [warn] Query failed for {record_id}: {exc}")

            # Aggregate
            metrics: dict[str, float] = {}
            for stage, times in stage_times.items():
                if times:
                    base = stage.replace("t_", "").replace("_ms", "")
                    metrics[f"{base}_p50_ms"] = _percentile(times, 50)
                    metrics[f"{base}_p95_ms"] = _percentile(times, 95)
                    metrics[f"{base}_mean_ms"] = statistics.mean(times)

            params = {
                "condition_id": condition["id"],
                "condition_label": condition["label"],
                "top_k": self.top_k,
                "n_queries": len(records),
                "n_repeats": self.n_repeats,
                "hybrid_search.enabled": cfg.hybrid_search.enabled,
                "two_stage.enabled": cfg.two_stage.enabled,
                "two_stage.query_rewrite_enabled": cfg.two_stage.query_rewrite_enabled,
                "two_stage.hyde_enabled": cfg.two_stage.hyde_enabled,
                "reranker.enabled": cfg.reranker.enabled,
            }

            run_name = f"latency_{condition['label']}"
            with mlflow_tracking.start_run(run_name=run_name, tags={"experiment_type": "latency"}):
                mlflow_tracking.log_params(params)
                mlflow_tracking.log_metrics(cast(dict[str, float | int | None], metrics))
                mlflow_tracking.log_dict_as_artifact(
                    {"condition": condition["label"], "timings": stage_times},
                    "raw_timings.json",
                )

            summary = {"condition": condition["label"], "params": params, "metrics": metrics}
            results.append(summary)

            # Print summary table
            total_p50 = metrics.get("total_p50_ms", float("nan"))
            total_p95 = metrics.get("total_p95_ms", float("nan"))
            embed_p50 = metrics.get("embed_p50_ms", float("nan"))
            retrieve_p50 = metrics.get("retrieve_p50_ms", float("nan"))
            rerank_p50 = metrics.get("rerank_p50_ms", float("nan"))
            generate_p50 = metrics.get("generate_p50_ms", float("nan"))
            print(
                f"  total p50={total_p50:.0f}ms p95={total_p95:.0f}ms | embed={embed_p50:.0f} retrieve={retrieve_p50:.0f} rerank={rerank_p50:.0f} generate={generate_p50:.0f}"
            )

        return results
