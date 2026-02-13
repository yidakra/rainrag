"""Retrieval evaluation metrics: Recall@k, Precision@k, MRR, NDCG@k, MAP.

All functions operate on pre-computed doc-ID lists and require no LLM calls,
so they are fast and cheap to run on every experiment condition.
"""
from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k results."""
    if not relevant:
        return 0.0
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    hits = set(retrieved[:k]) & relevant
    return len(hits) / k


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant doc."""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    retrieved_k = list(retrieved[:k])

    def dcg(items: list[str]) -> float:
        return sum(
            (1.0 / math.log2(i + 2)) if item in relevant else 0.0
            for i, item in enumerate(items)
        )

    actual_dcg = dcg(retrieved_k)
    # Ideal: all relevant docs first
    ideal_len = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    """Average Precision (area under the precision-recall curve)."""
    if not relevant:
        return 0.0
    num_hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            num_hits += 1
            precision_sum += num_hits / i
    return precision_sum / len(relevant)


def compute_all_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    ks: tuple[int, ...] = (3, 5, 10),
) -> dict[str, float]:
    """Compute the full set of retrieval metrics for a single query result.

    Args:
        retrieved_ids: Ordered list of doc IDs returned by the system (best first).
        relevant_ids: Ground-truth relevant doc IDs from the eval dataset.
        ks: Cut-offs for recall, precision, and NDCG.

    Returns:
        Dict mapping metric name → score (all in [0, 1]).
    """
    relevant = set(relevant_ids)
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
    metrics["mrr"] = mrr(retrieved_ids, relevant)
    metrics["map"] = average_precision(retrieved_ids, relevant)
    return metrics


def aggregate_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average all per-query metric dicts into a single summary dict."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(d[k] for d in per_query) / len(per_query) for k in keys}
