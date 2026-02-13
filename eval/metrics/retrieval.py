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


def percentile_at(values: list[float], p: float) -> float:
    """p-th percentile of *values* using linear interpolation, 0 < p <= 100."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def intent_coverage_at_k(
    variant_retrieved_ids: list[Sequence[str]],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of query variants with ≥1 relevant doc in their top-k.

    Operationalises the VRisk intuition (Takehi et al., WSDM 2026): a result
    set is robust only when *every interpretation* of an ambiguous query (each
    rewrite variant) is adequately served — not just the majority reading.

    Args:
        variant_retrieved_ids: One ordered list of retrieved doc IDs per query variant.
        relevant: Ground-truth relevant doc IDs for this query.
        k: Cut-off for the coverage check.

    Returns:
        Score in [0, 1]; 1.0 means every variant has a relevant doc in its top-k.
    """
    if not variant_retrieved_ids or not relevant:
        return 0.0
    covered = sum(
        1 for ids in variant_retrieved_ids
        if set(ids[:k]) & relevant
    )
    return covered / len(variant_retrieved_ids)


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
    """Macro-average all per-query metric dicts; also add p10 and p25 percentiles.

    In addition to the standard mean, two robustness quantiles are computed for
    every metric key:

        <metric>_p10  — 10th-percentile (worst-decile) performance
        <metric>_p25  — 25th-percentile performance

    These expose whether a condition improves average performance while leaving
    hard queries behind — a pattern that the mean alone would hide, and that
    VRisk-style analysis (Takehi et al., WSDM 2026) specifically targets.
    """
    if not per_query:
        return {}
    keys = list(per_query[0].keys())
    agg: dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in per_query if k in d]
        if not vals:
            continue
        agg[k] = sum(vals) / len(vals)
        agg[f"{k}_p10"] = percentile_at(vals, 10)
        agg[f"{k}_p25"] = percentile_at(vals, 25)
    return agg
