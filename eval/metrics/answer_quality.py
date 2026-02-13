"""Answer quality metrics: RAGAS (LLM-as-judge) and ROUGE-L.

RAGAS and rouge-score are optional dependencies. If missing the relevant
functions return NaN values so the rest of the eval pipeline keeps working.
"""
from __future__ import annotations

import math
from typing import Any

try:
    from rouge_score import rouge_scorer as _rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    # RAGAS 0.1.x / 0.2.x compatible import
    from ragas import evaluate as _ragas_evaluate
    from ragas.metrics import (
        answer_relevancy as _answer_relevancy,
        context_precision as _context_precision,
        context_recall as _context_recall,
        faithfulness as _faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 between generated answer and reference answer."""
    if not _ROUGE_AVAILABLE or not reference.strip():
        return math.nan
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(reference, hypothesis)
    return score["rougeL"].fmeasure


def compute_ragas_metrics(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute RAGAS metrics for a batch of eval records.

    Each record must contain:
        - ``question`` (str)
        - ``answer`` (str) — the system's generated answer
        - ``contexts`` (list[str]) — texts of the retrieved documents
        - ``ground_truth`` (str) — reference answer (may be empty string)

    Returns a dict of averaged metric scores. Keys are prefixed with ``ragas.``.
    Returns ``{"ragas.available": 0.0}`` when RAGAS is not installed.
    """
    if not _RAGAS_AVAILABLE:
        return {"ragas.available": 0.0}

    try:
        from datasets import Dataset  # type: ignore[import]
    except ImportError:
        return {"ragas.available": 0.0}

    rows = [
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r.get("ground_truth", ""),
        }
        for r in records
    ]
    dataset = Dataset.from_list(rows)

    metrics = [_faithfulness, _answer_relevancy, _context_precision, _context_recall]
    result = _ragas_evaluate(dataset, metrics=metrics)

    return {
        "ragas.faithfulness": float(result["faithfulness"]),
        "ragas.answer_relevance": float(result["answer_relevancy"]),
        "ragas.context_precision": float(result["context_precision"]),
        "ragas.context_recall": float(result["context_recall"]),
    }


def answer_length(answer: str) -> int:
    """Number of words in the generated answer (sanity check metric)."""
    return len(answer.split())
