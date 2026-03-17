"""Answer quality metrics: RAGAS (LLM-as-judge) and ROUGE-L.

RAGAS and rouge-score are optional dependencies. If missing the relevant
functions return NaN values so the rest of the eval pipeline keeps working.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any


logger = logging.getLogger(__name__)


_rouge_scorer: Any | None = None
_ragas_evaluate: Any | None = None
_answer_relevancy: Any | None = None
_context_precision: Any | None = None
_context_recall: Any | None = None
_faithfulness: Any | None = None
_rouge_available = False
_ragas_available = False


try:
    from rouge_score import rouge_scorer

    _rouge_scorer = rouge_scorer
    _rouge_available = True
except ImportError:
    _rouge_available = False

_ROUGE_AVAILABLE = _rouge_available

try:
    # RAGAS 0.1.x / 0.2.x compatible import
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    _ragas_evaluate = evaluate
    _answer_relevancy = answer_relevancy
    _context_precision = context_precision
    _context_recall = context_recall
    _faithfulness = faithfulness
    _ragas_available = True
except Exception:
    _ragas_available = False

_RAGAS_AVAILABLE = _ragas_available


try:
    # Avoid requiring datasets at import time; it is optional for evaluation.
    import datasets
except ImportError:  # pragma: no cover
    datasets = None


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 between generated answer and reference answer."""
    if not _ROUGE_AVAILABLE or _rouge_scorer is None:
        return math.nan
    if not reference or not reference.strip() or not hypothesis:
        return math.nan
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(reference, hypothesis)
    return score["rougeL"].fmeasure


def compute_ragas_metrics(
    records: list[Mapping[str, Any]],
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
    if not _RAGAS_AVAILABLE or datasets is None:
        return {"ragas.available": 0.0}
    assert _faithfulness is not None
    assert _answer_relevancy is not None
    assert _context_precision is not None
    assert _context_recall is not None
    assert _ragas_evaluate is not None

    rows: list[dict[str, Any]] = []
    malformed = 0
    for r in records:
        try:
            question = str(r["question"])
            answer = str(r["answer"])

            contexts_raw = r.get("contexts", [])
            if isinstance(contexts_raw, str):
                contexts: list[str] = [contexts_raw]
            elif isinstance(contexts_raw, list):
                contexts = [str(c) for c in contexts_raw]
            else:
                raise TypeError("contexts must be list or str")

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": str(r.get("ground_truth", "")),
                }
            )
        except Exception as exc:  # missing key or wrong type
            malformed += 1
            logger.warning("Skipping malformed record in compute_ragas_metrics: %s", exc)
    if malformed:
        logger.warning("%d malformed records were skipped", malformed)
    if not rows:
        logger.error("No valid records to evaluate")
        return {"ragas.available": 0.0}
    try:
        dataset = datasets.Dataset.from_list(rows)
        metrics = [_faithfulness, _answer_relevancy, _context_precision, _context_recall]
        result = _ragas_evaluate(dataset, metrics=metrics)
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {"ragas.available": 0.0}

    answer_relevancy_score = float(result["answer_relevancy"])
    return {
        "ragas.faithfulness": float(result["faithfulness"]),
        # Keep both spellings for backward compatibility with existing dashboards.
        "ragas.answer_relevance": answer_relevancy_score,
        "ragas.answer_relevancy": answer_relevancy_score,
        "ragas.context_precision": float(result["context_precision"]),
        "ragas.context_recall": float(result["context_recall"]),
    }


def answer_length(answer: str) -> int:
    """Number of words in the generated answer (sanity check metric)."""
    return len(answer.split())
