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
_ROUGE_L_SCORER: Any | None = None
_rouge_available = False
_ragas_available = False


try:
    from rouge_score import rouge_scorer

    _rouge_scorer = rouge_scorer
    _ROUGE_L_SCORER = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
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
except (ImportError, AttributeError):
    _ragas_available = False

_RAGAS_AVAILABLE = _ragas_available


try:
    # Avoid requiring datasets at import time; it is optional for evaluation.
    import datasets
except ImportError:  # pragma: no cover
    datasets = None


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 between generated answer and reference answer."""
    if not _ROUGE_AVAILABLE or _rouge_scorer is None or _ROUGE_L_SCORER is None:
        return math.nan
    # Normalize both inputs to avoid treating whitespace-only strings as valid.
    # The typing of this API is strict: hypothesis/reference are str.
    if not reference or not reference.strip() or not hypothesis or not hypothesis.strip():
        return math.nan
    score = _ROUGE_L_SCORER.score(reference, hypothesis)
    return score["rougeL"].fmeasure


def compute_ragas_metrics(
    records: list[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute RAGAS metrics for a batch of eval records.

    Each record must contain:
        - ``question`` (str)
        - ``answer`` (str) — the system's generated answer
        - ``contexts`` (list[str]) — texts of the retrieved documents
        - ``ground_truth`` (str) — reference answer (required for context_recall)
          (if omitted/empty, context_recall is skipped)

    Returns a dict of averaged metric scores. Keys are prefixed with ``ragas.``.
    Returns ``{"ragas.available": 0.0}`` when RAGAS is not installed.
    """
    if not _RAGAS_AVAILABLE or datasets is None:
        return {"ragas.available": 0.0}
    if _faithfulness is None:
        raise RuntimeError("_faithfulness is required but is None")
    if _answer_relevancy is None:
        raise RuntimeError("_answer_relevancy is required but is None")
    if _context_precision is None:
        raise RuntimeError("_context_precision is required but is None")
    if _context_recall is None:
        raise RuntimeError("_context_recall is required but is None")
    if _ragas_evaluate is None:
        raise RuntimeError("_ragas_evaluate is required but is None")

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

            if not contexts:
                raise ValueError("contexts must be a non-empty list or non-empty string")

            ground_truth = str(r.get("ground_truth", ""))
            has_ground_truth = bool(ground_truth and ground_truth.strip())
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "ragas_context_recall_enabled": has_ground_truth,
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
        context_recall_enabled = all(r.get("ragas_context_recall_enabled", False) for r in rows)

        metrics: list[Any] = [_faithfulness, _answer_relevancy, _context_precision]
        if context_recall_enabled:
            metrics.append(_context_recall)

        result = _ragas_evaluate(dataset, metrics=metrics)
        answer_relevancy_score = float(result["answer_relevancy"])

        output = {
            "ragas.faithfulness": float(result["faithfulness"]),
            # Keep both spellings for backward compatibility with existing dashboards.
            "ragas.answer_relevance": answer_relevancy_score,
            "ragas.answer_relevancy": answer_relevancy_score,
            "ragas.context_precision": float(result["context_precision"]),
        }

        if context_recall_enabled and "context_recall" in result:
            output["ragas.context_recall"] = float(result["context_recall"])
        else:
            output["ragas.context_recall"] = math.nan

        return output
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {"ragas.available": 0.0}


def answer_length(answer: str) -> int:
    """Number of words in the generated answer (sanity check metric)."""
    return len(answer.split())
