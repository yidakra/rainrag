"""Outcome-informed formulation selection, independent of retrieval and provider clients.

Models are fitted on archive judgments, never shipped benchmark coefficients. Features
are observed before candidate selection and before the shared final reranking stage.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "top1",
    "margin12",
    "margin13",
    "z1",
    "entropy",
    "maxp",
    "top5mass",
    "std",
    "count",
    "overlap10",
    "overlap100",
)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetrievalAlternative:
    """One ordered candidate ranking. ID is stable within a query's option pool."""

    id: str
    query: str
    documents: list[dict[str, Any]]


def configuration_signature(config: Any, candidate_depth: int | None = None) -> str:
    """Bind weights to retrieval configuration without serialising any credentials."""
    embedding_models = {
        "local": config.embedding.model_name,
        "mistral": config.mistral.embedding_model,
        "openai": config.openai.embedding_model,
        "gemini": config.gemini.embedding_model,
    }
    settings = {
        "collection": config.qdrant.collection_name,
        "corpus_revision": config.adaptive_retrieval.corpus_revision,
        "embedding_provider": config.embedding.provider,
        "embedding_model": embedding_models.get(config.embedding.provider),
        "hybrid": config.hybrid_search.model_dump(),
        "two_stage": config.two_stage.model_dump(),
        "reranker": config.reranker.model_dump(),
        "reranker_model": config.cohere.model_name,
        "llm_provider": config.llm.provider,
        "llm_model": getattr(config, config.llm.provider).model_name,
        "candidate_depth": candidate_depth
        if candidate_depth is not None
        else (
            config.reranker.initial_k
            if config.reranker.enabled
            else getattr(config, config.llm.provider).top_k
        ),
    }
    payload = json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _scores(alternative: RetrievalAlternative) -> np.ndarray:
    ids = [d["doc_id"] for d in alternative.documents]
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids) or len(set(ids)) != len(
        ids
    ):
        raise ValueError("Candidate document IDs must be non-empty and unique")
    scores = np.asarray([float(d["score"]) for d in alternative.documents], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Candidate scores must be finite")
    return scores


def feature_matrix(alternatives: list[RetrievalAlternative]) -> np.ndarray:
    """Stable score-distribution features and mean Jaccard agreement with other options.

    Score margins use descending scores, while agreement uses actual retrieval rank.
    Empty and short lists are supported. Features are not relevance probabilities.
    """
    if not alternatives or len({a.id for a in alternatives}) != len(alternatives):
        raise ValueError("Alternatives must be non-empty and have unique IDs")
    rows = []
    for alternative in alternatives:
        scores = np.sort(_scores(alternative))[::-1]
        if scores.size:
            exp = np.exp(scores - scores[0])
            probability = exp / exp.sum()
            std = float(scores.std())
            confidence = [
                float(scores[0]),
                float(scores[0] - scores[1]) if len(scores) > 1 else 0.0,
                float(scores[0] - scores[2]) if len(scores) > 2 else 0.0,
                float((scores[0] - scores.mean()) / (std + 1e-9)),
                float(-(probability * np.log(probability + 1e-12)).sum()),
                float(probability[0]),
                float(probability[:5].sum()),
                std,
                float(len(scores)),
            ]
        else:
            confidence = [0.0] * 9
        for k in (10, 100):
            own = {str(d["doc_id"]) for d in alternative.documents[:k]}
            overlaps = []
            for other in alternatives:
                if other.id == alternative.id:
                    continue
                theirs = {str(d["doc_id"]) for d in other.documents[:k]}
                union = own | theirs
                overlaps.append(len(own & theirs) / len(union) if union else 0.0)
            confidence.append(float(np.mean(overlaps)) if overlaps else 0.0)
        rows.append(confidence)
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Candidate features must be finite")
    return matrix


@dataclass(frozen=True)
class RidgeSelector:
    """Portable numeric artifact, without pickle or executable model deserialisation."""

    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    signature: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict within-query utility contrasts, not calibrated absolute utility."""
        centred = features - features.mean(axis=0)
        predictions = ((centred - self.mean) / self.scale) @ self.coefficients
        if not np.isfinite(predictions).all():
            raise ValueError("Non-finite selector predictions")
        return predictions

    def save(self, path: Path) -> None:
        """Write a versioned model with its exact feature order and retrieval binding."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "configuration_signature": self.signature,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
        }
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: Path, signature: str) -> RidgeSelector:
        """Reject incompatible, malformed and non-finite artifacts at startup."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Selector artifact must be an object")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported selector schema")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Selector feature order does not match")
        if payload.get("configuration_signature") != signature:
            raise ValueError("Selector retrieval configuration does not match")
        arrays = [
            np.asarray(payload[name], dtype=float) for name in ("mean", "scale", "coefficients")
        ]
        if any(a.shape != (len(FEATURE_NAMES),) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Invalid selector coefficients")
        if (arrays[1] <= 0).any():
            raise ValueError("Selector scales must be positive")
        return cls(arrays[0], arrays[1], arrays[2], signature)


def fit_ridge(
    matrices: list[np.ndarray], targets: list[np.ndarray], signature: str, alpha: float = 1.0
) -> RidgeSelector:
    """Fit one fixed-alpha ridge using only the supplied training queries.

    Centre each query's options and standardise on training data only. The offline
    pre-retrieval comparison uses this identical learner with a different matrix.
    """
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    if not matrices or len(matrices) != len(targets):
        raise ValueError("Training queries and targets must align")
    for features, utility in zip(matrices, targets, strict=True):
        if features.ndim != 2 or utility.shape != (len(features),) or not len(features):
            raise ValueError("Training option shapes do not align")
        if not np.isfinite(features).all() or not np.isfinite(utility).all():
            raise ValueError("Training features and targets must be finite")
    x = np.concatenate([m - m.mean(axis=0) for m in matrices])
    y = np.concatenate([t - t.mean() for t in targets])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    x = (x - mean) / scale
    coefficients = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    if not all(np.isfinite(a).all() for a in (mean, scale, coefficients)):
        raise ValueError("Non-finite fitted selector")
    return RidgeSelector(mean, scale, coefficients, signature)


def select_alternative(
    alternatives: list[RetrievalAlternative], model: RidgeSelector, min_gain: float = 0.0
) -> tuple[RetrievalAlternative, dict[str, Any]]:
    """Select against original; ties, empty options and small predicted gains keep it."""
    if not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError("Minimum predicted gain must be finite and non-negative")
    original = next((a for a in alternatives if a.id == "original"), None)
    if original is None:
        raise ValueError("The original query must remain an alternative")
    predictions = model.predict(feature_matrix(alternatives))
    eligible = [i for i, a in enumerate(alternatives) if a.documents]
    original_index = alternatives.index(original)
    chosen = max(eligible, key=lambda i: predictions[i]) if eligible else original_index
    gain = float(predictions[chosen] - predictions[original_index])
    if original.documents and gain <= min_gain:
        chosen = original_index
    return alternatives[chosen], {
        "selected_id": alternatives[chosen].id,
        "predicted_gain": float(predictions[chosen] - predictions[original_index]),
        "predictions": {a.id: float(predictions[i]) for i, a in enumerate(alternatives)},
    }


def alternative_snapshot(alternatives: list[RetrievalAlternative]) -> list[dict[str, Any]]:
    """Export scores and IDs, not candidate transcript text, for offline judgments."""
    return [
        {
            "id": a.id,
            "query": a.query,
            "documents": [
                {key: d[key] for key in ("doc_id", "video_id", "score", "score_type") if key in d}
                for d in a.documents
            ],
        }
        for a in alternatives
    ]
