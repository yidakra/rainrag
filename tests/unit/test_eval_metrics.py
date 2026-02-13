"""Unit tests for eval/metrics/retrieval.py and eval/metrics/answer_quality.py.

These tests require no external services (no Qdrant, no LLM APIs) and run
entirely in-process with deterministic inputs and expected outputs.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Allow imports from the repo root (eval/ lives alongside src/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.metrics.retrieval import (
    aggregate_metrics,
    average_precision,
    compute_all_metrics,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from eval.metrics.answer_quality import answer_length, rouge_l


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ids(*args: str) -> list[str]:
    """Convenience: build a list of doc-id strings."""
    return list(args)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_perfect_recall(self):
        """All relevant docs appear in top-k → recall = 1.0."""
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0

    def test_zero_recall(self):
        """No relevant doc in retrieved list → recall = 0.0."""
        assert recall_at_k(["x", "y", "z"], {"a", "b"}, k=3) == 0.0

    def test_partial_recall(self):
        """One of two relevant docs found → recall = 0.5."""
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5

    def test_cutoff_excludes_relevant(self):
        """Relevant doc beyond k → not counted."""
        assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0

    def test_empty_relevant_set(self):
        """No relevant docs defined → recall = 0.0 (nothing to find)."""
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], {"a"}, k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        """k > len(retrieved) is fine; just uses all retrieved docs."""
        assert recall_at_k(["a"], {"a", "b"}, k=100) == 0.5


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], {"a"}, k=2) == 0.5

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_cutoff(self):
        """precision@1 considers only the first result."""
        assert precision_at_k(["a", "x", "x"], {"a"}, k=1) == 1.0
        assert precision_at_k(["x", "a", "a"], {"a"}, k=1) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


class TestMRR:
    def test_first_rank(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_rank(self):
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_rank(self):
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_multiple_relevant_returns_first_hit(self):
        """MRR is based on first hit; second hit doesn't matter."""
        assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)

    def test_empty_retrieved(self):
        assert mrr([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert mrr(["a", "b"], set()) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNDCGAtK:
    def test_perfect_ndcg(self):
        """Relevant doc at rank 1 → NDCG = 1.0."""
        assert ndcg_at_k(["a", "x"], {"a"}, k=2) == 1.0

    def test_no_relevant(self):
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_relevant_at_rank_2(self):
        """Relevant doc at rank 2 (index 1): DCG = 1/log2(3) ≈ 0.631.
        Ideal DCG (rank 1) = 1/log2(2) = 1.0 → NDCG ≈ 0.631."""
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert ndcg_at_k(["x", "a"], {"a"}, k=2) == pytest.approx(expected, abs=1e-9)

    def test_multiple_relevant_perfect_order(self):
        """Both relevant docs at ranks 1 and 2 → NDCG = 1.0."""
        assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0

    def test_multiple_relevant_reversed_order(self):
        """Relevant docs at ranks 2, 3 instead of 1, 2 → NDCG < 1.0."""
        score_actual = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
        score_perfect = 1.0
        assert score_actual < score_perfect
        assert score_actual > 0.0

    def test_empty_retrieved(self):
        assert ndcg_at_k([], {"a"}, k=5) == 0.0

    def test_k_zero(self):
        assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------


class TestAveragePrecision:
    def test_single_relevant_at_rank_1(self):
        assert average_precision(["a", "x", "y"], {"a"}) == 1.0

    def test_single_relevant_at_rank_2(self):
        """Hit at rank 2: precision = 1/2. AP = 0.5."""
        assert average_precision(["x", "a", "y"], {"a"}) == pytest.approx(0.5)

    def test_two_relevant_perfect(self):
        """Hits at ranks 1 and 2: AP = (1/1 + 2/2) / 2 = 1.0."""
        assert average_precision(["a", "b", "x"], {"a", "b"}) == 1.0

    def test_two_relevant_non_contiguous(self):
        """Hits at ranks 1 and 3: AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 ≈ 0.833."""
        expected = (1.0 + 2 / 3) / 2
        assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(expected)

    def test_no_hit(self):
        assert average_precision(["x", "y"], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert average_precision(["a", "b"], set()) == 0.0

    def test_empty_retrieved(self):
        assert average_precision([], {"a"}) == 0.0


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        result = compute_all_metrics(["a", "b", "c"], ["a"], ks=(3, 5))
        expected_keys = {
            "recall@3", "recall@5",
            "precision@3", "precision@5",
            "ndcg@3", "ndcg@5",
            "mrr", "map",
        }
        assert set(result.keys()) == expected_keys

    def test_values_in_unit_interval(self):
        result = compute_all_metrics(["a", "x", "b", "y", "c"], ["a", "b"])
        for k, v in result.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} is out of [0, 1]"

    def test_perfect_retrieval(self):
        result = compute_all_metrics(["a", "b"], ["a", "b"], ks=(2,))
        assert result["recall@2"] == 1.0
        assert result["precision@2"] == 1.0
        assert result["ndcg@2"] == 1.0
        assert result["mrr"] == 1.0
        assert result["map"] == 1.0

    def test_no_relevant(self):
        result = compute_all_metrics(["x", "y", "z"], ["a"])
        for v in result.values():
            assert v == 0.0

    def test_empty_relevant_ids(self):
        result = compute_all_metrics(["a", "b"], [])
        # All metrics should be 0 (nothing to find)
        for v in result.values():
            assert v == 0.0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_empty_list(self):
        assert aggregate_metrics([]) == {}

    def test_single_entry(self):
        m = {"recall@5": 0.8, "mrr": 0.6}
        assert aggregate_metrics([m]) == pytest.approx(m)

    def test_averages_correctly(self):
        m1 = {"recall@5": 0.8, "mrr": 1.0}
        m2 = {"recall@5": 0.4, "mrr": 0.5}
        result = aggregate_metrics([m1, m2])
        assert result["recall@5"] == pytest.approx(0.6)
        assert result["mrr"] == pytest.approx(0.75)

    def test_three_entries(self):
        entries = [{"ndcg@5": v} for v in [0.3, 0.6, 0.9]]
        result = aggregate_metrics(entries)
        assert result["ndcg@5"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# answer_length
# ---------------------------------------------------------------------------


class TestAnswerLength:
    def test_normal_sentence(self):
        assert answer_length("The bridge will be repaired.") == 5

    def test_empty_string(self):
        # "".split() returns [] in Python → len = 0
        assert answer_length("") == 0

    def test_single_word(self):
        assert answer_length("Hello") == 1

    def test_whitespace_only(self):
        assert answer_length("   ") == 0


# ---------------------------------------------------------------------------
# rouge_l
# ---------------------------------------------------------------------------


class TestRougeL:
    """ROUGE-L tests. If rouge-score is not installed the function returns NaN."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_rouge(self):
        try:
            import rouge_score  # noqa: F401
        except ImportError:
            pytest.skip("rouge-score not installed")

    def test_identical_strings(self):
        score = rouge_l("The cat sat on the mat.", "The cat sat on the mat.")
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_completely_different(self):
        score = rouge_l("Hello world foo bar", "Completely unrelated text xyz.")
        assert 0.0 <= score < 0.5

    def test_partial_overlap(self):
        hypothesis = "The bridge will be repaired by autumn."
        reference = "The bridge will be repaired in autumn."
        score = rouge_l(hypothesis, reference)
        assert 0.5 < score <= 1.0

    def test_empty_reference_returns_nan(self):
        score = rouge_l("some answer", "")
        assert math.isnan(score)

    def test_returns_float_in_unit_interval(self):
        score = rouge_l("foo bar baz", "foo bar")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# answer_quality — NaN fallback when rouge-score is absent
# ---------------------------------------------------------------------------


class TestAnswerQualityFallback:
    def test_rouge_l_returns_nan_without_rouge_score(self, monkeypatch):
        """Simulate rouge-score not installed by patching the availability flag."""
        import eval.metrics.answer_quality as aq

        monkeypatch.setattr(aq, "_ROUGE_AVAILABLE", False)
        result = aq.rouge_l("hypothesis", "reference")
        assert math.isnan(result)

    def test_compute_ragas_returns_unavailable_sentinel(self, monkeypatch):
        """Simulate ragas not installed."""
        import eval.metrics.answer_quality as aq

        monkeypatch.setattr(aq, "_RAGAS_AVAILABLE", False)
        result = aq.compute_ragas_metrics([
            {"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt"}
        ])
        assert result == {"ragas.available": 0.0}
