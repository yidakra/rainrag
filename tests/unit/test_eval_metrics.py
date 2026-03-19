"""Unit tests for eval/metrics/retrieval.py and eval/metrics/answer_quality.py.

These tests require no external services (no Qdrant, no LLM APIs) and run
entirely in-process with deterministic inputs and expected outputs.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics.answer_quality import answer_length, rouge_l
from eval.metrics.cost import aggregate_costs, chars_to_tokens, estimate_query_cost
from eval.metrics.retrieval import (
    aggregate_metrics,
    average_precision,
    compute_all_metrics,
    intent_coverage_at_k,
    mrr,
    ndcg_at_k,
    percentile_at,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


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
            "recall@3",
            "recall@5",
            "precision@3",
            "precision@5",
            "ndcg@3",
            "ndcg@5",
            "mrr",
            "map",
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

    def test_single_entry_mean_values(self):
        m = {"recall@5": 0.8, "mrr": 0.6}
        result = aggregate_metrics([m])
        # Mean equals the single value
        assert result["recall@5"] == pytest.approx(0.8)
        assert result["mrr"] == pytest.approx(0.6)

    def test_single_entry_percentiles_equal_value(self):
        """Percentiles of a single-element list all equal that element."""
        m = {"recall@5": 0.8, "mrr": 0.6}
        result = aggregate_metrics([m])
        assert result["recall@5_p10"] == pytest.approx(0.8)
        assert result["recall@5_p25"] == pytest.approx(0.8)
        assert result["mrr_p10"] == pytest.approx(0.6)
        assert result["mrr_p25"] == pytest.approx(0.6)

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

    def test_percentile_keys_present_for_every_metric(self):
        entries = [{"recall@5": v, "mrr": v} for v in [0.2, 0.5, 0.8]]
        result = aggregate_metrics(entries)
        for base in ("recall@5", "mrr"):
            assert f"{base}_p10" in result, f"missing {base}_p10"
            assert f"{base}_p25" in result, f"missing {base}_p25"

    def test_p10_le_mean(self):
        """p10 must be ≤ mean for any distribution."""
        # 9 zeros + 1 one → mean = 0.1, p10 = 0.0
        vals = [0.0] * 9 + [1.0]
        entries = [{"m": v} for v in vals]
        result = aggregate_metrics(entries)
        assert result["m_p10"] <= result["m"]
        assert result["m_p25"] <= result["m"]


# ---------------------------------------------------------------------------
# percentile_at
# ---------------------------------------------------------------------------


class TestPercentileAt:
    def test_empty_list(self):
        assert percentile_at([], 50) == 0.0

    def test_single_value_all_percentiles_equal(self):
        for p in (0, 10, 50, 90, 100):
            assert percentile_at([0.7], p) == pytest.approx(0.7)

    def test_p0_equals_minimum(self):
        assert percentile_at([0.1, 0.5, 0.9], 0) == pytest.approx(0.1)

    def test_p100_equals_maximum(self):
        assert percentile_at([0.1, 0.5, 0.9], 100) == pytest.approx(0.9)

    def test_p50_median_odd(self):
        assert percentile_at([0.0, 0.5, 1.0], 50) == pytest.approx(0.5)

    def test_p50_median_even(self):
        # [0.0, 1.0] → p50 is midpoint = 0.5
        assert percentile_at([0.0, 1.0], 50) == pytest.approx(0.5)

    def test_linear_interpolation(self):
        # [0.0, 0.5, 1.0], p25: idx = 0.25 * 2 = 0.5 → lo=0, hi=1, frac=0.5
        # → 0.0 * 0.5 + 0.5 * 0.5 = 0.25
        assert percentile_at([0.0, 0.5, 1.0], 25) == pytest.approx(0.25)

    def test_unsorted_input_same_as_sorted(self):
        vals = [0.9, 0.1, 0.5, 0.3, 0.7]
        assert percentile_at(vals, 50) == pytest.approx(percentile_at(sorted(vals), 50))

    def test_all_same_values(self):
        assert percentile_at([0.4, 0.4, 0.4], 10) == pytest.approx(0.4)
        assert percentile_at([0.4, 0.4, 0.4], 90) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# intent_coverage_at_k
# ---------------------------------------------------------------------------


class TestIntentCoverageAtK:
    def test_all_variants_covered(self):
        """Every variant has a relevant doc in top-k → 1.0."""
        assert intent_coverage_at_k(
            [["a", "b"], ["a", "c"], ["b", "d"]], {"a", "b"}, k=2
        ) == pytest.approx(1.0)

    def test_no_variants_covered(self):
        """No variant has any relevant doc → 0.0."""
        assert intent_coverage_at_k([["x", "y"], ["z", "w"]], {"a", "b"}, k=5) == pytest.approx(0.0)

    def test_half_covered(self):
        """One of two variants has a relevant doc → 0.5."""
        assert intent_coverage_at_k([["a", "x"], ["x", "y"]], {"a"}, k=2) == pytest.approx(0.5)

    def test_cutoff_excludes_relevant(self):
        """Relevant doc at rank 3 is not counted when k=2."""
        assert intent_coverage_at_k([["x", "y", "a"]], {"a"}, k=2) == pytest.approx(0.0)

    def test_cutoff_includes_relevant(self):
        """Relevant doc at rank 2 is counted when k≥2."""
        assert intent_coverage_at_k([["x", "a", "y"]], {"a"}, k=2) == pytest.approx(1.0)

    def test_empty_relevant_returns_zero(self):
        assert intent_coverage_at_k([["a", "b"]], set(), k=5) == pytest.approx(0.0)

    def test_empty_variants_returns_zero(self):
        assert intent_coverage_at_k([], {"a"}, k=5) == pytest.approx(0.0)

    def test_single_variant_fully_covered(self):
        assert intent_coverage_at_k([["a", "b"]], {"a"}, k=1) == pytest.approx(1.0)

    def test_three_variants_two_covered(self):
        assert intent_coverage_at_k([["a"], ["b"], ["x"]], {"a", "b"}, k=1) == pytest.approx(2 / 3)


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
        # Use controlled token sequences with no overlap so the LCS is 0 regardless of
        # tokenization differences across rouge-score versions.
        score = rouge_l("a b c", "d e f")
        assert score == pytest.approx(0.0, abs=1e-8)

    def test_partial_overlap(self):
        # Use controlled token sequences with a single shared token so the expected
        # ROUGE-L score is stable even if tokenization changes.
        hypothesis = "a b c d"
        reference = "a x y z"
        score = rouge_l(hypothesis, reference)
        # With a single common token in 4-token inputs, the LCS-based ROUGE-L score is
        # expected to be around 0.25; allow a small range to remain robust.
        assert 0.2 < score < 0.3

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
        result = aq.compute_ragas_metrics(
            [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt"}]
        )
        assert result == {"ragas.available": 0.0}


# ---------------------------------------------------------------------------
# cost estimation
# ---------------------------------------------------------------------------


class TestCharsToTokens:
    def test_empty_string(self):
        assert chars_to_tokens("") == 0.0

    def test_none_returns_zero(self):
        assert chars_to_tokens(None) == 0.0

    def test_four_chars(self):
        assert chars_to_tokens("abcd") == pytest.approx(1.0)

    def test_scaling(self):
        assert chars_to_tokens("a" * 400) == pytest.approx(100.0)


class TestEstimateQueryCost:
    def _call(
        self,
        llm="openai",
        embed="openai",
        query="test",
        contexts=None,
        answer="ok",
        llm_query_rewrite_calls: int = 0,
        llm_hyde_calls: int = 0,
        reranker_calls: int = 0,
    ):
        return estimate_query_cost(
            query=query,
            contexts=contexts or ["context text"],
            answer=answer,
            llm_provider=llm,
            embed_provider=embed,
            llm_query_rewrite_calls=llm_query_rewrite_calls,
            llm_hyde_calls=llm_hyde_calls,
            reranker_calls=reranker_calls,
        )

    def test_returns_all_keys(self):
        result = self._call()
        expected = {
            "cost.input_tokens_est",
            "cost.output_tokens_est",
            "cost.embed_tokens_est",
            "cost.llm_base_usd_est",
            "cost.llm_rewrite_usd_est",
            "cost.llm_hyde_usd_est",
            "cost.reranker_usd_est",
            "cost.llm_usd_est",
            "cost.embed_usd_est",
            "cost.total_usd_est",
        }
        assert set(result.keys()) == expected

    def test_all_values_non_negative(self):
        result = self._call()
        for k, v in result.items():
            assert v >= 0.0, f"{k}={v} is negative"

    def test_total_equals_llm_plus_embed(self):
        result = self._call()
        assert result["cost.total_usd_est"] == pytest.approx(
            result["cost.llm_usd_est"] + result["cost.embed_usd_est"]
        )

    def test_additional_llm_calls_increase_total(self):
        base = self._call()
        rewrite = self._call(llm_query_rewrite_calls=1)
        hyde = self._call(llm_hyde_calls=1)

        assert rewrite["cost.llm_rewrite_usd_est"] > 0
        assert rewrite["cost.total_usd_est"] > base["cost.total_usd_est"]

        assert hyde["cost.llm_hyde_usd_est"] > 0
        assert hyde["cost.total_usd_est"] > base["cost.total_usd_est"]

    def test_unknown_provider_gives_zero_cost(self):
        result = estimate_query_cost(
            query="q",
            contexts=["c"],
            answer="a",
            llm_provider="unknown_llm",
            embed_provider="unknown_embed",
        )
        assert result["cost.llm_usd_est"] == pytest.approx(0.0)
        assert result["cost.embed_usd_est"] == pytest.approx(0.0)

    def test_local_embed_is_free(self):
        result = self._call(embed="local")
        assert result["cost.embed_usd_est"] == pytest.approx(0.0)

    def test_longer_answer_increases_output_tokens(self):
        short = self._call(answer="ok")
        long = self._call(answer="ok " * 200)
        assert long["cost.output_tokens_est"] > short["cost.output_tokens_est"]

    def test_more_context_increases_input_tokens(self):
        small = self._call(contexts=["a"])
        large = self._call(contexts=["a" * 1000])
        assert large["cost.input_tokens_est"] > small["cost.input_tokens_est"]

    def test_token_counts_are_integers(self):
        result = self._call()
        for k in ("cost.input_tokens_est", "cost.output_tokens_est", "cost.embed_tokens_est"):
            assert isinstance(result[k], int), f"{k} should be int"

    def test_empty_contexts(self):
        result = estimate_query_cost(
            query="hello",
            contexts=[],
            answer="world",
            llm_provider="claude",
            embed_provider="local",
        )
        assert result["cost.total_usd_est"] >= 0.0


class TestAggregateCosts:
    def test_empty_list_returns_empty(self):
        assert aggregate_costs([]) == {}

    def test_single_query(self):
        q = {
            "cost.llm_usd_est": 0.001,
            "cost.embed_usd_est": 0.0001,
            "cost.total_usd_est": 0.0011,
        }
        result = aggregate_costs([q])
        assert result["cost.total_usd_est"] == pytest.approx(0.0011)
        assert result["cost.aggregate_usd_est"] == pytest.approx(0.0011)
        assert result["cost.mean_usd_est_per_query"] == pytest.approx(0.0011)

    def test_averages_two_queries(self):
        q1 = {"cost.total_usd_est": 0.002}
        q2 = {"cost.total_usd_est": 0.004}
        result = aggregate_costs([q1, q2])
        assert result["cost.total_usd_est"] == pytest.approx(0.006)
        assert result["cost.aggregate_usd_est"] == pytest.approx(0.006)
        assert result["cost.mean_usd_est_per_query"] == pytest.approx(0.003)

    def test_all_cost_keys_aggregated(self):
        keys = [
            "cost.input_tokens_est",
            "cost.output_tokens_est",
            "cost.embed_tokens_est",
            "cost.llm_base_usd_est",
            "cost.llm_rewrite_usd_est",
            "cost.llm_hyde_usd_est",
            "cost.reranker_usd_est",
            "cost.llm_usd_est",
            "cost.embed_usd_est",
            "cost.total_usd_est",
        ]
        queries = [{k: 1.0 for k in keys}, {k: 3.0 for k in keys}]
        result = aggregate_costs(queries)
        for k in keys:
            assert result[k] == pytest.approx(4.0), f"total for {k}"
            if k == "cost.total_usd_est":
                assert result["cost.mean_usd_est_per_query"] == pytest.approx(2.0)
            else:
                assert result[f"{k}_per_query"] == pytest.approx(2.0), f"avg for {k}"
