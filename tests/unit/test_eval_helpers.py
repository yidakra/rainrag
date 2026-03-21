"""Unit tests for eval helper utilities.

Covers:
- eval.experiments.base.apply_overrides()
- eval.datasets.beir_adapter data containers (BEIRCorpus, BEIRQRels, BEIRAdapter)

No external services required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    @pytest.fixture
    def base_config(self, test_config):
        """Use the shared test_config fixture from conftest.py."""
        return test_config

    @pytest.fixture
    def minimal_config(self, base_config):
        """Alias for base_config to support tests expecting minimal_config."""
        return base_config

    def test_single_nested_override(self, base_config):
        from eval.experiments.base import apply_overrides

        result = apply_overrides(base_config, {"hybrid_search.enabled": True})
        assert result.hybrid_search.enabled is True

    def test_does_not_mutate_original(self, base_config):
        from eval.experiments.base import apply_overrides

        original_value = base_config.hybrid_search.enabled
        apply_overrides(base_config, {"hybrid_search.enabled": not original_value})
        assert base_config.hybrid_search.enabled == original_value

    def test_multiple_overrides(self, base_config):
        from eval.experiments.base import apply_overrides

        result = apply_overrides(
            base_config,
            {
                "hybrid_search.enabled": True,
                "hybrid_search.fusion_method": "weighted",
                "reranker.enabled": False,
                "two_stage.query_rewrite_variants": 3,
            },
        )
        assert result.hybrid_search.enabled is True
        assert result.hybrid_search.fusion_method == "weighted"
        assert result.reranker.enabled is False
        assert result.two_stage.query_rewrite_variants == 3

    def test_llm_provider_override(self, base_config):
        from eval.experiments.base import apply_overrides

        result = apply_overrides(base_config, {"llm.provider": "openai"})
        assert result.llm.provider == "openai"

    def test_apply_overrides_merge_strategy(self, minimal_config):
        """The new merge_strategy field must be settable via apply_overrides."""
        from eval.experiments.base import apply_overrides

        result = apply_overrides(minimal_config, {"two_stage.merge_strategy": "diverse_rrf"})
        assert result.two_stage.merge_strategy == "diverse_rrf"
        assert minimal_config.two_stage.merge_strategy == "coverage"

    def test_apply_overrides_merge_rrf_k(self, minimal_config):
        """The new merge_rrf_k field must be settable via apply_overrides."""
        from eval.experiments.base import apply_overrides

        result = apply_overrides(minimal_config, {"two_stage.merge_rrf_k": 20})
        assert result.two_stage.merge_rrf_k == 20
        assert minimal_config.two_stage.merge_rrf_k == 60

    def test_apply_overrides_prompt_doc_order(self, minimal_config):
        """prompt_doc_order must be settable via apply_overrides (Axis F)."""
        from eval.experiments.base import apply_overrides

        result = apply_overrides(minimal_config, {"two_stage.prompt_doc_order": "book_end"})
        assert result.two_stage.prompt_doc_order == "book_end"

    def test_invalid_two_stage_enum_values(self):
        """Assigning or constructing forbidden strings should trigger validation errors."""
        from pydantic import ValidationError

        from rainrag.config import TwoStageConfig

        with pytest.raises(ValidationError):
            TwoStageConfig(merge_strategy=cast(Any, "not_a_strategy"))
        with pytest.raises(ValidationError):
            TwoStageConfig(prompt_doc_order=cast(Any, "upside_down"))

    def test_apply_overrides_min_retrieval_score(self, minimal_config):
        """min_retrieval_score must be settable via apply_overrides."""
        from eval.experiments.base import apply_overrides

        result = apply_overrides(minimal_config, {"reranker.min_retrieval_score": 0.4})
        assert result.reranker.min_retrieval_score == pytest.approx(0.4)
        assert minimal_config.reranker.min_retrieval_score == pytest.approx(0.0)

    def test_empty_overrides_returns_equivalent_config(self, base_config):
        from eval.experiments.base import apply_overrides

        result = apply_overrides(base_config, {})
        # Same values, different object
        assert result is not base_config
        assert result.llm.provider == base_config.llm.provider
        assert result.hybrid_search.enabled == base_config.hybrid_search.enabled

    def test_apply_overrides_invalid_path_raises(self, base_config):
        from eval.experiments.base import apply_overrides

        with pytest.raises(ValueError, match=r"Invalid override path 'nonexistent\.field'"):
            apply_overrides(base_config, {"nonexistent.field": True})

    def test_boolean_false_override(self, base_config):
        import copy

        from eval.experiments.base import apply_overrides

        # Create a copy to avoid mutating shared fixture
        cfg = copy.deepcopy(base_config)
        cfg.reranker.enabled = True
        result = apply_overrides(cfg, {"reranker.enabled": False})
        assert result.reranker.enabled is False

    def test_float_override(self, base_config):
        from eval.experiments.base import apply_overrides

        result = apply_overrides(base_config, {"hybrid_search.bm25_weight": 0.42})
        assert result.hybrid_search.bm25_weight == pytest.approx(0.42)

    def test_deep_copy_isolation(self, base_config):
        """Modifying the returned config must not affect the original."""
        from eval.experiments.base import apply_overrides

        result = apply_overrides(base_config, {})
        result.hybrid_search.enabled = not result.hybrid_search.enabled
        assert result.hybrid_search.enabled != base_config.hybrid_search.enabled


# ---------------------------------------------------------------------------
# BEIRCorpus
# ---------------------------------------------------------------------------


class TestBEIRCorpus:
    def test_len(self):
        from eval.datasets.beir_adapter import BEIRCorpus

        corpus = BEIRCorpus(
            docs={"d1": {"title": "T", "text": "foo"}, "d2": {"title": "U", "text": "bar"}}
        )
        assert len(corpus) == 2

    def test_empty(self):
        from eval.datasets.beir_adapter import BEIRCorpus

        assert len(BEIRCorpus()) == 0


# ---------------------------------------------------------------------------
# BEIRQRels
# ---------------------------------------------------------------------------


class TestBEIRQRels:
    @pytest.fixture
    def qrels(self):
        from eval.datasets.beir_adapter import BEIRQRels

        return BEIRQRels(
            qrels={
                "q1": {"d1": 2, "d2": 1, "d3": 0},
                "q2": {"d4": 1},
                "q3": {},
            }
        )

    def test_relevant_default_min_score(self, qrels):
        result = qrels.relevant_doc_ids("q1", min_score=1)
        assert set(result) == {"d1", "d2"}

    def test_relevant_high_min_score(self, qrels):
        result = qrels.relevant_doc_ids("q1", min_score=2)
        assert set(result) == {"d1"}

    def test_zero_score_excluded(self, qrels):
        """Score 0 is not relevant."""
        result = qrels.relevant_doc_ids("q1", min_score=1)
        assert "d3" not in result

    def test_empty_qrels_for_query(self, qrels):
        assert qrels.relevant_doc_ids("q3") == []

    def test_unknown_query_returns_empty(self, qrels):
        assert qrels.relevant_doc_ids("q_unknown") == []

    def test_single_relevant(self, qrels):
        result = qrels.relevant_doc_ids("q2")
        assert set(result) == {"d4"}


# ---------------------------------------------------------------------------
# _embed_documents_local prefix handling
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class TestEmbedDocumentsLocal:
    def make_engine(self, prefix="", model_name="", has_name_attr=True):
        from types import SimpleNamespace

        from rainrag.config import EmbeddingConfig

        cfg = EmbeddingConfig(prefix=prefix)
        cfg.model_name = model_name
        model = SimpleNamespace()
        if has_name_attr:
            model.name = model_name
        # Intentionally mock encode as a pass-through so tests can assert that
        # _embed_documents_local (or other pre-processing) transforms inputs
        # before encoding. The output of encode is then easy to inspect.
        model.encode = lambda texts, **kwargs: texts  # echo back
        eng = SimpleNamespace(config=SimpleNamespace(embedding=cfg), embedding_model=model)
        return eng

    def test_respects_custom_prefix(self):
        engine = self.make_engine(prefix="pre: ", model_name="other")
        from eval.datasets.beir_adapter import _embed_documents_local

        texts = ["foo", "bar"]
        out = _embed_documents_local(engine, texts, batch_size=1)
        assert out == ["pre: foo", "pre: bar"]

    def test_auto_prefix_for_e5_model(self):
        engine = self.make_engine(prefix="", model_name="intfloat/multilingual-e5-large")
        from eval.datasets.beir_adapter import _embed_documents_local

        texts = ["foo"]
        out = _embed_documents_local(engine, texts, batch_size=1)
        assert out == ["passage: foo"]

    def test_no_prefix_for_non_e5_with_empty_config(self):
        engine = self.make_engine(prefix="", model_name="bert-base-uncased")
        from eval.datasets.beir_adapter import _embed_documents_local

        texts = ["foo"]
        out = _embed_documents_local(engine, texts, batch_size=1)
        assert out == ["foo"]


# ---------------------------------------------------------------------------
# BEIRAdapter
# ---------------------------------------------------------------------------


class TestBEIRAdapter:
    def test_require_loaded_raises_before_load(self):
        from eval.datasets.beir_adapter import BEIRAdapter

        adapter = BEIRAdapter("scifact")
        with pytest.raises(RuntimeError, match="Call .load\\(\\)"):
            adapter._require_loaded()

    def test_summary_before_load(self):
        from eval.datasets.beir_adapter import BEIRAdapter

        adapter = BEIRAdapter("scifact")
        assert "not loaded" in adapter.summary()

    def test_collection_name_default(self):
        from eval.datasets.beir_adapter import BEIRAdapter

        assert BEIRAdapter("nfcorpus").collection_name == "beir_nfcorpus"

    def test_collection_name_custom_suffix(self):
        from eval.datasets.beir_adapter import BEIRAdapter

        assert (
            BEIRAdapter("scifact", collection_suffix="test_run").collection_name == "beir_test_run"
        )

    def test_to_eval_jsonl_skips_queries_without_relevant(self, tmp_path):
        """Queries that have no relevant docs in qrels should be skipped."""
        from eval.datasets.beir_adapter import BEIRAdapter, BEIRCorpus, BEIRQRels, BEIRQueries

        adapter = BEIRAdapter("test")
        adapter._corpus = BEIRCorpus(docs={"d1": {"title": "T", "text": "foo"}})
        adapter._queries = BEIRQueries(queries={"q1": "What is foo?", "q2": "Orphan query"})
        adapter._qrels = BEIRQRels(qrels={"q1": {"d1": 1}})  # q2 has no qrel

        out = tmp_path / "out.jsonl"
        records = adapter.to_eval_jsonl(str(out))

        assert len(records) == 1
        assert records[0]["query"] == "What is foo?"
        assert records[0]["relevant_doc_ids"] == ["d1"]
        assert records[0]["reference_answer"] == ""
        assert records[0]["beir_dataset"] == "test"

        # Verify file matches returned records
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

    def test_to_eval_jsonl_all_fields_present(self, tmp_path):
        from eval.datasets.beir_adapter import BEIRAdapter, BEIRCorpus, BEIRQRels, BEIRQueries

        adapter = BEIRAdapter("scifact")
        adapter._corpus = BEIRCorpus(docs={"d1": {"title": "T", "text": "foo"}})
        adapter._queries = BEIRQueries(queries={"q1": "question text"})
        adapter._qrels = BEIRQRels(qrels={"q1": {"d1": 2}})

        out = tmp_path / "out.jsonl"
        records = adapter.to_eval_jsonl(str(out))
        rec = records[0]

        required_keys = {
            "query_id",
            "query",
            "language",
            "relevant_doc_ids",
            "reference_answer",
            "category",
            "temporal",
            "beir_dataset",
            "beir_query_id",
            "beir_collection",
        }
        assert required_keys.issubset(rec.keys())
        assert rec["language"] == "en"
        assert rec["category"] == "factual"
        assert rec["temporal"] is False

    def test_to_eval_jsonl_min_relevance_filter(self, tmp_path):
        """min_relevance=2 should exclude docs with score=1."""
        from eval.datasets.beir_adapter import BEIRAdapter, BEIRCorpus, BEIRQRels, BEIRQueries

        adapter = BEIRAdapter("test")
        adapter._corpus = BEIRCorpus(
            docs={"d1": {"title": "", "text": "a"}, "d2": {"title": "", "text": "b"}}
        )
        adapter._queries = BEIRQueries(queries={"q1": "question"})
        adapter._qrels = BEIRQRels(qrels={"q1": {"d1": 2, "d2": 1}})

        out = tmp_path / "out.jsonl"
        records = adapter.to_eval_jsonl(str(out), min_relevance=2)
        assert records[0]["relevant_doc_ids"] == ["d1"]

    def test_eval_bm25_baseline_perfect(self):
        """With a tiny corpus where the query perfectly matches the doc, BM25 should rank it first."""
        from eval.datasets.beir_adapter import BEIRAdapter, BEIRCorpus, BEIRQRels, BEIRQueries

        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError:
            pytest.skip("rank-bm25 not installed")

        adapter = BEIRAdapter("test")
        adapter._corpus = BEIRCorpus(
            docs={
                "d1": {"title": "cat", "text": "the cat sat on the mat"},
                "d2": {"title": "dog", "text": "the dog ran in the park"},
            }
        )
        adapter._queries = BEIRQueries(queries={"q1": "cat mat"})
        adapter._qrels = BEIRQRels(qrels={"q1": {"d1": 1}})

        metrics = adapter.eval_bm25_baseline(top_k=2)
        # d1 should be ranked first because it matches "cat mat"
        assert metrics["recall@2"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_eval_bm25_baseline_returns_standard_keys(self):
        """eval_bm25_baseline must return the standard retrieval metric keys."""
        from eval.datasets.beir_adapter import BEIRAdapter, BEIRCorpus, BEIRQRels, BEIRQueries

        try:
            from rank_bm25 import BM25Okapi  # noqa: F401
        except ImportError:
            pytest.skip("rank-bm25 not installed")

        adapter = BEIRAdapter("test")
        adapter._corpus = BEIRCorpus(docs={"d1": {"title": "", "text": "alpha beta gamma"}})
        adapter._queries = BEIRQueries(queries={"q1": "alpha"})
        adapter._qrels = BEIRQRels(qrels={"q1": {"d1": 1}})

        metrics = adapter.eval_bm25_baseline(top_k=1)
        for key in ("recall@1", "mrr"):
            assert key in metrics, f"Missing key: {key}"
        assert 0.0 <= metrics["mrr"] <= 1.0


# ---------------------------------------------------------------------------
# _build_summary cost and carbon coverage
# ---------------------------------------------------------------------------


class TestBuildSummary:
    """Unit-tests for BaseExperiment._build_summary aggregation logic.

    _build_summary is a pure function of its arguments, so we can drive it
    directly without touching any I/O or external services.
    """

    @pytest.fixture
    def minimal_condition(self):
        return {"id": "01", "label": "vector_only"}

    @pytest.fixture
    def minimal_config(self, test_config):
        return test_config

    @staticmethod
    def _make_valid_result(
        query_id: str = "q1",
        elapsed_ms: float = 100.0,
        retrieval_score: float = 0.8,
        ndcg5: float = 0.75,
        mrr: float = 0.9,
        cost_total_usd: float = 0.001,
        rouge_l: float | None = 0.5,
    ) -> dict:
        """Build a synthetic per-query result dict."""
        r: dict = {
            "query_id": query_id,
            "query": "test query",
            "language": "en",
            "elapsed_ms": elapsed_ms,
            "recall@3": retrieval_score,
            "recall@5": retrieval_score,
            "recall@10": retrieval_score,
            "precision@3": retrieval_score,
            "precision@5": retrieval_score,
            "ndcg@5": ndcg5,
            "ndcg@10": ndcg5,
            "mrr": mrr,
            "map": mrr,
            "cost.input_tokens_est": 250.0,
            "cost.output_tokens_est": 50.0,
            "cost.embed_tokens_est": 10.0,
            "cost.llm_usd_est": cost_total_usd * 0.9,
            "cost.embed_usd_est": cost_total_usd * 0.1,
            "cost.total_usd_est": cost_total_usd,
        }
        if rouge_l is not None:
            r["rouge_l"] = rouge_l
        return r

    def test_cost_metrics_in_summary(self, minimal_condition, minimal_config):
        from eval.experiments.base import BaseExperiment

        # Patch abstract method so we can instantiate
        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        results = [self._make_valid_result("q1", cost_total_usd=0.002)]
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results
        )

        assert "cost.total_usd_est" in summary["metrics"]
        assert summary["metrics"]["cost.total_usd_est"] == pytest.approx(0.002)
        assert "cost.aggregate_usd_est" in summary["metrics"]
        assert summary["metrics"]["cost.aggregate_usd_est"] == pytest.approx(0.002)
        assert "cost.mean_usd_est_per_query" in summary["metrics"]
        assert summary["metrics"]["cost.mean_usd_est_per_query"] == pytest.approx(0.002)

    def test_latency_percentiles(self, minimal_condition, minimal_config):
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        # Two results with known latencies so we can predict p50
        results = [
            self._make_valid_result("q1", elapsed_ms=100.0),
            self._make_valid_result("q2", elapsed_ms=200.0),
        ]
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results
        )

        assert "latency_p50_ms" in summary["metrics"]
        assert "latency_p95_ms" in summary["metrics"]
        assert "latency_mean_ms" in summary["metrics"]
        assert summary["metrics"]["latency_mean_ms"] == pytest.approx(150.0)

    def test_num_queries_and_errors(self, minimal_condition, minimal_config):
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        results = [
            self._make_valid_result("q1"),
            self._make_valid_result("q2"),
            {"query_id": "q3", "query": "bad", "language": "en", "error": "timeout"},
        ]
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results
        )

        assert summary["metrics"]["num_queries"] == 2.0
        assert summary["metrics"]["num_errors"] == 1.0

    def test_carbon_metrics_included_when_available(self, minimal_condition, minimal_config):
        from eval.experiments.base import BaseExperiment
        from eval.metrics.carbon import CarbonResult

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        results = [self._make_valid_result("q1")]

        carbon = CarbonResult(
            available=True,
            emissions_kg=0.0042,
            energy_kwh=0.01,
            duration_s=12.5,
        )
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results, carbon=carbon
        )

        assert "carbon.emissions_kg" in summary["metrics"]
        assert summary["metrics"]["carbon.emissions_kg"] == pytest.approx(0.0042)
        assert "carbon.emissions_g" in summary["metrics"]
        assert summary["metrics"]["carbon.energy_kwh"] == pytest.approx(0.01)
        assert summary["metrics"]["carbon.duration_s"] == pytest.approx(12.5)

    def test_carbon_none_does_not_crash(self, minimal_condition, minimal_config):
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        results = [self._make_valid_result("q1")]
        # carbon=None is the default — must not raise
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results, carbon=None
        )
        assert "num_queries" in summary["metrics"]

    def test_rouge_l_mean_in_summary(self, minimal_condition, minimal_config):
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        results = [
            self._make_valid_result("q1", rouge_l=0.4),
            self._make_valid_result("q2", rouge_l=0.6),
        ]
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results
        )
        assert "rouge_l" in summary["metrics"]
        assert summary["metrics"]["rouge_l"] == pytest.approx(0.5)

    def test_percentile_keys_in_summary(self, minimal_condition, minimal_config):
        """aggregate_metrics emits _p10/_p25 siblings; they must flow through _build_summary."""
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        # Three queries so the percentile is non-trivial
        results = [
            self._make_valid_result("q1", ndcg5=0.2),
            self._make_valid_result("q2", ndcg5=0.5),
            self._make_valid_result("q3", ndcg5=0.8),
        ]
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results
        )

        assert "ndcg@5_p10" in summary["metrics"], "ndcg@5_p10 must be present"
        assert "ndcg@5_p25" in summary["metrics"], "ndcg@5_p25 must be present"
        # p10 ≤ mean ≤ p90 for any real distribution
        assert summary["metrics"]["ndcg@5_p10"] <= summary["metrics"]["ndcg@5"]
        assert summary["metrics"]["ndcg@5_p25"] <= summary["metrics"]["ndcg@5"]

    def test_intent_coverage_flows_through_summary(self, minimal_condition, minimal_config):
        """intent_coverage@k added by _run_dataset must flow through _build_summary."""
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        exp = _Exp(dataset_path=None)
        # Inject intent_coverage keys directly (as _run_dataset would)
        results = [
            {**self._make_valid_result("q1"), "intent_coverage@3": 1.0, "intent_coverage@5": 1.0},
            {**self._make_valid_result("q2"), "intent_coverage@3": 0.5, "intent_coverage@5": 0.5},
        ]
        summary = exp._build_summary(
            minimal_condition, minimal_config, top_k=5, all_results=results
        )

        assert "intent_coverage@5" in summary["metrics"], "intent_coverage@5 must be aggregated"
        assert summary["metrics"]["intent_coverage@5"] == pytest.approx(0.75)
        assert "intent_coverage@3" in summary["metrics"]


# ---------------------------------------------------------------------------
# _print_result
# ---------------------------------------------------------------------------


class TestPrintResult:
    """Unit tests for BaseExperiment._print_result console output."""

    def _result(self, metrics: dict) -> dict:
        return {
            "condition_id": "test-01",
            "condition_label": "test_label",
            "top_k": 5,
            "metrics": metrics,
        }

    def _capture(self, metrics: dict, capsys) -> str:
        from eval.experiments.base import BaseExperiment

        BaseExperiment._print_result(self._result(metrics))
        return capsys.readouterr().out

    def test_contains_condition_id(self, capsys):
        out = self._capture({"recall@5": 0.5, "ndcg@5": 0.4, "mrr": 0.6, "rouge_l": 0.3}, capsys)
        assert "test-01" in out

    def test_contains_condition_label(self, capsys):
        out = self._capture({"recall@5": 0.5, "ndcg@5": 0.4, "mrr": 0.6, "rouge_l": 0.3}, capsys)
        assert "test_label" in out

    def test_contains_recall(self, capsys):
        out = self._capture({"recall@5": 0.750, "ndcg@5": 0.4, "mrr": 0.6, "rouge_l": 0.3}, capsys)
        assert "recall@5=0.750" in out

    def test_ndcg_shows_p10_when_present(self, capsys):
        out = self._capture(
            {"recall@5": 0.5, "ndcg@5": 0.6, "ndcg@5_p10": 0.3, "mrr": 0.5, "rouge_l": 0.4},
            capsys,
        )
        assert "p10=0.300" in out

    def test_ndcg_hides_p10_when_absent(self, capsys):
        out = self._capture({"recall@5": 0.5, "ndcg@5": 0.6, "mrr": 0.5, "rouge_l": 0.4}, capsys)
        assert "p10=" not in out

    def test_intent_coverage_shown_when_present(self, capsys):
        out = self._capture(
            {
                "recall@5": 0.5,
                "ndcg@5": 0.6,
                "mrr": 0.5,
                "rouge_l": 0.4,
                "intent_coverage@5": 0.875,
            },
            capsys,
        )
        assert "ic@5=0.875" in out

    def test_intent_coverage_hidden_when_absent(self, capsys):
        out = self._capture({"recall@5": 0.5, "ndcg@5": 0.6, "mrr": 0.5, "rouge_l": 0.4}, capsys)
        assert "ic@5" not in out

    def test_cost_shown_when_present(self, capsys):
        out = self._capture(
            {
                "recall@5": 0.5,
                "ndcg@5": 0.4,
                "mrr": 0.6,
                "rouge_l": 0.3,
                "cost.mean_usd_est_per_query": 0.0012,
            },
            capsys,
        )
        assert "$0.0012/q" in out

    def test_cost_na_when_absent(self, capsys):
        out = self._capture({"recall@5": 0.5, "ndcg@5": 0.4, "mrr": 0.6, "rouge_l": 0.3}, capsys)
        assert "cost=n/a" in out

    def test_latency_shown_when_present(self, capsys):
        out = self._capture(
            {"recall@5": 0.5, "ndcg@5": 0.4, "mrr": 0.6, "rouge_l": 0.3, "latency_p50_ms": 123.4},
            capsys,
        )
        assert "123ms" in out

    def test_single_line_output(self, capsys):
        out = self._capture({"recall@5": 0.5, "ndcg@5": 0.4, "mrr": 0.6, "rouge_l": 0.3}, capsys)
        assert out.count("\n") == 1


# ---------------------------------------------------------------------------
# results_to_csv (fallback plain-CSV path)
# ---------------------------------------------------------------------------


class TestResultsToCsv:
    """Tests for BaseExperiment.results_to_csv using the pandas-free fallback."""

    @staticmethod
    def _make_results(n: int = 2) -> list[dict]:
        return [
            {
                "condition_id": f"cond-{i}",
                "condition_label": f"label-{i}",
                "top_k": 5,
                "metrics": {
                    "recall@5": 0.5 + i * 0.1,
                    "ndcg@5": 0.4 + i * 0.1,
                    "mrr": 0.6,
                },
            }
            for i in range(n)
        ]

    @staticmethod
    def _exp():
        from eval.experiments.base import BaseExperiment

        class _Exp(BaseExperiment):
            def conditions(self):
                return []

        return _Exp(dataset_path=None)

    def test_csv_created(self, tmp_path):
        path = str(tmp_path / "out.csv")
        self._exp().results_to_csv(self._make_results(), path)
        assert Path(path).exists()

    def test_csv_has_header(self, tmp_path):
        path = str(tmp_path / "out.csv")
        self._exp().results_to_csv(self._make_results(), path)
        header = Path(path).read_text().splitlines()[0]
        assert "condition_id" in header
        assert "label" in header
        assert "top_k" in header

    def test_csv_metric_columns_present(self, tmp_path):
        path = str(tmp_path / "out.csv")
        self._exp().results_to_csv(self._make_results(), path)
        header = Path(path).read_text().splitlines()[0]
        assert "recall@5" in header
        assert "ndcg@5" in header

    def test_csv_row_count(self, tmp_path):
        path = str(tmp_path / "out.csv")
        self._exp().results_to_csv(self._make_results(n=3), path)
        lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
        # 1 header + 3 data rows
        assert len(lines) == 4

    def test_csv_values_correct(self, tmp_path):
        path = str(tmp_path / "out.csv")
        self._exp().results_to_csv(self._make_results(n=1), path)
        content = Path(path).read_text()
        assert "cond-0" in content
        assert "label-0" in content

    def test_empty_results_no_crash(self, tmp_path):
        path = str(tmp_path / "out.csv")
        # Should return without creating a file or raising
        self._exp().results_to_csv([], path)
        assert not Path(path).exists()
