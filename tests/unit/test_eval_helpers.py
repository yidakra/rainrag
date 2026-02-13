"""Unit tests for eval helper utilities.

Covers:
- eval.experiments.base.apply_overrides()
- eval.datasets.beir_adapter data containers (BEIRCorpus, BEIRQRels, BEIRAdapter)

No external services required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


class TestApplyOverrides:
    @pytest.fixture
    def base_config(self, test_config):
        """Use the shared test_config fixture from conftest.py."""
        return test_config

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

    def test_empty_overrides_returns_equivalent_config(self, base_config):
        from eval.experiments.base import apply_overrides

        result = apply_overrides(base_config, {})
        # Same values, different object
        assert result is not base_config
        assert result.llm.provider == base_config.llm.provider
        assert result.hybrid_search.enabled == base_config.hybrid_search.enabled

    def test_boolean_false_override(self, base_config):
        from eval.experiments.base import apply_overrides

        # Start with enabled=True, override to False
        base_config.reranker.enabled = True
        result = apply_overrides(base_config, {"reranker.enabled": False})
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

        corpus = BEIRCorpus(docs={"d1": {"title": "T", "text": "foo"}, "d2": {"title": "U", "text": "bar"}})
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
        assert result == ["d1"]

    def test_zero_score_excluded(self, qrels):
        """Score 0 is not relevant."""
        result = qrels.relevant_doc_ids("q1", min_score=1)
        assert "d3" not in result

    def test_empty_qrels_for_query(self, qrels):
        assert qrels.relevant_doc_ids("q3") == []

    def test_unknown_query_returns_empty(self, qrels):
        assert qrels.relevant_doc_ids("q_unknown") == []

    def test_single_relevant(self, qrels):
        assert qrels.relevant_doc_ids("q2") == ["d4"]


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

        assert BEIRAdapter("scifact", collection_suffix="test_run").collection_name == "beir_test_run"

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
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
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
            "query_id", "query", "language", "relevant_doc_ids",
            "reference_answer", "category", "temporal",
            "beir_dataset", "beir_query_id", "beir_collection",
        }
        assert required_keys.issubset(rec.keys())
        assert rec["language"] == "en"
        assert rec["category"] == "factual"
        assert rec["temporal"] is False

    def test_to_eval_jsonl_min_relevance_filter(self, tmp_path):
        """min_relevance=2 should exclude docs with score=1."""
        from eval.datasets.beir_adapter import BEIRAdapter, BEIRCorpus, BEIRQRels, BEIRQueries

        adapter = BEIRAdapter("test")
        adapter._corpus = BEIRCorpus(docs={"d1": {"title": "", "text": "a"}, "d2": {"title": "", "text": "b"}})
        adapter._queries = BEIRQueries(queries={"q1": "question"})
        adapter._qrels = BEIRQRels(qrels={"q1": {"d1": 2, "d2": 1}})

        out = tmp_path / "out.jsonl"
        records = adapter.to_eval_jsonl(str(out), min_relevance=2)
        assert records[0]["relevant_doc_ids"] == ["d1"]

    def test_eval_bm25_baseline_perfect(self, tmp_path):
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
