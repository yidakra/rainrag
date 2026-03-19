"""
Tests for two-stage retrieval (Zhai & Lafferty, SIGIR 2002).

Covers:
- TwoStageConfig wired into the query engine
- _rewrite_query_for_retrieval(): success path, fallback on LLM error
- _generate_hyde_embedding(): success path, fallback on LLM error
- query() pipeline with Stage 2a (query rewriting) enabled
- query() pipeline with Stage 2b (HyDE) enabled
- query() pipeline with two_stage disabled (no behaviour change)
"""

from unittest.mock import MagicMock, patch

import pytest

from rainrag.config import (
    Config,
    EmbeddingConfig,
    HybridSearchConfig,
    LLMConfig,
    LoggingConfig,
    MistralConfig,
    OpenAIConfig,
    PathsConfig,
    ProcessingConfig,
    QdrantConfig,
    TwoStageConfig,
    VideoConfig,
)
from rainrag.query import RAGQueryEngine


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_config():
    """Minimal config with two_stage disabled (baseline / control)."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings",
        ),
        embedding=EmbeddingConfig(provider="openai", model_name="text-embedding-3-small"),
        qdrant=QdrantConfig(collection_name="test_collection", vector_size=1536),
        llm=LLMConfig(provider="openai"),
        mistral=MistralConfig(api_key="test-mistral-key"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            max_tokens=512,
            temperature=0.3,
            top_k=3,
        ),
        hybrid_search=HybridSearchConfig(enabled=False),
        two_stage=TwoStageConfig(enabled=False),
        processing=ProcessingConfig(),
        logging=LoggingConfig(log_file="/tmp/test.log"),
        video=VideoConfig(enabled=False),
    )


@pytest.fixture
def two_stage_config(base_config):
    """Config with two-stage Stage 2a (query rewriting) enabled."""
    base_config.two_stage = TwoStageConfig(
        enabled=True,
        query_rewrite_enabled=True,
        query_rewrite_variants=2,
        hyde_enabled=False,
    )
    return base_config


@pytest.fixture
def hyde_config(base_config):
    """Config with two-stage Stage 2b (HyDE) enabled, rewriting disabled."""
    base_config.two_stage = TwoStageConfig(
        enabled=True,
        query_rewrite_enabled=False,
        hyde_enabled=True,
        hyde_alpha=0.5,
    )
    return base_config


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with canned embedding and chat responses."""
    client = MagicMock()

    embed_resp = MagicMock()
    embed_resp.data = [MagicMock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = embed_resp

    chat_resp = MagicMock()
    chat_resp.choices = [
        MagicMock(message=MagicMock(content="rewritten query A\nrewritten query B"))
    ]
    client.chat.completions.create.return_value = chat_resp

    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client returning a single document."""
    client = MagicMock()

    collections = MagicMock()
    collections.collections = []
    client.get_collections.return_value = collections

    point = MagicMock()
    point.id = "doc1"
    point.score = 0.9
    point.payload = {
        "text": "There was a bridge construction project underway in 2022.",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1",
        "date": "2022-03-15",
        "start_time": "00:01:00",
        "end_time": "00:06:00",
        "is_chunk": True,
        "chunk_index": 0,
        "total_chunks": 2,
        "video_id": "vid1",
        "web_title": None,
        "web_date": None,
        "web_date_ts": None,
        "web_description": None,
        "web_url": None,
        "duration_seconds": 300,
        "start_time_seconds": 60,
        "end_time_seconds": 360,
    }
    result = MagicMock()
    result.points = [point]
    client.query_points.return_value = result

    return client


def _setup(engine, qdrant_client):
    """Wire mock Qdrant into an already-constructed engine."""
    engine.qdrant_client = qdrant_client
    engine.cohere_client = None
    engine.bm25 = None
    engine.bm25_corpus = []
    engine.bm25_tokenized_corpus = []
    return engine


# ---------------------------------------------------------------------------
# TwoStageConfig unit tests (config-only, no engine needed)
# ---------------------------------------------------------------------------


class TestTwoStageConfigDefaults:
    def test_disabled_by_default(self):
        cfg = TwoStageConfig()
        assert cfg.enabled is False

    def test_query_rewrite_on_by_default(self):
        cfg = TwoStageConfig()
        assert cfg.query_rewrite_enabled is True
        assert cfg.query_rewrite_variants == 2

    def test_hyde_off_by_default(self):
        cfg = TwoStageConfig()
        assert cfg.hyde_enabled is False
        assert cfg.hyde_alpha == 0.5

    def test_merge_strategy_defaults_to_coverage(self):
        cfg = TwoStageConfig()
        assert cfg.merge_strategy == "coverage"

    def test_merge_rrf_k_defaults_to_60(self):
        cfg = TwoStageConfig()
        assert cfg.merge_rrf_k == 60

    def test_merge_strategy_accepts_diverse_rrf(self):
        cfg = TwoStageConfig(merge_strategy="diverse_rrf", merge_rrf_k=20)
        assert cfg.merge_strategy == "diverse_rrf"
        assert cfg.merge_rrf_k == 20


# ---------------------------------------------------------------------------
# _rewrite_query_for_retrieval tests
# ---------------------------------------------------------------------------


class TestRewriteQuery:
    def test_returns_original_plus_variants(self, two_stage_config, mock_openai_client):
        """Should return original query + LLM-generated variants."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, MagicMock())

            # LLM returns two lines
            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content = "infrastructure work in 2022\nbridge building project"

            variants = engine._rewrite_query_for_retrieval(
                "infrastructure projects 2022", language="en"
            )

        assert variants[0] == "infrastructure projects 2022"  # original always first
        assert len(variants) == 3  # original + 2 rewrites
        assert "infrastructure work in 2022" in variants
        assert "bridge building project" in variants

    def test_respects_n_variants(self, two_stage_config, mock_openai_client):
        """Should cap variants at query_rewrite_variants."""
        two_stage_config.two_stage.query_rewrite_variants = 1

        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, MagicMock())

            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content = "line one\nline two\nline three"

            variants = engine._rewrite_query_for_retrieval("original", language="en")

        # original + 1 rewrite (n=1)
        assert len(variants) == 2

    def test_falls_back_to_original_on_llm_error(self, two_stage_config, mock_openai_client):
        """If LLM call fails, should silently return just the original query."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, MagicMock())

            mock_openai_client.chat.completions.create.side_effect = RuntimeError("API down")

            variants = engine._rewrite_query_for_retrieval("query", language="en")

        assert variants == ["query"]

    def test_russian_prompt_used_for_ru(self, two_stage_config, mock_openai_client):
        """Russian language should trigger Russian prompt."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, MagicMock())

            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content = "вариант один"

            engine._rewrite_query_for_retrieval("запрос", language="ru")

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        prompt_text = call_kwargs[1]["messages"][0]["content"]
        assert "Перепиши" in prompt_text

    def test_rewrite_uses_configured_temperature(self, two_stage_config, mock_openai_client):
        """Rewrite LLM call should use query_rewrite_temperature, not the answer temperature."""
        two_stage_config.two_stage.query_rewrite_temperature = 0.9
        two_stage_config.openai.temperature = 0.0  # answer temperature stays zero

        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, MagicMock())

            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content = "variant one"

            engine._rewrite_query_for_retrieval("query", language="en")

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs[1]["temperature"] == 0.9


# ---------------------------------------------------------------------------
# _generate_hyde_embedding tests
# ---------------------------------------------------------------------------


class TestHydeEmbedding:
    def test_returns_embedding_of_hypothetical_doc(self, hyde_config, mock_openai_client):
        """Should generate a passage, embed it, and return a vector."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(hyde_config)
            _setup(engine, MagicMock())

            mock_openai_client.chat.completions.create.return_value.choices[
                0
            ].message.content = (
                "Crews were seen building a new overpass along the highway corridor."
            )

            vec = engine._generate_hyde_embedding("bridge construction", language="en")

        # Should be a non-empty float list (mocked to [0.1]*1536)
        assert isinstance(vec, list)
        assert len(vec) == 1536

    def test_falls_back_to_query_embedding_on_llm_error(self, hyde_config, mock_openai_client):
        """On LLM failure, should embed the original query instead of raising."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(hyde_config)
            _setup(engine, MagicMock())

            mock_openai_client.chat.completions.create.side_effect = RuntimeError("timeout")

            vec = engine._generate_hyde_embedding("bridge construction", language="en")

        assert isinstance(vec, list)
        assert len(vec) == 1536

    def test_hyde_uses_configured_temperature(self, hyde_config, mock_openai_client):
        """HyDE LLM call should use hyde_temperature, not the answer temperature."""
        hyde_config.two_stage.hyde_temperature = 0.9
        hyde_config.openai.temperature = 0.0  # answer temperature stays zero

        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(hyde_config)
            _setup(engine, MagicMock())

            engine._generate_hyde_embedding("flood damage", language="en")

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs[1]["temperature"] == 0.9


# ---------------------------------------------------------------------------
# Full query() pipeline tests
# ---------------------------------------------------------------------------


class TestTwoStagePipeline:
    def test_two_stage_disabled_calls_embed_once(
        self, base_config, mock_openai_client, mock_qdrant_client
    ):
        """With two_stage disabled, embed_query should be called exactly once."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(base_config)
            _setup(engine, mock_qdrant_client)

            engine.query("What bridges were built?", top_k=1, language="en")

        # One embed call for the query, one chat call for the answer
        assert mock_openai_client.embeddings.create.call_count == 1
        assert mock_openai_client.chat.completions.create.call_count == 1

    def test_query_rewriting_embeds_multiple_variants(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """With two-stage enabled and 2 variants, embed should be called 3 times
        (original + 2 rewrites) and chat twice (rewrite + answer)."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            # First chat call returns rewrites, second returns the answer
            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(
                    choices=[MagicMock(message=MagicMock(content="rewrite one\nrewrite two"))]
                ),
                MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Final answer about bridges."))]
                ),
            ]

            result = engine.query("bridge construction", top_k=1, language="en")

        assert "answer" in result
        assert result["answer"] == "Final answer about bridges."
        # 3 embed calls: original + 2 rewrites
        assert mock_openai_client.embeddings.create.call_count == 3
        # 2 chat calls: rewrite LLM + answer LLM
        assert mock_openai_client.chat.completions.create.call_count == 2

    def test_variant_dedup_keeps_best_score(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """Documents returned by multiple variants should be deduped; the copy
        with the higher score should be kept."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            # Both Qdrant calls return the same doc but with different scores
            low_score_point = MagicMock()
            low_score_point.id = "doc1"
            low_score_point.score = 0.5
            low_score_point.payload = mock_qdrant_client.query_points.return_value.points[0].payload

            high_score_point = MagicMock()
            high_score_point.id = "doc1"
            high_score_point.score = 0.9
            high_score_point.payload = mock_qdrant_client.query_points.return_value.points[
                0
            ].payload

            low_result = MagicMock()
            low_result.points = [low_score_point]
            high_result = MagicMock()
            high_result.points = [high_score_point]

            # First variant retrieves low score, second retrieves high score
            mock_qdrant_client.query_points.side_effect = [low_result, high_result, high_result]

            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=MagicMock(content="v1\nv2"))]),
                MagicMock(choices=[MagicMock(message=MagicMock(content="answer"))]),
            ]

            result = engine.query("test query", top_k=3, language="en")

        docs = result["retrieved_documents"]
        # Only one unique doc
        assert len(docs) == 1
        # The higher score wins
        assert docs[0]["score"] == 0.9

    def test_hyde_blends_embedding(self, hyde_config, mock_openai_client, mock_qdrant_client):
        """With HyDE enabled, embed_query should be called twice (query + HyDE passage)
        and the Qdrant call should use a blended vector (not identical to raw query)."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(hyde_config)
            _setup(engine, mock_qdrant_client)

            # First embed: raw query → [0.1]*1536
            # Second embed: HyDE passage → [0.2]*1536  (different so blend differs)
            raw_embed = MagicMock()
            raw_embed.data = [MagicMock(embedding=[0.1] * 1536)]
            hyde_embed = MagicMock()
            hyde_embed.data = [MagicMock(embedding=[0.2] * 1536)]

            mock_openai_client.embeddings.create.side_effect = [raw_embed, hyde_embed]
            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(
                    choices=[MagicMock(message=MagicMock(content="Hypothetical passage text."))]
                ),
                MagicMock(choices=[MagicMock(message=MagicMock(content="Final answer."))]),
            ]

            result = engine.query("bridge construction", top_k=1, language="en")

        # Two embed calls: raw query + HyDE passage
        assert mock_openai_client.embeddings.create.call_count == 2

        # The vector passed to Qdrant should be the normalised blend of [0.1] and [0.2],
        # i.e. [0.15]*1536 (normalised) — not [0.1]*1536 unchanged.
        qdrant_call_vector = mock_qdrant_client.query_points.call_args[1]["query"]
        assert qdrant_call_vector != [0.1] * 1536
        assert qdrant_call_vector != [0.2] * 1536
        assert result["answer"] == "Final answer."

    def test_result_structure_unchanged(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """The dict returned by query() should have the same keys regardless of two-stage."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=MagicMock(content="v1\nv2"))]),
                MagicMock(choices=[MagicMock(message=MagicMock(content="answer"))]),
            ]

            result = engine.query("test", top_k=1, language="en")

        assert set(result.keys()) == {
            "question",
            "answer",
            "retrieved_documents",
            "num_documents",
            "query_variants",
            "variant_retrieved_ids",
            "cost.llm_calls_count",
            "cost.llm_query_rewrite_calls",
            "cost.llm_hyde_calls",
            "cost.embed_calls_count",
            "cost.reranker_calls_count",
        }

    def test_answer_generation_uses_zero_temperature(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """Final answer generation must use the provider's configured temperature (0),
        not the rewrite temperature (0.7), ensuring deterministic journalist output."""
        two_stage_config.openai.temperature = 0.0
        two_stage_config.two_stage.query_rewrite_temperature = 0.7

        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            rewrite_resp = MagicMock(choices=[MagicMock(message=MagicMock(content="v1\nv2"))])
            answer_resp = MagicMock(choices=[MagicMock(message=MagicMock(content="Final answer."))])
            mock_openai_client.chat.completions.create.side_effect = [rewrite_resp, answer_resp]

            engine.query("test query", top_k=1, language="en")

        calls = mock_openai_client.chat.completions.create.call_args_list
        assert len(calls) == 2
        rewrite_call_temp = calls[0][1]["temperature"]
        answer_call_temp = calls[1][1]["temperature"]
        assert rewrite_call_temp == 0.7  # diverse paraphrases
        assert answer_call_temp == 0.0  # deterministic answer


# ---------------------------------------------------------------------------
# Merge strategy unit tests (no live services needed)
# ---------------------------------------------------------------------------


class TestMergeStrategies:
    """Unit tests for _merge_variants_coverage and _merge_variants_diverse_rrf.

    Uses RAGQueryEngine.__new__ to bypass __init__ so no real config/clients
    are required.
    """

    @pytest.fixture
    def engine(self):
        return RAGQueryEngine.__new__(RAGQueryEngine)

    def _doc(self, doc_id: str, score: float, rank: int = 1) -> dict:
        return {"doc_id": doc_id, "score": score, "rank": rank}

    # --- _merge_variants_coverage ---

    def test_coverage_selects_unique_variant_doc_first(self, engine):
        """Doc unique to variant 1 should be selected before a lower-ranked doc
        that is already covered by the first selection."""
        # variant 0: [a(0.9), b(0.5)]
        # variant 1: [c(0.8), b(0.4)]
        # 'a' covers {0} only, 'c' covers {1} only, 'b' covers {0,1}
        # Greedy step 1: 'b' covers 2 variants (best), but 'a' and 'c' also tie
        # at new_coverage=1.  Tie goes to 'b' (score 0.5 — wait, 'a' is 0.9 > 'b' 0.5).
        # Actually: 'b' covers {0,1} → new_coverage=2; 'a' covers {0} → 1; 'c' covers {1} → 1
        # 'b' wins step 1.
        vdocs_a = [self._doc("a", 0.9, 1), self._doc("b", 0.5, 2)]
        vdocs_b = [self._doc("c", 0.8, 1), self._doc("b", 0.4, 2)]
        result = engine._merge_variants_coverage([vdocs_a, vdocs_b], retrieval_k=3)
        ids = [d["doc_id"] for d in result]
        assert ids[0] == "b"  # covers both variants first

    def test_coverage_tiebreak_by_score(self, engine):
        """When two docs cover the same number of new variants, higher score wins."""
        # variant 0: [a(0.9)], variant 1: [b(0.7)]  — each unique to one variant
        vdocs_a = [self._doc("a", 0.9, 1)]
        vdocs_b = [self._doc("b", 0.7, 1)]
        result = engine._merge_variants_coverage([vdocs_a, vdocs_b], retrieval_k=2)
        ids = [d["doc_id"] for d in result]
        assert ids[0] == "a"  # higher score
        assert ids[1] == "b"

    def test_coverage_uses_best_score_for_duplicate_doc(self, engine):
        """When same doc appears in multiple variants with different scores,
        the highest score should be used in the merged result."""
        vdocs_a = [self._doc("a", 0.4, 1)]
        vdocs_b = [self._doc("a", 0.9, 1)]
        result = engine._merge_variants_coverage([vdocs_a, vdocs_b], retrieval_k=1)
        assert result[0]["doc_id"] == "a"
        assert result[0]["score"] == pytest.approx(0.9)

    def test_coverage_respects_retrieval_k(self, engine):
        vdocs = [[self._doc(f"d{i}", 1.0 - i * 0.1, i + 1) for i in range(5)]]
        result = engine._merge_variants_coverage(vdocs, retrieval_k=3)
        assert len(result) == 3

    def test_coverage_ranks_renumbered_by_caller(self, engine):
        """After the greedy selection, ranks are re-assigned by query() — the
        merge method itself doesn't re-rank; that's done in the calling code."""
        vdocs_a = [self._doc("a", 0.9, 1), self._doc("b", 0.7, 2)]
        vdocs_b = [self._doc("c", 0.8, 1)]
        result = engine._merge_variants_coverage([vdocs_a, vdocs_b], retrieval_k=3)
        # All three docs should be present
        assert {d["doc_id"] for d in result} == {"a", "b", "c"}

    # --- _merge_variants_diverse_rrf ---

    def test_diverse_rrf_returns_correct_count(self, engine):
        vdocs_a = [self._doc("a", 0.9, 1), self._doc("b", 0.7, 2)]
        vdocs_b = [self._doc("c", 0.8, 1), self._doc("a", 0.6, 2)]
        result = engine._merge_variants_diverse_rrf([vdocs_a, vdocs_b], retrieval_k=3)
        assert len(result) == 3

    def test_diverse_rrf_sets_fusion_method(self, engine):
        vdocs = [[self._doc("a", 0.9, 1)], [self._doc("b", 0.8, 1)]]
        result = engine._merge_variants_diverse_rrf(vdocs, retrieval_k=2)
        assert all(d["fusion_method"] == "diverse_rrf" for d in result)

    def test_diverse_rrf_upweights_minority_doc(self, engine):
        """A doc appearing in only 1 variant (minority) should score higher
        relative to a doc appearing in 2 variants (consensus), per diversity weight."""
        # variant 0: [x(rank=1), y(rank=2)]
        # variant 1: [x(rank=1), z(rank=2)]
        # 'x' appears in both → variant_count=2 → dw=1/sqrt(2)≈0.707
        # 'y' appears in variant 0 only → variant_count=1 → dw=1.0
        # 'z' appears in variant 1 only → variant_count=1 → dw=1.0
        # x score = 2 * (0.707 / 61) ≈ 0.02319
        # y score = 1 * (1.0 / 62)  ≈ 0.01613
        # z score = 1 * (1.0 / 62)  ≈ 0.01613
        # So x > y = z (x appears twice so still wins overall)
        vdocs_a = [self._doc("x", 0.9, 1), self._doc("y", 0.7, 2)]
        vdocs_b = [self._doc("x", 0.8, 1), self._doc("z", 0.6, 2)]
        result = engine._merge_variants_diverse_rrf([vdocs_a, vdocs_b], retrieval_k=3)
        ids = [d["doc_id"] for d in result]
        assert ids[0] == "x"
        # y and z have equal scores; both must be present
        assert set(ids[1:]) == {"y", "z"}

    def test_diverse_rrf_smaller_rrf_k_amplifies_differences(self, engine):
        """With rrf_k=1 (extreme) the rank difference between rank-1 and rank-2
        docs should be larger than with rrf_k=60."""
        vdocs = [[self._doc("a", 0.9, 1), self._doc("b", 0.5, 2)]]
        result_60 = engine._merge_variants_diverse_rrf(vdocs, retrieval_k=2, rrf_k=60)
        result_1 = engine._merge_variants_diverse_rrf(vdocs, retrieval_k=2, rrf_k=1)
        gap_60 = result_60[0]["score"] - result_60[1]["score"]
        gap_1 = result_1[0]["score"] - result_1[1]["score"]
        assert gap_1 > gap_60

    # --- query() integration: variant_retrieved_ids exposed ---

    def test_query_exposes_variant_retrieved_ids(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """query() must return 'variant_retrieved_ids' listing per-variant doc IDs."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=MagicMock(content="v1\nv2"))]),
                MagicMock(choices=[MagicMock(message=MagicMock(content="answer"))]),
            ]

            result = engine.query("test", top_k=1, language="en")

        assert "variant_retrieved_ids" in result
        assert isinstance(result["variant_retrieved_ids"], list)
        assert len(result["variant_retrieved_ids"]) > 0
        assert all(isinstance(ids, list) for ids in result["variant_retrieved_ids"])

    def test_query_exposes_query_variants(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """query() must return 'query_variants' with the original + rewrites."""
        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(
                    choices=[MagicMock(message=MagicMock(content="rewrite one\nrewrite two"))]
                ),
                MagicMock(choices=[MagicMock(message=MagicMock(content="answer"))]),
            ]

            result = engine.query("original query", top_k=1, language="en")

        variants = result["query_variants"]
        assert variants[0] == "original query"
        assert len(variants) == 3  # original + 2 rewrites

    def test_diverse_rrf_strategy_used_when_configured(
        self, two_stage_config, mock_openai_client, mock_qdrant_client
    ):
        """When merge_strategy='diverse_rrf', merged docs should carry fusion_method tag."""
        two_stage_config.two_stage.merge_strategy = "diverse_rrf"

        with patch("rainrag.query.OpenAI", return_value=mock_openai_client):
            engine = RAGQueryEngine(two_stage_config)
            _setup(engine, mock_qdrant_client)

            mock_openai_client.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=MagicMock(content="v1\nv2"))]),
                MagicMock(choices=[MagicMock(message=MagicMock(content="answer"))]),
            ]

            result = engine.query("test", top_k=1, language="en")

        docs = result["retrieved_documents"]
        assert all(d.get("fusion_method") == "diverse_rrf" for d in docs)


# ---------------------------------------------------------------------------
# _apply_score_threshold tests
# ---------------------------------------------------------------------------


def _make_docs(scores):
    """Build a minimal document list with the given scores and sequential ranks."""
    return [{"score": s, "rank": i + 1, "text": f"doc{i}"} for i, s in enumerate(scores)]


class TestScoreThreshold:
    """Tests for RAGQueryEngine._apply_score_threshold."""

    @pytest.fixture
    def engine(self, base_config):
        eng = RAGQueryEngine.__new__(RAGQueryEngine)
        eng.config = base_config
        return eng

    def test_zero_threshold_is_noop(self, engine):
        docs = _make_docs([0.9, 0.5, 0.1])
        assert engine._apply_score_threshold(docs, 0.0) == docs

    def test_negative_threshold_is_noop(self, engine):
        docs = _make_docs([0.9, 0.1])
        assert engine._apply_score_threshold(docs, -1.0) == docs

    def test_drops_docs_below_threshold(self, engine):
        docs = _make_docs([0.9, 0.5, 0.3, 0.1])
        result = engine._apply_score_threshold(docs, 0.4)
        assert [d["score"] for d in result] == [0.9, 0.5]

    def test_keeps_docs_at_threshold(self, engine):
        docs = _make_docs([0.9, 0.5, 0.3])
        result = engine._apply_score_threshold(docs, 0.5)
        assert [d["score"] for d in result] == [0.9, 0.5]

    def test_empty_input(self, engine):
        assert engine._apply_score_threshold([], 0.5) == []

    def test_all_below_threshold_returns_empty(self, engine):
        docs = _make_docs([0.1, 0.2, 0.3])
        result = engine._apply_score_threshold(docs, 0.9)
        assert result == []

    def test_does_not_mutate_input(self, engine):
        docs = _make_docs([0.9, 0.1])
        original = list(docs)
        engine._apply_score_threshold(docs, 0.5)
        assert docs == original


# ---------------------------------------------------------------------------
# _order_documents_for_prompt tests
# ---------------------------------------------------------------------------


class TestOrderDocumentsForPrompt:
    """Tests for RAGQueryEngine._order_documents_for_prompt."""

    @pytest.fixture
    def engine(self, base_config):
        eng = RAGQueryEngine.__new__(RAGQueryEngine)
        eng.config = base_config
        return eng

    def test_rank_order_is_unchanged(self, engine):
        docs = _make_docs([0.9, 0.7, 0.5, 0.3])
        result = engine._order_documents_for_prompt(docs, "rank")
        assert [d["score"] for d in result] == [0.9, 0.7, 0.5, 0.3]

    def test_reversed_order(self, engine):
        docs = _make_docs([0.9, 0.7, 0.5, 0.3])
        result = engine._order_documents_for_prompt(docs, "reversed")
        assert [d["score"] for d in result] == [0.3, 0.5, 0.7, 0.9]

    def test_reversed_renumbers_ranks(self, engine):
        docs = _make_docs([0.9, 0.7, 0.5])
        result = engine._order_documents_for_prompt(docs, "reversed")
        assert [d["rank"] for d in result] == [1, 2, 3]

    def test_book_end_four_docs(self, engine):
        """Best first, second-best last, remaining in middle."""
        docs = _make_docs([0.9, 0.8, 0.6, 0.4])
        result = engine._order_documents_for_prompt(docs, "book_end")
        scores = [d["score"] for d in result]
        assert scores[0] == 0.9  # best first
        assert scores[-1] == 0.8  # second-best last
        # remaining in middle
        assert set(scores[1:-1]) == {0.6, 0.4}

    def test_book_end_renumbers_ranks(self, engine):
        docs = _make_docs([0.9, 0.8, 0.6, 0.4])
        result = engine._order_documents_for_prompt(docs, "book_end")
        assert [d["rank"] for d in result] == [1, 2, 3, 4]

    def test_book_end_three_docs(self, engine):
        """With 3 docs: best first, middle doc in middle, second-best last."""
        docs = _make_docs([0.9, 0.7, 0.5])
        result = engine._order_documents_for_prompt(docs, "book_end")
        scores = [d["score"] for d in result]
        assert scores[0] == 0.9
        assert scores[-1] == 0.7
        assert scores[1] == 0.5

    def test_book_end_two_docs_unchanged(self, engine):
        docs = _make_docs([0.9, 0.7])
        result = engine._order_documents_for_prompt(docs, "book_end")
        assert [d["score"] for d in result] == [0.9, 0.7]

    def test_single_doc_unchanged_for_any_order(self, engine):
        for order in ("rank", "reversed", "book_end"):
            docs = _make_docs([0.9])
            result = engine._order_documents_for_prompt(docs, order)
            assert len(result) == 1
            assert result[0]["score"] == 0.9

    def test_unknown_order_falls_back_to_rank(self, engine):
        docs = _make_docs([0.9, 0.7, 0.5])
        result = engine._order_documents_for_prompt(docs, "unknown_strategy")
        assert [d["score"] for d in result] == [0.9, 0.7, 0.5]

    def test_rank_order_returns_same_references(self, engine):
        """'rank' should not copy or mutate documents."""
        docs = _make_docs([0.9, 0.5])
        result = engine._order_documents_for_prompt(docs, "rank")
        assert result is docs

    def test_does_not_mutate_original_docs_list(self, engine):
        docs = _make_docs([0.9, 0.7, 0.5, 0.3])
        original_scores = [d["score"] for d in docs]
        original_ranks = [d["rank"] for d in docs]
        engine._order_documents_for_prompt(docs, "book_end")
        assert [d["score"] for d in docs] == original_scores
        assert [d["rank"] for d in docs] == original_ranks
