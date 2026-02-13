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

from unittest.mock import MagicMock, call, patch

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
    chat_resp.choices = [MagicMock(message=MagicMock(content="rewritten query A\nrewritten query B"))]
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
            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
                "infrastructure work in 2022\nbridge building project"
            )

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

            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
                "line one\nline two\nline three"
            )

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

            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
                "вариант один"
            )

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

            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
                "variant one"
            )

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

            mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
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
            high_score_point.payload = mock_qdrant_client.query_points.return_value.points[0].payload

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

    def test_hyde_blends_embedding(
        self, hyde_config, mock_openai_client, mock_qdrant_client
    ):
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
                MagicMock(choices=[MagicMock(message=MagicMock(content="Hypothetical passage text."))]),
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

        assert set(result.keys()) == {"question", "answer", "retrieved_documents", "num_documents"}

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
        assert rewrite_call_temp == 0.7   # diverse paraphrases
        assert answer_call_temp == 0.0    # deterministic answer
