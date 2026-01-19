"""
Tests for hybrid search functionality (Vector + BM25).

This module tests:
- BM25 index building from Qdrant collection
- BM25 search with keyword matching
- Reciprocal Rank Fusion (RRF)
- Weighted score fusion
- Full hybrid search pipeline
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag.config import (
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
    VideoConfig,
)
from src.rainrag.query import RAGQueryEngine


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def hybrid_config():
    """Create test configuration with hybrid search enabled."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings",
        ),
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="intfloat/multilingual-e5-large",
            batch_size=32,
            device="cpu",
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine",
        ),
        llm=LLMConfig(provider="openai"),
        mistral=MistralConfig(
            api_key="test-mistral-key",
            model_name="mistral-small-latest",
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            max_tokens=512,
            temperature=0.3,
            top_k=5,
        ),
        hybrid_search=HybridSearchConfig(
            enabled=True,
            bm25_weight=0.3,
            top_k_multiplier=3,
            fusion_method="rrf",
            rrf_k=60,
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True),
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = MagicMock()

    # Mock embeddings response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = embedding_response

    # Mock chat completions response
    chat_response = MagicMock()
    chat_response.choices = [
        MagicMock(message=MagicMock(content="This is a test response from OpenAI."))
    ]
    client.chat.completions.create.return_value = chat_response

    return client


@pytest.fixture
def mock_qdrant_client_with_corpus():
    """Mock Qdrant client with a corpus of documents for BM25 indexing."""
    client = MagicMock()

    # Mock scroll results for BM25 index building
    # Qdrant scroll() returns a tuple of (points, next_offset)

    # First batch
    point1 = MagicMock()
    point1.id = "doc1"
    point1.payload = {
        "text": "Machine learning is a subset of artificial intelligence",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1",
    }
    point2 = MagicMock()
    point2.id = "doc2"
    point2.payload = {
        "text": "Deep learning uses neural networks for pattern recognition",
        "language": "en",
        "path": "/test/doc2.vtt",
        "doc_id": "doc2",
    }

    # Second batch
    point3 = MagicMock()
    point3.id = "doc3"
    point3.payload = {
        "text": "Natural language processing enables computers to understand text",
        "language": "en",
        "path": "/test/doc3.vtt",
        "doc_id": "doc3",
    }

    # Configure scroll to return tuples of (points, next_offset)
    client.scroll.side_effect = [
        ([point1, point2], "offset123"),  # First batch with offset
        ([point3], None),  # Second batch, no more results
    ]

    # Mock query_points results for vector search
    query_result = MagicMock()
    point = MagicMock()
    point.id = "doc1"
    point.score = 0.95
    point.payload = {
        "text": "Machine learning is a subset of artificial intelligence",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1",
    }
    query_result.points = [point]
    client.query_points.return_value = query_result

    return client


# ============================================================================
# BM25 Index Building Tests
# ============================================================================


def test_build_bm25_index_success(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test successful BM25 index building from Qdrant collection."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Verify BM25 index was built
            assert engine.bm25 is not None
            assert len(engine.bm25_corpus) == 3  # 3 documents
            assert len(engine.bm25_tokenized_corpus) == 3

            # Verify corpus contains expected documents
            assert any("machine learning" in doc["text"].lower() for doc in engine.bm25_corpus)
            assert any("deep learning" in doc["text"].lower() for doc in engine.bm25_corpus)
            assert any("natural language" in doc["text"].lower() for doc in engine.bm25_corpus)

            # Verify scroll was called
            assert mock_qdrant_client_with_corpus.scroll.called


def test_build_bm25_index_disabled(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test that BM25 index is not built when hybrid search is disabled."""
    hybrid_config.hybrid_search.enabled = False

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Verify BM25 index was NOT built
            assert engine.bm25 is None
            assert len(engine.bm25_corpus) == 0
            assert len(engine.bm25_tokenized_corpus) == 0

            # Verify scroll was NOT called
            assert not mock_qdrant_client_with_corpus.scroll.called


def test_build_bm25_index_empty_collection(hybrid_config, mock_openai_client):
    """Test BM25 index building with empty Qdrant collection."""
    client = MagicMock()
    # scroll() returns tuple of (points, next_offset)
    client.scroll.return_value = ([], None)

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=client):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Should handle empty collection gracefully
            # BM25 index is not created when there are no documents
            assert engine.bm25 is None
            assert len(engine.bm25_corpus) == 0
            assert len(engine.bm25_tokenized_corpus) == 0


# ============================================================================
# BM25 Search Tests
# ============================================================================


def test_search_bm25_keyword_matching(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test BM25 search finds documents by keyword matching."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Search for exact keyword
            results = engine._search_bm25("machine learning", top_k=3)

            # Verify results
            assert len(results) > 0
            assert (
                results[0]["doc_id"] == "doc1"
            )  # Should find doc1 which contains "machine learning"
            assert "score" in results[0]
            assert "rank" in results[0]


def test_search_bm25_multiple_keywords(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test BM25 search with multiple keywords."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Search for multiple keywords
            results = engine._search_bm25("neural networks pattern", top_k=3)

            # Verify results
            assert len(results) > 0
            # Should find doc2 which contains "neural networks" and "pattern"
            doc_ids = [r["doc_id"] for r in results]
            assert "doc2" in doc_ids


def test_search_bm25_no_matches(hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus):
    """Test BM25 search with query that has no keyword matches."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Search for non-existent keywords
            results = engine._search_bm25("quantum physics", top_k=3)

            # Should return documents (with low scores) or empty list
            # BM25 behavior varies, but should not crash
            assert isinstance(results, list)


# ============================================================================
# Score Fusion Tests - RRF
# ============================================================================


def test_fuse_scores_rrf_basic(hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus):
    """Test Reciprocal Rank Fusion (RRF) with basic inputs."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Create mock results
            vector_results = [
                {"doc_id": "doc1", "score": 0.95, "rank": 0, "text": "text1"},
                {"doc_id": "doc2", "score": 0.85, "rank": 1, "text": "text2"},
            ]
            bm25_results = [
                {"doc_id": "doc2", "score": 5.0, "rank": 0, "text": "text2"},
                {"doc_id": "doc3", "score": 3.0, "rank": 1, "text": "text3"},
            ]

            # Fuse scores
            fused = engine._fuse_scores_rrf(vector_results, bm25_results, k=60)

            # Verify results
            assert len(fused) == 3  # doc1, doc2, doc3
            # doc2 should be ranked highest (appears in both lists)
            assert fused[0]["doc_id"] == "doc2"
            assert "score" in fused[0]
            assert "fusion_method" in fused[0]
            assert fused[0]["fusion_method"] == "rrf"


def test_fuse_scores_rrf_single_list(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test RRF with results from only one source."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            vector_results = [
                {"doc_id": "doc1", "score": 0.95, "rank": 0, "text": "text1"},
            ]
            bm25_results = []

            # Fuse scores
            fused = engine._fuse_scores_rrf(vector_results, bm25_results, k=60)

            # Should still work with single list
            assert len(fused) == 1
            assert fused[0]["doc_id"] == "doc1"


def test_fuse_scores_rrf_empty_lists(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test RRF with empty input lists."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Fuse empty lists
            fused = engine._fuse_scores_rrf([], [], k=60)

            # Should return empty list
            assert len(fused) == 0


# ============================================================================
# Score Fusion Tests - Weighted
# ============================================================================


def test_fuse_scores_weighted_basic(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test weighted score fusion with basic inputs."""
    hybrid_config.hybrid_search.fusion_method = "weighted"

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Create mock results
            vector_results = [
                {"doc_id": "doc1", "score": 0.95, "rank": 0, "text": "text1"},
                {"doc_id": "doc2", "score": 0.85, "rank": 1, "text": "text2"},
            ]
            bm25_results = [
                {"doc_id": "doc2", "score": 5.0, "rank": 0, "text": "text2"},
                {"doc_id": "doc3", "score": 3.0, "rank": 1, "text": "text3"},
            ]

            # Fuse scores with 0.3 BM25 weight
            fused = engine._fuse_scores_weighted(vector_results, bm25_results, bm25_weight=0.3)

            # Verify results
            assert len(fused) == 3  # doc1, doc2, doc3
            assert all("score" in doc for doc in fused)
            assert all("fusion_method" in doc for doc in fused)
            assert all(doc["fusion_method"] == "weighted" for doc in fused)
            # doc2 should have high score (appears in both lists)
            doc2 = next(d for d in fused if d["doc_id"] == "doc2")
            assert doc2["score"] > 0


def test_fuse_scores_weighted_different_weights(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test weighted fusion with different BM25 weights."""
    hybrid_config.hybrid_search.fusion_method = "weighted"

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Use multiple documents with different scores to test weighting
            vector_results = [
                {"doc_id": "doc1", "score": 0.9, "rank": 0, "text": "text1"},
                {"doc_id": "doc2", "score": 0.5, "rank": 1, "text": "text2"},
            ]
            bm25_results = [
                {"doc_id": "doc1", "score": 3.0, "rank": 0, "text": "text1"},
                {"doc_id": "doc2", "score": 10.0, "rank": 1, "text": "text2"},
            ]

            # Test with low BM25 weight (0.1) - should favor vector scores
            fused_low = engine._fuse_scores_weighted(vector_results, bm25_results, bm25_weight=0.1)

            # Test with high BM25 weight (0.9) - should favor BM25 scores
            fused_high = engine._fuse_scores_weighted(vector_results, bm25_results, bm25_weight=0.9)

            # With low BM25 weight, doc1 should rank higher (has higher vector score)
            # With high BM25 weight, doc2 should rank higher (has higher BM25 score)
            assert fused_low[0]["doc_id"] != fused_high[0]["doc_id"]


# ============================================================================
# Full Hybrid Search Pipeline Tests
# ============================================================================


def test_retrieve_documents_hybrid_enabled(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test document retrieval with hybrid search enabled."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Retrieve documents with hybrid search
            query_vector = [0.1] * 1536
            documents = engine.retrieve_documents(
                query_vector=query_vector,
                top_k=3,
                query_text="machine learning artificial intelligence",
            )

            # Verify results
            assert isinstance(documents, list)
            # Should have called both vector and BM25 search
            assert mock_qdrant_client_with_corpus.query_points.called


def test_retrieve_documents_hybrid_disabled(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test document retrieval with hybrid search disabled."""
    hybrid_config.hybrid_search.enabled = False

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Retrieve documents without hybrid search
            query_vector = [0.1] * 1536
            documents = engine.retrieve_documents(
                query_vector=query_vector,
                top_k=3,
            )

            # Should only use vector search
            assert isinstance(documents, list)
            assert mock_qdrant_client_with_corpus.query_points.called
            # BM25 should not be built
            assert engine.bm25 is None


def test_query_full_pipeline_hybrid(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test full query pipeline with hybrid search."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Run full query with hybrid search
            result = engine.query(
                question="What is machine learning?",
                top_k=3,
                language="en",
            )

            # Verify result structure
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "num_documents" in result

            # Verify both embedding and chat APIs were called
            assert mock_openai_client.embeddings.create.called
            assert mock_openai_client.chat.completions.create.called


def test_hybrid_search_top_k_multiplier(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test that top_k_multiplier increases candidate retrieval."""
    hybrid_config.hybrid_search.top_k_multiplier = 5

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            engine = RAGQueryEngine(hybrid_config)
            engine.initialize()

            # Retrieve documents
            query_vector = [0.1] * 1536
            engine.retrieve_documents(
                query_vector=query_vector,
                top_k=3,
                query_text="test query",
            )

            # Verify query_points was called with top_k * multiplier
            call_args = mock_qdrant_client_with_corpus.query_points.call_args
            # Should request 3 * 5 = 15 candidates
            assert call_args[1]["limit"] == 15


def test_hybrid_search_rrf_vs_weighted(
    hybrid_config, mock_openai_client, mock_qdrant_client_with_corpus
):
    """Test that RRF and weighted fusion methods both work."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client_with_corpus):
            # Test with RRF
            hybrid_config.hybrid_search.fusion_method = "rrf"
            engine_rrf = RAGQueryEngine(hybrid_config)
            engine_rrf.initialize()

            result_rrf = engine_rrf.query(question="test query", top_k=3, language="en")

            # Reset mock
            mock_openai_client.reset_mock()
            mock_qdrant_client_with_corpus.reset_mock()

            # Reconfigure scroll side_effect to return tuples
            point1 = MagicMock()
            point1.id = "doc1"
            point1.payload = {
                "text": "Machine learning is a subset of artificial intelligence",
                "language": "en",
                "path": "/test/doc1.vtt",
                "doc_id": "doc1",
            }
            point2 = MagicMock()
            point2.id = "doc2"
            point2.payload = {
                "text": "Deep learning uses neural networks for pattern recognition",
                "language": "en",
                "path": "/test/doc2.vtt",
                "doc_id": "doc2",
            }
            point3 = MagicMock()
            point3.id = "doc3"
            point3.payload = {
                "text": "Natural language processing enables computers to understand text",
                "language": "en",
                "path": "/test/doc3.vtt",
                "doc_id": "doc3",
            }

            mock_qdrant_client_with_corpus.scroll.side_effect = [
                ([point1, point2], "offset123"),
                ([point3], None),
            ]

            query_result = MagicMock()
            point = MagicMock()
            point.id = "doc1"
            point.score = 0.95
            point.payload = {
                "text": "Machine learning is a subset of artificial intelligence",
                "language": "en",
                "path": "/test/doc1.vtt",
                "doc_id": "doc1",
            }
            query_result.points = [point]
            mock_qdrant_client_with_corpus.query_points.return_value = query_result

            # Reconfigure embeddings response
            embedding_response = MagicMock()
            embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_openai_client.embeddings.create.return_value = embedding_response

            # Reconfigure chat response
            chat_response = MagicMock()
            chat_response.choices = [
                MagicMock(message=MagicMock(content="This is a test response from OpenAI."))
            ]
            mock_openai_client.chat.completions.create.return_value = chat_response

            # Test with weighted
            hybrid_config.hybrid_search.fusion_method = "weighted"
            engine_weighted = RAGQueryEngine(hybrid_config)
            engine_weighted.initialize()

            result_weighted = engine_weighted.query(question="test query", top_k=3, language="en")

            # Both should return valid results
            assert "answer" in result_rrf
            assert "answer" in result_weighted
