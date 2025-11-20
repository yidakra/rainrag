"""
Tests for embedding provider integrations (Mistral and OpenAI API embeddings).

This module tests:
- Mistral API embeddings
- OpenAI API embeddings
- Error handling for embedding APIs
- Cache integration with API providers
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag.config import (
    Config,
    PathsConfig,
    EmbeddingConfig,
    QdrantConfig,
    LLMConfig,
    MistralConfig,
    OpenAIConfig,
    ProcessingConfig,
    LoggingConfig,
)
from src.rainrag.query import RAGQueryEngine


# ============================================================================
# Mistral Embedding Tests
# ============================================================================

@pytest.fixture
def mistral_config(tmp_path):
    """Create test configuration with Mistral embedding provider."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache=str(tmp_path / "embeddings")
        ),
        embedding=EmbeddingConfig(
            provider="mistral",
            model_name="intfloat/multilingual-e5-large",
            batch_size=32,
            device="cpu"
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=1024,
            distance="Cosine"
        ),
        llm=LLMConfig(provider="mistral"),
        mistral=MistralConfig(
            api_key="test-mistral-key",
            model_name="mistral-small-latest"
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log")
    )


@pytest.fixture
def mock_mistral_client():
    """Mock Mistral client for embeddings."""
    client = MagicMock()

    # Mock embeddings response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 1024)]
    client.embeddings.create.return_value = embedding_response

    return client


def test_mistral_embedding_initialization(mistral_config, mock_mistral_client):
    """Test Mistral embedding client initialization."""
    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            # Client should be initialized
            assert engine.mistral_client is not None


def test_mistral_embedding_single_text(mistral_config, mock_mistral_client):
    """Test embedding a single text with Mistral API."""
    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            # Embed single text
            query = "This is a test document about machine learning."
            embedding = engine.embed_query(query)

            # Verify
            assert len(embedding) == 1024  # Mistral embedding size

            # Verify API was called
            mock_mistral_client.embeddings.create.assert_called_once()
            call_args = mock_mistral_client.embeddings.create.call_args
            assert call_args[1]["inputs"] == [query]


def test_mistral_embedding_batch(mistral_config, mock_mistral_client):
    """Test embedding multiple texts with Mistral API - via multiple calls."""
    # Test that we can embed multiple queries by calling embed_query multiple times
    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            # Embed multiple queries
            queries = ["Text 1", "Text 2", "Text 3"]
            embeddings = [engine.embed_query(q) for q in queries]

            # Verify
            assert len(embeddings) == 3
            assert all(len(emb) == 1024 for emb in embeddings)
            assert mock_mistral_client.embeddings.create.call_count == 3


def test_mistral_embedding_model_selection(mistral_config, mock_mistral_client):
    """Test that correct Mistral embedding model is used."""
    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            engine.embed_query("test")

            # Verify model parameter (always uses "mistral-embed" for embeddings)
            call_args = mock_mistral_client.embeddings.create.call_args
            assert call_args[1]["model"] == "mistral-embed"


def test_mistral_embedding_api_error(mistral_config, mock_mistral_client):
    """Test Mistral embedding API error handling."""
    mock_mistral_client.embeddings.create.side_effect = Exception("API Error")

    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test")

            assert "Mistral embeddings API error" in str(exc_info.value)


def test_mistral_embedding_rate_limit(mistral_config, mock_mistral_client):
    """Test Mistral API rate limit handling."""
    mock_mistral_client.embeddings.create.side_effect = Exception("Rate limit exceeded")

    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test")

            assert "rate limit" in str(exc_info.value).lower()


def test_mistral_embedding_empty_input(mistral_config, mock_mistral_client):
    """Test Mistral embedding with empty query string."""
    # Mock empty embedding for empty string
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.0] * 1024)]
    mock_mistral_client.embeddings.create.return_value = embedding_response

    with patch('src.rainrag.query.Mistral', return_value=mock_mistral_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(mistral_config)
            engine.initialize()

            # Embed empty string
            embedding = engine.embed_query("")

            # Should return embedding (API will handle it)
            assert len(embedding) == 1024


# ============================================================================
# OpenAI Embedding Tests
# ============================================================================

@pytest.fixture
def openai_config(tmp_path):
    """Create test configuration with OpenAI embedding provider."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache=str(tmp_path / "embeddings")
        ),
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            batch_size=32,
            device="cpu"
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=1536,
            distance="Cosine"
        ),
        llm=LLMConfig(provider="openai"),
        mistral=MistralConfig(
            api_key="test-mistral-key",
            model_name="mistral-small-latest"
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log")
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for embeddings."""
    client = MagicMock()

    # Mock embeddings response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = embedding_response

    return client


def test_openai_embedding_initialization(openai_config, mock_openai_client):
    """Test OpenAI embedding client initialization."""
    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Client should be initialized
            assert engine.openai_client is not None


def test_openai_embedding_single_text(openai_config, mock_openai_client):
    """Test embedding a single text with OpenAI API."""
    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Embed single text
            query = "This is a test document about artificial intelligence."
            embedding = engine.embed_query(query)

            # Verify
            assert len(embedding) == 1536  # OpenAI embedding size

            # Verify API was called
            mock_openai_client.embeddings.create.assert_called_once()
            call_args = mock_openai_client.embeddings.create.call_args
            assert call_args[1]["input"] == query


def test_openai_embedding_batch(openai_config, mock_openai_client):
    """Test embedding multiple texts with OpenAI API - via multiple calls."""
    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Embed multiple queries
            queries = ["Text 1", "Text 2", "Text 3"]
            embeddings = [engine.embed_query(q) for q in queries]

            # Verify
            assert len(embeddings) == 3
            assert all(len(emb) == 1536 for emb in embeddings)
            assert mock_openai_client.embeddings.create.call_count == 3


def test_openai_embedding_different_model(openai_config, mock_openai_client):
    """Test OpenAI embedding with text-embedding-3-large model."""
    openai_config.openai.embedding_model = "text-embedding-3-large"

    # Mock larger embeddings
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 3072)]
    mock_openai_client.embeddings.create.return_value = embedding_response

    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            embedding = engine.embed_query("test")

            # Verify model and dimensions
            call_args = mock_openai_client.embeddings.create.call_args
            assert call_args[1]["model"] == "text-embedding-3-large"
            assert len(embedding) == 3072


def test_openai_embedding_api_error(openai_config, mock_openai_client):
    """Test OpenAI embedding API error handling."""
    mock_openai_client.embeddings.create.side_effect = Exception("API Error")

    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test")

            assert "OpenAI embeddings API error" in str(exc_info.value)


def test_openai_embedding_auth_error(openai_config, mock_openai_client):
    """Test OpenAI authentication error handling."""
    mock_openai_client.embeddings.create.side_effect = Exception("Invalid API key")

    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test")

            assert "api key" in str(exc_info.value).lower()


def test_openai_embedding_empty_input(openai_config, mock_openai_client):
    """Test OpenAI embedding with empty query string."""
    # Mock empty embedding for empty string
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.0] * 1536)]
    mock_openai_client.embeddings.create.return_value = embedding_response

    with patch('src.rainrag.query.OpenAI', return_value=mock_openai_client):
        with patch('src.rainrag.query.QdrantClient'):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Embed empty string
            embedding = engine.embed_query("")

            # Should return embedding (API will handle it)
            assert len(embedding) == 1536
