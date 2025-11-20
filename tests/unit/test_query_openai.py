"""
Tests for OpenAI provider integration in query module.

This module tests:
- OpenAI embedding generation
- OpenAI LLM generation
- Error handling
- Integration with RAG query engine
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
def openai_config():
    """Create test configuration with OpenAI provider."""
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
            vector_size=1536,  # OpenAI embedding size
            distance="Cosine",
        ),
        llm=LLMConfig(provider="openai"),
        mistral=MistralConfig(api_key="test-mistral-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            max_tokens=512,
            temperature=0.3,
            top_k=5,
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
def mock_qdrant_client():
    """Mock Qdrant client."""
    client = MagicMock()

    # Mock query_points results (not search)
    query_result = MagicMock()
    point = MagicMock()
    point.id = "doc1"
    point.score = 0.95
    point.payload = {
        "text": "Test document content",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1",
    }
    query_result.points = [point]
    client.query_points.return_value = query_result

    return client


# ============================================================================
# OpenAI Embedding Tests
# ============================================================================


def test_embed_query_openai_success(openai_config, mock_openai_client, mock_qdrant_client):
    """Test successful query embedding with OpenAI API."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Embed a query
            query = "What is machine learning?"
            embedding = engine.embed_query(query)

            # Verify
            assert embedding is not None
            assert len(embedding) == 1536  # OpenAI embedding size
            assert all(isinstance(x, float) for x in embedding)

            # Verify OpenAI API was called correctly
            mock_openai_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small", input=query
            )


def test_embed_query_openai_error(openai_config, mock_openai_client, mock_qdrant_client):
    """Test OpenAI embedding API error handling."""
    mock_openai_client.embeddings.create.side_effect = Exception("API Error")

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test query")

            assert "OpenAI embeddings API error" in str(exc_info.value)


def test_embed_query_openai_rate_limit(openai_config, mock_openai_client, mock_qdrant_client):
    """Test OpenAI rate limit error handling."""
    mock_openai_client.embeddings.create.side_effect = Exception("Rate limit exceeded")

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test query")

            assert "Rate limit exceeded" in str(exc_info.value)


def test_embed_query_openai_different_model(openai_config, mock_openai_client, mock_qdrant_client):
    """Test embedding with different OpenAI model."""
    # Use text-embedding-3-large instead
    openai_config.openai.embedding_model = "text-embedding-3-large"

    # Mock larger embedding
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 3072)]
    mock_openai_client.embeddings.create.return_value = embedding_response

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            embedding = engine.embed_query("test")

            assert len(embedding) == 3072
            mock_openai_client.embeddings.create.assert_called_with(
                model="text-embedding-3-large", input="test"
            )


# ============================================================================
# OpenAI LLM Generation Tests
# ============================================================================


def test_generate_answer_openai_success(openai_config, mock_openai_client, mock_qdrant_client):
    """Test successful answer generation with OpenAI API."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Generate answer
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"},
            ]

            answer = engine.generate_answer(messages)

            # Verify
            assert answer == "This is a test response from OpenAI."

            # Verify OpenAI API was called correctly
            mock_openai_client.chat.completions.create.assert_called_once_with(
                model="gpt-4o-mini", messages=messages, max_tokens=512, temperature=0.3
            )


def test_generate_answer_openai_different_model(
    openai_config, mock_openai_client, mock_qdrant_client
):
    """Test answer generation with different OpenAI model."""
    openai_config.openai.model_name = "gpt-4o"

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer(messages)

            # Verify correct model was used
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["model"] == "gpt-4o"


def test_generate_answer_openai_custom_params(
    openai_config, mock_openai_client, mock_qdrant_client
):
    """Test answer generation with custom parameters."""
    openai_config.openai.max_tokens = 1024
    openai_config.openai.temperature = 0.7

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer(messages)

            # Verify parameters
            call_args = mock_openai_client.chat.completions.create.call_args
            assert call_args[1]["max_tokens"] == 1024
            assert call_args[1]["temperature"] == 0.7


def test_generate_answer_openai_error(openai_config, mock_openai_client, mock_qdrant_client):
    """Test OpenAI LLM API error handling."""
    mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.generate_answer([{"role": "user", "content": "test"}])

            assert "OpenAI API error" in str(exc_info.value)


def test_generate_answer_openai_empty_response(
    openai_config, mock_openai_client, mock_qdrant_client
):
    """Test handling of empty response from OpenAI."""
    # Mock empty response
    chat_response = MagicMock()
    chat_response.choices = [MagicMock(message=MagicMock(content="  "))]
    mock_openai_client.chat.completions.create.return_value = chat_response

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            answer = engine.generate_answer([{"role": "user", "content": "test"}])

            # Should return empty string after strip
            assert answer == ""


# ============================================================================
# OpenAI Full Query Pipeline Tests
# ============================================================================


def test_query_openai_full_pipeline(openai_config, mock_openai_client, mock_qdrant_client):
    """Test full RAG query pipeline with OpenAI."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            # Run full query
            result = engine.query(question="What is machine learning?", top_k=3, language="en")

            # Verify result structure
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "num_documents" in result

            assert result["answer"] == "This is a test response from OpenAI."
            assert len(result["retrieved_documents"]) == 1
            assert result["num_documents"] == 1

            # Verify both embedding and chat APIs were called
            assert mock_openai_client.embeddings.create.called
            assert mock_openai_client.chat.completions.create.called


def test_query_openai_russian_language(openai_config, mock_openai_client, mock_qdrant_client):
    """Test OpenAI query with Russian language."""
    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            result = engine.query(question="Что такое машинное обучение?", top_k=5, language="ru")

            # Verify result
            assert result["answer"] is not None

            # Verify system prompt included Russian instruction (in Cyrillic)
            call_args = mock_openai_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            system_message = next(m for m in messages if m["role"] == "system")
            # Check for Russian text (the word "русском" means "Russian" in Russian)
            assert "русском" in system_message["content"]


def test_query_openai_no_documents_retrieved(openai_config, mock_openai_client, mock_qdrant_client):
    """Test OpenAI query when no documents are retrieved."""
    # Mock empty query_points results
    query_result = MagicMock()
    query_result.points = []
    mock_qdrant_client.query_points.return_value = query_result

    with patch("src.rainrag.query.OpenAI", return_value=mock_openai_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(openai_config)
            engine.initialize()

            result = engine.query(question="What is quantum physics?", top_k=5, language="en")

            # Should still generate answer (without context)
            assert result["answer"] is not None
            assert len(result["retrieved_documents"]) == 0
            assert result["num_documents"] == 0
