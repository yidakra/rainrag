"""
Tests for Claude/Anthropic provider integration in query module.

This module tests:
- Claude LLM generation
- System message extraction
- Message format conversion
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
    ClaudeConfig,
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
def claude_config():
    """Create test configuration with Claude provider."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings",
        ),
        embedding=EmbeddingConfig(
            provider="local",
            model_name="intfloat/multilingual-e5-large",
            batch_size=32,
            device="cpu",
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=1024,
            distance="Cosine",
        ),
        llm=LLMConfig(provider="claude"),
        mistral=MistralConfig(api_key="test-mistral-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        ),
        claude=ClaudeConfig(
            api_key="test-claude-key",
            model_name="claude-haiku-4-5-20251001",
            max_tokens=512,
            temperature=0.3,
            top_k=5,
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True),
    )


@pytest.fixture
def mock_claude_client():
    """Mock Claude client."""
    client = MagicMock()

    # Mock messages response
    message_response = MagicMock()
    message_response.content = [MagicMock(text="This is a test response from Claude.")]
    client.messages.create.return_value = message_response

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
        "text": "Test document content about AI and machine learning.",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1",
    }
    query_result.points = [point]
    client.query_points.return_value = query_result

    return client


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for local embeddings."""
    with patch("src.rainrag.query.SentenceTransformer") as mock_st:
        model = MagicMock()
        model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
        mock_st.return_value = model
        yield mock_st


# ============================================================================
# Claude LLM Generation Tests
# ============================================================================


def test_generate_answer_claude_success(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test successful answer generation with Claude API."""
    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            # Generate answer
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"},
            ]

            answer = engine.generate_answer(messages)

            # Verify
            assert answer == "This is a test response from Claude."

            # Verify Claude API was called correctly
            call_args = mock_claude_client.messages.create.call_args
            assert call_args[1]["model"] == "claude-haiku-4-5-20251001"
            assert call_args[1]["max_tokens"] == 512
            assert call_args[1]["temperature"] == 0.3


def test_generate_answer_claude_system_message_extraction(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test that system message is extracted correctly for Claude API."""
    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            # Messages with system message
            messages = [
                {"role": "system", "content": "You are an expert in machine learning."},
                {"role": "user", "content": "Explain neural networks."},
                {"role": "assistant", "content": "Neural networks are..."},
                {"role": "user", "content": "Tell me more."},
            ]

            engine.generate_answer(messages)

            # Verify system message was extracted
            call_args = mock_claude_client.messages.create.call_args
            assert call_args[1]["system"] == "You are an expert in machine learning."

            # Verify messages don't include system message
            claude_messages = call_args[1]["messages"]
            assert len(claude_messages) == 3  # Only user and assistant messages
            assert all(m["role"] != "system" for m in claude_messages)


def test_generate_answer_claude_no_system_message(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test Claude API call when no system message is present."""
    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            # Messages without system message
            messages = [{"role": "user", "content": "What is AI?"}]

            engine.generate_answer(messages)

            # Verify system parameter is empty string
            call_args = mock_claude_client.messages.create.call_args
            assert call_args[1]["system"] == ""


def test_generate_answer_claude_different_model(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test answer generation with different Claude model."""
    claude_config.claude.model_name = "claude-sonnet-4-5-20250514"

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer(messages)

            # Verify correct model was used
            call_args = mock_claude_client.messages.create.call_args
            assert call_args[1]["model"] == "claude-sonnet-4-5-20250514"


def test_generate_answer_claude_custom_params(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test answer generation with custom parameters."""
    claude_config.claude.max_tokens = 1024
    claude_config.claude.temperature = 0.7

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer(messages)

            # Verify parameters
            call_args = mock_claude_client.messages.create.call_args
            assert call_args[1]["max_tokens"] == 1024
            assert call_args[1]["temperature"] == 0.7


def test_generate_answer_claude_error(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test Claude API error handling."""
    mock_claude_client.messages.create.side_effect = Exception("API Error")

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.generate_answer([{"role": "user", "content": "test"}])

            assert "Claude API error" in str(exc_info.value)


def test_generate_answer_claude_rate_limit(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test Claude rate limit error handling."""
    mock_claude_client.messages.create.side_effect = Exception("Rate limit exceeded")

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.generate_answer([{"role": "user", "content": "test"}])

            assert "Rate limit exceeded" in str(exc_info.value)


def test_generate_answer_claude_empty_response(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test handling of empty response from Claude."""
    # Mock empty response
    message_response = MagicMock()
    message_response.content = [MagicMock(text="  ")]
    mock_claude_client.messages.create.return_value = message_response

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            answer = engine.generate_answer([{"role": "user", "content": "test"}])

            # Should return empty string after strip
            assert answer == ""


# ============================================================================
# Claude Full Query Pipeline Tests
# ============================================================================


def test_query_claude_full_pipeline(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test full RAG query pipeline with Claude."""
    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            # Run full query
            result = engine.query(question="What is machine learning?", top_k=3, language="en")

            # Verify result structure
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "num_documents" in result

            assert result["answer"] == "This is a test response from Claude."
            assert len(result["retrieved_documents"]) == 1
            assert result["num_documents"] == 1

            # Verify Claude API was called
            assert mock_claude_client.messages.create.called


def test_query_claude_russian_language(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test Claude query with Russian language."""
    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            result = engine.query(question="Что такое машинное обучение?", top_k=5, language="ru")

            # Verify result
            assert result["answer"] is not None

            # Verify system prompt included Russian instruction (in Cyrillic)
            call_args = mock_claude_client.messages.create.call_args
            system_message = call_args[1]["system"]
            # Check for Russian text (the word "русском" means "Russian" in Russian)
            assert "русском" in system_message


def test_query_claude_with_context(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test Claude query with retrieved context documents."""
    # Mock multiple query_points results
    points = []
    for i in range(3):
        point = MagicMock()
        point.id = f"doc{i}"
        point.score = 0.9 - (i * 0.1)
        point.payload = {
            "text": f"Document {i} content about machine learning.",
            "language": "en",
            "path": f"/test/doc{i}.vtt",
            "doc_id": f"doc{i}",
        }
        points.append(point)

    query_result = MagicMock()
    query_result.points = points
    mock_qdrant_client.query_points.return_value = query_result

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            result = engine.query(question="Tell me about machine learning", top_k=3, language="en")

            # Verify context was included
            assert len(result["retrieved_documents"]) == 3

            # Verify context was passed to Claude
            call_args = mock_claude_client.messages.create.call_args
            messages = call_args[1]["messages"]
            user_message = next(m for m in messages if m["role"] == "user")
            # Context should be included in the user message
            assert "Document" in user_message["content"]


def test_query_claude_no_documents_retrieved(
    claude_config, mock_claude_client, mock_qdrant_client, mock_sentence_transformer
):
    """Test Claude query when no documents are retrieved."""
    # Mock empty query_points results
    query_result = MagicMock()
    query_result.points = []
    mock_qdrant_client.query_points.return_value = query_result

    with patch("src.rainrag.query.Anthropic", return_value=mock_claude_client):
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            engine = RAGQueryEngine(claude_config)
            engine.initialize()

            result = engine.query(question="What is quantum physics?", top_k=5, language="en")

            # Should still generate answer (without context)
            assert result["answer"] is not None
            assert len(result["retrieved_documents"]) == 0
            assert result["num_documents"] == 0
