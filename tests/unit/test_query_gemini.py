"""
Tests for Google Gemini provider integration in query module.

This module tests:
- Gemini embedding generation
- Gemini LLM generation
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
    Config,
    EmbeddingConfig,
    GeminiConfig,
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
def gemini_config():
    """Create test configuration with Gemini provider."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings",
        ),
        embedding=EmbeddingConfig(
            provider="gemini",
            model_name="intfloat/multilingual-e5-large",
            batch_size=32,
            device="cpu",
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=768,  # Gemini embedding size
            distance="Cosine",
        ),
        llm=LLMConfig(provider="gemini"),
        mistral=MistralConfig(api_key="test-mistral-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        ),
        gemini=GeminiConfig(
            api_key="test-gemini-key",
            model_name="gemini-2.5-flash",
            embedding_model="models/text-embedding-004",
            max_tokens=512,
            temperature=0.3,
            top_k=5,
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True),
    )


@pytest.fixture
def mock_gemini_model():
    """Mock Gemini GenerativeModel."""
    model = MagicMock()

    # Mock generate_content response
    response = MagicMock()
    response.text = "This is a test response from Gemini."
    model.generate_content.return_value = response

    return model


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
        "text": "Test document content about AI.",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1",
    }
    query_result.points = [point]
    client.query_points.return_value = query_result

    return client


# ============================================================================
# Gemini Embedding Tests
# ============================================================================


def test_embed_query_gemini_success(gemini_config, mock_qdrant_client):
    """Test successful query embedding with Gemini API."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            # Mock the client and its methods
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_result = MagicMock()
            mock_result.embeddings = [MagicMock(values=[0.1] * 768)]
            mock_client.models.embed_content.return_value = mock_result

            # Mock EmbedContentConfig
            mock_config = MagicMock()
            mock_config.task_type = "RETRIEVAL_QUERY"
            mock_genai.EmbedContentConfig.return_value = mock_config

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Embed a query
            query = "What is machine learning?"
            embedding = engine.embed_query(query)

            # Verify
            assert embedding is not None
            assert len(embedding) == 768  # Gemini embedding size
            assert all(isinstance(x, float) for x in embedding)

            # Verify Gemini API was called correctly
            mock_client.models.embed_content.assert_called_once()
            call_args = mock_client.models.embed_content.call_args
            assert call_args[1]["model"] == "models/text-embedding-004"
            assert call_args[1]["contents"] == [query]
            assert call_args[1]["config"].task_type == "RETRIEVAL_QUERY"


def test_embed_query_gemini_error(gemini_config, mock_qdrant_client):
    """Test Gemini embedding API error handling."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.embed_content.side_effect = Exception("API Error")

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test query")

            assert "Gemini embeddings API error" in str(exc_info.value)


def test_embed_query_gemini_different_model(gemini_config, mock_qdrant_client):
    """Test embedding with different Gemini model."""
    gemini_config.gemini.embedding_model = "models/embedding-001"

    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_result = MagicMock()
            mock_result.embeddings = [MagicMock(values=[0.1] * 768)]
            mock_client.models.embed_content.return_value = mock_result

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            engine.embed_query("test")

            # Verify correct model was used
            call_args = mock_client.models.embed_content.call_args
            assert call_args[1]["model"] == "models/embedding-001"


def test_embed_query_gemini_task_type(gemini_config, mock_qdrant_client):
    """Test that Gemini embedding uses correct task_type."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_result = MagicMock()
            mock_result.embeddings = [MagicMock(values=[0.1] * 768)]
            mock_client.models.embed_content.return_value = mock_result

            # Mock EmbedContentConfig
            mock_config = MagicMock()
            mock_config.task_type = "RETRIEVAL_QUERY"
            mock_genai.EmbedContentConfig.return_value = mock_config

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            engine.embed_query("test query")

            # Verify task_type is set for retrieval
            mock_genai.EmbedContentConfig.assert_called_once()
            config_call_args = mock_genai.EmbedContentConfig.call_args
            assert config_call_args[1]["task_type"] == "RETRIEVAL_QUERY"


# ============================================================================
# Gemini LLM Generation Tests
# ============================================================================


def test_generate_answer_gemini_success(gemini_config, mock_qdrant_client):
    """Test successful answer generation with Gemini API."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "This is a test response from Gemini."
            mock_client.models.generate_content.return_value = mock_response

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Generate answer
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"},
            ]

            answer = engine.generate_answer(messages)

            # Verify
            assert answer == "This is a test response from Gemini."

            # Verify Gemini API was called correctly
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            assert call_args[1]["model"] == "gemini-2.5-flash"
            assert (
                len(call_args[1]["contents"]) == 1
            )  # system message gets combined with user message


def test_generate_answer_gemini_system_message_handling(gemini_config, mock_qdrant_client):
    """Test that system message is combined with user message for Gemini."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            # Mock Content and Part
            mock_part = MagicMock()
            mock_part.text = "You are an expert in machine learning.\n\nExplain neural networks."
            mock_content = MagicMock()
            mock_content.role = "user"
            mock_content.parts = [mock_part]
            mock_genai.Content.return_value = mock_content

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Messages with system message
            messages = [
                {"role": "system", "content": "You are an expert in machine learning."},
                {"role": "user", "content": "Explain neural networks."},
            ]

            engine.generate_answer(messages)

            # Verify system message was combined with user content
            call_args = mock_client.models.generate_content.call_args
            contents = call_args[1]["contents"]
            assert len(contents) == 1  # Only user message after combining
            user_content = contents[0]
            assert user_content.role == "user"
            assert "You are an expert in machine learning." in user_content.parts[0].text
            assert "Explain neural networks." in user_content.parts[0].text


def test_generate_answer_gemini_no_system_message(gemini_config, mock_qdrant_client):
    """Test Gemini generation when no system message is present."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            # Mock Content and Part
            mock_part = MagicMock()
            mock_part.text = "What is AI?"
            mock_content = MagicMock()
            mock_content.role = "user"
            mock_content.parts = [mock_part]
            mock_genai.Content.return_value = mock_content

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Messages without system message
            messages = [{"role": "user", "content": "What is AI?"}]

            engine.generate_answer(messages)

            # Verify only user message is in contents
            call_args = mock_client.models.generate_content.call_args
            contents = call_args[1]["contents"]
            assert len(contents) == 1
            assert contents[0].role == "user"
            assert contents[0].parts[0].text == "What is AI?"


def test_generate_answer_gemini_config_params(gemini_config, mock_qdrant_client):
    """Test that generation config parameters are passed correctly."""
    gemini_config.gemini.max_tokens = 1024
    gemini_config.gemini.temperature = 0.7

    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            # Mock GenerateContentConfig
            mock_config = MagicMock()
            mock_config.max_output_tokens = 1024
            mock_config.temperature = 0.7
            mock_genai.GenerateContentConfig.return_value = mock_config

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer(messages)

            # Verify config was passed with correct params
            call_args = mock_client.models.generate_content.call_args
            config = call_args[1]["config"]
            assert config.max_output_tokens == 1024
            assert config.temperature == 0.7


def test_generate_answer_gemini_different_model(gemini_config, mock_qdrant_client):
    """Test answer generation with different Gemini model."""
    gemini_config.gemini.model_name = "gemini-2.5-pro"

    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "Response"
            mock_client.models.generate_content.return_value = mock_response

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer(messages)

            # Verify correct model was used
            call_args = mock_client.models.generate_content.call_args
            assert call_args[1]["model"] == "gemini-2.5-pro"


def test_generate_answer_gemini_error(gemini_config, mock_qdrant_client):
    """Test Gemini LLM API error handling."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.generate_content.side_effect = Exception("API Error")

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.generate_answer([{"role": "user", "content": "test"}])

            assert "Gemini API error" in str(exc_info.value)


def test_generate_answer_gemini_empty_response(gemini_config, mock_qdrant_client):
    """Test handling of empty response from Gemini."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            # Mock empty response
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_response = MagicMock()
            mock_response.text = "  "
            mock_client.models.generate_content.return_value = mock_response

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            answer = engine.generate_answer([{"role": "user", "content": "test"}])

            # Should return empty string after strip
            assert answer == ""


# ============================================================================
# Gemini Full Query Pipeline Tests
# ============================================================================


def test_query_gemini_full_pipeline(gemini_config, mock_qdrant_client):
    """Test full RAG query pipeline with Gemini."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            # Mock client and methods
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            # Mock embeddings
            mock_embed_result = MagicMock()
            mock_embed_result.embeddings = [MagicMock(values=[0.1] * 768)]
            mock_client.models.embed_content.return_value = mock_embed_result

            # Mock LLM response
            mock_llm_response = MagicMock()
            mock_llm_response.text = "This is a test response from Gemini."
            mock_client.models.generate_content.return_value = mock_llm_response

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Run full query
            result = engine.query(question="What is machine learning?", top_k=3, language="en")

            # Verify result structure
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "num_documents" in result

            assert result["answer"] == "This is a test response from Gemini."
            assert len(result["retrieved_documents"]) == 1
            assert result["num_documents"] == 1

            # Verify both embedding and generation were called
            mock_client.models.embed_content.assert_called()
            mock_client.models.generate_content.assert_called()


def test_query_gemini_russian_language(gemini_config, mock_qdrant_client):
    """Test Gemini query with Russian language."""
    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            # Mock client and methods
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            # Mock embeddings
            mock_embed_result = MagicMock()
            mock_embed_result.embeddings = [MagicMock(values=[0.1] * 768)]
            mock_client.models.embed_content.return_value = mock_embed_result

            # Mock LLM response
            mock_llm_response = MagicMock()
            mock_llm_response.text = "Ответ на русском языке"
            mock_client.models.generate_content.return_value = mock_llm_response

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            result = engine.query(question="Что такое машинное обучение?", top_k=5, language="ru")

            # Verify result
            assert result["answer"] == "Ответ на русском языке"
            assert result["answer"] is not None

            # Verify both embedding and generation were called
            mock_client.models.embed_content.assert_called()
            mock_client.models.generate_content.assert_called()


def test_query_gemini_no_documents_retrieved(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test Gemini query when no documents are retrieved."""
    # Mock empty query_points results
    query_result = MagicMock()
    query_result.points = []
    mock_qdrant_client.query_points.return_value = query_result

    with patch("src.rainrag.query.genai") as mock_genai:
        with patch("src.rainrag.query.QdrantClient", return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            result = engine.query(question="What is quantum physics?", top_k=5, language="en")

            # Should still generate answer (without context)
            assert result["answer"] is not None
            assert len(result["retrieved_documents"]) == 0
            assert result["num_documents"] == 0
