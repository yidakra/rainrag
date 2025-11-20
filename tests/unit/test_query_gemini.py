"""
Tests for Google Gemini provider integration in query module.

This module tests:
- Gemini embedding generation
- Gemini LLM generation
- Message format conversion
- Error handling
- Integration with RAG query engine
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag.config import (
    MistralConfig,
    OpenAIConfig,
    Config,
    PathsConfig,
    EmbeddingConfig,
    QdrantConfig,
    LLMConfig,
    GeminiConfig,
    ProcessingConfig,
    LoggingConfig,
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
            embeddings_cache="/test/embeddings"
        ),
        embedding=EmbeddingConfig(
            provider="gemini",
            model_name="intfloat/multilingual-e5-large",
            batch_size=32,
            device="cpu"
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            vector_size=768,  # Gemini embedding size
            distance="Cosine"
        ),
        llm=LLMConfig(provider="gemini"),
        gemini=GeminiConfig(
            api_key="test-gemini-key",
            model_name="gemini-2.5-flash",
            embedding_model="models/text-embedding-004",
            max_tokens=512,
            temperature=0.3,
            top_k=5
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
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

    # Mock search results
    search_result = MagicMock()
    search_result.id = "doc1"
    search_result.score = 0.95
    search_result.payload = {
        "text": "Test document content about AI.",
        "language": "en",
        "path": "/test/doc1.vtt"
    }
    client.search.return_value = [search_result]

    return client


# ============================================================================
# Gemini Embedding Tests
# ============================================================================

def test_embed_query_gemini_success(gemini_config, mock_qdrant_client):
    """Test successful query embedding with Gemini API."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            # Mock embed_content
            mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
            mock_genai.configure = MagicMock()

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
            mock_genai.embed_content.assert_called_once_with(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )


def test_embed_query_gemini_error(gemini_config, mock_qdrant_client):
    """Test Gemini embedding API error handling."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.embed_content.side_effect = Exception("API Error")
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Should raise RuntimeError
            with pytest.raises(RuntimeError) as exc_info:
                engine.embed_query("test query")

            assert "Gemini embeddings API error" in str(exc_info.value)


def test_embed_query_gemini_different_model(gemini_config, mock_qdrant_client):
    """Test embedding with different Gemini model."""
    gemini_config.gemini.embedding_model = "models/embedding-001"

    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            engine.embed_query("test")

            # Verify correct model was used
            call_args = mock_genai.embed_content.call_args
            assert call_args[1]["model"] == "models/embedding-001"


def test_embed_query_gemini_task_type(gemini_config, mock_qdrant_client):
    """Test that Gemini embedding uses correct task_type."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            engine.embed_query("test query")

            # Verify task_type is set for retrieval
            call_args = mock_genai.embed_content.call_args
            assert call_args[1]["task_type"] == "retrieval_query"


# ============================================================================
# Gemini LLM Generation Tests
# ============================================================================

def test_generate_answer_gemini_success(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test successful answer generation with Gemini API."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Generate answer
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is AI?"}
            ]

            answer = engine.generate_answer("test", messages)

            # Verify
            assert answer == "This is a test response from Gemini."

            # Verify Gemini model was created correctly
            mock_genai.GenerativeModel.assert_called_with("gemini-2.5-flash")


def test_generate_answer_gemini_system_message_handling(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test that system message is combined with user message for Gemini."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Messages with system message
            messages = [
                {"role": "system", "content": "You are an expert in machine learning."},
                {"role": "user", "content": "Explain neural networks."}
            ]

            engine.generate_answer("test", messages)

            # Verify system message was combined with user prompt
            call_args = mock_gemini_model.generate_content.call_args
            prompt = call_args[0][0]
            assert "You are an expert in machine learning." in prompt
            assert "Explain neural networks." in prompt


def test_generate_answer_gemini_no_system_message(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test Gemini generation when no system message is present."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Messages without system message
            messages = [
                {"role": "user", "content": "What is AI?"}
            ]

            engine.generate_answer("test", messages)

            # Verify only user message is in prompt
            call_args = mock_gemini_model.generate_content.call_args
            prompt = call_args[0][0]
            assert prompt == "What is AI?"


def test_generate_answer_gemini_config_params(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test that generation config parameters are passed correctly."""
    gemini_config.gemini.max_tokens = 1024
    gemini_config.gemini.temperature = 0.7

    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            generation_config = MagicMock()
            mock_genai.GenerationConfig.return_value = generation_config
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer("test", messages)

            # Verify GenerationConfig was created with correct params
            mock_genai.GenerationConfig.assert_called_with(
                max_output_tokens=1024,
                temperature=0.7
            )


def test_generate_answer_gemini_different_model(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test answer generation with different Gemini model."""
    gemini_config.gemini.model_name = "gemini-2.5-pro"

    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            messages = [{"role": "user", "content": "test"}]
            engine.generate_answer("test", messages)

            # Verify correct model was used
            mock_genai.GenerativeModel.assert_called_with("gemini-2.5-pro")


def test_generate_answer_gemini_error(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test Gemini LLM API error handling."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_gemini_model.generate_content.side_effect = Exception("API Error")
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            with pytest.raises(RuntimeError) as exc_info:
                engine.generate_answer("test", [{"role": "user", "content": "test"}])

            assert "Gemini API error" in str(exc_info.value)


def test_generate_answer_gemini_empty_response(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test handling of empty response from Gemini."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            # Mock empty response
            response = MagicMock()
            response.text = "  "
            mock_gemini_model.generate_content.return_value = response
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            answer = engine.generate_answer("test", [{"role": "user", "content": "test"}])

            # Should return empty string after strip
            assert answer == ""


# ============================================================================
# Gemini Full Query Pipeline Tests
# ============================================================================

def test_query_gemini_full_pipeline(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test full RAG query pipeline with Gemini."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            # Run full query
            result = engine.query(
                question="What is machine learning?",
                top_k=3,
                language="en"
            )

            # Verify result structure
            assert "answer" in result
            assert "retrieved_documents" in result
            assert "num_documents" in result

            assert result["answer"] == "This is a test response from Gemini."
            assert len(result["retrieved_documents"]) == 1
            assert result["num_documents"] == 1

            # Verify both embedding and generation were called
            assert mock_genai.embed_content.called
            assert mock_gemini_model.generate_content.called


def test_query_gemini_russian_language(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test Gemini query with Russian language."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            result = engine.query(
                question="Что такое машинное обучение?",
                top_k=5,
                language="ru"
            )

            # Verify result
            assert result["answer"] is not None

            # Verify system prompt included Russian instruction
            call_args = mock_gemini_model.generate_content.call_args
            prompt = call_args[0][0]
            assert "Russian" in prompt or "russian" in prompt.lower()


def test_query_gemini_no_documents_retrieved(gemini_config, mock_gemini_model, mock_qdrant_client):
    """Test Gemini query when no documents are retrieved."""
    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            # Mock empty search results
            mock_qdrant_client.search.return_value = []

            mock_genai.GenerativeModel.return_value = mock_gemini_model
            mock_genai.GenerationConfig = MagicMock()
            mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
            mock_genai.configure = MagicMock()

            engine = RAGQueryEngine(gemini_config)
            engine.initialize()

            result = engine.query(
                question="What is quantum physics?",
                top_k=5,
                language="en"
            )

            # Should still generate answer (without context)
            assert result["answer"] is not None
            assert len(result["retrieved_documents"]) == 0
            assert result["num_documents"] == 0
