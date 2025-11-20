"""
Integration tests for complete RAG pipelines with different providers.

This module tests:
- Full query pipeline (embedding + retrieval + LLM) for each provider
- Provider switching
- Mixed provider configurations (e.g., Mistral embeddings + OpenAI LLM)
- End-to-end workflows
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag.config import (
    Config, PathsConfig, EmbeddingConfig, QdrantConfig,
    LLMConfig, MistralConfig, OpenAIConfig, ClaudeConfig, GeminiConfig,
    ProcessingConfig, LoggingConfig, VideoConfig
)
from src.rainrag.query import RAGQueryEngine


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for integration tests."""
    client = MagicMock()

    # Mock query_points results (updated for new API)
    query_result = MagicMock()
    point = MagicMock()
    point.id = "doc1"
    point.score = 0.95
    point.payload = {
        "text": "Machine learning is a subset of artificial intelligence.",
        "language": "en",
        "path": "/test/doc1.vtt",
        "doc_id": "doc1"
    }
    query_result.points = [point]
    client.query_points.return_value = query_result

    return client


# ============================================================================
# Mistral Full Pipeline Tests
# ============================================================================

def test_mistral_full_pipeline_embeddings_and_llm(mock_qdrant_client):
    """Test full pipeline with Mistral for both embeddings and LLM."""
    config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
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
            api_key="test-key",
            model_name="mistral-small-latest"
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.Mistral') as mock_mistral_class:
        with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
            # Mock Mistral client
            mock_mistral = MagicMock()

            # Mock embedding response
            embedding_response = MagicMock()
            embedding_response.data = [MagicMock(embedding=[0.1] * 1024)]
            mock_mistral.embeddings.create.return_value = embedding_response

            # Mock chat response
            chat_response = MagicMock()
            chat_response.choices = [
                MagicMock(message=MagicMock(content="Machine learning is a subset of AI that enables computers to learn from data."))
            ]
            mock_mistral.chat.complete.return_value = chat_response

            mock_mistral_class.return_value = mock_mistral

            # Run full pipeline
            engine = RAGQueryEngine(config)
            engine.initialize()

            result = engine.query(
                question="What is machine learning?",
                top_k=5,
                language="en"
            )

            # Verify result
            assert result["answer"] is not None
            assert "machine learning" in result["answer"].lower()
            assert result["num_documents"] == 1

            # Verify both embeddings and chat were called
            assert mock_mistral.embeddings.create.called
            assert mock_mistral.chat.complete.called


# ============================================================================
# Mixed Provider Tests
# ============================================================================

def test_mistral_llm_with_local_embeddings(mock_qdrant_client):
    """Test Mistral LLM with local embeddings."""
    config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
        ),
        embedding=EmbeddingConfig(
            provider="local",
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
            api_key="test-key",
            model_name="mistral-small-latest"
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.SentenceTransformer') as mock_st:
        with patch('src.rainrag.query.Mistral') as mock_mistral_class:
            with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                # Mock local embeddings
                mock_model = MagicMock()
                mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
                mock_st.return_value = mock_model

                # Mock Mistral LLM
                mock_mistral = MagicMock()
                chat_response = MagicMock()
                chat_response.choices = [MagicMock(message=MagicMock(content="Test answer"))]
                mock_mistral.chat.complete.return_value = chat_response
                mock_mistral_class.return_value = mock_mistral

                # Run pipeline
                engine = RAGQueryEngine(config)
                engine.initialize()

                result = engine.query("test question", top_k=3, language="en")

                # Verify local embeddings were used
                assert mock_model.encode.called
                # Verify Mistral LLM was used
                assert mock_mistral.chat.complete.called
                # Verify result
                assert result["answer"] == "Test answer"


def test_openai_llm_with_mistral_embeddings(mock_qdrant_client):
    """Test OpenAI LLM with Mistral embeddings."""
    config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
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
        llm=LLMConfig(provider="openai"),
        mistral=MistralConfig(api_key="test-mistral-key", model_name="mistral-embed"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.Mistral') as mock_mistral_class:
        with patch('src.rainrag.query.OpenAI') as mock_openai_class:
            with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                # Mock Mistral embeddings
                mock_mistral = MagicMock()
                embedding_response = MagicMock()
                embedding_response.data = [MagicMock(embedding=[0.1] * 1024)]
                mock_mistral.embeddings.create.return_value = embedding_response
                mock_mistral_class.return_value = mock_mistral

                # Mock OpenAI LLM
                mock_openai = MagicMock()
                chat_response = MagicMock()
                chat_response.choices = [MagicMock(message=MagicMock(content="OpenAI answer"))]
                mock_openai.chat.completions.create.return_value = chat_response
                mock_openai_class.return_value = mock_openai

                # Run pipeline
                engine = RAGQueryEngine(config)
                engine.initialize()

                result = engine.query("test", top_k=5, language="en")

                # Verify Mistral embeddings were used
                assert mock_mistral.embeddings.create.called
                # Verify OpenAI LLM was used
                assert mock_openai.chat.completions.create.called
                # Verify result
                assert result["answer"] == "OpenAI answer"


def test_claude_llm_with_gemini_embeddings(mock_qdrant_client):
    """Test Claude LLM with Gemini embeddings."""
    config = Config(
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
            vector_size=768,
            distance="Cosine"
        ),
        llm=LLMConfig(provider="claude"),
        mistral=MistralConfig(
            api_key="test-mistral-key",
            model_name="mistral-small-latest"
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        claude=ClaudeConfig(
            api_key="test-claude-key",
            model_name="claude-haiku-4-5-20251001"
        ),
        gemini=GeminiConfig(
            api_key="test-gemini-key",
            model_name="gemini-2.5-flash",
            embedding_model="models/text-embedding-004"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.genai') as mock_genai:
        with patch('src.rainrag.query.Anthropic') as mock_anthropic_class:
            with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                # Mock Gemini embeddings
                mock_genai.embed_content.return_value = {'embedding': [0.1] * 768}
                mock_genai.configure = MagicMock()

                # Mock Claude LLM
                mock_claude = MagicMock()
                message_response = MagicMock()
                message_response.content = [MagicMock(text="Claude answer")]
                mock_claude.messages.create.return_value = message_response
                mock_anthropic_class.return_value = mock_claude

                # Run pipeline
                engine = RAGQueryEngine(config)
                engine.initialize()

                result = engine.query("test", top_k=5, language="en")

                # Verify Gemini embeddings were used
                assert mock_genai.embed_content.called
                # Verify Claude LLM was used
                assert mock_claude.messages.create.called
                # Verify result
                assert result["answer"] == "Claude answer"


# ============================================================================
# End-to-End Multilingual Tests
# ============================================================================

def test_multilingual_query_english(mock_qdrant_client):
    """Test end-to-end query in English."""
    config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
        ),
        embedding=EmbeddingConfig(
            provider="local",
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
        mistral=MistralConfig(api_key="test-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.SentenceTransformer') as mock_st:
        with patch('src.rainrag.query.Mistral') as mock_mistral_class:
            with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                # Mock embeddings
                mock_model = MagicMock()
                mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
                mock_st.return_value = mock_model

                # Mock Mistral
                mock_mistral = MagicMock()
                chat_response = MagicMock()
                chat_response.choices = [
                    MagicMock(message=MagicMock(content="Machine learning enables computers to learn from data without explicit programming."))
                ]
                mock_mistral.chat.complete.return_value = chat_response
                mock_mistral_class.return_value = mock_mistral

                # Run query in English
                engine = RAGQueryEngine(config)
                engine.initialize()

                result = engine.query(
                    question="What is machine learning?",
                    top_k=5,
                    language="en"
                )

                # Verify English response
                assert result["answer"] is not None
                assert len(result["answer"]) > 0


def test_multilingual_query_russian(mock_qdrant_client):
    """Test end-to-end query in Russian."""
    config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
        ),
        embedding=EmbeddingConfig(
            provider="local",
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
        llm=LLMConfig(provider="gemini"),
        mistral=MistralConfig(
            api_key="test-mistral-key",
            model_name="mistral-small-latest"
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        gemini=GeminiConfig(
            api_key="test-key",
            model_name="gemini-2.5-flash",
            embedding_model="models/text-embedding-004"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.SentenceTransformer') as mock_st:
        with patch('src.rainrag.query.genai') as mock_genai:
            with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                # Mock embeddings
                mock_model = MagicMock()
                mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
                mock_st.return_value = mock_model

                # Mock Gemini
                mock_gemini_model = MagicMock()
                response = MagicMock()
                response.text = "Машинное обучение - это метод искусственного интеллекта."
                mock_gemini_model.generate_content.return_value = response
                mock_genai.GenerativeModel.return_value = mock_gemini_model
                mock_genai.GenerationConfig = MagicMock()
                mock_genai.configure = MagicMock()

                # Run query in Russian
                engine = RAGQueryEngine(config)
                engine.initialize()

                result = engine.query(
                    question="Что такое машинное обучение?",
                    top_k=5,
                    language="ru"
                )

                # Verify Russian response
                assert result["answer"] is not None
                assert len(result["answer"]) > 0


# ============================================================================
# Provider Switching Tests
# ============================================================================

def test_switching_llm_providers(mock_qdrant_client):
    """Test switching between LLM providers without re-indexing."""
    base_config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
        ),
        embedding=EmbeddingConfig(
            provider="local",
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
        mistral=MistralConfig(api_key="test-mistral", model_name="mistral-small-latest"),
        openai=OpenAIConfig(api_key="test-openai", model_name="gpt-4o-mini", embedding_model="text-embedding-3-small"),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.SentenceTransformer') as mock_st:
        with patch('src.rainrag.query.Mistral') as mock_mistral_class:
            with patch('src.rainrag.query.OpenAI') as mock_openai_class:
                with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                    # Mock embeddings (same for both)
                    mock_model = MagicMock()
                    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
                    mock_st.return_value = mock_model

                    # Test with Mistral
                    mock_mistral = MagicMock()
                    chat_response = MagicMock()
                    chat_response.choices = [MagicMock(message=MagicMock(content="Mistral answer"))]
                    mock_mistral.chat.complete.return_value = chat_response
                    mock_mistral_class.return_value = mock_mistral

                    engine = RAGQueryEngine(base_config)
                    engine.initialize()
                    result1 = engine.query("test", top_k=3, language="en")
                    assert result1["answer"] == "Mistral answer"

                    # Switch to OpenAI (same embeddings, different LLM)
                    base_config.llm.provider = "openai"

                    mock_openai = MagicMock()
                    chat_response2 = MagicMock()
                    chat_response2.choices = [MagicMock(message=MagicMock(content="OpenAI answer"))]
                    mock_openai.chat.completions.create.return_value = chat_response2
                    mock_openai_class.return_value = mock_openai

                    engine2 = RAGQueryEngine(base_config)
                    engine2.initialize()
                    result2 = engine2.query("test", top_k=3, language="en")
                    assert result2["answer"] == "OpenAI answer"


# ============================================================================
# Error Recovery Tests
# ============================================================================

def test_pipeline_handles_empty_results(mock_qdrant_client):
    """Test pipeline when no documents are retrieved."""
    # Mock empty query_points results
    query_result = MagicMock()
    query_result.points = []
    mock_qdrant_client.query_points.return_value = query_result

    config = Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings"
        ),
        embedding=EmbeddingConfig(
            provider="local",
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
        mistral=MistralConfig(api_key="test-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small"
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True)
    )

    with patch('src.rainrag.query.SentenceTransformer') as mock_st:
        with patch('src.rainrag.query.Mistral') as mock_mistral_class:
            with patch('src.rainrag.query.QdrantClient', return_value=mock_qdrant_client):
                mock_model = MagicMock()
                mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1] * 1024)
                mock_st.return_value = mock_model

                mock_mistral = MagicMock()
                chat_response = MagicMock()
                chat_response.choices = [
                    MagicMock(message=MagicMock(content="I don't have information about that."))
                ]
                mock_mistral.chat.complete.return_value = chat_response
                mock_mistral_class.return_value = mock_mistral

                engine = RAGQueryEngine(config)
                engine.initialize()

                result = engine.query("unknown topic", top_k=5, language="en")

                # Should still return an answer (without context)
                assert result["answer"] is not None
                assert result["num_documents"] == 0
                assert len(result["retrieved_documents"]) == 0
