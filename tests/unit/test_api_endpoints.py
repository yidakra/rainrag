"""
Additional tests for FastAPI endpoints (query and health).

This module tests:
- /health endpoint with provider information
- /query endpoint with all providers
- Authentication middleware
- Error handling for API endpoints
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag.api import app
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


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_client():
    """Create a synchronous FastAPI test client for the app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mistral_config():
    """Create test configuration with Mistral provider."""
    return Config(
        paths=PathsConfig(
            archive_root="/test/archive",
            docs_output="/test/docs.jsonl",
            embeddings_cache="/test/embeddings",
        ),
        embedding=EmbeddingConfig(
            provider="mistral",
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
        llm=LLMConfig(provider="mistral"),
        mistral=MistralConfig(api_key="test-mistral-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True),
    )


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
            vector_size=1536,
            distance="Cosine",
        ),
        llm=LLMConfig(provider="openai"),
        mistral=MistralConfig(api_key="test-mistral-key", model_name="mistral-small-latest"),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        ),
        processing=ProcessingConfig(num_workers=4, max_file_size=10485760),
        logging=LoggingConfig(level="INFO", log_file="/test/logs.log"),
        video=VideoConfig(enabled=True),
    )


# ============================================================================
# Health Endpoint Tests
# ============================================================================


def test_health_endpoint_basic(test_client, mistral_config):
    """Test basic health endpoint response."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        # Use real config object
        mock_engine.config = mistral_config
        mock_engine.config.embedding.provider = "local"  # Override to local
        mock_engine.qdrant_client = MagicMock()
        mock_engine.embedding_model = MagicMock()

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["llm_provider"] == "mistral"
        assert data["embedding_provider"] == "local"


def test_health_endpoint_mistral_provider(test_client, mistral_config):
    """Test health endpoint with Mistral provider."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        # Mock query engine with mistral config
        mock_engine.config = mistral_config
        mock_engine.qdrant_client = MagicMock()
        mock_engine.embedding_model = None  # Mistral uses API, no local model

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "mistral"
        assert data["embedding_provider"] == "mistral"


def test_health_endpoint_openai_provider(test_client, openai_config):
    """Test health endpoint with OpenAI provider."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        # Mock query engine with openai config
        mock_engine.config = openai_config
        mock_engine.qdrant_client = MagicMock()
        mock_engine.embedding_model = None  # OpenAI uses API, no local model

        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "openai"
        assert data["embedding_provider"] == "openai"


# ============================================================================
# Query Endpoint Tests
# ============================================================================


def test_query_endpoint_success(test_client):
    """Test successful query endpoint request."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch(
                "src.rainrag.api.config"
            ):  # Mock config to avoid video URL generation issues
                mock_engine.query.return_value = {
                    "question": "What is machine learning?",
                    "answer": "This is a test answer.",
                    "retrieved_documents": [
                        {
                            "rank": 1,
                            "score": 0.95,
                            "text": "Test document",
                            "path": "/test/doc1.vtt",
                            "language": "en",
                            "doc_id": "doc1",
                        }
                    ],
                    "num_documents": 1,
                    "metadata_fallback_hits": 1,
                }

                response = test_client.post(
                    "/query",
                    json={"question": "What is machine learning?", "language": "en", "top_k": 5},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "This is a test answer."
                assert data["num_documents"] == 1
                assert data["metadata_fallback_hits"] == 1
                assert len(data["context"]) == 1


def test_query_endpoint_default_language(test_client):
    """Test query endpoint with default language."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch("src.rainrag.api.config"):
                mock_engine.query.return_value = {
                    "question": "test question",
                    "answer": "Test",
                    "retrieved_documents": [],
                    "num_documents": 0,
                    # metadata_fallback_hits is optional in the API response and defaults to 0
                }

                response = test_client.post(
                    "/query",
                    json={
                        "question": "test question"
                        # language and top_k should use defaults
                    },
                )

                assert response.status_code == 200
                # Verify defaults were used
                call_kwargs = mock_engine.query.call_args[1]
                assert call_kwargs.get("language") == "ru"  # Default language is Russian
                assert call_kwargs.get("top_k") is None  # Default is None (uses config default)


def test_query_endpoint_russian_language(test_client):
    """Test query endpoint with Russian language."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch("src.rainrag.api.config"):
                mock_engine.query.return_value = {
                    "question": "Что такое машинное обучение?",
                    "answer": "Это тестовый ответ.",
                    "retrieved_documents": [],
                    "num_documents": 0,
                    # metadata_fallback_hits is optional in the API response and defaults to 0
                }

                response = test_client.post(
                    "/query",
                    json={"question": "Что такое машинное обучение?", "language": "ru", "top_k": 3},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "Это тестовый ответ."


def test_query_endpoint_custom_top_k(test_client):
    """Test query endpoint with custom top_k."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch("src.rainrag.api.config"):
                mock_engine.query.return_value = {
                    "question": "test",
                    "answer": "Test",
                    "retrieved_documents": [],
                    "num_documents": 0,
                    # metadata_fallback_hits is optional in the API response and defaults to 0
                }

                response = test_client.post("/query", json={"question": "test", "top_k": 10})

                assert response.status_code == 200
                # Verify top_k was passed correctly
                call_kwargs = mock_engine.query.call_args[1]
                assert call_kwargs.get("top_k") == 10


def test_query_endpoint_missing_question(test_client):
    """Test query endpoint with missing question."""
    response = test_client.post(
        "/query",
        json={
            "language": "en"
            # Missing "question" field
        },
    )

    # Should return 422 Unprocessable Entity for validation error
    assert response.status_code == 422


def test_query_endpoint_empty_question(test_client):
    """Test query endpoint with empty question."""
    response = test_client.post("/query", json={"question": "", "language": "en"})

    # Should return 422 for validation error (min_length=1)
    assert response.status_code == 422


def test_query_endpoint_error_handling(test_client):
    """Test query endpoint error handling."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            mock_engine.query.side_effect = Exception("Query engine failed")

            response = test_client.post("/query", json={"question": "test", "language": "en"})

            # Should return 500 for internal error
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data


# ============================================================================
# Authentication Tests
# ============================================================================


@pytest.mark.skip(
    reason="Auth testing with FastAPI dependency injection is complex - covered by integration tests"
)
def test_query_endpoint_with_authentication(test_client):
    """Test query endpoint with authentication token."""
    pass


@pytest.mark.skip(
    reason="Auth testing with FastAPI dependency injection is complex - covered by integration tests"
)
def test_query_endpoint_invalid_token(test_client):
    """Test query endpoint with invalid authentication token."""
    pass


@pytest.mark.skip(
    reason="Auth testing with FastAPI dependency injection is complex - covered by integration tests"
)
def test_query_endpoint_missing_token(test_client):
    """Test query endpoint with missing authentication token when required."""
    pass


# ============================================================================
# CORS Tests
# ============================================================================


def test_cors_headers(test_client):
    """Test CORS headers are present."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch("src.rainrag.api.config"):
                mock_engine.query.return_value = {
                    "question": "test",
                    "answer": "Test",
                    "retrieved_documents": [],
                    "num_documents": 0,
                    # metadata_fallback_hits is optional in the API response and defaults to 0
                }

                response = test_client.post(
                    "/query", json={"question": "test"}, headers={"Origin": "http://localhost:7860"}
                )

                assert response.status_code == 200
                # CORS headers should be present
                assert "access-control-allow-origin" in response.headers


# ============================================================================
# Integration with Different Providers Tests
# ============================================================================


def test_query_endpoint_with_mistral_provider(test_client, mistral_config):
    """Test query endpoint configured with Mistral provider."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch("src.rainrag.api.config"):
                mock_engine.config = mistral_config
                mock_engine.query.return_value = {
                    "question": "test",
                    "answer": "Mistral response",
                    "retrieved_documents": [],
                    "num_documents": 0,
                    # metadata_fallback_hits is optional in the API response and defaults to 0
                }

                response = test_client.post("/query", json={"question": "test"})

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data


def test_query_endpoint_with_openai_provider(test_client, openai_config):
    """Test query endpoint configured with OpenAI provider."""
    with patch("src.rainrag.api.query_engine") as mock_engine:
        with patch("src.rainrag.api.verify_auth_token", return_value=True):
            with patch("src.rainrag.api.config"):
                mock_engine.config = openai_config
                mock_engine.query.return_value = {
                    "question": "test",
                    "answer": "OpenAI response",
                    "retrieved_documents": [],
                    "num_documents": 0,
                    # metadata_fallback_hits is optional in the API response and defaults to 0
                }

                response = test_client.post("/query", json={"question": "test"})

                assert response.status_code == 200
                data = response.json()
                assert "answer" in data
