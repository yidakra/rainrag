"""
Provider test fixtures for mocking external API calls.

This module provides pytest fixtures for mocking:
- Mistral API (embeddings and LLM)
- OpenAI API (embeddings and LLM)
- Claude/Anthropic API (LLM)
- Google Gemini API (embeddings and LLM)
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock


# ============================================================================
# Mistral API Fixtures
# ============================================================================

@pytest.fixture
def mock_mistral_embedding_response():
    """Mock Mistral embedding API response."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1024)  # 1024-dim embedding
    ]
    return mock_response


@pytest.fixture
def mock_mistral_chat_response():
    """Mock Mistral chat completion API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="This is a test response from Mistral."))
    ]
    return mock_response


@pytest.fixture
def mock_mistral_client(mock_mistral_embedding_response, mock_mistral_chat_response):
    """Mock Mistral client with embeddings and chat methods."""
    client = MagicMock()
    client.embeddings.create.return_value = mock_mistral_embedding_response
    client.chat.complete.return_value = mock_mistral_chat_response
    return client


# ============================================================================
# OpenAI API Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_embedding_response():
    """Mock OpenAI embedding API response."""
    mock_response = MagicMock()
    mock_response.data = [
        MagicMock(embedding=[0.1] * 1536)  # text-embedding-3-small is 1536-dim
    ]
    return mock_response


@pytest.fixture
def mock_openai_chat_response():
    """Mock OpenAI chat completion API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="This is a test response from OpenAI."))
    ]
    return mock_response


@pytest.fixture
def mock_openai_client(mock_openai_embedding_response, mock_openai_chat_response):
    """Mock OpenAI client with embeddings and chat methods."""
    client = MagicMock()
    client.embeddings.create.return_value = mock_openai_embedding_response
    client.chat.completions.create.return_value = mock_openai_chat_response
    return client


# ============================================================================
# Claude/Anthropic API Fixtures
# ============================================================================

@pytest.fixture
def mock_claude_message_response():
    """Mock Claude message API response."""
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(text="This is a test response from Claude.")
    ]
    return mock_response


@pytest.fixture
def mock_claude_client(mock_claude_message_response):
    """Mock Claude client with messages method."""
    client = MagicMock()
    client.messages.create.return_value = mock_claude_message_response
    return client


# ============================================================================
# Google Gemini API Fixtures
# ============================================================================

@pytest.fixture
def mock_gemini_embedding_response():
    """Mock Gemini embedding API response."""
    mock_response = {
        'embedding': [0.1] * 768  # text-embedding-004 is 768-dim
    }
    return mock_response


@pytest.fixture
def mock_gemini_generate_response():
    """Mock Gemini generate_content API response."""
    mock_response = MagicMock()
    mock_response.text = "This is a test response from Gemini."
    return mock_response


@pytest.fixture
def mock_gemini_model(mock_gemini_generate_response):
    """Mock Gemini GenerativeModel."""
    model = MagicMock()
    model.generate_content.return_value = mock_gemini_generate_response
    return model


# ============================================================================
# Provider Error Fixtures
# ============================================================================

@pytest.fixture
def mock_api_error():
    """Mock API error for testing error handling."""
    error = Exception("API request failed")
    return error


@pytest.fixture
def mock_rate_limit_error():
    """Mock rate limit error."""
    error = Exception("Rate limit exceeded")
    return error


@pytest.fixture
def mock_auth_error():
    """Mock authentication error."""
    error = Exception("Invalid API key")
    return error


# ============================================================================
# Multi-Provider Test Data
# ============================================================================

@pytest.fixture
def provider_configs():
    """Configuration data for all providers."""
    return {
        "mistral": {
            "api_key": "test-mistral-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5
        },
        "openai": {
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5
        },
        "claude": {
            "api_key": "test-claude-key",
            "model_name": "claude-haiku-4-5-20251001",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5
        },
        "gemini": {
            "api_key": "test-gemini-key",
            "model_name": "gemini-2.5-flash",
            "embedding_model": "models/text-embedding-004",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5
        }
    }


@pytest.fixture
def sample_query_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]


@pytest.fixture
def sample_embedding_texts():
    """Sample texts for embedding generation."""
    return [
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of AI.",
        "Natural language processing enables computers to understand human language."
    ]


@pytest.fixture
def sample_retrieved_documents():
    """Sample retrieved documents for RAG context."""
    return [
        {
            "id": "doc1",
            "text": "Paris is the capital of France.",
            "score": 0.95,
            "metadata": {"language": "en", "path": "/test/doc1.vtt"}
        },
        {
            "id": "doc2",
            "text": "France is located in Western Europe.",
            "score": 0.85,
            "metadata": {"language": "en", "path": "/test/doc2.vtt"}
        }
    ]
