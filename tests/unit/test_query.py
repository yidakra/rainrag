"""Unit tests for query module."""

from typing import Any, Dict, List
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest
import requests

from rainrag.config import Config
from rainrag.query import RAGQueryEngine, run_query


class TestRAGQueryEngine:
    """Tests for RAGQueryEngine class."""

    def test_init(self, test_config: Config) -> None:
        """Test engine initialization."""
        engine = RAGQueryEngine(test_config)

        assert engine.config == test_config
        assert engine.embedding_model is None
        assert engine.qdrant_client is None
        assert engine.vllm_url == "http://localhost:8000/v1/completions"
        assert engine.session is not None  # Verify session is created

    def test_vllm_url_custom_config(self, test_config: Config) -> None:
        """Test vLLM URL construction with custom config."""
        test_config.vllm.host = "vllm-server"
        test_config.vllm.port = 9000

        engine = RAGQueryEngine(test_config)

        assert engine.vllm_url == "http://vllm-server:9000/v1/completions"

    @patch("rainrag.query.SentenceTransformer")
    @patch("rainrag.query.QdrantClient")
    def test_initialize(
        self, mock_qdrant_client: Mock, mock_sentence_transformer: Mock, test_config: Config
    ) -> None:
        """Test engine initialization."""
        # Setup mocks
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_qdrant_client.return_value = mock_client

        # Initialize engine
        engine = RAGQueryEngine(test_config)
        engine.initialize()

        # Verify model loading
        mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        assert engine.embedding_model == mock_model

        # Verify Qdrant connection
        mock_qdrant_client.assert_called_once_with(
            host="localhost",
            port=6333,
        )
        assert engine.qdrant_client == mock_client
        mock_client.get_collections.assert_called_once()

    @patch("rainrag.query.QdrantClient")
    def test_initialize_qdrant_connection_error(
        self, mock_qdrant_client: Mock, test_config: Config
    ) -> None:
        """Test initialization with Qdrant connection error."""
        mock_client = Mock()
        mock_client.get_collections.side_effect = Exception("Connection failed")
        mock_qdrant_client.return_value = mock_client

        engine = RAGQueryEngine(test_config)

        with patch("rainrag.query.SentenceTransformer"):
            with pytest.raises(Exception, match="Connection failed"):
                engine.initialize()

    def test_embed_query_not_initialized(self, test_config: Config) -> None:
        """Test embed_query without initialization."""
        engine = RAGQueryEngine(test_config)

        with pytest.raises(RuntimeError, match="Embedding model not initialized"):
            engine.embed_query("test query")

    def test_embed_query(self, test_config: Config) -> None:
        """Test query embedding."""
        engine = RAGQueryEngine(test_config)

        # Mock embedding model
        mock_model = Mock()
        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_model.encode.return_value = mock_embedding
        engine.embedding_model = mock_model

        # Embed query
        result = engine.embed_query("What is this about?")

        # Verify
        mock_model.encode.assert_called_once_with(
            "query: What is this about?",
            normalize_embeddings=True,
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_query_with_e5_prefix(self, test_config: Config) -> None:
        """Test that query is prefixed with 'query:' for E5 model."""
        engine = RAGQueryEngine(test_config)

        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        engine.embedding_model = mock_model

        engine.embed_query("test")

        # Check that the query was prefixed
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == "query: test"

    def test_retrieve_documents_not_initialized(self, test_config: Config) -> None:
        """Test retrieve_documents without initialization."""
        engine = RAGQueryEngine(test_config)

        with pytest.raises(RuntimeError, match="Qdrant client not initialized"):
            engine.retrieve_documents([0.1, 0.2], top_k=5)

    def test_retrieve_documents(self, test_config: Config) -> None:
        """Test document retrieval from Qdrant."""
        engine = RAGQueryEngine(test_config)

        # Mock Qdrant client
        mock_client = Mock()
        mock_hit1 = Mock()
        mock_hit1.score = 0.95
        mock_hit1.payload = {
            "text": "This is about energy",
            "path": "/path/to/file1.vtt",
            "language": "en",
            "doc_id": "doc1",
        }

        mock_hit2 = Mock()
        mock_hit2.score = 0.87
        mock_hit2.payload = {
            "text": "More about renewable energy",
            "path": "/path/to/file2.vtt",
            "language": "en",
            "doc_id": "doc2",
        }

        mock_client.search.return_value = [mock_hit1, mock_hit2]
        engine.qdrant_client = mock_client

        # Retrieve documents
        query_vector = [0.1, 0.2, 0.3]
        documents = engine.retrieve_documents(query_vector, top_k=2)

        # Verify search call
        mock_client.search.assert_called_once_with(
            collection_name="test_collection",
            query_vector=query_vector,
            limit=2,
        )

        # Verify results
        assert len(documents) == 2

        assert documents[0]["rank"] == 1
        assert documents[0]["score"] == 0.95
        assert documents[0]["text"] == "This is about energy"
        assert documents[0]["path"] == "/path/to/file1.vtt"
        assert documents[0]["language"] == "en"
        assert documents[0]["doc_id"] == "doc1"

        assert documents[1]["rank"] == 2
        assert documents[1]["score"] == 0.87

    def test_retrieve_documents_qdrant_error(self, test_config: Config) -> None:
        """Test retrieval with Qdrant error."""
        engine = RAGQueryEngine(test_config)

        mock_client = Mock()
        mock_client.search.side_effect = Exception("Search failed")
        engine.qdrant_client = mock_client

        with pytest.raises(Exception, match="Search failed"):
            engine.retrieve_documents([0.1, 0.2], top_k=5)

    def test_build_prompt(self, test_config: Config) -> None:
        """Test prompt building with context."""
        engine = RAGQueryEngine(test_config)

        documents = [
            {
                "rank": 1,
                "score": 0.95,
                "text": "Energy is important for sustainability.",
                "path": "/archive/energy_ep1.vtt",
                "language": "en",
                "doc_id": "doc1",
            },
            {
                "rank": 2,
                "score": 0.87,
                "text": "Renewable energy sources include solar and wind.",
                "path": "/archive/energy_ep2.vtt",
                "language": "en",
                "doc_id": "doc2",
            },
        ]

        question = "What is renewable energy?"
        prompt = engine.build_prompt(question, documents)

        # Verify prompt structure
        assert "You are an assistant" in prompt
        assert "video transcripts" in prompt
        assert question in prompt

        # Verify context includes both documents
        assert "[Document 1]" in prompt
        assert "[Document 2]" in prompt
        assert "Energy is important for sustainability." in prompt
        assert "Renewable energy sources include solar and wind." in prompt
        assert "/archive/energy_ep1.vtt" in prompt
        assert "/archive/energy_ep2.vtt" in prompt

    def test_build_prompt_empty_documents(self, test_config: Config) -> None:
        """Test prompt building with no documents."""
        engine = RAGQueryEngine(test_config)

        prompt = engine.build_prompt("What is energy?", [])

        assert "What is energy?" in prompt
        # Should still have prompt structure but no context
        assert "You are an assistant" in prompt

    def test_generate_answer(self, test_config: Config) -> None:
        """Test answer generation via vLLM."""
        engine = RAGQueryEngine(test_config)

        # Mock the session.post method
        with patch.object(engine.session, 'post') as mock_post:
            # Mock successful vLLM response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{"text": "  This is the generated answer.  "}]
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Generate answer
            prompt = "Test prompt with context"
            answer = engine.generate_answer(prompt)

            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:8000/v1/completions"

            payload = call_args[1]["json"]
            assert payload["model"] == "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
            assert payload["prompt"] == prompt
            assert payload["max_tokens"] == 512
            assert payload["temperature"] == 0.3
            assert payload["stream"] is False
            assert call_args[1]["timeout"] == 30  # Verify timeout is 30s

            # Verify answer is stripped
            assert answer == "This is the generated answer."

    def test_generate_answer_connection_error(self, test_config: Config) -> None:
        """Test answer generation with connection error and exception chaining."""
        engine = RAGQueryEngine(test_config)

        with patch.object(engine.session, 'post') as mock_post:
            original_error = requests.exceptions.ConnectionError("Connection failed")
            mock_post.side_effect = original_error

            with pytest.raises(RuntimeError, match="Cannot connect to vLLM server") as exc_info:
                engine.generate_answer("test prompt")

            # Verify exception chaining - original error is preserved
            assert exc_info.value.__cause__ is original_error

    def test_generate_answer_timeout(self, test_config: Config) -> None:
        """Test answer generation with timeout and exception chaining."""
        engine = RAGQueryEngine(test_config)

        with patch.object(engine.session, 'post') as mock_post:
            original_error = requests.exceptions.Timeout("Timeout")
            mock_post.side_effect = original_error

            with pytest.raises(RuntimeError, match="timed out after 30 seconds") as exc_info:
                engine.generate_answer("test prompt")

            # Verify exception chaining - original error is preserved
            assert exc_info.value.__cause__ is original_error

    def test_generate_answer_http_error(self, test_config: Config) -> None:
        """Test answer generation with HTTP error and exception chaining."""
        engine = RAGQueryEngine(test_config)

        with patch.object(engine.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"
            original_error = requests.exceptions.HTTPError("500 Error")
            original_error.response = mock_response
            mock_response.raise_for_status.side_effect = original_error
            mock_post.return_value = mock_response

            with pytest.raises(RuntimeError, match="vLLM server returned HTTP error") as exc_info:
                engine.generate_answer("test prompt")

            # Verify exception chaining - original error is preserved
            assert exc_info.value.__cause__ is original_error

    def test_query_full_pipeline(self, test_config: Config) -> None:
        """Test complete query pipeline."""
        engine = RAGQueryEngine(test_config)

        # Mock embedding model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        engine.embedding_model = mock_model

        # Mock Qdrant client
        mock_client = Mock()
        mock_hit = Mock()
        mock_hit.score = 0.95
        mock_hit.payload = {
            "text": "Test document content",
            "path": "/test.vtt",
            "language": "en",
            "doc_id": "doc1",
        }
        mock_client.search.return_value = [mock_hit]
        engine.qdrant_client = mock_client

        # Mock vLLM
        with patch.object(engine.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"text": "Generated answer"}]}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Execute query
            result = engine.query("What is this about?", top_k=1)

            # Verify result structure
            assert result["question"] == "What is this about?"
            assert result["answer"] == "Generated answer"
            assert result["num_documents"] == 1
            assert len(result["retrieved_documents"]) == 1
            assert result["retrieved_documents"][0]["text"] == "Test document content"

    def test_query_custom_top_k(self, test_config: Config) -> None:
        """Test query with custom top_k."""
        engine = RAGQueryEngine(test_config)

        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        engine.embedding_model = mock_model

        mock_client = Mock()
        mock_client.search.return_value = []
        engine.qdrant_client = mock_client

        with patch.object(engine.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"text": "Answer"}]}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            engine.query("Test?", top_k=10)

            # Verify search was called with top_k=10
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args
            assert call_args[1]["limit"] == 10

    def test_query_default_top_k(self, test_config: Config) -> None:
        """Test query with default top_k from config."""
        engine = RAGQueryEngine(test_config)

        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2])
        engine.embedding_model = mock_model

        mock_client = Mock()
        mock_client.search.return_value = []
        engine.qdrant_client = mock_client

        with patch.object(engine.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {"choices": [{"text": "Answer"}]}
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            engine.query("Test?")

            # Verify search was called with default top_k=5 from config
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args
            assert call_args[1]["limit"] == 5


class TestRunQuery:
    """Tests for run_query convenience function."""

    @patch("rainrag.query.RAGQueryEngine")
    @patch("rainrag.query.load_config")
    def test_run_query(self, mock_load_config: Mock, mock_engine_class: Mock) -> None:
        """Test run_query function."""
        # Setup mocks
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        mock_engine = Mock()
        mock_engine.query.return_value = {
            "question": "Test?",
            "answer": "Test answer",
            "retrieved_documents": [],
            "num_documents": 0,
        }
        mock_engine_class.return_value = mock_engine

        # Run query
        result = run_query("config.yaml", "Test?", top_k=3)

        # Verify
        mock_load_config.assert_called_once_with("config.yaml")
        mock_engine_class.assert_called_once_with(mock_config)
        mock_engine.initialize.assert_called_once()
        mock_engine.query.assert_called_once_with("Test?", 3)

        assert result["answer"] == "Test answer"

    @patch("rainrag.query.RAGQueryEngine")
    @patch("rainrag.query.load_config")
    def test_run_query_default_top_k(
        self, mock_load_config: Mock, mock_engine_class: Mock
    ) -> None:
        """Test run_query with default top_k."""
        mock_config = Mock()
        mock_load_config.return_value = mock_config

        mock_engine = Mock()
        mock_engine.query.return_value = {"answer": "Test"}
        mock_engine_class.return_value = mock_engine

        run_query("config.yaml", "Test?")

        # Verify query called with None (uses config default)
        mock_engine.query.assert_called_once_with("Test?", None)
