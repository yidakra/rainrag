"""Unit tests for MCP server module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from rainrag.mcp_server import get_query_engine, initialize_server


class TestMCPServerInitialization:
    """Tests for MCP server initialization."""

    def test_initialize_server(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test MCP server initialization with config."""
        # Create a temporary config file
        config_data = {
            "paths": {
                "archive_root": "/test/archive",
                "docs_output": "/test/docs.jsonl",
                "embeddings_cache": "/test/embeddings",
            },
            "embedding": {
                "provider": "mistral",
                "model_name": "intfloat/multilingual-e5-large",
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 1024,
            },
            "llm": {"provider": "mistral"},
            "mistral": {
                "api_key": "test-key",
                "model_name": "mistral-small-latest",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
            "openai": {"api_key": "test-openai-key"},
            "processing": {"num_workers": 4},
            "logging": {"level": "INFO"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Mock the RAGQueryEngine and its initialize method
            with patch("rainrag.mcp_server.RAGQueryEngine") as mock_engine_class:
                mock_engine = MagicMock()
                mock_engine_class.return_value = mock_engine

                # Initialize the server
                initialize_server(config_path)

                # Verify the engine was created and initialized
                mock_engine_class.assert_called_once()
                mock_engine.initialize.assert_called_once()

                # Verify we can get the engine
                engine = get_query_engine()
                assert engine is mock_engine

        finally:
            Path(config_path).unlink()

    def test_get_query_engine_before_init(self) -> None:
        """Test getting query engine before initialization."""
        # Reset the global engine
        import rainrag.mcp_server

        rainrag.mcp_server._query_engine = None

        with pytest.raises(RuntimeError, match="Query engine not initialized"):
            get_query_engine()


class TestMCPServerTools:
    """Tests for MCP server tools."""

    def test_query_rag_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test query_rag tool function."""
        # Mock the query engine
        mock_engine = MagicMock()
        mock_engine.query.return_value = {
            "question": "test question",
            "answer": "test answer",
            "retrieved_documents": [
                {
                    "rank": 1,
                    "score": 0.95,
                    "text": "test document",
                    "path": "/test/file.vtt",
                    "language": "en",
                    "doc_id": "doc1",
                }
            ],
            "num_documents": 1,
        }

        # Import after mocking to avoid initialization
        import rainrag.mcp_server

        rainrag.mcp_server._query_engine = mock_engine

        from rainrag.mcp_server import query_rag

        # Call the tool
        result = query_rag(question="test question", language="en", top_k=5)

        # Verify the engine was called correctly
        mock_engine.query.assert_called_once_with(
            question="test question", top_k=5, language="en", date_from=None, date_to=None
        )

        # Verify the result
        assert result["question"] == "test question"
        assert result["answer"] == "test answer"
        assert result["num_documents"] == 1
        assert len(result["retrieved_documents"]) == 1

    def test_retrieve_documents_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test retrieve_documents tool function."""
        # Mock the query engine
        mock_engine = MagicMock()
        mock_engine.embed_query.return_value = [0.1] * 1024
        mock_engine.retrieve_documents.return_value = [
            {
                "rank": 1,
                "score": 0.95,
                "text": "test document 1",
                "path": "/test/file1.vtt",
                "language": "en",
                "doc_id": "doc1",
            },
            {
                "rank": 2,
                "score": 0.85,
                "text": "test document 2",
                "path": "/test/file2.vtt",
                "language": "en",
                "doc_id": "doc2",
            },
        ]

        # Import after mocking to avoid initialization
        import rainrag.mcp_server

        rainrag.mcp_server._query_engine = mock_engine

        from rainrag.mcp_server import retrieve_documents

        # Call the tool
        result = retrieve_documents(question="test query", top_k=3)

        # Verify the engine methods were called correctly
        mock_engine.embed_query.assert_called_once_with("test query")
        mock_engine.retrieve_documents.assert_called_once_with(
            [0.1] * 1024, 3, date_from=None, date_to=None
        )

        # Verify the result
        assert result["question"] == "test query"
        assert result["num_documents"] == 2
        assert len(result["documents"]) == 2
        assert result["documents"][0]["rank"] == 1
        assert result["documents"][1]["rank"] == 2

    def test_get_current_config_resource(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_current_config resource function."""
        # Mock the config
        from rainrag.config import (
            Config,
            EmbeddingConfig,
            LLMConfig,
            LoggingConfig,
            MistralConfig,
            OpenAIConfig,
            PathsConfig,
            ProcessingConfig,
            QdrantConfig,
        )

        mock_config = Config(
            paths=PathsConfig(
                archive_root="/test/archive",
                docs_output="/test/docs.jsonl",
                embeddings_cache="/test/embeddings",
            ),
            embedding=EmbeddingConfig(provider="mistral"),
            qdrant=QdrantConfig(
                host="localhost",
                port=6333,
                collection_name="test_collection",
                vector_size=1024,
            ),
            llm=LLMConfig(provider="mistral"),
            mistral=MistralConfig(api_key="test-key", model_name="mistral-small-latest"),
            openai=OpenAIConfig(api_key="test-key"),
            processing=ProcessingConfig(),
            logging=LoggingConfig(),
        )

        # Import after mocking
        import rainrag.mcp_server

        rainrag.mcp_server._config = mock_config

        from rainrag.mcp_server import get_current_config

        # Call the resource function
        result = get_current_config()

        # Verify the result contains expected information
        assert "RainRAG Configuration" in result
        assert "Embedding Provider: mistral" in result
        assert "LLM Provider: mistral" in result
        assert "Vector Database: Qdrant at localhost:6333" in result
        assert "Collection: test_collection" in result
        assert "Mistral Model: mistral-small-latest" in result


class TestMCPServerRunning:
    """Tests for running MCP server."""

    def test_run_server_with_stdio(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test running MCP server with stdio transport."""
        # Create a temporary config file
        config_data = {
            "paths": {
                "archive_root": "/test/archive",
                "docs_output": "/test/docs.jsonl",
                "embeddings_cache": "/test/embeddings",
            },
            "embedding": {"provider": "mistral"},
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 1024,
            },
            "llm": {"provider": "mistral"},
            "mistral": {
                "api_key": "test-key",
                "model_name": "mistral-small-latest",
            },
            "openai": {"api_key": "test-openai-key"},
            "processing": {},
            "logging": {},
            "mcp": {
                "transport": "stdio",
                "host": "localhost",
                "port": 8000,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Mock dependencies
            with patch("rainrag.mcp_server.initialize_server") as mock_init:
                with patch("rainrag.mcp_server._config") as mock_config:
                    with patch("rainrag.mcp_server.mcp") as mock_mcp:
                        # Setup mock config
                        mock_mcp_config = MagicMock()
                        mock_mcp_config.transport = "stdio"
                        mock_mcp_config.host = "localhost"
                        mock_mcp_config.port = 8000
                        mock_config.mcp = mock_mcp_config

                        from rainrag.mcp_server import run_server

                        # Run the server
                        run_server(config_path, transport="stdio")

                        # Verify initialization was called
                        mock_init.assert_called_once_with(config_path)

                        # Verify mcp.run was called with correct transport
                        mock_mcp.run.assert_called_once()
                        call_kwargs = mock_mcp.run.call_args.kwargs
                        assert call_kwargs.get("transport") == "stdio"

        finally:
            Path(config_path).unlink()

    def test_run_server_with_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test running MCP server with HTTP transport."""
        # Create a temporary config file
        config_data = {
            "paths": {
                "archive_root": "/test/archive",
                "docs_output": "/test/docs.jsonl",
                "embeddings_cache": "/test/embeddings",
            },
            "embedding": {"provider": "mistral"},
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 1024,
            },
            "llm": {"provider": "mistral"},
            "mistral": {
                "api_key": "test-key",
                "model_name": "mistral-small-latest",
            },
            "openai": {"api_key": "test-openai-key"},
            "processing": {},
            "logging": {},
            "mcp": {
                "transport": "streamable-http",
                "host": "0.0.0.0",
                "port": 9000,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            # Mock dependencies
            with patch("rainrag.mcp_server.initialize_server") as mock_init:
                with patch("rainrag.mcp_server._config") as mock_config:
                    with patch("rainrag.mcp_server.mcp") as mock_mcp:
                        with patch("rainrag.mcp_server.uvicorn") as mock_uvicorn:
                            # Setup mock config
                            mock_mcp_config = MagicMock()
                            mock_mcp_config.transport = "streamable-http"
                            mock_mcp_config.host = "0.0.0.0"
                            mock_mcp_config.port = 9000
                            mock_config.mcp = mock_mcp_config

                            # Mock the ASGI app
                            mock_asgi_app = MagicMock()
                            mock_mcp.streamable_http_app.return_value = mock_asgi_app

                            from rainrag.mcp_server import run_server

                            # Run the server with HTTP transport
                            run_server(
                                config_path,
                                transport="streamable-http",
                                host="0.0.0.0",
                                port=9000,
                            )

                            # Verify initialization was called
                            mock_init.assert_called_once_with(config_path)

                            # Verify streamable_http_app was called to get ASGI app
                            mock_mcp.streamable_http_app.assert_called_once()

                            # Verify uvicorn.run was called with correct parameters
                            mock_uvicorn.run.assert_called_once_with(
                                mock_asgi_app, host="0.0.0.0", port=9000
                            )

        finally:
            Path(config_path).unlink()
