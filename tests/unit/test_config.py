"""Unit tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from rainrag.config import (
    Config,
    EmbeddingConfig,
    LoggingConfig,
    PathsConfig,
    ProcessingConfig,
    QdrantConfig,
    MistralConfig,
    load_config,
)


class TestPathsConfig:
    """Tests for PathsConfig model."""

    def test_paths_config_creation(self) -> None:
        """Test creating paths configuration."""
        config = PathsConfig(
            archive_root="/data/archive",
            docs_output="/data/docs.jsonl",
            embeddings_cache="/data/embeddings",
        )

        assert config.archive_root == "/data/archive"
        assert config.docs_output == "/data/docs.jsonl"
        assert config.embeddings_cache == "/data/embeddings"


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_embedding_config_defaults(self) -> None:
        """Test default embedding configuration."""
        config = EmbeddingConfig()

        assert config.model_name == "intfloat/multilingual-e5-large"
        assert config.batch_size == 32
        assert config.max_seq_length == 512
        assert config.device == "cuda"
        assert config.normalize_embeddings is True

    def test_embedding_config_custom(self) -> None:
        """Test custom embedding configuration."""
        config = EmbeddingConfig(
            model_name="custom-model",
            batch_size=64,
            device="cpu",
        )

        assert config.model_name == "custom-model"
        assert config.batch_size == 64
        assert config.device == "cpu"


class TestQdrantConfig:
    """Tests for QdrantConfig model."""

    def test_qdrant_config_defaults(self) -> None:
        """Test default Qdrant configuration."""
        config = QdrantConfig()

        assert config.host == "localhost"
        assert config.port == 6333
        assert config.collection_name == "broadcast_transcripts"
        assert config.vector_size == 1024
        assert config.distance == "Cosine"
        assert config.recreate_collection is False

    def test_qdrant_config_custom(self) -> None:
        """Test custom Qdrant configuration."""
        config = QdrantConfig(
            host="qdrant-server",
            port=6334,
            collection_name="test_collection",
            vector_size=384,
            distance="Euclidean",
        )

        assert config.host == "qdrant-server"
        assert config.port == 6334
        assert config.collection_name == "test_collection"
        assert config.vector_size == 384
        assert config.distance == "Euclidean"


class TestMistralConfig:
    """Tests for MistralConfig model."""

    def test_mistral_config_defaults(self) -> None:
        """Test default Mistral configuration."""
        config = MistralConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.model_name == "mistral-small-latest"
        assert config.max_tokens == 512
        assert config.temperature == 0.3
        assert config.top_k == 5

    def test_mistral_config_custom(self) -> None:
        """Test custom Mistral configuration."""
        config = MistralConfig(
            api_key="custom-key",
            model_name="mistral-large-latest",
            max_tokens=1024,
            temperature=0.7,
            top_k=10,
        )

        assert config.api_key == "custom-key"
        assert config.model_name == "mistral-large-latest"
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_k == 10


class TestProcessingConfig:
    """Tests for ProcessingConfig model."""

    def test_processing_config_defaults(self) -> None:
        """Test default processing configuration."""
        config = ProcessingConfig()

        assert config.num_workers == 4
        assert config.max_file_size == 10485760
        assert config.min_text_length == 50


class TestLoggingConfig:
    """Tests for LoggingConfig model."""

    def test_logging_config_defaults(self) -> None:
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.log_file == "./logs/rainrag.log"


class TestConfig:
    """Tests for main Config model."""

    def test_config_creation(self) -> None:
        """Test creating complete configuration."""
        config = Config(
            paths=PathsConfig(
                archive_root="/data/archive",
                docs_output="/data/docs.jsonl",
                embeddings_cache="/data/embeddings",
            ),
            embedding=EmbeddingConfig(),
            qdrant=QdrantConfig(),
            mistral=MistralConfig(api_key="test-key"),
            processing=ProcessingConfig(),
            logging=LoggingConfig(),
        )

        assert config.paths.archive_root == "/data/archive"
        assert config.embedding.model_name == "intfloat/multilingual-e5-large"
        assert config.qdrant.host == "localhost"
        assert config.mistral.api_key == "test-key"
        assert config.mistral.model_name == "mistral-small-latest"
        assert config.processing.num_workers == 4
        assert config.logging.level == "INFO"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_file(self) -> None:
        """Test loading configuration from YAML file."""
        config_data = {
            "paths": {
                "archive_root": "/test/archive",
                "docs_output": "/test/docs.jsonl",
                "embeddings_cache": "/test/embeddings",
            },
            "embedding": {
                "model_name": "test-model",
                "batch_size": 16,
                "max_seq_length": 256,
                "device": "cpu",
                "normalize_embeddings": False,
            },
            "qdrant": {
                "host": "test-host",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 384,
                "distance": "Cosine",
                "recreate_collection": True,
            },
            "mistral": {
                "api_key": "test-key",
                "model_name": "mistral-large-latest",
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_k": 10,
            },
            "processing": {
                "num_workers": 2,
                "max_file_size": 1048576,
                "min_text_length": 10,
            },
            "logging": {
                "level": "DEBUG",
                "format": "{message}",
                "log_file": "/test/logs/test.log",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)

            assert config.paths.archive_root == "/test/archive"
            assert config.embedding.model_name == "test-model"
            assert config.embedding.batch_size == 16
            assert config.qdrant.host == "test-host"
            assert config.qdrant.recreate_collection is True
            assert config.mistral.api_key == "test-key"
            assert config.mistral.model_name == "mistral-large-latest"
            assert config.mistral.temperature == 0.7
            assert config.processing.num_workers == 2
            assert config.logging.level == "DEBUG"

        finally:
            Path(config_path).unlink()

    def test_load_config_file_not_found(self) -> None:
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_config_invalid_yaml(self) -> None:
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            config_path = f.name

        try:
            with pytest.raises(Exception):  # YAML parsing error
                load_config(config_path)
        finally:
            Path(config_path).unlink()

    def test_load_config_missing_required_fields(self) -> None:
        """Test loading config with missing required fields."""
        config_data = {
            "paths": {
                "archive_root": "/test/archive",
                # Missing required fields
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(Exception):  # Pydantic validation error
                load_config(config_path)
        finally:
            Path(config_path).unlink()
