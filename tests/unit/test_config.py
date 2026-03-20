"""Unit tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rainrag.config import (
    ClaudeConfig,
    Config,
    EmbeddingConfig,
    GeminiConfig,
    LLMConfig,
    LoggingConfig,
    MCPConfig,
    MistralConfig,
    OpenAIConfig,
    PathsConfig,
    ProcessingConfig,
    QdrantConfig,
    TwoStageConfig,
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

        assert config.provider == "local"
        assert config.model_name == "intfloat/multilingual-e5-large"
        assert config.batch_size == 32
        assert config.max_seq_length == 512
        assert config.device == "auto"
        assert config.normalize_embeddings is True
        assert config.prefix == ""  # default empty, no prefix applied

    def test_embedding_config_custom(self) -> None:
        """Test custom embedding configuration."""
        config = EmbeddingConfig(
            model_name="custom-model",
            batch_size=64,
            device="cpu",
            prefix="my-prefix: ",
        )

        assert config.model_name == "custom-model"
        assert config.batch_size == 64
        assert config.device == "cpu"
        assert config.prefix == "my-prefix: "


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
        assert config.max_tokens == 2048
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


class TestOpenAIConfig:
    """Tests for OpenAIConfig model."""

    def test_openai_config_defaults(self) -> None:
        """Test default OpenAI configuration."""
        config = OpenAIConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.model_name == "gpt-4o-mini"
        assert config.embedding_model == "text-embedding-3-small"
        assert config.max_tokens == 2048
        assert config.temperature == 0.3
        assert config.top_k == 5

    def test_openai_config_custom(self) -> None:
        """Test custom OpenAI configuration."""
        config = OpenAIConfig(
            api_key="custom-key",
            model_name="gpt-4o",
            embedding_model="text-embedding-3-large",
            max_tokens=1024,
            temperature=0.7,
            top_k=10,
        )

        assert config.api_key == "custom-key"
        assert config.model_name == "gpt-4o"
        assert config.embedding_model == "text-embedding-3-large"
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_k == 10


class TestClaudeConfig:
    """Tests for ClaudeConfig model."""

    def test_claude_config_defaults(self) -> None:
        """Test default Claude configuration."""
        config = ClaudeConfig()

        assert config.api_key == ""
        assert config.model_name == "claude-3-5-sonnet-20240620"
        assert config.max_tokens == 2048
        assert config.temperature == 0.3
        assert config.top_k == 5

    def test_claude_config_custom(self) -> None:
        """Test custom Claude configuration."""
        config = ClaudeConfig(
            api_key="custom-key",
            model_name="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0.7,
            top_k=10,
        )

        assert config.api_key == "custom-key"
        assert config.model_name == "claude-3-opus-20240229"
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_k == 10


class TestGeminiConfig:
    """Tests for GeminiConfig model."""

    def test_gemini_config_defaults(self) -> None:
        """Test default Gemini configuration."""
        config = GeminiConfig()

        assert config.api_key == ""
        assert config.model_name == "gemini-1.5-flash"
        assert config.embedding_model == "models/text-embedding-004"
        assert config.max_tokens == 2048
        assert config.temperature == 0.3
        assert config.top_k == 5

    def test_gemini_config_custom(self) -> None:
        """Test custom Gemini configuration."""
        config = GeminiConfig(
            api_key="custom-key",
            model_name="gemini-1.5-pro",
            embedding_model="models/embedding-001",
            max_tokens=1024,
            temperature=0.7,
            top_k=10,
        )

        assert config.api_key == "custom-key"
        assert config.model_name == "gemini-1.5-pro"
        assert config.embedding_model == "models/embedding-001"
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_k == 10


class TestTwoStageConfig:
    """Tests for TwoStageConfig model."""

    def test_two_stage_config_defaults(self) -> None:
        """Test default two-stage configuration."""
        config = TwoStageConfig()

        assert config.enabled is False
        assert config.query_rewrite_enabled is True
        assert config.query_rewrite_variants == 2
        assert config.query_rewrite_temperature == 0.7
        assert config.hyde_enabled is False
        assert config.hyde_alpha == 0.5
        assert config.hyde_temperature == 0.7

    def test_two_stage_config_custom(self) -> None:
        """Test custom two-stage configuration."""
        config = TwoStageConfig(
            enabled=True,
            query_rewrite_enabled=True,
            query_rewrite_variants=3,
            hyde_enabled=True,
            hyde_alpha=0.7,
        )

        assert config.enabled is True
        assert config.query_rewrite_enabled is True
        assert config.query_rewrite_variants == 3
        assert config.hyde_enabled is True
        assert config.hyde_alpha == 0.7

    def test_two_stage_config_alpha_bounds(self) -> None:
        """Test that hyde_alpha is bounded to [0, 1]."""
        # out-of-range values should raise
        with pytest.raises(ValidationError):
            TwoStageConfig(hyde_alpha=1.5)
        with pytest.raises(ValidationError):
            TwoStageConfig(hyde_alpha=-0.1)
        # boundary values should be accepted and preserved
        cfg_low = TwoStageConfig(hyde_alpha=0)
        cfg_high = TwoStageConfig(hyde_alpha=1)
        assert cfg_low.hyde_alpha == 0
        assert cfg_high.hyde_alpha == 1

    def test_two_stage_config_variants_bounds(self) -> None:
        """Test that query_rewrite_variants is bounded to [1, 5]."""
        # values outside range should raise
        with pytest.raises(ValidationError):
            TwoStageConfig(query_rewrite_variants=0)
        with pytest.raises(ValidationError):
            TwoStageConfig(query_rewrite_variants=6)
        # boundary values should be accepted and preserved (not silently clamped)
        cfg_low = TwoStageConfig(query_rewrite_variants=1)
        cfg_high = TwoStageConfig(query_rewrite_variants=5)
        assert cfg_low.query_rewrite_variants == 1
        assert cfg_high.query_rewrite_variants == 5

    def test_two_stage_config_rewrite_temperature_bounds(self) -> None:
        """Test that query_rewrite_temperature is bounded to [0, 2]."""
        with pytest.raises(ValidationError):
            TwoStageConfig(query_rewrite_temperature=-0.1)
        with pytest.raises(ValidationError):
            TwoStageConfig(query_rewrite_temperature=2.1)
        # Boundary values should be valid and preserved
        cfg_low = TwoStageConfig(query_rewrite_temperature=0.0)
        cfg_high = TwoStageConfig(query_rewrite_temperature=2.0)
        assert cfg_low.query_rewrite_temperature == 0.0
        assert cfg_high.query_rewrite_temperature == 2.0

    def test_two_stage_config_hyde_temperature_bounds(self) -> None:
        """Test that hyde_temperature is bounded to [0, 2]."""
        with pytest.raises(ValidationError):
            TwoStageConfig(hyde_temperature=-0.1)
        with pytest.raises(ValidationError):
            TwoStageConfig(hyde_temperature=2.1)
        # Boundary values should be valid and preserved
        cfg_low = TwoStageConfig(hyde_temperature=0.0)
        cfg_high = TwoStageConfig(hyde_temperature=2.0)
        assert cfg_low.hyde_temperature == 0.0
        assert cfg_high.hyde_temperature == 2.0
        # Boundary values should be valid
        TwoStageConfig(hyde_temperature=0.0)
        TwoStageConfig(hyde_temperature=2.0)


class TestMCPConfig:
    """Tests for MCPConfig model."""

    def test_mcp_config_defaults(self) -> None:
        """Test default MCP configuration."""
        config = MCPConfig()

        assert config.transport == "stdio"
        assert config.host == "localhost"
        assert config.port == 8000

    def test_mcp_config_custom(self) -> None:
        """Test custom MCP configuration."""
        config = MCPConfig(
            transport="streamable-http",
            host="0.0.0.0",
            port=9000,
        )

        assert config.transport == "streamable-http"
        assert config.host == "0.0.0.0"
        assert config.port == 9000


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_llm_config_defaults(self) -> None:
        """Test default LLM configuration."""
        config = LLMConfig()

        assert config.provider == "mistral"

    def test_llm_config_custom(self) -> None:
        """Test custom LLM configuration."""
        config = LLMConfig(provider="openai")

        assert config.provider == "openai"


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
            llm=LLMConfig(),
            mistral=MistralConfig(api_key="test-key"),
            openai=OpenAIConfig(api_key="test-openai-key"),
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

    def test_load_config_from_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading configuration from YAML file."""
        # Mock load_dotenv to prevent loading from .env file
        import rainrag.config

        monkeypatch.setattr(rainrag.config, "load_dotenv", lambda: None)

        # Clear environment variables that would override config values
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        config_data = {
            "paths": {
                "archive_root": "/test/archive",
                "docs_output": "/test/docs.jsonl",
                "embeddings_cache": "/test/embeddings",
            },
            "embedding": {
                "provider": "local",
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
            "llm": {
                "provider": "mistral",
            },
            "mistral": {
                "api_key": "test-key",
                "model_name": "mistral-large-latest",
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_k": 10,
            },
            "openai": {
                "api_key": "test-openai-key",
                "model_name": "gpt-4o",
                "embedding_model": "text-embedding-3-large",
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
            assert config.mcp.transport == "streamable-http"
            assert config.mcp.host == "0.0.0.0"
            assert config.mcp.port == 9000

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
