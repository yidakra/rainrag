"""Configuration management for RainRAG."""

import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    archive_root: str = Field(description="Root directory containing .vtt files")
    docs_output: str = Field(description="Output path for parsed documents")
    embeddings_cache: str = Field(description="Directory for cached embeddings")
    video_root: str = Field(
        default="",
        description="Root directory containing video files (same as archive_root if not specified)",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: str = Field(
        default="local",
        description="Embedding provider: 'local' for local model, 'mistral' for Mistral API, 'openai' for OpenAI API, 'gemini' for Google Gemini API",
    )
    model_name: str = Field(default="intfloat/multilingual-e5-large")
    batch_size: int = Field(default=32)
    max_seq_length: int = Field(
        default=512,
        description="Maximum sequence length in tokens for embedding model (multilingual-e5-large supports up to 512)",
    )
    device: str = Field(default="cuda")
    normalize_embeddings: bool = Field(default=True)
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for transient embedding API failures",
    )
    retry_backoff_factor: float = Field(
        default=2.0, gt=0, description="Exponential backoff factor for retries"
    )


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    collection_name: str = Field(default="broadcast_transcripts")
    vector_size: int = Field(default=1024)
    distance: str = Field(default="Cosine")
    recreate_collection: bool = Field(default=False)


class MistralConfig(BaseModel):
    """Configuration for Mistral API."""

    api_key: str = Field(description="Mistral API key")
    model_name: str = Field(
        default="mistral-small-latest",
        description="Mistral model to use: mistral-small-latest, mistral-medium-latest, mistral-large-latest, etc.",
    )
    embedding_model: str = Field(
        default="mistral-embed",
        description="Mistral embedding model",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API."""

    api_key: str = Field(description="OpenAI API key")
    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model: text-embedding-3-small, text-embedding-3-large, etc.",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class ClaudeConfig(BaseModel):
    """Configuration for Anthropic Claude API."""

    api_key: str = Field(default="", description="Anthropic API key")
    model_name: str = Field(
        default="claude-3-5-sonnet-20240620",
        description="Claude model to use: claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-haiku-20240307, etc.",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API."""

    api_key: str = Field(default="", description="Google API key")
    model_name: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp, etc.",
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model: models/text-embedding-004, models/embedding-001, etc.",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class LLMConfig(BaseModel):
    """Configuration for LLM provider selection."""

    provider: str = Field(
        default="mistral",
        description="LLM provider: 'mistral' for Mistral API, 'openai' for OpenAI API, 'claude' for Anthropic Claude API, 'gemini' for Google Gemini API",
    )


class ChunkingConfig(BaseModel):
    """Configuration for VTT chunking."""

    enabled: bool = Field(default=True, description="Enable chunking of VTT files")
    strategy: Literal["time", "token", "hybrid"] = Field(
        default="hybrid",
        description="Chunking strategy: 'time' (time-based only), 'token' (token-based only), 'hybrid' (time-based with token validation)",
    )
    chunk_duration_seconds: int = Field(
        default=300,
        description="Duration of each chunk in seconds (default: 5 minutes) - used for 'time' and 'hybrid' strategies",
    )
    overlap_seconds: int = Field(
        default=30,
        description="Overlap between adjacent chunks in seconds (default: 30). Prevents information loss at chunk boundaries.",
    )
    min_chunk_tokens: int = Field(
        default=50,
        description="Minimum tokens per chunk (chunks smaller than this may be merged with neighbors)",
    )
    max_chunk_tokens: int | None = Field(
        default=None,
        description="Maximum tokens per chunk (auto-detected from embedding model if not set). Chunks exceeding this will be split.",
    )
    token_buffer: int = Field(
        default=50,
        description="Safety buffer to reserve for special tokens (e.g., 'passage:' prefix)",
    )


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search (vector + BM25)."""

    enabled: bool = Field(
        default=False,
        description="Enable hybrid search combining vector similarity and BM25 keyword matching",
    )
    bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 scores in hybrid search (0.0-1.0). Vector weight = 1 - bm25_weight",
    )
    top_k_multiplier: int = Field(
        default=3,
        description="Retrieve top_k * multiplier documents before reranking (to get more BM25 candidates)",
    )
    fusion_method: str = Field(
        default="rrf",
        description="Score fusion method: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (weighted sum)",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant (default: 60, standard value from literature)",
    )


class ProcessingConfig(BaseModel):
    """Configuration for data processing."""

    num_workers: int = Field(default=4)
    max_file_size: int = Field(default=10485760)  # 10 MB
    min_text_length: int = Field(default=50)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    log_file: str = Field(default="./logs/rainrag.log")


class VideoConfig(BaseModel):
    """Configuration for video file serving."""

    enabled: bool = Field(default=True, description="Enable video file serving")
    extensions: list[str] = Field(
        default=[".mp4", ".mkv", ".webm", ".avi", ".mov"],
        description="Video file extensions to look for",
    )
    vtt_extensions: list[str] = Field(
        default=[".vtt", ".en.vtt", ".ru.vtt"],
        description="VTT file extensions to serve",
    )


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = Field(
        default="stdio",
        description='Transport protocol: "stdio", "sse", or "streamable-http"',
    )
    host: str = Field(default="localhost")
    port: int = Field(default=8000)


class CohereConfig(BaseModel):
    """Configuration for Cohere API (reranking and embeddings)."""

    api_key: str = Field(default="", description="Cohere API key")
    model_name: str = Field(
        default="rerank-v3.5",
        description="Cohere rerank model: rerank-v3.5, rerank-english-v3.0, rerank-multilingual-v3.0",
    )
    embedding_model: str = Field(
        default="embed-multilingual-v3.0",
        description="Cohere embedding model: embed-multilingual-v3.0, embed-english-v3.0, etc.",
    )


class RerankerConfig(BaseModel):
    """Configuration for reranking."""

    enabled: bool = Field(default=False, description="Enable reranking of search results")
    provider: str = Field(default="cohere", description="Reranker provider: 'cohere'")
    top_n: int = Field(default=5, description="Number of documents to return after reranking")
    initial_k: int = Field(
        default=20, description="Number of candidates to retrieve before reranking"
    )


class Config(BaseModel):
    """Main configuration model."""

    paths: PathsConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    llm: LLMConfig
    mistral: MistralConfig
    openai: OpenAIConfig
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    cohere: CohereConfig = Field(default_factory=CohereConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    processing: ProcessingConfig
    logging: LoggingConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    def get_max_chunk_tokens(self) -> int:
        """
        Get the maximum chunk size in tokens based on embedding model.

        Returns the configured max_chunk_tokens if set, otherwise auto-detects
        based on the embedding model's known limits.

        Returns:
            Maximum tokens per chunk (accounting for token_buffer)
        """
        # Use explicit config if set
        if self.chunking.max_chunk_tokens is not None:
            return self.chunking.max_chunk_tokens

        # Auto-detect based on embedding model
        model_limits = {
            # Local models
            "intfloat/multilingual-e5-large": 512,
            "intfloat/multilingual-e5-base": 512,
            "intfloat/multilingual-e5-small": 512,
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "sentence-transformers/all-mpnet-base-v2": 384,
            # OpenAI models
            "text-embedding-3-large": 8191,
            "text-embedding-3-small": 8191,
            "text-embedding-ada-002": 8191,
            # Cohere models
            "embed-english-v3.0": 512,
            "embed-multilingual-v3.0": 512,
            "embed-english-light-v3.0": 512,
            "embed-multilingual-light-v3.0": 512,
            # Mistral models
            "mistral-embed": 512,
            # Gemini models
            "models/text-embedding-004": 3072,
            "models/embedding-001": 2048,
        }

        # Get model name based on provider
        if self.embedding.provider == "local":
            model_name = self.embedding.model_name
        elif self.embedding.provider == "openai":
            model_name = self.openai.embedding_model
        elif self.embedding.provider == "gemini":
            model_name = self.gemini.embedding_model
        elif self.embedding.provider == "mistral":
            model_name = self.mistral.embedding_model
        elif self.embedding.provider == "cohere":
            model_name = self.cohere.embedding_model
        else:
            # Default to configured max_seq_length
            return max(1, self.embedding.max_seq_length - self.chunking.token_buffer)

        # Look up model limit
        limit = model_limits.get(model_name, self.embedding.max_seq_length)

        # Subtract safety buffer
        return max(1, limit - self.chunking.token_buffer)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    # Load environment variables from .env file if it exists
    load_dotenv()

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config_data = cast(dict[str, Any], yaml.safe_load(f))

    # Override Mistral API key from environment variable if set
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if mistral_api_key:
        if "mistral" not in config_data:
            config_data["mistral"] = {}
        config_data["mistral"]["api_key"] = mistral_api_key

    # Override OpenAI API key from environment variable if set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        if "openai" not in config_data:
            config_data["openai"] = {}
        config_data["openai"]["api_key"] = openai_api_key

    # Override Anthropic API key from environment variable if set
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        if "claude" not in config_data:
            config_data["claude"] = {}
        config_data["claude"]["api_key"] = anthropic_api_key

    # Override Google API key from environment variable if set
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        if "gemini" not in config_data:
            config_data["gemini"] = {}
        config_data["gemini"]["api_key"] = google_api_key

    # Override Cohere API key from environment variable if set
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if cohere_api_key:
        if "cohere" not in config_data:
            config_data["cohere"] = {}
        config_data["cohere"]["api_key"] = cohere_api_key

    return Config(**config_data)
