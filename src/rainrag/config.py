"""Configuration management for RainRAG."""

import os
from pathlib import Path
from typing import Any

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
    max_seq_length: int = Field(default=512)
    device: str = Field(default="cuda")
    normalize_embeddings: bool = Field(default=True)


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
    """Configuration for Cohere Rerank API."""

    api_key: str = Field(default="", description="Cohere API key")
    model_name: str = Field(
        default="rerank-v3.5",
        description="Cohere rerank model: rerank-v3.5, rerank-english-v3.0, rerank-multilingual-v3.0",
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
    processing: ProcessingConfig
    logging: LoggingConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)


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
        config_data: dict[str, Any] = yaml.safe_load(f)

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
