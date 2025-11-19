"""Configuration management for RainRAG."""

import os
from pathlib import Path
from typing import Any, Dict

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
        description="Embedding provider: 'local' for local model, 'mistral' for Mistral API, 'openai' for OpenAI API"
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
    max_tokens: int = Field(default=512)
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
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class LLMConfig(BaseModel):
    """Configuration for LLM provider selection."""

    provider: str = Field(
        default="mistral",
        description="LLM provider: 'mistral' for Mistral API, 'openai' for OpenAI API"
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


class Config(BaseModel):
    """Main configuration model."""

    paths: PathsConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    llm: LLMConfig
    mistral: MistralConfig
    openai: OpenAIConfig
    processing: ProcessingConfig
    logging: LoggingConfig
    video: VideoConfig = Field(default_factory=VideoConfig)


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

    with open(config_file, "r") as f:
        config_data: Dict[str, Any] = yaml.safe_load(f)

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

    return Config(**config_data)
