"""Configuration management for RainRAG."""

from pathlib import Path
from typing import Any, Dict

import yaml
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


class VLLMConfig(BaseModel):
    """Configuration for vLLM inference server."""

    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    model_name: str = Field(default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    use_chat_completions: bool = Field(
        default=True,
        description="Use chat completions API (/v1/chat/completions) instead of completions API. "
        "Set to true for instruction-tuned models, false for base models or if chat API is not available.",
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
    vllm: VLLMConfig
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
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config_data: Dict[str, Any] = yaml.safe_load(f)

    return Config(**config_data)
