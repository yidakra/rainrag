"""Pytest configuration and fixtures."""

import tempfile
from collections.abc import Generator
from pathlib import Path

# The package import path is configured via pytest.ini (pythonpath = src).
# Avoid runtime sys.path mutation to keep tests non-invasive and deterministic.
import pytest

from rainrag.config import Config


@pytest.fixture(autouse=True)
def disable_api_startup_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable heavy API lifespan initialization during unit/integration tests."""
    monkeypatch.setenv("RAINRAG_SKIP_API_STARTUP_INIT", "1")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_vtt_en() -> str:
    """Sample English VTT file content."""
    return """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello, this is a test subtitle.

00:00:05.000 --> 00:00:10.000
This is the second line of text.

00:00:10.000 --> 00:00:15.000
<v Speaker>And this has markup tags</v> that should be removed.
"""


@pytest.fixture
def sample_vtt_ru() -> str:
    """Sample Russian VTT file content."""
    return """WEBVTT

1
00:00:00.000 --> 00:00:05.000
Привет, это тестовые субтитры.

2
00:00:05.000 --> 00:00:10.000
Это вторая строка текста.

NOTE Some comment

3
00:00:10.000 --> 00:00:15.000
Текст с <c>разметкой</c> который должен быть очищен.
"""


@pytest.fixture
def invalid_vtt() -> str:
    """Invalid VTT file content (missing WEBVTT header)."""
    return """
00:00:00.000 --> 00:00:05.000
This is not a valid VTT file.
"""


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Create a test configuration."""
    archive_dir = temp_dir / "archive"
    archive_dir.mkdir()

    data_dir = temp_dir / "data"
    data_dir.mkdir()

    embeddings_dir = temp_dir / "embeddings"
    embeddings_dir.mkdir()

    return Config.model_validate(
        {
            "paths": {
                "archive_root": str(archive_dir),
                "docs_output": str(data_dir / "docs.jsonl"),
                "embeddings_cache": str(embeddings_dir),
            },
            "embedding": {
                "provider": "local",  # Use local model for tests
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
                "batch_size": 8,
                "max_seq_length": 128,
                "device": "cpu",
                "normalize_embeddings": True,
            },
            "qdrant": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 384,  # all-MiniLM-L6-v2 dimension
                "distance": "Cosine",
                "recreate_collection": False,
            },
            "llm": {
                "provider": "mistral",
            },
            "mistral": {
                "api_key": "test-api-key",
                "model_name": "mistral-small-latest",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
            "openai": {
                "api_key": "test-api-key",
                "model_name": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
            "claude": {
                "api_key": "test-api-key",
                "model_name": "claude-3-5-sonnet-20240620",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
            "gemini": {
                "api_key": "test-api-key",
                "model_name": "gemini-1.5-flash",
                "embedding_model": "models/text-embedding-004",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
            "processing": {
                "num_workers": 2,
                "max_file_size": 1048576,  # 1MB
                "min_text_length": 10,
            },
            "logging": {
                "level": "ERROR",  # Reduce noise during tests
                "format": "{message}",
                "log_file": str(temp_dir / "test.log"),
            },
            "web_metadata": {
                "enabled": False,  # Disable web metadata for tests
                "path": str(temp_dir / "web_metadata"),
                "min_content_length": 10,
            },
        }
    )


@pytest.fixture
def archive_with_vtt_files(temp_dir: Path, sample_vtt_en: str, sample_vtt_ru: str) -> Path:
    """Create an archive directory with sample VTT files."""
    archive_dir = temp_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Create English VTT files
    en_dir = archive_dir / "english" / "broadcast_001"
    en_dir.mkdir(parents=True)
    (en_dir / "subtitle_en.vtt").write_text(sample_vtt_en)

    # Create Russian VTT files
    ru_dir = archive_dir / "russian" / "broadcast_002"
    ru_dir.mkdir(parents=True)
    (ru_dir / "subtitle_ru.vtt").write_text(sample_vtt_ru)

    # Create a mixed directory
    mixed_dir = archive_dir / "mixed"
    mixed_dir.mkdir()
    (mixed_dir / "test_english.vtt").write_text(sample_vtt_en)
    (mixed_dir / "test_russian.vtt").write_text(sample_vtt_ru)

    return archive_dir
