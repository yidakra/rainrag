"""Tests for the FastAPI backend."""

# asyncio not needed in these tests
from pathlib import Path
from typing import cast

import anyio
import httpx
import pytest

from rainrag.api import app, find_video_file  # get_video_base_name imported in individual tests
from rainrag.config import Config, EmbeddingConfig, PathsConfig, QdrantConfig, LLMConfig, MistralConfig, OpenAIConfig


class _SyncASGIClient:
    """Synchronous facade over httpx.AsyncClient for ASGI app tests."""

    def __init__(self, asgi_app):
        # call superclass init to satisfy linters
        super().__init__()
        self._transport = httpx.ASGITransport(app=asgi_app)
        self._base_url = "http://testserver"

    def request(self, method: str, url: str, **kwargs):
        async def _request():
            async with httpx.AsyncClient(
                transport=self._transport,
                base_url=self._base_url,
            ) as client:
                return await client.request(method, url, **kwargs)

        return anyio.run(_request)

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def test_client():
    """Create a FastAPI TestClient for the app."""
    return _SyncASGIClient(app)


@pytest.fixture
def test_config_with_video(temp_dir: Path) -> Config:
    """Create a test configuration with video settings."""
    archive_dir = temp_dir / "archive"
    archive_dir.mkdir()

    data_dir = temp_dir / "data"
    data_dir.mkdir()

    embeddings_dir = temp_dir / "embeddings"
    embeddings_dir.mkdir()

    return Config(
        paths=cast(
            PathsConfig,
            {
                "archive_root": str(archive_dir),
                "docs_output": str(data_dir / "docs.jsonl"),
                "embeddings_cache": str(embeddings_dir),
                "video_root": str(archive_dir),
            },
        ),
        embedding=cast(
            EmbeddingConfig,
            {
                "provider": "local",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 8,
                "max_seq_length": 128,
                "device": "cpu",
                "normalize_embeddings": True,
            },
        ),
        qdrant=cast(
            QdrantConfig,
            {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection",
                "vector_size": 384,
                "distance": "Cosine",
                "recreate_collection": False,
            },
        ),
        llm=cast(
            LLMConfig,
            {
                "provider": "mistral",
            },
        ),
        mistral=cast(
            MistralConfig,
            {
                "api_key": "test-key",
                "model_name": "mistral-small-latest",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
        ),
        openai=cast(
            OpenAIConfig,
            {
                "api_key": "test-openai-key",
                "model_name": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
        ),
        processing={
            "num_workers": 2,
            "max_file_size": 1048576,
            "min_text_length": 10,
        },
        logging={
            "level": "ERROR",
            "format": "{message}",
            "log_file": str(temp_dir / "test.log"),
        },
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )


@pytest.fixture
def archive_with_videos(temp_dir: Path, sample_vtt_en: str) -> Path:
    """Create an archive directory with VTT files and corresponding videos."""
    archive_dir = temp_dir / "archive"
    archive_dir.mkdir()

    # Create a directory with VTT and video files
    test_dir = archive_dir / "test_videos"
    test_dir.mkdir()

    # Create multi-resolution video setup (like the actual archive)
    hash_name = "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd"
    (test_dir / f"{hash_name}.en.vtt").write_text(sample_vtt_en)
    (test_dir / f"{hash_name}.ru.vtt").write_text(sample_vtt_en)

    # Create multiple resolution video files
    (test_dir / f"{hash_name}_1080p.mp4").write_bytes(b"1080p video")
    (test_dir / f"{hash_name}_720p.mp4").write_bytes(b"720p video")
    (test_dir / f"{hash_name}_480p.mp4").write_bytes(b"480p video")
    (test_dir / f"{hash_name}_360p.mp4").write_bytes(b"360p video")
    (test_dir / f"{hash_name}_180p.mp4").write_bytes(b"180p video")

    # Create VTT files with old-style naming (exact match)
    (test_dir / "video1.en.vtt").write_text(sample_vtt_en)
    (test_dir / "video1.mp4").write_bytes(b"old style video")

    # Create VTT without video
    (test_dir / "video3.vtt").write_text(sample_vtt_en)

    # Create multi-resolution video setup (like the actual archive)
    hash_name = "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd"
    (test_dir / f"{hash_name}.en.vtt").write_text(sample_vtt_en)
    (test_dir / f"{hash_name}.ru.vtt").write_text(sample_vtt_en)

    # Create multiple resolution video files
    (test_dir / f"{hash_name}_1080p.mp4").write_bytes(b"1080p video")
    (test_dir / f"{hash_name}_720p.mp4").write_bytes(b"720p video")
    (test_dir / f"{hash_name}_480p.mp4").write_bytes(b"480p video")
    (test_dir / f"{hash_name}_360p.mp4").write_bytes(b"360p video")
    (test_dir / f"{hash_name}_180p.mp4").write_bytes(b"180p video")

    return archive_dir


def test_find_video_file_mp4(temp_dir: Path, archive_with_videos: Path):
    """Test finding MP4 video file for VTT."""

    # Set up config
    test_cfg = Config(
        paths=cast(
            PathsConfig,
            {
                "archive_root": str(archive_with_videos),
                "docs_output": "./data/docs.jsonl",
                "embeddings_cache": "./embeddings",
                "video_root": str(archive_with_videos),
            },
        ),
        embedding=cast(
            EmbeddingConfig,
            {
                "provider": "local",
                "model_name": "test",
                "batch_size": 8,
                "max_seq_length": 128,
                "device": "cpu",
                "normalize_embeddings": True,
            },
        ),
        qdrant=cast(
            QdrantConfig,
            {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test",
                "vector_size": 384,
                "distance": "Cosine",
                "recreate_collection": False,
            },
        ),
        llm=cast(LLMConfig, {"provider": "mistral"}),
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai=cast(
            OpenAIConfig,
            {
                "api_key": "test-openai-key",
                "model_name": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
                "max_tokens": 512,
                "temperature": 0.3,
                "top_k": 5,
            },
        ),
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )

    # Temporarily set the global config
    import rainrag.api as api_module

    original_config = api_module.config
    api_module.config = test_cfg

    try:
        vtt_path = str(archive_with_videos / "test_videos" / "video1.en.vtt")
        video_file = find_video_file(vtt_path)

        assert video_file is not None
        assert video_file.endswith("video1.mp4")
        assert Path(video_file).exists()
    finally:
        api_module.config = original_config


def test_find_video_file_mkv(temp_dir: Path, archive_with_videos: Path):
    """Test finding video file for VTT with hash-based naming (multi-resolution)."""
    import rainrag.api as api_module

    original_config = api_module.config

    test_cfg = Config(
        paths=cast(
            PathsConfig,
            {
                "archive_root": str(archive_with_videos),
                "docs_output": "./data/docs.jsonl",
                "embeddings_cache": "./embeddings",
                "video_root": str(archive_with_videos),
            },
        ),
        embedding=cast(
            EmbeddingConfig,
            {
                "provider": "local",
                "model_name": "test",
                "batch_size": 8,
                "max_seq_length": 128,
                "device": "cpu",
                "normalize_embeddings": True,
            },
        ),
        qdrant=cast(
            QdrantConfig,
            {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test",
                "vector_size": 384,
                "distance": "Cosine",
                "recreate_collection": False,
            },
        ),
        llm={"provider": "mistral"},
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai={
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )

    api_module.config = test_cfg

    try:
        # Test with hash-based naming that has multiple resolutions
        hash_name = "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd"
        vtt_path = str(archive_with_videos / "test_videos" / f"{hash_name}.ru.vtt")
        video_file = find_video_file(vtt_path)

        assert video_file is not None
        # Should find one of the resolution variants
        assert hash_name in video_file
        assert Path(video_file).exists()
    finally:
        api_module.config = original_config


def test_find_video_file_not_found(temp_dir: Path, archive_with_videos: Path):
    """Test when video file doesn't exist."""
    import rainrag.api as api_module

    original_config = api_module.config

    test_cfg = Config(
        paths={
            "archive_root": str(archive_with_videos),
            "docs_output": "./data/docs.jsonl",
            "embeddings_cache": "./embeddings",
            "video_root": str(archive_with_videos),
        },
        embedding=cast(
            EmbeddingConfig,
            {
                "provider": "local",
                "model_name": "test",
                "batch_size": 8,
                "max_seq_length": 128,
                "device": "cpu",
                "normalize_embeddings": True,
            },
        ),
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
            "vector_size": 384,
            "distance": "Cosine",
            "recreate_collection": False,
        },
        llm={"provider": "mistral"},
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai={
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )

    api_module.config = test_cfg

    try:
        vtt_path = str(archive_with_videos / "test_videos" / "video3.vtt")
        video_file = find_video_file(vtt_path)

        assert video_file is None
    finally:
        api_module.config = original_config


def test_api_root_endpoint(test_client):
    """Test the root endpoint returns API information."""
    response = test_client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "RainRAG API"
    assert "endpoints" in data
    assert "POST /query" in data["endpoints"]
    assert "Submit a question" in data["endpoints"]["POST /query"]


def test_video_endpoint_security(test_client, temp_dir: Path, archive_with_videos: Path):
    """Test that video endpoint prevents path traversal attacks."""
    import rainrag.api as api_module

    original_config = api_module.config

    test_cfg = Config(
        paths={
            "archive_root": str(archive_with_videos),
            "docs_output": "./data/docs.jsonl",
            "embeddings_cache": "./embeddings",
            "video_root": str(archive_with_videos),
        },
        embedding={
            "provider": "local",
            "model_name": "test",
            "batch_size": 8,
            "max_seq_length": 128,
            "device": "cpu",
            "normalize_embeddings": True,
        },
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
            "vector_size": 384,
            "distance": "Cosine",
            "recreate_collection": False,
        },
        llm={"provider": "mistral"},
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai={
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )

    api_module.config = test_cfg

    try:
        # Try path traversal attack
        response = test_client.get("/video/../../../etc/passwd")
        assert response.status_code in [400, 403, 404]  # Should be rejected

        # Try another path traversal
        response = test_client.get("/video/../../sensitive_file.txt")
        assert response.status_code in [400, 403, 404]  # Should be rejected
    finally:
        api_module.config = original_config


def test_vtt_endpoint_security(test_client, temp_dir: Path, archive_with_videos: Path):
    """Test that VTT endpoint prevents path traversal attacks."""
    import rainrag.api as api_module

    original_config = api_module.config

    test_cfg = Config(
        paths={
            "archive_root": str(archive_with_videos),
            "docs_output": "./data/docs.jsonl",
            "embeddings_cache": "./embeddings",
            "video_root": str(archive_with_videos),
        },
        embedding={
            "provider": "local",
            "model_name": "test",
            "batch_size": 8,
            "max_seq_length": 128,
            "device": "cpu",
            "normalize_embeddings": True,
        },
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
            "vector_size": 384,
            "distance": "Cosine",
            "recreate_collection": False,
        },
        llm={"provider": "mistral"},
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai={
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )

    api_module.config = test_cfg

    try:
        # Try path traversal attack
        response = test_client.get("/vtt/../../../etc/passwd")
        assert response.status_code in [400, 403, 404]  # Should be rejected
    finally:
        api_module.config = original_config


def test_video_disabled(temp_dir: Path):
    """Test video serving when disabled in config."""
    import rainrag.api as api_module

    original_config = api_module.config

    test_cfg = Config(
        paths=cast(
            PathsConfig,
            {
                "archive_root": str(temp_dir),
                "docs_output": "./data/docs.jsonl",
                "embeddings_cache": "./embeddings",
                "video_root": str(temp_dir),
            },
        ),
        embedding=cast(
            EmbeddingConfig,
            {
                "provider": "local",
                "model_name": "test",
                "batch_size": 8,
                "max_seq_length": 128,
                "device": "cpu",
                "normalize_embeddings": True,
            },
        ),
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
            "vector_size": 384,
            "distance": "Cosine",
            "recreate_collection": False,
        },
        llm={"provider": "mistral"},
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai={
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={"enabled": False, "extensions": [".mp4"], "vtt_extensions": [".vtt"]},
    )

    api_module.config = test_cfg

    try:
        # use the FastAPI TestClient to simulate requests
        client = _SyncASGIClient(app)
        response = client.get("/video/test.mp4")
        assert response.status_code == 404
        assert "disabled" in response.json()["detail"].lower()
    finally:
        api_module.config = original_config


def test_find_video_file_multi_resolution(temp_dir: Path, archive_with_videos: Path):
    """Test finding highest resolution video file for VTT (prefers 1080p)."""
    import rainrag.api as api_module

    original_config = api_module.config

    test_cfg = Config(
        paths={
            "archive_root": str(archive_with_videos),
            "docs_output": "./data/docs.jsonl",
            "embeddings_cache": "./embeddings",
            "video_root": str(archive_with_videos),
        },
        embedding={
            "provider": "local",
            "model_name": "test",
            "batch_size": 8,
            "max_seq_length": 128,
            "device": "cpu",
            "normalize_embeddings": True,
        },
        qdrant={
            "host": "localhost",
            "port": 6333,
            "collection_name": "test",
            "vector_size": 384,
            "distance": "Cosine",
            "recreate_collection": False,
        },
        llm={"provider": "mistral"},
        mistral={
            "api_key": "test-key",
            "model_name": "mistral-small-latest",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        openai={
            "api_key": "test-openai-key",
            "model_name": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_k": 5,
        },
        processing={"num_workers": 2, "max_file_size": 1048576, "min_text_length": 10},
        logging={"level": "ERROR", "format": "{message}", "log_file": "./test.log"},
        video={
            "enabled": True,
            "extensions": [".mp4", ".mkv", ".webm"],
            "vtt_extensions": [".vtt", ".en.vtt", ".ru.vtt"],
        },
    )

    api_module.config = test_cfg

    try:
        # Test with English VTT
        hash_name = "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd"
        vtt_path_en = str(archive_with_videos / "test_videos" / f"{hash_name}.en.vtt")
        video_file_en = find_video_file(vtt_path_en)

        assert video_file_en is not None
        assert f"{hash_name}_1080p.mp4" in video_file_en  # Should prefer 1080p
        assert Path(video_file_en).exists()

        # Test with Russian VTT (should find same video)
        vtt_path_ru = str(archive_with_videos / "test_videos" / f"{hash_name}.ru.vtt")
        video_file_ru = find_video_file(vtt_path_ru)

        assert video_file_ru is not None
        assert f"{hash_name}_1080p.mp4" in video_file_ru
        assert video_file_en == video_file_ru  # Same video for both languages
    finally:
        api_module.config = original_config


def test_get_video_base_name_english():
    """Test extracting base name from English VTT file."""
    from rainrag.api import get_video_base_name

    vtt_path = (
        "/mnt/vod/srv/storage/transcoded/3b/10/f9/3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd.en.vtt"
    )
    base_name = get_video_base_name(vtt_path)

    # Should remove .en suffix and include parent directory
    assert "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd" in base_name
    assert ".en" not in base_name


def test_get_video_base_name_russian():
    """Test extracting base name from Russian VTT file."""
    from rainrag.api import get_video_base_name

    vtt_path = (
        "/mnt/vod/srv/storage/transcoded/3b/10/f9/3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd.ru.vtt"
    )
    base_name = get_video_base_name(vtt_path)

    # Should remove .ru suffix and include parent directory
    assert "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd" in base_name
    assert ".ru" not in base_name


def test_get_video_base_name_grouping():
    """Test that English and Russian versions get the same base name."""
    from rainrag.api import get_video_base_name

    vtt_path_en = "/archive/videos/test_hash.en.vtt"
    vtt_path_ru = "/archive/videos/test_hash.ru.vtt"

    base_name_en = get_video_base_name(vtt_path_en)
    base_name_ru = get_video_base_name(vtt_path_ru)

    # Both should have the same base name for grouping
    assert base_name_en == base_name_ru
    assert "test_hash" in base_name_en
