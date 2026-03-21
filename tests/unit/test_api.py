"""Tests for the FastAPI backend."""

# asyncio not needed in these tests
from contextlib import contextmanager
from pathlib import Path

import anyio
import httpx
import pytest

from rainrag.api import app, find_video_file  # get_video_base_name imported in individual tests
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
    VideoConfig,
)


@contextmanager
def override_api_config(test_cfg):
    """Temporarily override rainrag.api.config in context, restoring after exit."""
    import rainrag.api as api_module

    original_config = api_module.config
    api_module.config = test_cfg
    try:
        yield
    finally:
        api_module.config = original_config


class _SyncASGIClient:
    """Synchronous facade over httpx.AsyncClient for ASGI app tests."""

    def __init__(self, asgi_app):
        # call parent constructor (object) to satisfy linters/analysis tools
        super().__init__()
        # only holds transport/base URL; every request creates its own AsyncClient
        self._transport = httpx.ASGITransport(app=asgi_app)
        self._base_url = "http://testserver"

    def request(self, method: str, url: str, **kwargs):
        async def _request():
            async with httpx.AsyncClient(
                transport=self._transport,
                base_url=self._base_url,
            ) as client:
                return await client.request(method, url, **kwargs)

        # If we're already running in an AnyIO async task, use from_thread.run so
        # we don't attempt to create a new event loop from within an existing one.
        # Otherwise, run a fresh event loop (sync tests).
        try:
            anyio.get_current_task()
            in_async_task = True
        except RuntimeError:
            in_async_task = False

        if not in_async_task:
            return anyio.run(_request)

        # `from_thread.run` exists at runtime but isn’t exposed in the stubs,
        # so silence the type checker here.
        return anyio.from_thread.run(_request)  # type: ignore[attr-defined]

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def test_client():
    """Create a sync ASGI test client (returns _SyncASGIClient) for the app."""
    return _SyncASGIClient(app)


def _build_config(
    archive_root: str,
    docs_output: str = "./data/docs.jsonl",
    embeddings_cache: str = "./embeddings",
    video_root: str | None = None,
    video_enabled: bool = True,
) -> Config:
    """Build a Config with shared defaults for tests."""
    if video_root is None:
        video_root = archive_root

    return Config(
        paths=PathsConfig(
            archive_root=archive_root,
            docs_output=docs_output,
            embeddings_cache=embeddings_cache,
            video_root=video_root,
        ),
        embedding=EmbeddingConfig(
            provider="local",
            model_name="test",
            batch_size=8,
            max_seq_length=128,
            device="cpu",
            normalize_embeddings=True,
        ),
        qdrant=QdrantConfig(
            host="localhost",
            port=6333,
            collection_name="test",
            vector_size=384,
            distance="Cosine",
            recreate_collection=False,
        ),
        llm=LLMConfig(provider="mistral"),
        mistral=MistralConfig(
            api_key="test-key",
            model_name="mistral-small-latest",
            max_tokens=512,
            temperature=0.3,
            top_k=5,
        ),
        openai=OpenAIConfig(
            api_key="test-openai-key",
            model_name="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
            max_tokens=512,
            temperature=0.3,
            top_k=5,
        ),
        processing=ProcessingConfig(
            num_workers=2,
            max_file_size=1048576,
            min_text_length=10,
        ),
        logging=LoggingConfig(
            level="ERROR",
            format="{message}",
            log_file="./test.log",
        ),
        video=VideoConfig(
            enabled=video_enabled,
            extensions=[".mp4", ".mkv", ".webm"],
            vtt_extensions=[".vtt", ".en.vtt", ".ru.vtt"],
        ),
    )


def make_test_config(archive_root: str, video_enabled: bool = True) -> Config:
    """Create a test Config matching the common defaults used in these tests."""
    return _build_config(archive_root, video_enabled=video_enabled)


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

    test_cfg = make_test_config(str(archive_with_videos))

    with override_api_config(test_cfg):
        vtt_path = str(archive_with_videos / "test_videos" / "video1.en.vtt")
        video_file = find_video_file(vtt_path)

        assert video_file is not None
        assert video_file.endswith("video1.mp4")
        assert Path(video_file).exists()


def test_find_video_file_mkv(temp_dir: Path, archive_with_videos: Path):
    """Test finding video file for VTT with hash-based naming (multi-resolution)."""
    test_cfg = make_test_config(str(archive_with_videos))

    with override_api_config(test_cfg):
        # Test with hash-based naming that has multiple resolutions
        hash_name = "3b10f9b81a130d9ed9bb81c3f4a304c9f3641dfd"
        vtt_path = str(archive_with_videos / "test_videos" / f"{hash_name}.ru.vtt")
        video_file = find_video_file(vtt_path)

        assert video_file is not None
        # Should find one of the resolution variants
        assert hash_name in video_file
        assert Path(video_file).exists()


def test_find_video_file_not_found(temp_dir: Path, archive_with_videos: Path):
    """Test when video file doesn't exist."""
    test_cfg = make_test_config(str(archive_with_videos))

    with override_api_config(test_cfg):
        vtt_path = str(archive_with_videos / "test_videos" / "video3.vtt")
        video_file = find_video_file(vtt_path)

        assert video_file is None


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
    test_cfg = make_test_config(str(archive_with_videos))

    with override_api_config(test_cfg):
        # Try path traversal attack
        response = test_client.get("/video/../../../etc/passwd")
        assert response.status_code in [400, 403, 404]  # Should be rejected

        # Try another path traversal
        response = test_client.get("/video/../../sensitive_file.txt")
        assert response.status_code in [400, 403, 404]  # Should be rejected


def test_vtt_endpoint_security(test_client, temp_dir: Path, archive_with_videos: Path):
    """Test that VTT endpoint prevents path traversal attacks."""
    test_cfg = make_test_config(str(archive_with_videos))

    with override_api_config(test_cfg):
        # Try path traversal attack
        response = test_client.get("/vtt/../../../etc/passwd")
        assert response.status_code in [400, 403, 404]  # Should be rejected


def test_video_disabled(test_client, temp_dir: Path):
    """Test video serving when disabled in config."""
    test_cfg = make_test_config(str(temp_dir), video_enabled=False)

    with override_api_config(test_cfg):
        # use the shared fixture client
        response = test_client.get("/video/test.mp4")
        assert response.status_code == 404
        assert "disabled" in response.json()["detail"].lower()


def test_find_video_file_multi_resolution(temp_dir: Path, archive_with_videos: Path):
    """Test finding highest resolution video file for VTT (prefers 1080p)."""
    test_cfg = make_test_config(str(archive_with_videos))

    with override_api_config(test_cfg):
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
