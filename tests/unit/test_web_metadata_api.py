"""Unit tests for the web metadata API client and hybrid loader."""

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rainrag.ingest import WebMetadataLoader
from rainrag.web_metadata_api import WebMetadataAPIClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ARTICLE = {
    "id": 568518,
    "date_active_start": "2026-01-21T16:13:32Z",
    "name": "Test article title",
    "preview_text": "<p>Preview text</p>",
    "detail_text": "<p>Detail text</p>",
    "url": "https://tvrain.tv/teleshow/test/article-568518/",
    "video_hash": "abc123def456",
    "teleshow": {"id": 1, "name": "Test Show"},
    "authors": [],
    "presentors": [],
}


def _make_zip(articles: list[dict]) -> bytes:
    """Create an in-memory ZIP with one JSON per article."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, article in enumerate(articles):
            zf.writestr(f"article_{i}.json", json.dumps(article))
    return buf.getvalue()


@pytest.fixture
def api_client() -> WebMetadataAPIClient:
    return WebMetadataAPIClient(base_url="https://library.example.com", token="test-token")


@pytest.fixture
def meta_dir(tmp_path: Path) -> Path:
    d = tmp_path / "web_metadata"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# WebMetadataAPIClient tests
# ---------------------------------------------------------------------------


class TestWebMetadataAPIClient:
    def test_from_env_missing_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="LIBRARY_API_TOKEN"):
                WebMetadataAPIClient.from_env()

    def test_from_env_success(self) -> None:
        with patch.dict("os.environ", {"MY_TOKEN": "secret"}, clear=False):
            client = WebMetadataAPIClient.from_env(
                base_url="https://example.com", token_env="MY_TOKEN"
            )
            assert client.base_url == "https://example.com"

    def test_fetch_by_hash_success(self, api_client: WebMetadataAPIClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_ARTICLE

        with patch("rainrag.web_metadata_api.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = api_client.fetch_by_hash("abc123def456")
            assert result == SAMPLE_ARTICLE

    def test_fetch_by_hash_404(self, api_client: WebMetadataAPIClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("rainrag.web_metadata_api.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            result = api_client.fetch_by_hash("nonexistent")
            assert result is None

    def test_parse_zip(self) -> None:
        articles = [SAMPLE_ARTICLE, {**SAMPLE_ARTICLE, "id": 2, "video_hash": "xyz789"}]
        zip_data = _make_zip(articles)

        parsed = WebMetadataAPIClient._parse_zip(zip_data)
        assert len(parsed) == 2
        assert parsed[0]["id"] == 568518
        assert parsed[1]["video_hash"] == "xyz789"

    def test_parse_zip_skips_non_json(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("article.json", json.dumps(SAMPLE_ARTICLE))
            zf.writestr("readme.txt", "not json")
            zf.writestr("subdir/", "")
        parsed = WebMetadataAPIClient._parse_zip(buf.getvalue())
        assert len(parsed) == 1

    def test_parse_zip_handles_list_format(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("batch.json", json.dumps([SAMPLE_ARTICLE, SAMPLE_ARTICLE]))
        parsed = WebMetadataAPIClient._parse_zip(buf.getvalue())
        assert len(parsed) == 2

    def test_sync_to_local(self, api_client: WebMetadataAPIClient, meta_dir: Path) -> None:
        zip_data = _make_zip([SAMPLE_ARTICLE])
        mock_resp = MagicMock()
        mock_resp.content = zip_data
        mock_resp.raise_for_status = MagicMock()

        with patch("rainrag.web_metadata_api.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            written = api_client.sync_to_local(meta_dir)

        assert written == 1
        out_file = meta_dir / "abc123def456.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["name"] == "Test article title"

    def test_sync_to_local_skips_no_hash(
        self, api_client: WebMetadataAPIClient, meta_dir: Path
    ) -> None:
        article_no_hash = {**SAMPLE_ARTICLE}
        del article_no_hash["video_hash"]
        zip_data = _make_zip([article_no_hash])
        mock_resp = MagicMock()
        mock_resp.content = zip_data
        mock_resp.raise_for_status = MagicMock()

        with patch("rainrag.web_metadata_api.httpx.Client") as mock_client_cls:
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
            mock_client_cls.return_value.__enter__.return_value.get.return_value = mock_resp

            written = api_client.sync_to_local(meta_dir)

        assert written == 0


# ---------------------------------------------------------------------------
# WebMetadataLoader multi-source tests
# ---------------------------------------------------------------------------


class TestWebMetadataLoaderSources:
    def test_local_mode_reads_file(self, meta_dir: Path) -> None:
        (meta_dir / "abc123.json").write_text(json.dumps(SAMPLE_ARTICLE))
        loader = WebMetadataLoader(meta_dir, source="local")

        result = loader.load_metadata("abc123")
        assert result is not None
        assert result["name"] == "Test article title"

    def test_local_mode_returns_none_when_missing(self, meta_dir: Path) -> None:
        loader = WebMetadataLoader(meta_dir, source="local")
        assert loader.load_metadata("nonexistent") is None

    def test_local_mode_ignores_api(self, meta_dir: Path) -> None:
        mock_api = MagicMock()
        loader = WebMetadataLoader(meta_dir, source="local", api_client=mock_api)

        assert loader.load_metadata("nonexistent") is None
        mock_api.fetch_by_hash.assert_not_called()

    def test_api_mode_fetches_and_caches(self, meta_dir: Path) -> None:
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = SAMPLE_ARTICLE
        loader = WebMetadataLoader(meta_dir, source="api", api_client=mock_api)

        result = loader.load_metadata("newvideo")
        assert result is not None
        assert result["name"] == "Test article title"
        mock_api.fetch_by_hash.assert_called_once_with("newvideo")

        # Verify cache file was written
        assert (meta_dir / "newvideo.json").exists()

    def test_api_mode_uses_cache(self, meta_dir: Path) -> None:
        (meta_dir / "cached.json").write_text(json.dumps(SAMPLE_ARTICLE))
        mock_api = MagicMock()
        loader = WebMetadataLoader(meta_dir, source="api", api_client=mock_api)

        result = loader.load_metadata("cached")
        assert result is not None
        mock_api.fetch_by_hash.assert_not_called()

    def test_api_mode_returns_none_on_404(self, meta_dir: Path) -> None:
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = None
        loader = WebMetadataLoader(meta_dir, source="api", api_client=mock_api)

        assert loader.load_metadata("missing") is None

    def test_hybrid_prefers_local(self, meta_dir: Path) -> None:
        (meta_dir / "local_hash.json").write_text(json.dumps(SAMPLE_ARTICLE))
        mock_api = MagicMock()
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        result = loader.load_metadata("local_hash")
        assert result is not None
        mock_api.fetch_by_hash.assert_not_called()

    def test_hybrid_falls_back_to_api(self, meta_dir: Path) -> None:
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = SAMPLE_ARTICLE
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        result = loader.load_metadata("api_only")
        assert result is not None
        mock_api.fetch_by_hash.assert_called_once_with("api_only")
        # Verify caching
        assert (meta_dir / "api_only.json").exists()

    def test_hybrid_returns_none_when_both_miss(self, meta_dir: Path) -> None:
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = None
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        assert loader.load_metadata("nowhere") is None

    def test_api_error_handled_gracefully(self, meta_dir: Path) -> None:
        mock_api = MagicMock()
        mock_api.fetch_by_hash.side_effect = RuntimeError("connection refused")
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        # Should not raise, just return None
        assert loader.load_metadata("broken") is None
