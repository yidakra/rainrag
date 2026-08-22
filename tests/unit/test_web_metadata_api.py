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
    # The live API always emits these, empty or not; their absence marks a cache
    # file written before the taxonomy shipped.
    "tags": [{"id": 258, "name": "Украина", "category": "theme"}],
    "stories": [],
    "tableOfContents": [],
}

# A cache file from before the taxonomy existed: no `tags` key at all.
LEGACY_CACHED_ARTICLE = {
    key: value for key, value in SAMPLE_ARTICLE.items() if key not in ("tags", "stories")
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
def mock_httpx_client():
    mock_instance = MagicMock()
    mock_client_cls = MagicMock()
    mock_client_cls.return_value.__enter__.return_value = mock_instance
    mock_client_cls.return_value.__exit__.return_value = False

    with patch("rainrag.web_metadata_api.httpx.Client", mock_client_cls):
        yield mock_client_cls, mock_instance


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

    def test_fetch_by_hash_success(
        self,
        api_client: WebMetadataAPIClient,
        mock_httpx_client,
    ) -> None:
        _mock_client_cls, mock_instance = mock_httpx_client
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_ARTICLE
        mock_instance.get.return_value = mock_resp

        result = api_client.fetch_by_hash("abc123def456")
        assert result == SAMPLE_ARTICLE

    def test_fetch_by_hash_404(
        self,
        api_client: WebMetadataAPIClient,
        mock_httpx_client,
    ) -> None:
        _mock_client_cls, mock_instance = mock_httpx_client
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_instance.get.return_value = mock_resp

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

    def test_sync_to_local(
        self,
        api_client: WebMetadataAPIClient,
        meta_dir: Path,
        mock_httpx_client,
    ) -> None:
        _mock_client_cls, mock_instance = mock_httpx_client
        zip_data = _make_zip([SAMPLE_ARTICLE])
        mock_resp = MagicMock()
        mock_resp.content = zip_data
        mock_resp.raise_for_status = MagicMock()
        mock_instance.get.return_value = mock_resp

        written = api_client.sync_to_local(meta_dir)

        assert written == 1
        out_file = meta_dir / "abc123def456.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["name"] == "Test article title"

    def test_sync_to_local_skips_no_hash(
        self,
        api_client: WebMetadataAPIClient,
        meta_dir: Path,
        mock_httpx_client,
    ) -> None:
        _mock_client_cls, mock_instance = mock_httpx_client
        article_no_hash = {**SAMPLE_ARTICLE}
        del article_no_hash["video_hash"]
        zip_data = _make_zip([article_no_hash])
        mock_resp = MagicMock()
        mock_resp.content = zip_data
        mock_resp.raise_for_status = MagicMock()
        mock_instance.get.return_value = mock_resp

        written = api_client.sync_to_local(meta_dir)

        assert written == 0


# ---------------------------------------------------------------------------
# WebMetadataLoader multi-source tests
# ---------------------------------------------------------------------------


class TestWebMetadataLoaderSources:
    def test_local_mode_reads_file(self, meta_dir: Path) -> None:
        (meta_dir / "abc123.json").write_text(json.dumps(SAMPLE_ARTICLE), encoding="utf-8")
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
        (meta_dir / "cached.json").write_text(json.dumps(SAMPLE_ARTICLE), encoding="utf-8")
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
        (meta_dir / "local_hash.json").write_text(json.dumps(SAMPLE_ARTICLE), encoding="utf-8")
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


class TestStaleCacheRefetch:
    """A cache file predating the taxonomy must not silently index without tags."""

    def test_api_mode_refetches_a_legacy_cache_entry(self, meta_dir: Path) -> None:
        (meta_dir / "legacy.json").write_text(json.dumps(LEGACY_CACHED_ARTICLE), encoding="utf-8")
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = SAMPLE_ARTICLE
        loader = WebMetadataLoader(meta_dir, source="api", api_client=mock_api)

        result = loader.load_metadata("legacy")

        mock_api.fetch_by_hash.assert_called_once_with("legacy")
        assert result is not None and result["tags"]
        # The refreshed article replaces the legacy file on disk.
        assert "tags" in json.loads((meta_dir / "legacy.json").read_text(encoding="utf-8"))

    def test_hybrid_mode_refetches_a_legacy_cache_entry(self, meta_dir: Path) -> None:
        (meta_dir / "legacy.json").write_text(json.dumps(LEGACY_CACHED_ARTICLE), encoding="utf-8")
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = SAMPLE_ARTICLE
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        assert loader.load_metadata("legacy")["tags"]
        mock_api.fetch_by_hash.assert_called_once_with("legacy")

    def test_failed_refetch_falls_back_to_the_legacy_entry(self, meta_dir: Path) -> None:
        """Stale title/date/programme still beats indexing nothing."""
        (meta_dir / "legacy.json").write_text(json.dumps(LEGACY_CACHED_ARTICLE), encoding="utf-8")
        mock_api = MagicMock()
        mock_api.fetch_by_hash.return_value = None
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        result = loader.load_metadata("legacy")

        assert result is not None
        assert result["name"] == "Test article title"
        assert "tags" not in result

    def test_local_mode_warns_up_front_about_a_stale_cache(
        self, meta_dir: Path, monkeypatch
    ) -> None:
        """Local mode cannot refetch, so the stale cache is reported once per run."""
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        for name in ("a", "b", "c"):
            (meta_dir / f"{name}.json").write_text(
                json.dumps(LEGACY_CACHED_ARTICLE), encoding="utf-8"
            )
        warnings: list[str] = []
        monkeypatch.setattr(
            "rainrag.ingest.logger.warning", lambda message, *a, **k: warnings.append(str(message))
        )

        assert warn_if_metadata_cache_predates_taxonomy(meta_dir) is True
        assert len(warnings) == 1
        assert "sync-metadata" in warnings[0]

    def test_no_warning_when_any_sampled_file_is_current(self, meta_dir: Path) -> None:
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        (meta_dir / "legacy.json").write_text(json.dumps(LEGACY_CACHED_ARTICLE), encoding="utf-8")
        (meta_dir / "current.json").write_text(json.dumps(SAMPLE_ARTICLE), encoding="utf-8")

        assert warn_if_metadata_cache_predates_taxonomy(meta_dir) is False

    def test_no_warning_for_an_empty_or_missing_cache_dir(self, meta_dir: Path) -> None:
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        assert warn_if_metadata_cache_predates_taxonomy(meta_dir) is False
        assert warn_if_metadata_cache_predates_taxonomy(meta_dir / "nope") is False

    def test_a_current_cache_entry_is_not_refetched(self, meta_dir: Path) -> None:
        """An article with no tags still has the key, so it is not stale."""
        untagged = {**SAMPLE_ARTICLE, "tags": []}
        (meta_dir / "untagged.json").write_text(json.dumps(untagged), encoding="utf-8")
        mock_api = MagicMock()
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=mock_api)

        assert loader.load_metadata("untagged") is not None
        mock_api.fetch_by_hash.assert_not_called()
