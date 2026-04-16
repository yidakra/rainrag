"""HTTP client for the library.tvrain.tv web metadata API.

Provides two access patterns:

* **Single lookup** -- ``fetch_by_hash(video_hash)`` calls
  ``GET /video/{hash}/article`` and returns parsed JSON.
* **Batch export** -- ``export_batch()`` calls ``GET /article/export``,
  downloads a ZIP archive of article data, and returns a list of parsed
  article dicts.

Both methods require a Bearer token passed via the ``Authorization`` header.
The token is read from the environment variable configured in
``WebMetadataConfig.api_token_env`` (default ``LIBRARY_API_TOKEN``).
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import httpx
from loguru import logger


_DEFAULT_TIMEOUT = 60.0  # seconds
_BATCH_TIMEOUT = 300.0  # batch download can be large


class WebMetadataAPIClient:
    """Client for the library.tvrain.tv metadata API."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        batch_timeout: float = _BATCH_TIMEOUT,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        self._timeout = timeout
        self._batch_timeout = batch_timeout

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        base_url: str = "https://library.tvrain.tv",
        token_env: str = "LIBRARY_API_TOKEN",
    ) -> WebMetadataAPIClient:
        """Create a client reading the Bearer token from an env var.

        Raises ``ValueError`` if the env var is not set or empty.
        """
        token = os.environ.get(token_env, "").strip()
        if not token:
            raise ValueError(
                f"Environment variable {token_env} is required for the web metadata API "
                + "but is not set or empty."
            )
        return cls(base_url=base_url, token=token)

    # ------------------------------------------------------------------
    # Single-hash lookup
    # ------------------------------------------------------------------

    def fetch_by_hash(self, video_hash: str) -> dict[str, Any] | None:
        """Fetch article metadata for a single video hash.

        Calls ``GET /video/{hash}/article`` with ``Accept: application/json``.

        Returns:
            Parsed article dict, or *None* if the API returned 404.

        Raises:
            httpx.HTTPStatusError: on non-404 HTTP errors.
        """
        url = f"{self.base_url}/video/{video_hash}/article"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, headers=self._headers)
                if resp.status_code == 404:
                    logger.debug(f"API returned 404 for video hash {video_hash}")
                    return None
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError:
            raise
        except httpx.HTTPError as exc:
            logger.error(f"HTTP error fetching metadata for {video_hash}: {exc}")
            raise

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    def export_batch(
        self,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict[str, Any]]:
        """Download batch article export as a ZIP and parse contained JSONs.

        Calls ``GET /article/export`` (returns a ZIP archive).
        By default the API returns articles for the last 180 days.
        ``start_time`` / ``end_time`` are Unix timestamps that narrow the
        window (max span: 180 days).

        Returns:
            List of parsed article dicts extracted from the ZIP.
        """
        url = f"{self.base_url}/article/export"
        params: dict[str, int] = {}
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time

        # For batch we don't send Accept: application/json -- the endpoint
        # returns a ZIP regardless.  Clone headers without Accept override.
        headers = {k: v for k, v in self._headers.items() if k.lower() != "accept"}

        with httpx.Client(timeout=self._batch_timeout) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()

        return self._parse_zip(resp.content)

    @staticmethod
    def _parse_zip(data: bytes) -> list[dict[str, Any]]:
        """Extract JSON article dicts from a ZIP archive."""
        articles: list[dict[str, Any]] = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                # Skip directories and non-JSON files
                if name.endswith("/") or not name.lower().endswith(".json"):
                    continue
                try:
                    raw = zf.read(name)
                    article = json.loads(raw)
                    if isinstance(article, dict):
                        articles.append(article)
                    elif isinstance(article, list):
                        # Some exports wrap articles in a list
                        for item in article:
                            if isinstance(item, dict):
                                articles.append(item)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping unparseable entry {name} in ZIP: {exc}")
        logger.info(f"Parsed {len(articles)} articles from batch export ZIP")
        return articles

    # ------------------------------------------------------------------
    # Sync to local directory
    # ------------------------------------------------------------------

    def sync_to_local(
        self,
        output_dir: Path,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> int:
        """Download batch export and write individual ``{hash}.json`` files.

        This bridges the API into the existing file-based
        ``WebMetadataLoader`` workflow.

        Returns:
            Number of metadata files written.
        """
        articles = self.export_batch(start_time=start_time, end_time=end_time)
        output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        for article in articles:
            video_hash = article.get("video_hash", "").strip()
            if not video_hash:
                logger.debug("Skipping article without video_hash field")
                continue
            # Sanitize: use only the basename to prevent path traversal
            safe_hash = Path(video_hash).name
            if not safe_hash or safe_hash != video_hash:
                logger.warning(f"Skipping article with invalid video_hash: {video_hash!r}")
                continue
            target = output_dir / f"{safe_hash}.json"
            target.write_text(
                json.dumps(article, ensure_ascii=False, indent=None), encoding="utf-8"
            )
            written += 1

        logger.info(f"Wrote {written} metadata files to {output_dir}")
        return written
