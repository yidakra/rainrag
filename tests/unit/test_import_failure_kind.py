"""Tests for classifying video-import failures.

The distinction matters to the person pasting the link: "the platform is
throttling us, try later", "this is region-locked, it will never work here" and
"there is no video at that link" are three different pieces of advice. Getting
it wrong is what made a real user on 2026-08-19 see "no downloadable video
found" when YouTube had in fact returned HTTP 403.
"""

from __future__ import annotations

import pytest

from rainrag.api import _download_failure_kind


def _err(message: str) -> Exception:
    """A stand-in for whatever the downloader raises; only str() is inspected."""
    return RuntimeError(message)


class TestBlocked:
    """Platform refused us; a later retry may succeed."""

    @pytest.mark.parametrize(
        "message",
        [
            # The exact failure a journalist hit on 2026-08-19.
            "ERROR: unable to download video data: HTTP Error 403: Forbidden",
            "ERROR: Sign in to confirm you're not a bot. Use --cookies",
            "ERROR: HTTP Error 429: Too Many Requests",
            "ERROR: rate limit exceeded, retry later",
            "ERROR: Too Many Requests",
        ],
    )
    def test_classified_as_blocked(self, message: str):
        assert _download_failure_kind(_err(message)) == "blocked"


class TestGeo:
    """Region locked; retrying from this server changes nothing."""

    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: The uploader has not made this video available from your location",
            "ERROR: The uploader has blocked it in your country",
            "ERROR: This video is not available in your country",
            "ERROR: video is geo restricted",
            "ERROR: content is geo-restricted",
        ],
    )
    def test_classified_as_geo(self, message: str):
        assert _download_failure_kind(_err(message)) == "geo"

    def test_geo_wins_over_blocked(self):
        """A message mentioning both must give the region advice, not "retry"."""
        both = "ERROR: HTTP Error 403: Forbidden - not available from your location"
        assert _download_failure_kind(_err(both)) == "geo"


class TestOrdinaryFailures:
    """Everything else: the link really is wrong, or something else broke."""

    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: Unable to download webpage: Not Found (404)",
            "ERROR: [twitter] No video could be found in this tweet",
            "ERROR: Video unavailable",
            "ERROR: Failed to resolve 'nope.example.com'",
            "",
        ],
    )
    def test_classified_as_failed(self, message: str):
        assert _download_failure_kind(_err(message)) == "failed"

    def test_matching_is_case_insensitive(self):
        assert _download_failure_kind(_err("HTTP ERROR 403: FORBIDDEN")) == "blocked"
