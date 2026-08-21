"""Tests for the expiring media tokens used in media URLs.

These tokens are the credential that travels in the query string, because
``<video>`` and HLS segment requests cannot send an Authorization header. Two
properties have to hold together: an unexpired token must be accepted (or every
viewer gets 401 and video breaks for the whole newsroom), and an expired or
tampered one must not be.
"""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from rainrag.api import (
    _media_token_is_valid,
    issue_media_token,
    verify_auth_token,
)


SECRET = "test-secret-value"


def _ui_style_token(secret: str, ttl: int = 3600) -> str:
    """Mint a token the way app.py does, independently of the API helper.

    The UI and API sign separately; if the two ever drift, every media URL
    starts returning 401. This reproduces the UI's construction by hand so the
    test fails on drift rather than passing because both call the same code.
    """
    payload = str(int(time.time()) + ttl)
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"v1.{payload}.{signature}"


class TestAccepted:
    def test_freshly_issued_token_is_valid(self):
        assert _media_token_is_valid(issue_media_token(secret=SECRET), SECRET)

    def test_ui_and_api_agree(self):
        """The independently constructed UI form must verify against the API."""
        assert _media_token_is_valid(_ui_style_token(SECRET), SECRET)

    def test_shape_is_versioned(self):
        token = issue_media_token(secret=SECRET)
        version, payload, signature = token.split(".")
        assert version == "v1"
        assert payload.isdigit()
        assert len(signature) == 32


class TestRejected:
    def test_expired(self):
        assert not _media_token_is_valid(issue_media_token(-60, secret=SECRET), SECRET)

    def test_expiring_exactly_now(self):
        """A token whose expiry has just passed must not squeak through."""
        payload = str(int(time.time()) - 1)
        sig = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
        assert not _media_token_is_valid(f"v1.{payload}.{sig}", SECRET)

    def test_tampered_signature(self):
        token = issue_media_token(secret=SECRET)
        flipped = token[:-1] + ("0" if token[-1] != "0" else "1")
        assert not _media_token_is_valid(flipped, SECRET)

    def test_tampered_expiry_extends_nothing(self):
        """Pushing the expiry out must invalidate the signature."""
        _, payload, sig = issue_media_token(secret=SECRET).split(".")
        forged = f"v1.{int(payload) + 86400}.{sig}"
        assert not _media_token_is_valid(forged, SECRET)

    def test_signed_with_a_different_secret(self):
        assert not _media_token_is_valid(issue_media_token(secret="other"), SECRET)

    @pytest.mark.parametrize(
        "token",
        ["", "nonsense", "v1.only-two", "v2.123.abc", "v1.notanumber.abc", "v1..", SECRET],
    )
    def test_malformed(self, token: str):
        assert not _media_token_is_valid(token, SECRET)

    def test_the_standing_secret_is_not_a_media_token(self):
        """It is still accepted by verify_auth_token, but not via this path."""
        assert not _media_token_is_valid(SECRET, SECRET)


class TestAuthDisabled:
    def test_issuing_returns_empty_without_a_secret(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
        assert issue_media_token() == ""


class TestVerifyAuthTokenIntegration:
    def test_media_token_is_accepted_as_the_query_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", SECRET)
        assert verify_auth_token(access_token=issue_media_token(secret=SECRET)) is True

    def test_standing_secret_still_accepted(self, monkeypatch: pytest.MonkeyPatch):
        """API clients and existing integrations must keep working."""
        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", SECRET)
        assert verify_auth_token(access_token=SECRET) is True

    def test_expired_media_token_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        from fastapi import HTTPException

        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", SECRET)
        with pytest.raises(HTTPException) as excinfo:
            verify_auth_token(access_token=issue_media_token(-60, secret=SECRET))
        assert excinfo.value.status_code == 401
