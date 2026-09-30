"""Tests for POST /telegram/download, the download-only Telegram endpoint.

It exists so another trusted service (the danbi try page) can use rainrag's
Telegram login without a copy of the session leaving this server. The download
itself is faked; these tests pin the contract around it: auth, link checks,
error mapping, and that nothing is left on disk after the response.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rainrag.api import app
from rainrag.telegram_media import TelegramNotDownloadableError


def _manager(tmp_path: Path, enabled: bool = True) -> SimpleNamespace:
    cfg = SimpleNamespace(
        telegram_enabled=enabled,
        telegram_api_id_env="TG_TEST_API_ID",
        telegram_api_hash_env="TG_TEST_API_HASH",
        telegram_session_path=str(tmp_path / "telegram.session"),
        telegram_flood_sleep_threshold=60,
        max_upload_mb=512,
        tmp_root=str(tmp_path / "tmp"),
    )
    return SimpleNamespace(cfg=cfg)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TG_TEST_API_ID", "12345")
    monkeypatch.setenv("TG_TEST_API_HASH", "not-a-real-hash")
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    with TestClient(app) as c:
        yield c


def _fake_download(payload: bytes = b"fake mp4 bytes"):
    """A stand-in for download_telegram_video that writes a file where told."""

    async def fake(ref, dest_dir, **kwargs):
        out = Path(dest_dir) / "telegram_video.mp4"
        out.write_bytes(payload)
        fake.calls.append({"ref": ref, "dest": Path(dest_dir), **kwargs})
        return out

    fake.calls = []
    return fake


def test_returns_the_file_and_leaves_nothing_behind(client, tmp_path):
    """A good link returns the bytes, and the work directory is gone afterwards."""
    fake = _fake_download()
    with (
        patch("rainrag.api.video_session_manager", _manager(tmp_path)),
        patch("rainrag.telegram_media.download_telegram_video", fake),
    ):
        r = client.post("/telegram/download", json={"url": "https://t.me/somechannel/123"})
    assert r.status_code == 200
    assert r.content == b"fake mp4 bytes"
    assert r.headers["content-type"].startswith("video/")
    assert len(fake.calls) == 1
    assert not fake.calls[0]["dest"].exists()


def test_max_mb_cannot_exceed_the_server_cap(client, tmp_path):
    """The caller may lower the byte cap, never raise it above max_upload_mb."""
    fake = _fake_download()
    with (
        patch("rainrag.api.video_session_manager", _manager(tmp_path)),
        patch("rainrag.telegram_media.download_telegram_video", fake),
    ):
        client.post("/telegram/download", json={"url": "https://t.me/c/1/2", "max_mb": 100000})
        client.post("/telegram/download", json={"url": "https://t.me/c/1/2", "max_mb": 50})
    assert fake.calls[0]["max_bytes"] == 512 * 1024 * 1024
    assert fake.calls[1]["max_bytes"] == 50 * 1024 * 1024


def test_a_non_telegram_link_is_refused(client, tmp_path):
    """Only t.me posts: this endpoint is not a general downloader."""
    with patch("rainrag.api.video_session_manager", _manager(tmp_path)):
        r = client.post("/telegram/download", json={"url": "https://www.youtube.com/watch?v=abc"})
    assert r.status_code == 400


def test_disabled_telegram_is_a_503(client, tmp_path):
    """With telegram_enabled off the endpoint says so instead of guessing."""
    with patch("rainrag.api.video_session_manager", _manager(tmp_path, enabled=False)):
        r = client.post("/telegram/download", json={"url": "https://t.me/somechannel/123"})
    assert r.status_code == 503


def test_a_private_address_is_refused_before_any_download(client, tmp_path):
    """The shared URL validator runs first."""
    fake = _fake_download()
    with (
        patch("rainrag.api.video_session_manager", _manager(tmp_path)),
        patch("rainrag.telegram_media.download_telegram_video", fake),
    ):
        r = client.post("/telegram/download", json={"url": "http://127.0.0.1/x"})
    assert r.status_code == 400
    assert fake.calls == []


def test_a_post_without_video_is_a_422_with_the_reason(client, tmp_path):
    """TelegramNotDownloadableError keeps its message, as in the session import."""

    async def no_video(ref, dest_dir, **kwargs):
        raise TelegramNotDownloadableError("That Telegram post has no video.")

    with (
        patch("rainrag.api.video_session_manager", _manager(tmp_path)),
        patch("rainrag.telegram_media.download_telegram_video", no_video),
    ):
        r = client.post("/telegram/download", json={"url": "https://t.me/somechannel/9"})
    assert r.status_code == 422
    assert "no video" in r.json()["detail"]


def test_auth_is_required_when_a_token_is_configured(client, tmp_path, monkeypatch):
    """The endpoint hands out media, so it sits behind the API token like the rest."""
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "test-token")
    fake = _fake_download()
    with (
        patch("rainrag.api.video_session_manager", _manager(tmp_path)),
        patch("rainrag.telegram_media.download_telegram_video", fake),
    ):
        anon = client.post("/telegram/download", json={"url": "https://t.me/somechannel/123"})
        ok = client.post(
            "/telegram/download",
            json={"url": "https://t.me/somechannel/123"},
            headers={"Authorization": "Bearer test-token"},
        )
    assert anon.status_code == 401
    assert ok.status_code == 200
