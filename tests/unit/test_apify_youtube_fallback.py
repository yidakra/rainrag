"""Tests for the Apify fallback on blocked YouTube imports.

YouTube refuses downloads from this server's datacenter IP (#41). When the env
var named by ``video_upload.apify_token_env`` holds a token, a YouTube import
that yt-dlp reports as blocked is fetched through the johnvc/youtube-download-api
Apify Actor instead; without a token nothing changes. Apify is faked here: these
tests pin the gating, the request, and the hand-off to the session manager.
"""

from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from rainrag.api import (
    _apify_quality,
    _apify_youtube_download,
    _refuse_non_public_hop,
    app,
)


YOUTUBE_URL = "https://youtu.be/FMbQXEoaCJc"
FILE_URL = "https://api.apify.com/v2/key-value-stores/kv1/records/video.mp4?signature=abc"
# The yt-dlp failure seen on 2026-08-16.
BLOCKED = "ERROR: [youtube] FMbQXEoaCJc: Sign in to confirm you're not a bot"
UNAVAILABLE = "ERROR: [youtube] FMbQXEoaCJc: Video unavailable"
FILE_ROW = {
    "videoId": "FMbQXEoaCJc",
    "title": "Some title",
    "container": "mp4",
    "fileUrl": FILE_URL,
}


class _FakeResponse:
    """A stand-in for an httpx response: a JSON payload, file bytes or an error."""

    def __init__(
        self, payload: object = None, body: bytes = b"", error: Exception | None = None
    ) -> None:
        self._payload = payload
        self._body = body
        self._error = error

    def raise_for_status(self) -> None:
        """Raise the configured error, if any."""
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        """Return the canned payload."""
        return self._payload

    def iter_bytes(self):
        """Yield the canned file body."""
        yield self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeApify:
    """Plays the Apify API: canned JSON per request, then the file for fileUrl."""

    def __init__(self, responses: list[_FakeResponse], body: bytes = b"fake mp4") -> None:
        self.responses = list(responses)
        self.body = body
        self.calls: list[tuple[str, str, dict]] = []
        self.clients: list[dict] = []

    def client(self, **kwargs: object) -> _FakeApify:
        """Replace httpx.Client: record its settings and serve requests in order."""
        self.clients.append(kwargs)
        return self

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        """Record a POST and return the next canned response."""
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        """Record a GET and return the next canned response."""
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeResponse:
        """Replace httpx.stream: record the download and return the file body."""
        self.calls.append((method, url, kwargs))
        return _FakeResponse(body=self.body)

    def __enter__(self) -> _FakeApify:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _cfg(**overrides: object) -> SimpleNamespace:
    """The video_upload settings the fallback reads."""
    values: dict[str, object] = {
        "apify_token_env": "APIFY_TEST_TOKEN",
        "max_video_height": 720,
        "max_upload_mb": 512,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _run(status: str) -> _FakeResponse:
    """An Apify run object in the given state."""
    return _FakeResponse({"data": {"id": "run1", "status": status, "defaultDatasetId": "ds1"}})


@pytest.fixture
def fake_apify(monkeypatch):
    """Route httpx through a _FakeApify and set a token; tests fill in responses."""
    fake = _FakeApify([])
    monkeypatch.setenv("APIFY_TEST_TOKEN", "test-token")
    monkeypatch.setattr(httpx, "Client", fake.client)
    monkeypatch.setattr(httpx, "stream", fake.stream)
    return fake


class TestApifyDownload:
    def test_without_a_token_nothing_is_requested(self, fake_apify, monkeypatch, tmp_path):
        monkeypatch.delenv("APIFY_TEST_TOKEN")
        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None
        assert fake_apify.calls == []

    def test_config_without_the_setting_is_tolerated(self, fake_apify, tmp_path):
        cfg = SimpleNamespace(max_upload_mb=512)
        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, cfg) is None
        assert fake_apify.calls == []

    def test_other_platforms_are_not_sent_to_apify(self, fake_apify, tmp_path):
        assert _apify_youtube_download("https://vk.com/video-1_2", tmp_path, _cfg()) is None
        assert fake_apify.calls == []

    def test_polls_the_run_and_saves_the_video(self, fake_apify, tmp_path):
        fake_apify.responses = [_run("RUNNING"), _run("SUCCEEDED"), _FakeResponse([FILE_ROW])]

        info = _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg())

        assert info == {"title": "Some title"}
        assert (tmp_path / "video.mp4").read_bytes() == b"fake mp4"
        assert fake_apify.clients[0]["headers"] == {"Authorization": "Bearer test-token"}
        method, path, kwargs = fake_apify.calls[0]
        assert (method, path) == ("POST", "/acts/johnvc~youtube-download-api/runs")
        assert kwargs["json"] == {
            "videoUrls": [YOUTUBE_URL],
            "format": "video",
            "quality": "720p",
            "videoCodec": "h264",
            "maxMegabytes": 512,
        }
        assert fake_apify.calls[1][1] == "/actor-runs/run1"
        assert fake_apify.calls[2][1] == "/datasets/ds1/items"
        # The signed fileUrl is fetched by its own client, without the token. It
        # follows the redirect Apify may answer with and checks every hop.
        assert fake_apify.calls[3] == ("GET", FILE_URL, {})
        media_client = fake_apify.clients[1]
        assert "headers" not in media_client
        assert media_client["follow_redirects"] is True
        assert media_client["event_hooks"] == {"request": [_refuse_non_public_hop]}

    def test_a_skipped_video_returns_none(self, fake_apify, tmp_path):
        skipped = {"resultType": "skipped", "error": "OVER_MAX_MEGABYTES"}
        fake_apify.responses = [_run("SUCCEEDED"), _FakeResponse([skipped])]
        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None
        assert not (tmp_path / "video.mp4").exists()

    def test_an_api_error_returns_none(self, fake_apify, tmp_path):
        request = httpx.Request("POST", "https://api.apify.com/v2/acts")
        error = httpx.HTTPStatusError(
            "401 Unauthorized", request=request, response=httpx.Response(401)
        )
        fake_apify.responses = [_FakeResponse(error=error)]
        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None

    @pytest.mark.parametrize(
        "payload",
        [{"error": "not a list"}, [None, "row", 5], [{"title": "no fileUrl"}]],
    )
    def test_a_malformed_dataset_returns_none(self, fake_apify, tmp_path, payload):
        fake_apify.responses = [_run("SUCCEEDED"), _FakeResponse(payload)]
        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None
        assert not (tmp_path / "video.mp4").exists()

    def test_a_malformed_run_returns_none(self, fake_apify, tmp_path):
        fake_apify.responses = [_FakeResponse({"data": None})]
        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None

    def test_download_stops_one_byte_past_the_limit(self, fake_apify, tmp_path):
        limit = 1024 * 1024
        fake_apify.body = b"x" * (limit + 4096)
        fake_apify.responses = [_run("SUCCEEDED"), _FakeResponse([FILE_ROW])]

        info = _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg(max_upload_mb=1))

        # One byte over is enough for the endpoint's size check to answer 413.
        assert info == {"title": "Some title"}
        assert (tmp_path / "video.mp4").stat().st_size == limit + 1

    def test_failure_log_leaves_out_the_signed_media_url(self, fake_apify, monkeypatch, tmp_path):
        """A failed media download logs the error type and status, never the signed URL."""
        request = httpx.Request("GET", FILE_URL)
        error = httpx.HTTPStatusError(
            f"403 Forbidden for url '{FILE_URL}'", request=request, response=httpx.Response(403)
        )
        monkeypatch.setattr(fake_apify, "stream", lambda *a, **k: _FakeResponse(error=error))
        fake_apify.responses = [_run("SUCCEEDED"), _FakeResponse([FILE_ROW])]

        with patch("rainrag.api.logger") as log:
            assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None

        logged = " ".join(str(arg) for call in log.warning.call_args_list for arg in call.args)
        assert "signature" not in logged
        assert "HTTPStatusError" in logged
        assert "403" in logged

    def test_a_run_past_the_deadline_is_aborted(self, fake_apify, monkeypatch, tmp_path):
        monkeypatch.setattr("rainrag.api._APIFY_WAIT_SECONDS", -1)
        fake_apify.responses = [_run("RUNNING"), _FakeResponse({"data": {}})]

        assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None
        assert fake_apify.calls[1][:2] == ("POST", "/actor-runs/run1/abort")
        assert len(fake_apify.calls) == 2

    def test_a_slow_media_download_gives_up(self, fake_apify, monkeypatch, tmp_path):
        """A media download past _APIFY_MEDIA_SECONDS stops and returns None."""
        monkeypatch.setattr("rainrag.api._APIFY_MEDIA_SECONDS", -1)
        fake_apify.responses = [_run("SUCCEEDED"), _FakeResponse([FILE_ROW])]

        with patch("rainrag.api.logger") as log:
            assert _apify_youtube_download(YOUTUBE_URL, tmp_path, _cfg()) is None

        logged = " ".join(str(arg) for call in log.warning.call_args_list for arg in call.args)
        assert "ReadTimeout" in logged
        assert (tmp_path / "video.mp4").stat().st_size == 0

    def test_media_hops_to_internal_addresses_are_refused(self):
        """The hop check refuses metadata and loopback targets."""
        for target in ("http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:6333/"):
            with pytest.raises(httpx.RequestError):
                _refuse_non_public_hop(httpx.Request("GET", target))

    def test_media_hops_to_public_hosts_pass(self, public_dns):
        _refuse_non_public_hop(httpx.Request("GET", FILE_URL))

    @pytest.mark.parametrize(
        ("max_height", "quality"),
        [(720, "720p"), (480, "480p"), (1000, "720p"), (100, "144p"), (0, "1080p")],
    )
    def test_quality_follows_max_video_height(self, max_height, quality):
        assert _apify_quality(max_height) == quality


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host name to a public address, so no test needs real DNS."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        """Answer every lookup with one public IPv4 address."""
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture
def client(public_dns, monkeypatch):
    """A test client with no API token configured."""
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    with TestClient(app) as c:
        yield c


def _manager(tmp_path: Path, sessions: list) -> SimpleNamespace:
    """A stand-in video manager that records what reaches create_session."""
    cfg = _cfg(tmp_root=str(tmp_path / "tmp"), yt_dlp_cookies_path="", telegram_enabled=False)

    def create_session(path: Path, filename: str) -> SimpleNamespace:
        """Record the file and name the endpoint hands over."""
        sessions.append((Path(path).read_bytes(), filename))
        return SimpleNamespace(public_dict=lambda: {"id": "s1", "filename": filename})

    return SimpleNamespace(cfg=cfg, create_session=create_session)


def _apify_writes_video(url: str, dest: Path, cfg: object) -> dict:
    """A successful Apify download."""
    (dest / "video.mp4").write_bytes(b"apify mp4")
    return {"title": "Apify title"}


class TestImportFallback:
    def test_blocked_youtube_import_is_served_by_apify(self, client, tmp_path):
        sessions: list = []
        with (
            patch("rainrag.api.video_session_manager", _manager(tmp_path, sessions)),
            patch("rainrag.api._yt_dlp_download", side_effect=RuntimeError(BLOCKED)),
            patch("rainrag.api._apify_youtube_download", side_effect=_apify_writes_video) as apify,
        ):
            r = client.post("/video-sessions/from-url", json={"url": YOUTUBE_URL})
        assert r.status_code == 200
        assert sessions == [(b"apify mp4", "Apify title.mp4")]
        assert apify.call_args.args[0] == YOUTUBE_URL

    def test_block_stays_a_503_when_apify_has_no_file(self, client, tmp_path):
        sessions: list = []
        with (
            patch("rainrag.api.video_session_manager", _manager(tmp_path, sessions)),
            patch("rainrag.api._yt_dlp_download", side_effect=RuntimeError(BLOCKED)),
            patch("rainrag.api._apify_youtube_download", return_value=None),
        ):
            r = client.post("/video-sessions/from-url", json={"url": YOUTUBE_URL})
        assert r.status_code == 503
        assert sessions == []

    def test_other_failures_do_not_try_apify(self, client, tmp_path):
        sessions: list = []
        with (
            patch("rainrag.api.video_session_manager", _manager(tmp_path, sessions)),
            patch("rainrag.api._yt_dlp_download", side_effect=RuntimeError(UNAVAILABLE)),
            patch("rainrag.api._apify_youtube_download") as apify,
        ):
            r = client.post("/video-sessions/from-url", json={"url": YOUTUBE_URL})
        assert r.status_code == 422
        apify.assert_not_called()
