"""Tests for _validate_video_url, the SSRF guard on user-supplied video URLs.

POST /video-sessions/from-url and POST /telegram/download pass the URL to
yt-dlp or Telethon, which fetch it from inside the network. The guard must
refuse a host that is, or resolves to, a non-public address. Every test here
patches socket.getaddrinfo, so none of them depends on real DNS.
"""

from __future__ import annotations

import socket

import pytest
from fastapi import HTTPException

from rainrag.api import _validate_video_url


def _answer(*addresses: str):
    """Return a fake getaddrinfo that answers with ``addresses``."""

    def fake(host, port, *args, **kwargs):
        fake.calls.append(host)
        out = []
        for addr in addresses:
            if ":" in addr:
                out.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (addr, port or 0, 0, 0)))
            else:
                out.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port or 0)))
        return out

    fake.calls = []
    return fake


def _unresolvable(host, port, *args, **kwargs):
    raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")


def _refused(url: str) -> str:
    with pytest.raises(HTTPException) as info:
        _validate_video_url(url)
    assert info.value.status_code == 400
    return str(info.value.detail)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/video.mp4",
        "http://10.1.2.3/video.mp4",
        "http://172.16.0.1/video.mp4",
        "http://192.168.1.1/video.mp4",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/video.mp4",
        "http://[fe80::1%25eth0]/video.mp4",
        "http://[::ffff:10.0.0.1]/video.mp4",
    ],
)
def test_literal_private_ip_is_refused_without_a_lookup(monkeypatch, url):
    fake = _answer("93.184.216.34")
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    assert _refused(url) == "URL targets a non-public address"
    assert fake.calls == []


def test_literal_public_ip_passes_without_a_lookup(monkeypatch):
    fake = _answer("10.0.0.5")
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    _validate_video_url("https://93.184.216.34/video.mp4")
    assert fake.calls == []


def test_localhost_is_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answer("127.0.0.1", "::1"))
    assert _refused("http://localhost:8080/video.mp4") == "URL targets a non-public address"


@pytest.mark.parametrize("address", ["10.0.0.5", "::1", "169.254.169.254", "172.16.5.4"])
def test_name_that_resolves_to_a_private_address_is_refused(monkeypatch, address):
    fake = _answer(address)
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    assert _refused("https://evil.example.com/v.mp4") == "URL targets a non-public address"
    assert fake.calls == ["evil.example.com"]


def test_name_that_resolves_to_a_public_address_passes(monkeypatch):
    fake = _answer("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    _validate_video_url("https://www.youtube.com/watch?v=abc")
    assert fake.calls == ["www.youtube.com"]


def test_mixed_answer_with_one_private_address_is_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answer("93.184.216.34", "10.0.0.5"))
    assert _refused("https://mixed.example.com/v.mp4") == "URL targets a non-public address"


def test_unresolvable_name_is_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _unresolvable)
    assert _refused("https://no-such-host.invalid/v.mp4") == "URL host name does not resolve"


def test_empty_answer_is_refused(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", _answer())
    assert _refused("https://empty.example.com/v.mp4") == "URL host name does not resolve"


@pytest.mark.parametrize(
    ("url", "detail"),
    [
        ("ftp://example.com/v.mp4", "Only http and https URLs are supported"),
        ("file:///etc/passwd", "Only http and https URLs are supported"),
        ("https://user:pw@example.com/v.mp4", "URLs with embedded credentials are not supported"),
        ("http:///v.mp4", "URL has no host"),
        ("http://example.com:99999/v.mp4", "Invalid URL"),
    ],
)
def test_existing_checks_still_refuse(monkeypatch, url, detail):
    monkeypatch.setattr(socket, "getaddrinfo", _answer("93.184.216.34"))
    assert _refused(url) == detail
