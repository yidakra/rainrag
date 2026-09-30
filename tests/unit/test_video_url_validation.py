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


@pytest.fixture(autouse=True)
def default_internal_hosts(monkeypatch):
    """Start every test from the default allowlist, whatever the shell has set."""
    monkeypatch.delenv("RAINRAG_URL_INTERNAL_HOSTS", raising=False)


def _answer(*addresses: str):
    """Return a fake getaddrinfo that answers with ``addresses``."""

    def fake(host, port, *args, **kwargs):
        """Record the host asked for and answer with the given addresses on its port."""
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
    """Fail the lookup the way getaddrinfo does for an unknown name."""
    raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")


def _refused(url: str) -> str:
    """Assert that ``url`` is a 400 and return the error detail for a message check."""
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
    """A non-public IP literal is a 400 before any DNS lookup, zone ids and IPv4-mapped forms included."""
    fake = _answer("93.184.216.34")
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    assert _refused(url) == "URL targets a non-public address"
    assert fake.calls == []


def test_literal_public_ip_passes_without_a_lookup(monkeypatch):
    """A public IP literal passes, and the resolver is not asked about it."""
    fake = _answer("10.0.0.5")
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    _validate_video_url("https://93.184.216.34/video.mp4")
    assert fake.calls == []


def test_localhost_is_refused(monkeypatch):
    """A name that resolves to loopback on both families is a 400."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("127.0.0.1", "::1"))
    assert _refused("http://localhost:8080/video.mp4") == "URL targets a non-public address"


@pytest.mark.parametrize("address", ["10.0.0.5", "::1", "169.254.169.254", "172.16.5.4"])
def test_name_that_resolves_to_a_private_address_is_refused(monkeypatch, address):
    """A public-looking name that resolves to a private, loopback or metadata address is a 400."""
    fake = _answer(address)
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    assert _refused("https://evil.example.com/v.mp4") == "URL targets a non-public address"
    assert fake.calls == ["evil.example.com"]


def test_name_that_resolves_to_a_public_address_passes(monkeypatch):
    """A name whose IPv4 and IPv6 answers are all public passes after one lookup."""
    fake = _answer("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")
    monkeypatch.setattr(socket, "getaddrinfo", fake)
    _validate_video_url("https://www.youtube.com/watch?v=abc")
    assert fake.calls == ["www.youtube.com"]


def test_mixed_answer_with_one_private_address_is_refused(monkeypatch):
    """One private address among public ones is enough to refuse the URL."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("93.184.216.34", "10.0.0.5"))
    assert _refused("https://mixed.example.com/v.mp4") == "URL targets a non-public address"


def test_unresolvable_name_is_refused(monkeypatch):
    """A lookup failure is a 400 with its own message, not a pass."""
    monkeypatch.setattr(socket, "getaddrinfo", _unresolvable)
    assert _refused("https://no-such-host.invalid/v.mp4") == "URL host name does not resolve"


def test_empty_answer_is_refused(monkeypatch):
    """A lookup that returns no address is a 400, not a pass."""
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
    """The scheme, credential, empty-host and port checks keep their 400 messages."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("93.184.216.34"))
    assert _refused(url) == detail


# Split DNS: tvrain.tv resolves to an internal address from the TV Rain server.


@pytest.mark.parametrize("host", ["tvrain.tv", "www.tvrain.tv", "WWW.TvRain.tv", "tvrain.tv."])
def test_allowlisted_name_may_resolve_to_a_private_address(monkeypatch, host):
    """Split DNS: tvrain.tv and its subdomains pass with an RFC 1918 answer, in any case or with a trailing dot."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    _validate_video_url(f"https://{host}/news/some-story")


def test_allowlisted_name_may_resolve_to_a_ula_address(monkeypatch):
    """An allowlisted name also passes with an IPv6 ULA answer."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("fd12:3456:789a::1"))
    _validate_video_url("https://www.tvrain.tv/news/some-story")


@pytest.mark.parametrize("host", ["eviltvrain.tv", "tvrain.tv.evil.com", "nottvrain.tv"])
def test_lookalike_name_is_not_allowlisted(monkeypatch, host):
    """Only an exact match or a real subdomain uses the allowlist; a lookalike is a 400."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    assert _refused(f"https://{host}/v.mp4") == "URL targets a non-public address"


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "127.8.8.8", "::1", "169.254.169.254", "fe80::1", "0.0.0.0", "100.64.0.1"],
)
def test_allowlisted_name_never_admits_loopback_or_link_local(monkeypatch, address):
    """The allowlist tolerates RFC 1918 and ULA only; loopback, link-local and other non-global answers stay refused."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer(address))
    assert _refused("https://tvrain.tv/v.mp4") == "URL targets a non-public address"


def test_allowlisted_name_with_one_loopback_answer_is_refused(monkeypatch):
    """One loopback answer next to a tolerated private one still refuses an allowlisted name."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100", "127.0.0.1"))
    assert _refused("https://tvrain.tv/v.mp4") == "URL targets a non-public address"


def test_allowlisted_name_still_needs_to_resolve(monkeypatch):
    """The allowlist does not skip the lookup; an unresolvable allowlisted name is a 400."""
    monkeypatch.setattr(socket, "getaddrinfo", _unresolvable)
    assert _refused("https://tvrain.tv/v.mp4") == "URL host name does not resolve"


@pytest.mark.parametrize(
    ("url", "detail"),
    [
        ("ftp://tvrain.tv/v.mp4", "Only http and https URLs are supported"),
        ("https://user:pw@tvrain.tv/v.mp4", "URLs with embedded credentials are not supported"),
        ("https://tvrain.tv:99999/v.mp4", "Invalid URL"),
    ],
)
def test_allowlisted_name_keeps_the_other_checks(monkeypatch, url, detail):
    """The scheme, credential and port checks still apply to an allowlisted name."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    assert _refused(url) == detail


def test_env_var_replaces_the_default(monkeypatch):
    """RAINRAG_URL_INTERNAL_HOSTS replaces the default list, with spaces and leading dots trimmed."""
    monkeypatch.setenv("RAINRAG_URL_INTERNAL_HOSTS", " intranet.example.org , .other.test ")
    monkeypatch.setattr(socket, "getaddrinfo", _answer("10.20.30.40"))
    _validate_video_url("https://intranet.example.org/v.mp4")
    _validate_video_url("https://media.other.test/v.mp4")
    assert _refused("https://tvrain.tv/v.mp4") == "URL targets a non-public address"


def test_empty_env_var_allowlists_nothing(monkeypatch):
    """An empty RAINRAG_URL_INTERNAL_HOSTS turns the allowlist off, default included."""
    monkeypatch.setenv("RAINRAG_URL_INTERNAL_HOSTS", "")
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    assert _refused("https://tvrain.tv/v.mp4") == "URL targets a non-public address"


@pytest.mark.parametrize(
    "entry", ["172.16.50.100", "10.0.0.5", "127.0.0.1", "::1", "[::1]", "169.254.169.254"]
)
def test_ip_literal_in_env_var_has_no_effect(monkeypatch, entry):
    """An IP literal in the variable neither admits that IP nor keeps the default names."""
    monkeypatch.setenv("RAINRAG_URL_INTERNAL_HOSTS", entry)
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    host = f"[{entry.strip('[]')}]" if ":" in entry else entry
    assert _refused(f"http://{host}/v.mp4") == "URL targets a non-public address"
    # The literal also does not stand in for the default name list.
    assert _refused("https://tvrain.tv/v.mp4") == "URL targets a non-public address"


# An allowlisted internal server may run Qdrant (6333/6334) or Redis (6379) on
# 0.0.0.0, so an internal answer is accepted only on the scheme's default port.


@pytest.mark.parametrize(
    "url",
    [
        "http://tvrain.tv:6333/collections",
        "http://rag.tvrain.tv:6334/",
        "http://tvrain.tv:6379/",
        "https://www.tvrain.tv:8080/v.mp4",
        "http://tvrain.tv:443/v.mp4",
        "https://tvrain.tv:80/v.mp4",
    ],
)
def test_allowlisted_internal_answer_on_another_port_is_refused(monkeypatch, url):
    """An internal answer on a port other than the scheme default is a 400, so Qdrant and Redis stay out of reach."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    assert _refused(url) == "URL targets a non-public address"


@pytest.mark.parametrize(
    "url",
    [
        "http://tvrain.tv/v.mp4",
        "https://tvrain.tv/v.mp4",
        "http://tvrain.tv:80/v.mp4",
        "https://www.tvrain.tv:443/v.mp4",
    ],
)
def test_allowlisted_internal_answer_on_the_default_port_passes(monkeypatch, url):
    """An internal answer passes with no port, or with 80 on http and 443 on https."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("172.16.50.100"))
    _validate_video_url(url)


@pytest.mark.parametrize(
    "url", ["https://www.youtube.com:8080/v.mp4", "http://tvrain.tv:8080/v.mp4"]
)
def test_public_answer_keeps_any_port(monkeypatch, url):
    """The port rule applies only to internal answers; a public answer passes on any port."""
    monkeypatch.setattr(socket, "getaddrinfo", _answer("93.184.216.34"))
    _validate_video_url(url)
