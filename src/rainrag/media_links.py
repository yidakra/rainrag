"""Browser-facing URLs for archive media, for clients with no CMS card to link.

An episode with a CMS article links to its page on the site. The 27% of the
index with no article has no page to link to, but the media itself is always
reachable: the archive path falls out of the video hash alone. These helpers
turn that path into a URL an editor can open.

The ``auth`` token is minted the way ``rainrag.api.issue_media_token`` does,
because that is the function verifying it -- same payload, same secret, same
truncation. ``api.py`` and ``app.py`` still carry their own copies: api.py is
the verifier and owns the format, and app.py talks to the API over HTTP
without importing this package at all. This module is deliberately a leaf
(stdlib only) so the Streamlit pages can import it without dragging in the
query engine.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


MEDIA_TOKEN_TTL_SECONDS = int(os.getenv("RAINRAG_MEDIA_TOKEN_TTL_SECONDS", str(12 * 3600)))


def asset_base() -> str:
    """Public base URL media is served from, or "" when it is not configured.

    Empty is a normal state, not an error: a dev box without the archive, or a
    deployment that does not expose media, simply shows no links.
    """
    return os.getenv("RAINRAG_ASSET_URL", "").rstrip("/")


def issue_media_token() -> str:
    """A time-limited token for the ``auth`` query parameter, "" when auth is off.

    <video> and subtitle requests cannot set an Authorization header, so the
    credential travels in the URL. A signed expiry means a link pasted into a
    document or a chat stops working on its own instead of granting archive
    access forever.
    """
    secret = os.getenv("RAINRAG_AUTH_TOKEN")
    if not secret:
        return ""
    expires = int(time.time()) + MEDIA_TOKEN_TTL_SECONDS
    payload = str(expires)
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"v1.{payload}.{signature}"


def append_auth_query(url: str) -> str:
    """Add the expiring ``auth`` parameter, preserving any existing query and fragment."""
    token = issue_media_token()
    if not token:
        return url
    parts = urlsplit(url)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params.setdefault("auth", token)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def archive_media_url(rel_path: str, kind: str = "video") -> str | None:
    """URL for an archive-relative media path, or None when it cannot be built.

    ``rel_path`` is relative to the archive root, as the precompute scripts
    record it (``0c/14/…/<hash>_720p.mp4``). Returns None when no asset base is
    configured, so callers fall back to plain text rather than emitting a
    broken link.
    """
    base = asset_base()
    if not base or not rel_path:
        return None
    if kind not in {"video", "vtt"}:
        raise ValueError(f"unknown media kind: {kind!r}")
    # The path is segment-encoded, not quoted whole: the slashes are structure.
    encoded = "/".join(quote(seg, safe="") for seg in rel_path.strip("/").split("/"))
    return append_auth_query(f"{base}/{kind}/{encoded}")
