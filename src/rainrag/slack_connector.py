"""Slack connector for RainRAG.

Lets journalists use the archive from Slack instead of the Streamlit UI, at
feature parity with it: mention the bot in a channel, DM it, or use the
/rainrag slash command.

Supported interactions (all bilingual, answer language follows the question):

- Archive Q&A with inline ``from:``/``to:`` date filters, ``top:N`` retrieval
  depth and ``lang:ru|en`` override -- the sidebar controls of the web UI.
- A context reply with transcript excerpts, scores, timecodes and expiring
  media links -- the expandable context panel of the web UI.
- "Find related" buttons on each excerpt -- the related-chunks explorer.
- ``name: <title>`` -- the search-by-name mode.
- ``video: <url>`` or an attached video file -- the upload mode: the video is
  imported and transcribed, and the Slack thread becomes the scoped Q&A
  session for it.
- ``status`` -- the backend health summary from the web UI sidebar.

This is a thin front-end, deliberately parallel to the Streamlit app: it calls
the existing FastAPI backend over HTTP rather than embedding a RAGQueryEngine,
so it inherits the backend's auth, concurrency limits, timeouts and usage
accounting, and stays light enough to run anywhere.

No Slack SDK is used. The Events API is plain HTTPS webhooks whose
authenticity check is a single HMAC, and replies are one POST to
``chat.postMessage`` -- httpx and the standard library cover both, which keeps
the dependency surface (and its supply chain) unchanged.

Slack requires webhook acknowledgement within 3 seconds while a RAG query
takes tens of seconds, so every handler acks immediately and does the actual
work in a background task, posting the answer when it is ready.
"""

import asyncio
import hashlib
import hmac
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger


SLACK_API_BASE = os.getenv("SLACK_API_BASE", "https://slack.com/api").rstrip("/")

# Slack rejects webhooks it cannot deliver in 3s and retries; the query itself
# runs in the background against the RainRAG API, whose own timeout is 240s by
# default, so this must comfortably exceed it.
QUERY_HTTP_TIMEOUT_SECONDS = float(os.getenv("RAINRAG_SLACK_QUERY_TIMEOUT_SECONDS", "300"))
SLACK_POST_TIMEOUT_SECONDS = 15.0

# Video import mirrors the web UI's budgets: the from-url request holds the
# connection open while yt-dlp downloads, and transcription continues
# server-side afterwards, observed by polling.
IMPORT_TIMEOUT_SECONDS = float(os.getenv("RAINRAG_SLACK_IMPORT_TIMEOUT_SECONDS", "1800"))
UPLOAD_TIMEOUT_SECONDS = float(os.getenv("RAINRAG_SLACK_UPLOAD_TIMEOUT_SECONDS", "1200"))
SESSION_WAIT_SECONDS = float(os.getenv("RAINRAG_SLACK_SESSION_WAIT_SECONDS", "3600"))
SESSION_POLL_SECONDS = float(os.getenv("RAINRAG_SLACK_SESSION_POLL_SECONDS", "10"))
MAX_UPLOAD_MB = int(os.getenv("RAINRAG_SLACK_MAX_UPLOAD_MB", "512"))

# Signed requests older than this are replays as far as we are concerned.
# Five minutes is the window Slack itself documents.
SIGNATURE_MAX_AGE_SECONDS = 300

# Slack limits a section block to 3000 characters of mrkdwn. Splitting a long
# answer across blocks preserves it; truncation would silently lose the tail.
SLACK_BLOCK_TEXT_LIMIT = 3000
MAX_ANSWER_BLOCKS = 5
MAX_SOURCES = 3
MAX_CONTEXT_CHUNKS = 5
MAX_NAME_RESULTS = 10
CONTEXT_EXCERPT_CHARS = 400
RELATED_TOP_K = 3  # same depth the web UI's "Find Related" button uses

_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

# How many of the top retrieved chunks get their actual footage posted into
# the thread as an inline clip (0 disables). One is the deliberate default:
# the best match plays right in Slack, the rest stay links -- five videos per
# answer would bury the conversation.
CLIP_CHUNKS = int(os.getenv("RAINRAG_SLACK_CLIP_CHUNKS", "1"))
CLIP_MAX_MB = int(os.getenv("RAINRAG_SLACK_CLIP_MAX_MB", "50"))
CLIP_FETCH_TIMEOUT_SECONDS = float(os.getenv("RAINRAG_SLACK_CLIP_FETCH_TIMEOUT_SECONDS", "180"))

_HELP_TEXT = (
    "Задайте вопрос по архиву Дождя — я найду ответ в транскриптах эфиров.\n"
    "Ask a question about the TV Rain archive and I'll answer from broadcast transcripts.\n\n"
    "• В канале: `@RainRAG когда впервые обсуждали закон об иноагентах?`\n"
    "• В личных сообщениях: просто напишите вопрос\n"
    "• Фильтры: `from:2021-01-01 to:2021-12-31`, `top:10`, `lang:en`\n"
    "• Поиск по названию: `name: вечернее шоу` / `название: вечернее шоу`\n"
    "• Импорт видео: `video: <ссылка>` или пришлите файл — после обработки\n"
    "  задавайте вопросы по этому видео в той же ветке\n"
    "• Статус системы: `status`"
)

_BUSY_TEXT = "Ищу в архиве… / Searching the archive…"


# --- Configuration -------------------------------------------------------------
# Secrets are read per-request rather than at import time so a unit test (or a
# credential rotation followed by service restart) never depends on import
# order.


def _signing_secret() -> str:
    secret = os.getenv("SLACK_SIGNING_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="SLACK_SIGNING_SECRET is not configured")
    return secret


def _bot_token() -> str | None:
    return os.getenv("SLACK_BOT_TOKEN") or None


def _api_base() -> str:
    return os.getenv("RAINRAG_API_URL", "http://localhost:8001").rstrip("/")


def _auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    token = os.getenv("RAINRAG_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", **_auth_headers()}


def _default_top_k() -> int | None:
    raw = os.getenv("RAINRAG_SLACK_TOP_K", "")
    try:
        return int(raw) if raw.strip() else None
    except ValueError:
        logger.warning("Invalid RAINRAG_SLACK_TOP_K={!r}, using backend default", raw)
        return None


def _show_context() -> bool:
    return os.getenv("RAINRAG_SLACK_SHOW_CONTEXT", "true").lower() not in {"0", "false", "no"}


def _asset_base() -> str:
    """Public browser-facing base URL for media links, e.g. https://rag.tvrain.tv.

    Reuses the RAINRAG_ASSET_URL the web UI already documents. Empty means no
    media links in Slack messages -- the answer and sources still work.
    """
    return os.getenv("RAINRAG_ASSET_URL", "").rstrip("/")


# --- Request authenticity ------------------------------------------------------


def verify_slack_signature(body: bytes, timestamp: str | None, signature: str | None) -> bool:
    """Check Slack's v0 request signature over the raw body.

    Must run on the raw bytes before any parsing: the signature covers the
    body exactly as sent, and re-serialising parsed JSON would not round-trip.
    """
    if not timestamp or not signature:
        return False
    try:
        ts = int(float(timestamp))
    except (ValueError, OverflowError):
        return False
    if abs(time.time() - ts) > SIGNATURE_MAX_AGE_SECONDS:
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(_signing_secret().encode(), base, hashlib.sha256).hexdigest()
    try:
        return hmac.compare_digest(expected, signature)
    except TypeError:
        # A non-ASCII signature header is by definition forged, not a fault.
        return False


async def _verified_body(request: Request) -> bytes:
    body = await request.body()
    if not verify_slack_signature(
        body,
        request.headers.get("X-Slack-Request-Timestamp"),
        request.headers.get("X-Slack-Signature"),
    ):
        raise HTTPException(status_code=401, detail="Invalid Slack signature")
    return body


# --- Event de-duplication ------------------------------------------------------
# Slack retries any event not acked within 3 seconds and redelivers on network
# blips. Each retry carries the same event_id, so a small TTL'd in-memory set
# is enough to make processing effectively once-per-event for a single worker.

_SEEN_EVENTS_TTL_SECONDS = 900
_SEEN_EVENTS_MAX = 4096
_seen_events: dict[str, float] = {}
_seen_events_lock = threading.Lock()


def _is_duplicate_event(event_id: str) -> bool:
    """Record an event id; True when it was already seen recently."""
    now = time.time()
    with _seen_events_lock:
        for key, seen_at in list(_seen_events.items()):
            if now - seen_at > _SEEN_EVENTS_TTL_SECONDS:
                del _seen_events[key]
        if event_id in _seen_events:
            return True
        if len(_seen_events) >= _SEEN_EVENTS_MAX:
            oldest = min(_seen_events, key=_seen_events.__getitem__)
            del _seen_events[oldest]
        _seen_events[event_id] = now
        return False


# --- Video-session thread registry ----------------------------------------------
# The web UI's upload mode is a sidebar toggle; the Slack analogue is a thread.
# Importing a video binds its Slack thread to the backend session, and every
# later message in that thread is answered from that video's transcript alone.
# In-memory like the dedup store: sessions expire server-side anyway, and a
# connector restart merely means asking in a fresh thread.

_VIDEO_THREAD_TTL_SECONDS = 24 * 3600
_video_threads: dict[tuple[str, str], tuple[str, float]] = {}
_video_threads_lock = threading.Lock()


def bind_video_thread(channel: str, thread_root: str, session_id: str) -> None:
    with _video_threads_lock:
        _video_threads[(channel, thread_root)] = (session_id, time.time())


def video_session_for_thread(channel: str, thread_root: str) -> str | None:
    now = time.time()
    with _video_threads_lock:
        for key, (_, bound_at) in list(_video_threads.items()):
            if now - bound_at > _VIDEO_THREAD_TTL_SECONDS:
                del _video_threads[key]
        entry = _video_threads.get((channel, thread_root))
        return entry[0] if entry else None


def unbind_video_thread(channel: str, thread_root: str) -> None:
    with _video_threads_lock:
        _video_threads.pop((channel, thread_root), None)


# --- Message parsing -------------------------------------------------------------

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
# Slack wraps URLs in <...> and may append a |label; unwrap before parsing so
# `video: <https://youtu.be/x|clip>` yields the bare URL.
_SLACK_LINK_RE = re.compile(r"<(https?://[^|>]+)(?:\|[^>]*)?>")
_DATE_FILTER_RE = re.compile(r"\b(from|to):(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_TOP_K_RE = re.compile(r"\btop:(\d{1,3})\b", re.IGNORECASE)
_LANG_RE = re.compile(r"\blang:(ru|en)\b", re.IGNORECASE)
_NAME_PREFIX_RE = re.compile(r"^(?:name|название)\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_VIDEO_PREFIX_RE = re.compile(r"^(?:video|видео)\s*:\s*(\S+)", re.IGNORECASE)
_STATUS_RE = re.compile(r"^(?:status|статус)$", re.IGNORECASE)
_HELP_RE = re.compile(r"^(?:help|помощь|\?)$", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)


@dataclass
class ParsedMessage:
    """A Slack message reduced to an intent plus the web UI's sidebar knobs."""

    mode: str  # "ask" | "name" | "video" | "status" | "help"
    text: str = ""  # question, name query or video URL depending on mode
    language: str | None = None  # explicit lang: override, else auto-detected
    top_k: int | None = None
    date_from: str | None = None
    date_to: str | None = None


def parse_message(raw_text: str) -> ParsedMessage:
    """Turn raw Slack message text into a ParsedMessage.

    Strips bot mentions and Slack link markup, then reads the inline options
    (from:/to: dates, top:N, lang:xx) and the mode prefixes (name:, video:,
    status, help). Dates that only look like dates (from:2024-02-31) stay in
    the question text instead of being forwarded as filters the backend
    rejects.
    """
    text = _MENTION_RE.sub("", raw_text or "")
    text = _SLACK_LINK_RE.sub(r"\1", text)

    language: str | None = None
    top_k: int | None = None
    date_from: str | None = None
    date_to: str | None = None

    def _capture_lang(match: re.Match[str]) -> str:
        nonlocal language
        language = match.group(1).lower()
        return ""

    # lang: applies to every mode (a video import's progress messages need a
    # language too), so it is extracted before the mode prefixes.
    text = _LANG_RE.sub(_capture_lang, text)
    text = " ".join(text.split())

    if not text:
        return ParsedMessage(mode="help", language=language)
    if _HELP_RE.match(text):
        return ParsedMessage(mode="help", text=text, language=language)
    if _STATUS_RE.match(text):
        return ParsedMessage(mode="status", text=text, language=language)

    video_match = _VIDEO_PREFIX_RE.match(text)
    if video_match:
        return ParsedMessage(mode="video", text=video_match.group(1), language=language)

    name_match = _NAME_PREFIX_RE.match(text)
    if name_match:
        return ParsedMessage(
            mode="name", text=" ".join(name_match.group(1).split()), language=language
        )

    def _capture_top(match: re.Match[str]) -> str:
        nonlocal top_k
        # Same bounds the API enforces (ge=1, le=20); out-of-range values stay
        # in the text so the user sees why nothing changed.
        value = int(match.group(1))
        if 1 <= value <= 20:
            top_k = value
            return ""
        return match.group(0)

    def _capture_date(match: re.Match[str]) -> str:
        nonlocal date_from, date_to
        try:
            date.fromisoformat(match.group(2))
        except ValueError:
            # Shaped like a date but not one (from:2024-02-31): keep it in
            # the question rather than forward a filter the backend rejects.
            return match.group(0)
        if match.group(1).lower() == "from":
            date_from = match.group(2)
        else:
            date_to = match.group(2)
        return ""

    text = _TOP_K_RE.sub(_capture_top, text)
    text = _DATE_FILTER_RE.sub(_capture_date, text)
    question = " ".join(text.split())

    if not question:
        return ParsedMessage(mode="help")
    return ParsedMessage(
        mode="ask",
        text=question,
        language=language,
        top_k=top_k,
        date_from=date_from,
        date_to=date_to,
    )


def detect_language(text: str) -> str:
    """Pick the answer language from the question's script.

    The corpus and audience are bilingual; any Cyrillic means the asker reads
    Russian, which is also the backend default.
    """
    return "ru" if _CYRILLIC_RE.search(text) else "en"


def _resolve_language(parsed: ParsedMessage) -> str:
    return parsed.language or detect_language(parsed.text)


# --- Expiring media links --------------------------------------------------------
# Media URLs carry their credential in the query string, because <video>
# requests cannot send an Authorization header. Minting a short-lived signed
# token (same scheme as rainrag.api / app.py, duplicated for the same reason:
# importing the API module would pull in the whole query engine) means a link
# pasted into Slack stops working on its own instead of granting archive
# access forever.

MEDIA_TOKEN_TTL_SECONDS = int(os.getenv("RAINRAG_MEDIA_TOKEN_TTL_SECONDS", str(12 * 3600)))


def issue_media_token() -> str:
    """Mint a time-limited token for media URLs, or "" when auth is disabled."""
    secret = os.getenv("RAINRAG_AUTH_TOKEN")
    if not secret:
        return ""
    expires = int(time.time()) + MEDIA_TOKEN_TTL_SECONDS
    payload = str(expires)
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"v1.{payload}.{signature}"


def append_auth_query(url: str) -> str:
    """Append an expiring auth token query param when API auth is configured."""
    token = issue_media_token()
    if not token:
        return url
    parts = urlsplit(url)
    query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
    query_params.setdefault("auth", token)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query_params), parts.fragment)
    )


def public_media_url(path: str | None) -> str | None:
    """Turn an API-relative media path (/video/...#t=42) into a browser URL.

    Returns None when no public asset base is configured -- Slack messages
    then simply omit the media links, same as the web UI with video disabled.
    """
    base = _asset_base()
    if not path or not base:
        return None
    return append_auth_query(f"{base}{path}")


# --- RainRAG API client -----------------------------------------------------------


async def query_rainrag(
    question: str,
    language: str,
    top_k: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Run one question through the RainRAG API and return its JSON response."""
    payload: dict[str, Any] = {"question": question, "language": language}
    if top_k is None:
        top_k = _default_top_k()
    if top_k is not None:
        payload["top_k"] = top_k
    if date_from:
        payload["date_from"] = date_from
    if date_to:
        payload["date_to"] = date_to

    async with httpx.AsyncClient(timeout=QUERY_HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(f"{_api_base()}/query", json=payload, headers=_api_headers())
        response.raise_for_status()
        return response.json()


async def query_video_session(
    session_id: str, question: str, language: str, top_k: int | None = None
) -> dict[str, Any]:
    """Ask a question scoped to one uploaded video's transcript."""
    payload: dict[str, Any] = {"question": question, "language": language}
    if top_k is not None:
        payload["top_k"] = top_k
    async with httpx.AsyncClient(timeout=QUERY_HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{_api_base()}/video-sessions/{session_id}/query",
            json=payload,
            headers=_api_headers(),
        )
        response.raise_for_status()
        return response.json()


async def search_by_name_api(query: str) -> dict[str, Any]:
    """Search videos by title via /search-by-name."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{_api_base()}/search-by-name",
            params={"q": query, "limit": MAX_NAME_RESULTS},
            headers=_auth_headers(),
        )
        response.raise_for_status()
        return response.json()


async def related_chunks_api(chunk_id: str) -> list[dict[str, Any]]:
    """Fetch chunks related to one retrieved chunk via /related-chunks."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{_api_base()}/related-chunks",
            json={"chunk_id": chunk_id, "top_k": RELATED_TOP_K, "same_video_only": False},
            headers=_api_headers(),
        )
        response.raise_for_status()
        return response.json().get("related_chunks", [])


async def api_health() -> dict[str, Any] | None:
    """Fetch /health, or None when the API is unreachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_api_base()}/health", headers=_auth_headers())
            return response.json() if response.status_code == 200 else None
    except Exception as exc:
        logger.error("Health check failed: {}", exc)
        return None


async def create_video_session_from_url(url: str) -> dict[str, Any]:
    """Import a video by URL; returns the new session dict."""
    async with httpx.AsyncClient(timeout=IMPORT_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{_api_base()}/video-sessions/from-url",
            json={"url": url},
            headers=_api_headers(),
        )
        response.raise_for_status()
        return response.json()


async def create_video_session_from_file(
    file_path: Path, filename: str, content_type: str
) -> dict[str, Any]:
    """Upload a video file; returns the new session dict."""
    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT_SECONDS) as client:
        with open(file_path, "rb") as fh:
            response = await client.post(
                f"{_api_base()}/video-sessions",
                files={"file": (filename, fh, content_type)},
                headers=_auth_headers(),
            )
        response.raise_for_status()
        return response.json()


async def get_video_session(session_id: str) -> dict[str, Any]:
    """Fetch one upload session's current status."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{_api_base()}/video-sessions/{session_id}", headers=_auth_headers()
        )
        response.raise_for_status()
        return response.json()


def _timecode_to_seconds(timecode: str | None) -> float | None:
    """Convert HH:MM:SS (or MM:SS) to seconds; None when unparsable."""
    if not timecode:
        return None
    parts = timecode.split(":")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if len(values) == 3:
        return values[0] * 3600 + values[1] * 60 + values[2]
    if len(values) == 2:
        return values[0] * 60 + values[1]
    return None


def _clip_request_for(doc: dict[str, Any]) -> tuple[str, float, float] | None:
    """Derive (relative video path, start, end) for a chunk's clip, or None.

    The chunk's video_url is an API-relative path like /video/ab/cd.mp4#t=600;
    the fragment is for browsers and is dropped here.
    """
    video_url = doc.get("video_url") or ""
    if not video_url.startswith("/video/"):
        return None
    rel = video_url[len("/video/") :].split("#", 1)[0]
    start = _timecode_to_seconds(doc.get("start_time"))
    end = _timecode_to_seconds(doc.get("end_time"))
    if rel and start is not None and end is not None and end > start:
        return rel, start, end
    return None


async def fetch_video_clip(rel_path: str, start: float, end: float) -> bytes | None:
    """Fetch one chunk's footage from /video-clip; None on any failure.

    A clip is a bonus on top of the answer, so failures are logged and
    swallowed rather than surfaced -- the message already carries the links.
    """
    max_bytes = CLIP_MAX_MB * 1024 * 1024
    try:
        async with httpx.AsyncClient(timeout=CLIP_FETCH_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "GET",
                f"{_api_base()}/video-clip/{rel_path}",
                params={"start": start, "end": end},
                headers=_auth_headers(),
            ) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                received = 0
                async for chunk in response.aiter_bytes(1024 * 1024):
                    received += len(chunk)
                    if received > max_bytes:
                        logger.info("Clip for {} exceeds {} MB; skipping", rel_path, CLIP_MAX_MB)
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
    except Exception as exc:
        logger.warning("Clip fetch failed for {}: {}", rel_path, exc)
        return None


async def upload_clip_to_slack(
    channel: str, thread_ts: str | None, filename: str, data: bytes, title: str
) -> bool:
    """Share a video clip into a channel/thread via Slack's external upload flow.

    Three steps (files.getUploadURLExternal → raw POST → completeUploadExternal)
    because the single-call files.upload API was retired. Needs files:write.
    """
    token = _bot_token()
    if not token:
        return False
    auth = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient(timeout=CLIP_FETCH_TIMEOUT_SECONDS) as client:
            ticket = (
                await client.post(
                    f"{SLACK_API_BASE}/files.getUploadURLExternal",
                    data={"filename": filename, "length": str(len(data))},
                    headers=auth,
                )
            ).json()
            if not ticket.get("ok"):
                logger.error("getUploadURLExternal rejected: {}", ticket.get("error"))
                return False

            upload = await client.post(ticket["upload_url"], content=data)
            if upload.status_code != 200:
                logger.error("Clip byte upload failed with HTTP {}", upload.status_code)
                return False

            complete_payload: dict[str, Any] = {
                "files": [{"id": ticket["file_id"], "title": title}],
                "channel_id": channel,
            }
            if thread_ts:
                complete_payload["thread_ts"] = thread_ts
            done = (
                await client.post(
                    f"{SLACK_API_BASE}/files.completeUploadExternal",
                    json=complete_payload,
                    headers=auth,
                )
            ).json()
            if not done.get("ok"):
                logger.error("completeUploadExternal rejected: {}", done.get("error"))
                return False
            return True
    except Exception as exc:
        logger.error("Clip upload to Slack failed: {}", exc)
        return False


async def _post_top_clips(
    context: list[dict[str, Any]], channel: str, thread_ts: str | None
) -> None:
    """Post the top chunks' actual footage into the thread as inline clips."""
    posted = 0
    for doc in context:
        if posted >= CLIP_CHUNKS:
            break
        clip_request = _clip_request_for(doc)
        if clip_request is None:
            continue
        rel_path, start, end = clip_request
        data = await fetch_video_clip(rel_path, start, end)
        if data is None:
            continue
        title = _chunk_title(doc)
        span = f"{doc.get('start_time')}–{doc.get('end_time')}"
        if await upload_clip_to_slack(
            channel, thread_ts, f"{Path(rel_path).stem}_{int(start)}.mp4", data, f"{title} · {span}"
        ):
            posted += 1


# --- Formatting --------------------------------------------------------------------


def _escape_mrkdwn(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_MD_BOLD_RE = re.compile(r"(?<!\*)\*\*(?!\s)(.+?)(?<!\s)\*\*(?!\*)", re.DOTALL)
_MD_BOLD_UNDERSCORE_RE = re.compile(r"(?<!_)__(?!\s)(.+?)(?<!\s)__(?!_)", re.DOTALL)
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")


def markdown_to_mrkdwn(text: str) -> str:
    """Convert the markdown LLMs habitually emit into Slack mrkdwn.

    LLM answers arrive as standard markdown, but Slack renders mrkdwn:
    ``**bold**`` shows its literal asterisks, headings show their hashes.
    Escaping runs first, so transcript text quoted in an answer cannot smuggle
    in a ``<!channel>`` ping or a fake link; the conversions below then insert
    the only angle brackets in the string. Deliberately minimal -- bold,
    headings and links cover what the answer prompts actually produce, and a
    missed construct degrades to visible punctuation, never to broken text.
    """
    text = _escape_mrkdwn(text)
    text = _MD_LINK_RE.sub(r"<\2|\1>", text)
    text = _MD_HEADING_RE.sub(r"*\1*", text)
    text = _MD_BOLD_RE.sub(r"*\1*", text)
    text = _MD_BOLD_UNDERSCORE_RE.sub(r"*\1*", text)
    return text


def _split_for_blocks(text: str, limit: int = SLACK_BLOCK_TEXT_LIMIT) -> list[str]:
    """Split text into block-sized pieces, preferring paragraph then word breaks."""
    pieces: list[str] = []
    remaining = text.strip()
    while remaining and len(pieces) < MAX_ANSWER_BLOCKS:
        if len(remaining) <= limit:
            pieces.append(remaining)
            remaining = ""
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        pieces.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        # Ran out of blocks with text left over: truncation is now the only
        # option, but make it visible.
        pieces[-1] = pieces[-1][: limit - 2].rstrip() + " …"
    return pieces


def _chunk_title(doc: dict[str, Any]) -> str:
    return doc.get("web_title") or doc.get("filename") or "—"


def _chunk_date(doc: dict[str, Any]) -> str | None:
    return doc.get("web_date") or doc.get("date")


def _source_lines(context: list[dict[str, Any]]) -> list[str]:
    """Render the top retrieved chunks as source attributions.

    Chunks are deduplicated by group_id so two hits in the same broadcast show
    as one source. Only public web URLs are linked here; the media links with
    expiring tokens live in the detailed context message.
    """
    lines: list[str] = []
    seen_groups: set[str] = set()
    for doc in context:
        group = doc.get("group_id") or doc.get("filename") or ""
        if group in seen_groups:
            continue
        seen_groups.add(group)

        title = _chunk_title(doc)
        chunk_date = _chunk_date(doc)
        url = doc.get("web_url")

        label = title if not chunk_date else f"{title} ({chunk_date})"
        label = _escape_mrkdwn(label)
        lines.append(f"• <{url}|{label}>" if url else f"• {label}")
        if len(lines) >= MAX_SOURCES:
            break
    return lines


def format_answer(result: dict[str, Any], language: str) -> tuple[str, list[dict[str, Any]]]:
    """Build (fallback text, Block Kit blocks) for a completed query."""
    answer = (result.get("answer") or "").strip() or (
        "Не нашёл ответа в архиве." if language == "ru" else "No answer found in the archive."
    )
    answer = markdown_to_mrkdwn(answer)
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": piece}}
        for piece in _split_for_blocks(answer)
    ]

    sources = _source_lines(result.get("context") or [])
    if sources:
        heading = "Источники" if language == "ru" else "Sources"
        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"*{heading}:*\n" + "\n".join(sources)}],
            }
        )

    # The plain-text fallback shows in notifications and clients without Block
    # Kit; Slack truncates it itself, so only the answer needs to be there.
    return answer[:SLACK_BLOCK_TEXT_LIMIT], blocks


def format_context_blocks(
    context: list[dict[str, Any]], language: str
) -> tuple[str, list[dict[str, Any]]] | None:
    """Build the transcript-excerpts message: the web UI's context panel.

    Each retrieved chunk gets its excerpt, date, timecodes and score, plus
    expiring media links when a public asset base is configured, and a "find
    related" button carrying the chunk's doc_id for the interactivity handler.
    Returns None when there is nothing to show.
    """
    chunks = (context or [])[:MAX_CONTEXT_CHUNKS]
    if not chunks:
        return None

    heading = "Контекст" if language == "ru" else "Context"
    video_label = "▶ Видео" if language == "ru" else "▶ Video"
    page_label = "Страница" if language == "ru" else "Page"
    related_label = "Похожее" if language == "ru" else "Related"

    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{heading}:*"}}
    ]
    buttons: list[dict[str, Any]] = []

    for i, doc in enumerate(chunks, start=1):
        title = _escape_mrkdwn(_chunk_title(doc))
        meta_bits: list[str] = []
        chunk_date = _chunk_date(doc)
        if chunk_date:
            meta_bits.append(str(chunk_date))
        start, end = doc.get("start_time"), doc.get("end_time")
        if start and end:
            meta_bits.append(f"{start}–{end}")
        score = doc.get("rerank_score") or doc.get("score")
        if isinstance(score, int | float):
            meta_bits.append(f"score {score:.2f}")
        meta = " · ".join(meta_bits)

        excerpt = " ".join((doc.get("text") or "").split())
        if len(excerpt) > CONTEXT_EXCERPT_CHARS:
            excerpt = excerpt[: CONTEXT_EXCERPT_CHARS - 2].rstrip() + " …"
        excerpt = _escape_mrkdwn(excerpt)

        links: list[str] = []
        video_link = public_media_url(doc.get("video_url"))
        if video_link:
            links.append(f"<{video_link}|{video_label}>")
        vtt_link = public_media_url(doc.get("vtt_url"))
        if vtt_link:
            links.append(f"<{vtt_link}|VTT>")
        if doc.get("web_url"):
            links.append(f"<{doc['web_url']}|{page_label}>")

        lines = [f"*{i}. {title}*" + (f"  _{meta}_" if meta else "")]
        if excerpt:
            lines.append(f"> {excerpt}")
        if links:
            lines.append(" · ".join(links))
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}})

        doc_id = doc.get("doc_id")
        if doc_id:
            buttons.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": f"{related_label} {i}"},
                    "action_id": f"related_{i}",
                    "value": doc_id,
                }
            )

    if buttons:
        blocks.append({"type": "actions", "elements": buttons})

    fallback = f"{heading}: {len(chunks)}"
    return fallback, blocks


def format_related_blocks(
    chunks: list[dict[str, Any]], language: str
) -> tuple[str, list[dict[str, Any]]]:
    """Format /related-chunks results, reusing the context layout sans buttons."""
    heading = "Похожий контент" if language == "ru" else "Related content"
    formatted = format_context_blocks(chunks, language)
    if formatted is None:
        text = "Похожих фрагментов не найдено" if language == "ru" else "No related chunks found"
        return text, [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]
    _, blocks = formatted
    blocks[0] = {"type": "section", "text": {"type": "mrkdwn", "text": f"*{heading}:*"}}
    # Related results are a leaf view in the web UI too: no further buttons.
    blocks = [b for b in blocks if b.get("type") != "actions"]
    return heading, blocks


def format_name_results(result: dict[str, Any], language: str) -> tuple[str, list[dict[str, Any]]]:
    """Format /search-by-name results as a linked list."""
    results = result.get("results") or []
    if not results:
        text = (
            "Ничего не найдено по названию." if language == "ru" else "No videos match that name."
        )
        return text, [{"type": "section", "text": {"type": "mrkdwn", "text": text}}]

    heading = "Найдено по названию" if language == "ru" else "Found by name"
    lines: list[str] = []
    for item in results[:MAX_NAME_RESULTS]:
        name = _escape_mrkdwn(item.get("name") or item.get("video_hash") or "—")
        item_date = item.get("date")
        label = name if not item_date else f"{name} ({item_date})"
        line = f"• <{item['web_url']}|{label}>" if item.get("web_url") else f"• {label}"
        show = item.get("teleshow_name")
        if show:
            line += f" — {_escape_mrkdwn(show)}"
        media_bits: list[str] = []
        for lang_code, media in (item.get("languages") or {}).items():
            video_link = public_media_url((media or {}).get("video_url"))
            if video_link:
                media_bits.append(f"<{video_link}|▶ {lang_code.upper()}>")
        if media_bits:
            line += "  " + " · ".join(media_bits)
        lines.append(line)

    text = "\n".join(lines)
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"*{heading}:*\n{text}"}}]
    return heading, blocks


def format_status(health: dict[str, Any] | None, language: str) -> str:
    """Render /health as the short status summary the web UI sidebar shows."""
    if not health:
        return "API недоступен." if language == "ru" else "The API is unreachable."
    if language == "ru":
        labels = ("Статус", "LLM", "Эмбеддинги", "Коллекция", "Гибридный поиск", "Реранкер")
        on, off = "вкл", "выкл"
    else:
        labels = ("Status", "LLM", "Embeddings", "Collection", "Hybrid search", "Reranker")
        on, off = "on", "off"
    return (
        f"{labels[0]}: {health.get('status', '?')}\n"
        f"{labels[1]}: {health.get('llm_provider', '?')} / {health.get('llm_model', '?')}\n"
        f"{labels[2]}: {health.get('embedding_provider', '?')} / {health.get('embedding_model', '?')}\n"
        f"{labels[3]}: {health.get('qdrant_collection', '?')}\n"
        f"{labels[4]}: {on if health.get('hybrid_search_enabled') else off} · "
        f"{labels[5]}: {on if health.get('reranker_enabled') else off}"
    )


def _error_text(language: str) -> str:
    if language == "ru":
        return "Не получилось выполнить поиск — попробуйте ещё раз чуть позже."
    return "The search failed — please try again in a moment."


def _import_error_text(status_code: int, language: str) -> str:
    """Map video-import HTTP failures to the same advice the web UI gives."""
    ru = {
        413: "Видео слишком большое для загрузки.",
        422: "По этой ссылке не нашлось видео — проверьте её.",
        451: "Это видео недоступно из региона сервера.",
        503: "Платформа отклонила скачивание — попробуйте позже.",
    }
    en = {
        413: "The video is too large to import.",
        422: "No downloadable video found at that link — please check it.",
        451: "This video is not available in the server's region.",
        503: "The video platform refused the download — try again later.",
    }
    table = ru if language == "ru" else en
    return table.get(
        status_code,
        "Не получилось импортировать видео." if language == "ru" else "Video import failed.",
    )


# --- Slack delivery ------------------------------------------------------------


async def post_slack_message(
    channel: str,
    text: str,
    blocks: list[dict[str, Any]] | None = None,
    thread_ts: str | None = None,
) -> bool:
    """Post a message via chat.postMessage; returns False on any failure.

    Slack reports API errors inside a 200 response ({"ok": false, ...}), so
    the status code alone proves nothing.
    """
    token = _bot_token()
    if not token:
        logger.error("SLACK_BOT_TOKEN is not configured; dropping reply to {}", channel)
        return False

    payload: dict[str, Any] = {"channel": channel, "text": text}
    if blocks:
        payload["blocks"] = blocks
    if thread_ts:
        payload["thread_ts"] = thread_ts

    try:
        async with httpx.AsyncClient(timeout=SLACK_POST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{SLACK_API_BASE}/chat.postMessage",
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            )
        data = response.json()
    except Exception as exc:
        logger.error("chat.postMessage failed for channel {}: {}", channel, exc)
        return False

    if not data.get("ok"):
        logger.error("chat.postMessage rejected for channel {}: {}", channel, data.get("error"))
        return False
    return True


async def post_response_url(
    response_url: str,
    text: str,
    blocks: list[dict[str, Any]] | None = None,
    response_type: str = "ephemeral",
) -> bool:
    """Deliver a slash-command result via its response_url."""
    payload: dict[str, Any] = {"response_type": response_type, "text": text}
    if blocks:
        payload["blocks"] = blocks
    try:
        async with httpx.AsyncClient(timeout=SLACK_POST_TIMEOUT_SECONDS) as client:
            response = await client.post(response_url, json=payload)
        return response.status_code == 200
    except Exception as exc:
        logger.error("response_url delivery failed: {}", exc)
        return False


# --- Question processing (background) -------------------------------------------

SendFn = Callable[[str, "list[dict[str, Any]] | None"], Awaitable[None]]


def _log_usage(event: str, outcome: str, started: float, **fields: Any) -> None:
    """One flat "[usage]" line per attempt, matching the backend's format so
    scripts/usage_report.py counts Slack traffic alongside web traffic."""
    rendered = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None and v != "")
    logger.info(
        "[usage] event={} outcome={} seconds={:.1f} {}",
        event,
        outcome,
        time.monotonic() - started,
        rendered,
    )


async def _handle_ask(
    parsed: ParsedMessage,
    send: SendFn,
    channel: str | None = None,
    thread_root: str | None = None,
) -> None:
    """Archive Q&A: answer, transcript-context message, then inline clips.

    Clips need a channel to upload into, so they only happen on the events
    path (mentions/DMs); slash commands answer ephemerally, where Slack has
    nowhere to attach a file.
    """
    language = _resolve_language(parsed)
    started = time.monotonic()
    outcome = "ok"
    docs = 0
    try:
        result = await query_rainrag(
            parsed.text, language, parsed.top_k, parsed.date_from, parsed.date_to
        )
        docs = int(result.get("num_documents") or 0)
        text, blocks = format_answer(result, language)
        await send(text, blocks)
        context = result.get("context") or []
        if _show_context():
            formatted = format_context_blocks(context, language)
            if formatted:
                await send(*formatted)
        if channel and CLIP_CHUNKS > 0:
            await _post_top_clips(context, channel, thread_root)
    except httpx.HTTPStatusError as exc:
        outcome = f"http_{exc.response.status_code}"
        logger.error("RainRAG API returned {} for Slack query", exc.response.status_code)
        await send(_error_text(language), None)
    except Exception:
        outcome = "error"
        logger.exception("Slack query processing failed")
        await send(_error_text(language), None)
    finally:
        _log_usage("slack_query", outcome, started, mode="corpus", lang=language, docs=docs)


async def _handle_session_ask(
    session_id: str, parsed: ParsedMessage, send: SendFn, channel: str, thread_root: str
) -> None:
    """Q&A scoped to an uploaded video's transcript (the web UI's video mode)."""
    language = _resolve_language(parsed)
    started = time.monotonic()
    outcome = "ok"
    try:
        result = await query_video_session(session_id, parsed.text, language, parsed.top_k)
        text, blocks = format_answer(result, language)
        await send(text, blocks)
        if _show_context():
            formatted = format_context_blocks(result.get("context") or [], language)
            if formatted:
                await send(*formatted)
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        outcome = f"http_{code}"
        if code == 409:
            await send(
                "Видео ещё обрабатывается — попробуйте чуть позже."
                if language == "ru"
                else "The video is still processing — try again shortly.",
                None,
            )
        elif code == 404:
            unbind_video_thread(channel, thread_root)
            await send(
                "Эта видео-сессия истекла. Импортируйте видео заново."
                if language == "ru"
                else "This video session has expired. Import the video again.",
                None,
            )
        else:
            await send(_error_text(language), None)
    except Exception:
        outcome = "error"
        logger.exception("Slack video-session query failed")
        await send(_error_text(language), None)
    finally:
        _log_usage("slack_query", outcome, started, mode="session", lang=language)


async def _handle_name(parsed: ParsedMessage, send: SendFn) -> None:
    """Search-by-name mode."""
    language = _resolve_language(parsed)
    started = time.monotonic()
    outcome = "ok"
    try:
        result = await search_by_name_api(parsed.text)
        await send(*format_name_results(result, language))
    except Exception:
        outcome = "error"
        logger.exception("Slack name search failed")
        await send(_error_text(language), None)
    finally:
        _log_usage("slack_name_search", outcome, started, lang=language)


async def _handle_status(send: SendFn, language: str) -> None:
    health = await api_health()
    await send(format_status(health, language), None)


async def _watch_video_session(
    session_id: str, channel: str, thread_root: str, language: str
) -> None:
    """Poll an upload session until it is ready and announce the outcome.

    Polling rather than pushing: the API has no callback channel, and the web
    UI observes sessions exactly the same way.
    """
    deadline = time.monotonic() + SESSION_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            session = await get_video_session(session_id)
        except Exception:
            logger.exception("Polling video session {} failed", session_id)
            await asyncio.sleep(SESSION_POLL_SECONDS)
            continue
        status = session.get("status")
        if status == "ready":
            await post_slack_message(
                channel,
                "Видео готово! Задавайте вопросы по нему в этой ветке."
                if language == "ru"
                else "The video is ready! Ask questions about it in this thread.",
                thread_ts=thread_root,
            )
            return
        if status == "error":
            unbind_video_thread(channel, thread_root)
            detail = session.get("error") or ""
            await post_slack_message(
                channel,
                (
                    "Не получилось обработать видео."
                    if language == "ru"
                    else "Video processing failed."
                )
                + (f" ({detail})" if detail else ""),
                thread_ts=thread_root,
            )
            return
        await asyncio.sleep(SESSION_POLL_SECONDS)

    unbind_video_thread(channel, thread_root)
    await post_slack_message(
        channel,
        "Обработка видео заняла слишком много времени — попробуйте ещё раз."
        if language == "ru"
        else "Video processing took too long — please try again.",
        thread_ts=thread_root,
    )


async def _handle_video_url(url: str, channel: str, thread_root: str, language: str) -> None:
    """Import a video by URL and bind its thread for scoped Q&A."""
    started = time.monotonic()
    outcome = "ok"
    await post_slack_message(
        channel,
        "Импортирую видео — это может занять несколько минут…"
        if language == "ru"
        else "Importing the video — this can take a few minutes…",
        thread_ts=thread_root,
    )
    try:
        session = await create_video_session_from_url(url)
    except httpx.HTTPStatusError as exc:
        outcome = f"http_{exc.response.status_code}"
        await post_slack_message(
            channel, _import_error_text(exc.response.status_code, language), thread_ts=thread_root
        )
        return
    except Exception:
        outcome = "error"
        logger.exception("Slack video import failed")
        await post_slack_message(channel, _error_text(language), thread_ts=thread_root)
        return
    finally:
        _log_usage("slack_video_import", outcome, started, via="url")

    bind_video_thread(channel, thread_root, session["id"])
    await _watch_video_session(session["id"], channel, thread_root, language)


def _is_slack_host(url: str) -> bool:
    """True when the URL points at Slack itself (slack.com or a subdomain)."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "slack.com" or host.endswith(".slack.com")


def _is_video_file(file_info: dict[str, Any]) -> bool:
    if (file_info.get("mimetype") or "").startswith("video/"):
        return True
    name = (file_info.get("name") or "").lower()
    return name.endswith(_VIDEO_EXTENSIONS)


async def _handle_video_file(
    file_info: dict[str, Any], channel: str, thread_root: str, language: str
) -> None:
    """Import an attached video file and bind its thread for scoped Q&A.

    Slack private file URLs need the bot token; the file is streamed to a temp
    path (the size cap enforced during download, not after) and re-uploaded to
    the API, which applies its own limit as well.
    """
    started = time.monotonic()
    outcome = "ok"
    token = _bot_token()
    download_url = file_info.get("url_private_download") or file_info.get("url_private")
    if not token or not download_url:
        await post_slack_message(channel, _error_text(language), thread_ts=thread_root)
        return
    if not _is_slack_host(download_url):
        # Defense in depth: events are already signature-verified, but the bot
        # token goes into this request's Authorization header, so never send
        # it anywhere but Slack's own file hosts.
        logger.warning(
            "Refusing file download from non-Slack host: {}",
            (urlsplit(download_url).hostname or "?"),
        )
        await post_slack_message(channel, _error_text(language), thread_ts=thread_root)
        return

    size = int(file_info.get("size") or 0)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if size > max_bytes:
        await post_slack_message(
            channel,
            f"Файл больше лимита в {MAX_UPLOAD_MB} МБ."
            if language == "ru"
            else f"The file exceeds the {MAX_UPLOAD_MB} MB limit.",
            thread_ts=thread_root,
        )
        return

    await post_slack_message(
        channel,
        "Загружаю видео — это может занять несколько минут…"
        if language == "ru"
        else "Uploading the video — this can take a few minutes…",
        thread_ts=thread_root,
    )

    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(suffix=".slackupload")
        tmp_path = Path(tmp_name)
        written = 0
        # fdopen wraps the descriptor before any network I/O: a request that
        # fails early (expired URL, connect timeout) must not leak the fd.
        with os.fdopen(fd, "wb") as out:
            async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "GET", download_url, headers={"Authorization": f"Bearer {token}"}
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError("file exceeds size limit")
                        out.write(chunk)

        session = await create_video_session_from_file(
            tmp_path,
            file_info.get("name") or "upload.mp4",
            file_info.get("mimetype") or "video/mp4",
        )
    except ValueError:
        outcome = "too_large"
        await post_slack_message(
            channel,
            f"Файл больше лимита в {MAX_UPLOAD_MB} МБ."
            if language == "ru"
            else f"The file exceeds the {MAX_UPLOAD_MB} MB limit.",
            thread_ts=thread_root,
        )
        return
    except httpx.HTTPStatusError as exc:
        outcome = f"http_{exc.response.status_code}"
        await post_slack_message(
            channel, _import_error_text(exc.response.status_code, language), thread_ts=thread_root
        )
        return
    except Exception:
        outcome = "error"
        logger.exception("Slack file upload failed")
        await post_slack_message(channel, _error_text(language), thread_ts=thread_root)
        return
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        _log_usage("slack_video_import", outcome, started, via="file")

    bind_video_thread(channel, thread_root, session["id"])
    await _watch_video_session(session["id"], channel, thread_root, language)


async def process_related_request(
    doc_id: str,
    channel: str,
    thread_ts: str | None,
    response_url: str | None,
    ephemeral: bool = False,
) -> None:
    """Handle a "find related" button click: fetch and post related chunks.

    A click on an ephemeral message (slash-command answers) must stay
    ephemeral -- replying in_channel would publish content the asker chose to
    keep to themselves.
    """
    started = time.monotonic()
    outcome = "ok"
    language = "ru"  # the button lives on a message we authored in this language
    try:
        chunks = await related_chunks_api(doc_id)
        # Excerpt text tells us the real language better than a guess.
        if chunks and detect_language(chunks[0].get("text") or "") == "en":
            language = "en"
        text, blocks = format_related_blocks(chunks, language)
    except Exception:
        outcome = "error"
        logger.exception("Related-chunks lookup failed")
        text, blocks = _error_text(language), None
    finally:
        _log_usage("slack_related", outcome, started)

    if ephemeral and response_url:
        await post_response_url(response_url, text, blocks, response_type="ephemeral")
    elif channel:
        await post_slack_message(channel, text, blocks, thread_ts)
    elif response_url:
        await post_response_url(response_url, text, blocks, response_type="ephemeral")


async def _dispatch(
    parsed: ParsedMessage,
    send: SendFn,
    *,
    channel: str | None = None,
    thread_root: str | None = None,
    reply_thread: str | None = None,
) -> None:
    """Route a parsed message to its handler. Video import needs a thread to
    bind, so it is only reachable from events (mentions/DMs), not slash
    commands -- the slash path passes channel=None and gets a hint instead."""
    if parsed.mode == "help":
        await send(_HELP_TEXT, None)
    elif parsed.mode == "status":
        await _handle_status(send, _resolve_language(parsed))
    elif parsed.mode == "name":
        await _handle_name(parsed, send)
    elif parsed.mode == "video":
        if channel and thread_root:
            # A bare URL says nothing about the asker's language; Russian is
            # the newsroom default unless lang: says otherwise.
            await _handle_video_url(parsed.text, channel, thread_root, parsed.language or "ru")
        else:
            await send(
                "Импорт видео работает через упоминание или личное сообщение боту — "
                "так я смогу отвечать в ветке. / Video import works via a mention or DM "
                "so I can answer in a thread.",
                None,
            )
    else:
        # Clips follow the same threading as the text replies (reply_thread is
        # None for top-level DM answers), not the video-binding root.
        await _handle_ask(parsed, send, channel=channel, thread_root=reply_thread)


async def process_event_question(event: dict[str, Any]) -> None:
    """Answer an app_mention or DM message event in its channel/thread."""
    channel = event.get("channel", "")
    is_dm = event.get("channel_type") == "im"
    # Replying in the existing thread (or starting one on the asking message
    # in channels) keeps long answers from burying the channel. DMs have no
    # threads worth forcing, so top-level replies read naturally there --
    # except for video sessions, which are thread-scoped by design.
    thread_root = event.get("thread_ts") or event.get("ts") or ""
    reply_thread = event.get("thread_ts") if is_dm else thread_root

    async def send(text: str, blocks: list[dict[str, Any]] | None) -> None:
        await post_slack_message(channel, text, blocks, reply_thread)

    parsed = parse_message(event.get("text", ""))

    # An attached video starts an upload session regardless of the text.
    video_files = [f for f in (event.get("files") or []) if _is_video_file(f)]
    if video_files:
        text = (event.get("text") or "").strip()
        language = parsed.language or (detect_language(text) if text else "ru")
        await _handle_video_file(video_files[0], channel, thread_root, language)
        return

    # A message inside a thread bound to a video session queries that video.
    # `send` already replies into that thread: thread_root is the thread's
    # root whenever the event carries thread_ts.
    if event.get("thread_ts"):
        session_id = video_session_for_thread(channel, event["thread_ts"])
        if session_id and parsed.mode == "ask":
            await _handle_session_ask(session_id, parsed, send, channel, event["thread_ts"])
            return

    await _dispatch(
        parsed, send, channel=channel, thread_root=thread_root, reply_thread=reply_thread
    )


async def process_command_question(text: str, response_url: str) -> None:
    """Answer a /rainrag slash command via its response_url."""
    response_type = os.getenv("RAINRAG_SLACK_COMMAND_RESPONSE", "ephemeral")
    if response_type not in {"ephemeral", "in_channel"}:
        response_type = "ephemeral"

    async def send(answer: str, blocks: list[dict[str, Any]] | None) -> None:
        await post_response_url(response_url, answer, blocks, response_type)

    await _dispatch(parse_message(text), send)


# --- FastAPI app -----------------------------------------------------------------

app = FastAPI(
    title="RainRAG Slack Connector",
    description="Slack front-end for the RainRAG query API",
    version="0.2.0",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness plus configuration sanity, without leaking secret values."""
    return {
        "status": "ok",
        "signing_secret_configured": bool(os.getenv("SLACK_SIGNING_SECRET")),
        "bot_token_configured": bool(_bot_token()),
        "api_url": _api_base(),
        "asset_url_configured": bool(_asset_base()),
    }


@app.post("/slack/events")
async def slack_events(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Slack Events API webhook: URL verification, mentions and DMs."""
    body = await _verified_body(request)

    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if payload.get("type") == "url_verification":
        return JSONResponse({"challenge": payload.get("challenge", "")})

    if payload.get("type") != "event_callback":
        return JSONResponse({"ok": True})

    event = payload.get("event") or {}
    event_id = payload.get("event_id") or ""

    # A retry means our first ack was slow, not that Slack wants the answer
    # twice; the event_id check below also covers redeliveries without the
    # retry header.
    if request.headers.get("X-Slack-Retry-Num"):
        return JSONResponse({"ok": True})
    if event_id and _is_duplicate_event(event_id):
        return JSONResponse({"ok": True})

    # Never answer bots (including ourselves: our own posts come back as
    # message events, and answering them would loop forever). Of the message
    # subtypes, only file_share is a user action we handle (video upload);
    # edits, joins and the rest are noise.
    if event.get("bot_id"):
        return JSONResponse({"ok": True})
    if event.get("subtype") and event.get("subtype") != "file_share":
        return JSONResponse({"ok": True})

    event_type = event.get("type")
    is_mention = event_type == "app_mention"
    is_dm = event_type == "message" and event.get("channel_type") == "im"
    if not (is_mention or is_dm):
        return JSONResponse({"ok": True})

    background_tasks.add_task(process_event_question, event)
    return JSONResponse({"ok": True})


@app.post("/slack/commands")
async def slack_commands(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Slash command webhook (/rainrag <question>)."""
    body = await _verified_body(request)

    form = {k: v[0] for k, v in parse_qs(body.decode("utf-8", errors="replace")).items()}
    text = form.get("text", "").strip()
    response_url = form.get("response_url", "")

    if not text:
        return JSONResponse({"response_type": "ephemeral", "text": _HELP_TEXT})
    if not response_url:
        raise HTTPException(status_code=400, detail="Missing response_url")

    background_tasks.add_task(process_command_question, text, response_url)
    # The immediate ack doubles as the "working on it" message; the real
    # answer follows via response_url when the query completes.
    return JSONResponse({"response_type": "ephemeral", "text": _BUSY_TEXT})


@app.post("/slack/interactive")
async def slack_interactive(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    """Interactivity webhook: the "find related" buttons on context messages."""
    body = await _verified_body(request)

    form = {k: v[0] for k, v in parse_qs(body.decode("utf-8", errors="replace")).items()}
    try:
        payload = json.loads(form.get("payload", "{}"))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")

    if payload.get("type") != "block_actions":
        return JSONResponse({})

    actions = payload.get("actions") or []
    action = actions[0] if actions else {}
    if not str(action.get("action_id", "")).startswith("related_"):
        return JSONResponse({})

    doc_id = action.get("value") or ""
    if not doc_id:
        return JSONResponse({})

    channel = (payload.get("channel") or {}).get("id") or ""
    container = payload.get("container") or {}
    thread_ts = container.get("thread_ts") or container.get("message_ts")
    response_url = payload.get("response_url")
    ephemeral = bool(container.get("is_ephemeral"))

    background_tasks.add_task(
        process_related_request, doc_id, channel, thread_ts, response_url, ephemeral
    )
    return JSONResponse({})


def run_server(host: str = "127.0.0.1", port: int = 8002) -> None:
    """Run the Slack connector under uvicorn (blocking)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


__all__ = [
    "app",
    "append_auth_query",
    "bind_video_thread",
    "detect_language",
    "format_answer",
    "format_context_blocks",
    "format_name_results",
    "format_status",
    "issue_media_token",
    "parse_message",
    "post_slack_message",
    "public_media_url",
    "query_rainrag",
    "run_server",
    "verify_slack_signature",
    "video_session_for_thread",
]
