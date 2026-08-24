"""Slack connector for RainRAG.

Lets journalists query the archive from Slack instead of the Streamlit UI:
mention the bot in a channel, DM it, or use the /rainrag slash command.

This is a thin front-end, deliberately parallel to the Streamlit app: it calls
the existing FastAPI backend over HTTP (``/query``) rather than embedding a
RAGQueryEngine, so it inherits the backend's auth, concurrency limits,
timeouts and usage accounting, and stays light enough to run anywhere.

No Slack SDK is used. The Events API is plain HTTPS webhooks whose
authenticity check is a single HMAC, and replies are one POST to
``chat.postMessage`` -- httpx and the standard library cover both, which keeps
the dependency surface (and its supply chain) unchanged.

Slack requires webhook acknowledgement within 3 seconds while a RAG query
takes tens of seconds, so every handler acks immediately and does the actual
work in a background task, posting the answer when it is ready.
"""

import hashlib
import hmac
import json
import os
import re
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any
from urllib.parse import parse_qs

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

# Signed requests older than this are replays as far as we are concerned.
# Five minutes is the window Slack itself documents.
SIGNATURE_MAX_AGE_SECONDS = 300

# Slack limits a section block to 3000 characters of mrkdwn. Splitting a long
# answer across blocks preserves it; truncation would silently lose the tail.
SLACK_BLOCK_TEXT_LIMIT = 3000
MAX_ANSWER_BLOCKS = 5
MAX_SOURCES = 3

_HELP_TEXT = (
    "Задайте вопрос по архиву Дождя — я найду ответ в транскриптах эфиров.\n"
    "Ask a question about the TV Rain archive and I'll answer from broadcast transcripts.\n\n"
    "• В канале: `@RainRAG когда впервые обсуждали закон об иноагентах?`\n"
    "• В личных сообщениях: просто напишите вопрос\n"
    "• Фильтр по датам: добавьте `from:2021-01-01 to:2021-12-31`"
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


def _api_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("RAINRAG_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _default_top_k() -> int | None:
    raw = os.getenv("RAINRAG_SLACK_TOP_K", "")
    try:
        return int(raw) if raw.strip() else None
    except ValueError:
        logger.warning("Invalid RAINRAG_SLACK_TOP_K={!r}, using backend default", raw)
        return None


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


# --- Question parsing ----------------------------------------------------------

_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")
_DATE_FILTER_RE = re.compile(r"\b(from|to):(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)


def parse_question(raw_text: str) -> tuple[str, str | None, str | None]:
    """Strip bot mentions and extract from:/to: date filters.

    Returns (question, date_from, date_to). Dates use the same YYYY-MM-DD
    format the backend validates, so malformed values simply stay part of the
    question text instead of failing the query.
    """
    text = _MENTION_RE.sub("", raw_text or "")
    date_from: str | None = None
    date_to: str | None = None

    def _capture(match: re.Match[str]) -> str:
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

    text = _DATE_FILTER_RE.sub(_capture, text)
    return " ".join(text.split()), date_from, date_to


def detect_language(text: str) -> str:
    """Pick the answer language from the question's script.

    The corpus and audience are bilingual; any Cyrillic means the asker reads
    Russian, which is also the backend default.
    """
    return "ru" if _CYRILLIC_RE.search(text) else "en"


# --- Backend query and answer formatting ---------------------------------------


async def query_rainrag(
    question: str,
    language: str,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Run one question through the RainRAG API and return its JSON response."""
    payload: dict[str, Any] = {"question": question, "language": language}
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


def _source_lines(context: list[dict[str, Any]]) -> list[str]:
    """Render the top retrieved chunks as source attributions.

    Chunks are deduplicated by group_id so two hits in the same broadcast show
    as one source. Only public web URLs are linked -- media URLs from the API
    are relative paths that would need the auth token appended, and a standing
    credential must not be pasted into Slack messages.
    """
    lines: list[str] = []
    seen_groups: set[str] = set()
    for doc in context:
        group = doc.get("group_id") or doc.get("filename") or ""
        if group in seen_groups:
            continue
        seen_groups.add(group)

        title = doc.get("web_title") or doc.get("filename") or "—"
        date = doc.get("web_date") or doc.get("date")
        url = doc.get("web_url")

        label = title if not date else f"{title} ({date})"
        # Escape Slack mrkdwn control characters in external text.
        label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        lines.append(f"• <{url}|{label}>" if url else f"• {label}")
        if len(lines) >= MAX_SOURCES:
            break
    return lines


def format_answer(result: dict[str, Any], language: str) -> tuple[str, list[dict[str, Any]]]:
    """Build (fallback text, Block Kit blocks) for a completed query."""
    answer = (result.get("answer") or "").strip() or (
        "Не нашёл ответа в архиве." if language == "ru" else "No answer found in the archive."
    )
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


def _error_text(language: str) -> str:
    if language == "ru":
        return "Не получилось выполнить поиск — попробуйте ещё раз чуть позже."
    return "The search failed — please try again in a moment."


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


async def _answer_question(raw_text: str, send: SendFn) -> None:
    """Parse, query and deliver one question via the given async send(text, blocks).

    Mirrors the backend's usage_span accounting with one flat "[usage]" line
    per attempt, so scripts/usage_report.py can count Slack traffic alongside
    web traffic from the same journal format.
    """
    question, date_from, date_to = parse_question(raw_text)
    if not question:
        await send(_HELP_TEXT, None)
        return

    language = detect_language(question)
    started = time.monotonic()
    outcome = "ok"
    docs = 0
    try:
        result = await query_rainrag(question, language, date_from, date_to)
        docs = int(result.get("num_documents") or 0)
        text, blocks = format_answer(result, language)
        await send(text, blocks)
    except httpx.HTTPStatusError as exc:
        outcome = f"http_{exc.response.status_code}"
        logger.error("RainRAG API returned {} for Slack query", exc.response.status_code)
        await send(_error_text(language), None)
    except Exception:
        outcome = "error"
        logger.exception("Slack query processing failed")
        await send(_error_text(language), None)
    finally:
        logger.info(
            "[usage] event=slack_query outcome={} seconds={:.1f} lang={} docs={}",
            outcome,
            time.monotonic() - started,
            language,
            docs,
        )


async def process_event_question(event: dict[str, Any]) -> None:
    """Answer an app_mention or DM message event in its channel/thread."""
    channel = event.get("channel", "")
    # Replying in the existing thread (or starting one on the asking message
    # in channels) keeps long answers from burying the channel. DMs have no
    # threads worth forcing, so top-level replies read naturally there.
    thread_ts = (
        event.get("thread_ts") or event.get("ts")
        if event.get("channel_type") != "im"
        else event.get("thread_ts")
    )

    async def send(text: str, blocks: list[dict[str, Any]] | None) -> None:
        await post_slack_message(channel, text, blocks, thread_ts)

    await _answer_question(event.get("text", ""), send)


async def process_command_question(text: str, response_url: str) -> None:
    """Answer a /rainrag slash command via its response_url."""
    response_type = os.getenv("RAINRAG_SLACK_COMMAND_RESPONSE", "ephemeral")
    if response_type not in {"ephemeral", "in_channel"}:
        response_type = "ephemeral"

    async def send(answer: str, blocks: list[dict[str, Any]] | None) -> None:
        await post_response_url(response_url, answer, blocks, response_type)

    await _answer_question(text, send)


# --- FastAPI app -----------------------------------------------------------------

app = FastAPI(
    title="RainRAG Slack Connector",
    description="Slack front-end for the RainRAG query API",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness plus configuration sanity, without leaking secret values."""
    return {
        "status": "ok",
        "signing_secret_configured": bool(os.getenv("SLACK_SIGNING_SECRET")),
        "bot_token_configured": bool(_bot_token()),
        "api_url": _api_base(),
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
    # message events, and answering them would loop forever).
    if event.get("bot_id") or event.get("subtype"):
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


def run_server(host: str = "127.0.0.1", port: int = 8002) -> None:
    """Run the Slack connector under uvicorn (blocking)."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


__all__ = [
    "app",
    "detect_language",
    "format_answer",
    "parse_question",
    "post_slack_message",
    "query_rainrag",
    "run_server",
    "verify_slack_signature",
]
