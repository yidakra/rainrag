"""Unit tests for the Slack connector.

Covers the security-critical paths (signature verification, replay window,
event de-duplication, bot-loop prevention) and the message plumbing
(question parsing, answer formatting, webhook handlers).
"""

import asyncio
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag import slack_connector
from src.rainrag.slack_connector import (
    _is_duplicate_event,
    _seen_events,
    app,
    detect_language,
    format_answer,
    parse_question,
    verify_slack_signature,
)


SIGNING_SECRET = "test-signing-secret"


@pytest.fixture(autouse=True)
def slack_env(monkeypatch):
    """Provide Slack credentials and a clean dedup store for every test."""
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    _seen_events.clear()
    yield
    _seen_events.clear()


@pytest.fixture
def client():
    return TestClient(app)


def sign(body: bytes, timestamp: str | None = None) -> dict[str, str]:
    """Produce valid Slack signature headers for a request body."""
    ts = timestamp if timestamp is not None else str(int(time.time()))
    base = b"v0:" + ts.encode() + b":" + body
    digest = hmac.new(SIGNING_SECRET.encode(), base, hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": f"v0={digest}"}


def event_body(event: dict, event_id: str = "Ev001") -> bytes:
    return json.dumps({"type": "event_callback", "event_id": event_id, "event": event}).encode()


# ============================================================================
# Signature verification
# ============================================================================


class TestSignatureVerification:
    def test_valid_signature_accepted(self):
        body = b'{"type":"url_verification"}'
        headers = sign(body)
        assert verify_slack_signature(
            body, headers["X-Slack-Request-Timestamp"], headers["X-Slack-Signature"]
        )

    def test_tampered_body_rejected(self):
        headers = sign(b"original")
        assert not verify_slack_signature(
            b"tampered", headers["X-Slack-Request-Timestamp"], headers["X-Slack-Signature"]
        )

    def test_missing_headers_rejected(self):
        assert not verify_slack_signature(b"body", None, None)

    def test_stale_timestamp_rejected(self):
        stale = str(int(time.time()) - 3600)
        headers = sign(b"body", timestamp=stale)
        assert not verify_slack_signature(
            b"body", headers["X-Slack-Request-Timestamp"], headers["X-Slack-Signature"]
        )

    def test_garbage_timestamp_rejected(self):
        assert not verify_slack_signature(b"body", "not-a-number", "v0=abc")

    def test_infinite_timestamp_rejected(self):
        assert not verify_slack_signature(b"body", "inf", "v0=abc")

    def test_non_ascii_signature_rejected(self):
        headers = sign(b"body")
        assert not verify_slack_signature(
            b"body", headers["X-Slack-Request-Timestamp"], "v0=пё" + "0" * 60
        )

    def test_unsigned_request_gets_401(self, client):
        response = client.post("/slack/events", content=b"{}")
        assert response.status_code == 401


# ============================================================================
# Question parsing and language detection
# ============================================================================


class TestParseQuestion:
    def test_strips_bot_mention(self):
        question, _, _ = parse_question("<@U0123ABCD> когда были протесты?")
        assert question == "когда были протесты?"

    def test_extracts_date_filters(self):
        question, date_from, date_to = parse_question("protests from:2020-01-01 to:2020-12-31")
        assert question == "protests"
        assert date_from == "2020-01-01"
        assert date_to == "2020-12-31"

    def test_malformed_date_stays_in_question(self):
        question, date_from, _ = parse_question("news from:yesterday")
        assert "from:yesterday" in question
        assert date_from is None

    def test_invalid_calendar_date_stays_in_question(self):
        # Shaped like a date but not a real one: must not be stripped or
        # forwarded as a filter the backend would reject.
        question, date_from, date_to = parse_question("news from:2024-02-31 to:2024-13-01")
        assert "from:2024-02-31" in question
        assert "to:2024-13-01" in question
        assert date_from is None
        assert date_to is None

    def test_empty_after_mention(self):
        question, _, _ = parse_question("<@U0123ABCD>")
        assert question == ""


class TestDetectLanguage:
    def test_cyrillic_is_russian(self):
        assert detect_language("что случилось в Москве?") == "ru"

    def test_latin_is_english(self):
        assert detect_language("what happened in Moscow?") == "en"

    def test_mixed_prefers_russian(self):
        assert detect_language("what about Навальный?") == "ru"


# ============================================================================
# Event de-duplication
# ============================================================================


class TestEventDedup:
    def test_first_event_not_duplicate(self):
        assert not _is_duplicate_event("Ev123")

    def test_second_event_is_duplicate(self):
        _is_duplicate_event("Ev123")
        assert _is_duplicate_event("Ev123")

    def test_distinct_events_independent(self):
        _is_duplicate_event("Ev123")
        assert not _is_duplicate_event("Ev456")


# ============================================================================
# Answer formatting
# ============================================================================


class TestFormatAnswer:
    def test_answer_and_sources(self):
        result = {
            "answer": "Ответ по архиву.",
            "context": [
                {
                    "group_id": "g1",
                    "web_title": "Вечернее шоу",
                    "web_date": "2021-05-01",
                    "web_url": "https://tvrain.tv/x",
                },
                {"group_id": "g1", "web_title": "Вечернее шоу"},
                {"group_id": "g2", "filename": "archive/abc.ru.vtt", "date": "2020-01-01"},
            ],
        }
        text, blocks = format_answer(result, "ru")
        assert text == "Ответ по архиву."
        assert blocks[0]["text"]["text"] == "Ответ по архиву."
        context_block = blocks[-1]
        assert context_block["type"] == "context"
        rendered = context_block["elements"][0]["text"]
        assert "Источники" in rendered
        # Duplicate group_id collapses to one source line.
        assert rendered.count("Вечернее шоу") == 1
        assert "<https://tvrain.tv/x|Вечернее шоу (2021-05-01)>" in rendered
        # Source without web_url is shown unlinked.
        assert "archive/abc.ru.vtt (2020-01-01)" in rendered

    def test_long_answer_split_across_blocks(self):
        result = {"answer": "слово " * 2000, "context": []}
        _, blocks = format_answer(result, "ru")
        assert len(blocks) > 1
        assert all(len(b["text"]["text"]) <= 3000 for b in blocks)

    def test_answer_filling_all_blocks_exactly_is_not_truncated(self):
        result = {"answer": ("x" * 2500 + "\n") * 5, "context": []}
        _, blocks = format_answer(result, "en")
        assert len(blocks) == 5
        assert not blocks[-1]["text"]["text"].endswith("…")

    def test_oversized_answer_truncated_with_ellipsis(self):
        result = {"answer": "x" * 20000, "context": []}
        _, blocks = format_answer(result, "en")
        assert len(blocks) == 5
        assert blocks[-1]["text"]["text"].endswith("…")

    def test_empty_answer_gets_fallback(self):
        text, _ = format_answer({"answer": "", "context": []}, "en")
        assert text == "No answer found in the archive."

    def test_mrkdwn_escaped_in_titles(self):
        result = {
            "answer": "a",
            "context": [{"group_id": "g", "web_title": "<b> & Co", "web_url": "https://e.x"}],
        }
        _, blocks = format_answer(result, "en")
        rendered = blocks[-1]["elements"][0]["text"]
        assert "&lt;b&gt; &amp; Co" in rendered


# ============================================================================
# Events endpoint
# ============================================================================


class TestEventsEndpoint:
    def test_url_verification_challenge(self, client):
        body = json.dumps({"type": "url_verification", "challenge": "ch-42"}).encode()
        response = client.post("/slack/events", content=body, headers=sign(body))
        assert response.status_code == 200
        assert response.json() == {"challenge": "ch-42"}

    def test_invalid_json_rejected(self, client):
        body = b"not json"
        response = client.post("/slack/events", content=body, headers=sign(body))
        assert response.status_code == 400

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_app_mention_processed(self, mock_process, client):
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> q"}
        body = event_body(event)
        response = client.post("/slack/events", content=body, headers=sign(body))
        assert response.status_code == 200
        mock_process.assert_called_once_with(event)

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_dm_message_processed(self, mock_process, client):
        event = {"type": "message", "channel_type": "im", "channel": "D1", "text": "q"}
        body = event_body(event)
        response = client.post("/slack/events", content=body, headers=sign(body))
        assert response.status_code == 200
        mock_process.assert_called_once()

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_bot_message_ignored(self, mock_process, client):
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "text": "q",
            "bot_id": "B99",
        }
        body = event_body(event)
        client.post("/slack/events", content=body, headers=sign(body))
        mock_process.assert_not_called()

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_channel_message_without_mention_ignored(self, mock_process, client):
        event = {"type": "message", "channel_type": "channel", "channel": "C1", "text": "q"}
        body = event_body(event)
        client.post("/slack/events", content=body, headers=sign(body))
        mock_process.assert_not_called()

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_duplicate_event_id_processed_once(self, mock_process, client):
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "q"}
        body = event_body(event, event_id="EvSame")
        client.post("/slack/events", content=body, headers=sign(body))
        client.post("/slack/events", content=body, headers=sign(body))
        assert mock_process.call_count == 1

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_slack_retry_header_acked_without_processing(self, mock_process, client):
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "q"}
        body = event_body(event)
        headers = {**sign(body), "X-Slack-Retry-Num": "1"}
        response = client.post("/slack/events", content=body, headers=headers)
        assert response.status_code == 200
        mock_process.assert_not_called()


# ============================================================================
# Slash command endpoint
# ============================================================================


class TestCommandsEndpoint:
    @patch("src.rainrag.slack_connector.process_command_question", new_callable=AsyncMock)
    def test_command_acks_and_processes(self, mock_process, client):
        body = urlencode(
            {"text": "когда были выборы?", "response_url": "https://hooks.slack.com/r/1"}
        ).encode()
        response = client.post(
            "/slack/commands",
            content=body,
            headers={**sign(body), "Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert response.json()["response_type"] == "ephemeral"
        mock_process.assert_called_once_with("когда были выборы?", "https://hooks.slack.com/r/1")

    @patch("src.rainrag.slack_connector.process_command_question", new_callable=AsyncMock)
    def test_empty_command_returns_help(self, mock_process, client):
        body = urlencode({"text": "", "response_url": "https://hooks.slack.com/r/1"}).encode()
        response = client.post(
            "/slack/commands",
            content=body,
            headers={**sign(body), "Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        assert "RainRAG" in response.json()["text"] or "вопрос" in response.json()["text"]
        mock_process.assert_not_called()


# ============================================================================
# Background question processing
# ============================================================================


class TestProcessEventQuestion:
    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_channel_mention_replies_in_thread(self, mock_query, mock_post):
        mock_query.return_value = {"answer": "ответ", "context": [], "num_documents": 0}
        event = {
            "type": "app_mention",
            "channel": "C1",
            "ts": "111.222",
            "channel_type": "channel",
            "text": "<@U1> что нового?",
        }
        asyncio.run(slack_connector.process_event_question(event))
        mock_query.assert_called_once()
        assert mock_query.call_args.args[0] == "что нового?"
        assert mock_query.call_args.args[1] == "ru"
        assert mock_post.call_args.args[0] == "C1"
        assert mock_post.call_args.args[3] == "111.222"  # threaded on the asking message

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_dm_reply_not_threaded(self, mock_query, mock_post):
        mock_query.return_value = {"answer": "answer", "context": [], "num_documents": 0}
        event = {
            "type": "message",
            "channel": "D1",
            "ts": "1.0",
            "channel_type": "im",
            "text": "what is new?",
        }
        asyncio.run(slack_connector.process_event_question(event))
        assert mock_post.call_args.args[3] is None

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_query_failure_posts_error_message(self, mock_query, mock_post):
        mock_query.side_effect = RuntimeError("backend down")
        event = {
            "type": "app_mention",
            "channel": "C1",
            "ts": "1.0",
            "text": "<@U1> вопрос",
        }
        asyncio.run(slack_connector.process_event_question(event))
        posted_text = mock_post.call_args.args[1]
        assert "Не получилось" in posted_text

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_empty_question_posts_help(self, mock_query, mock_post):
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1>"}
        asyncio.run(slack_connector.process_event_question(event))
        mock_query.assert_not_called()
        assert "вопрос" in mock_post.call_args.args[1]
