"""Unit tests for the Slack connector.

Covers the security-critical paths (signature verification, replay window,
event de-duplication, bot-loop prevention), message parsing for every mode,
formatting, media links, the video-session thread registry, and all four
webhook endpoints.
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

import httpx
import pytest
from fastapi.testclient import TestClient


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag import slack_connector
from src.rainrag.slack_connector import (
    _is_duplicate_event,
    _seen_events,
    _video_threads,
    app,
    append_auth_query,
    bind_video_thread,
    detect_language,
    format_answer,
    format_context_blocks,
    format_name_results,
    format_status,
    issue_media_token,
    parse_message,
    public_media_url,
    verify_slack_signature,
    video_session_for_thread,
)


SIGNING_SECRET = "test-signing-secret"


@pytest.fixture(autouse=True)
def slack_env(monkeypatch):
    """Provide Slack credentials and clean in-memory stores for every test."""
    monkeypatch.setenv("SLACK_SIGNING_SECRET", SIGNING_SECRET)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAINRAG_ASSET_URL", raising=False)
    _seen_events.clear()
    _video_threads.clear()
    yield
    _seen_events.clear()
    _video_threads.clear()


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
# Message parsing and language detection
# ============================================================================


class TestParseMessage:
    def test_strips_bot_mention(self):
        parsed = parse_message("<@U0123ABCD> когда были протесты?")
        assert parsed.mode == "ask"
        assert parsed.text == "когда были протесты?"

    def test_extracts_date_filters(self):
        parsed = parse_message("protests from:2020-01-01 to:2020-12-31")
        assert parsed.text == "protests"
        assert parsed.date_from == "2020-01-01"
        assert parsed.date_to == "2020-12-31"

    def test_malformed_date_stays_in_question(self):
        parsed = parse_message("news from:yesterday")
        assert "from:yesterday" in parsed.text
        assert parsed.date_from is None

    def test_invalid_calendar_date_stays_in_question(self):
        # Shaped like a date but not a real one: must not be stripped or
        # forwarded as a filter the backend would reject.
        parsed = parse_message("news from:2024-02-31 to:2024-13-01")
        assert "from:2024-02-31" in parsed.text
        assert "to:2024-13-01" in parsed.text
        assert parsed.date_from is None
        assert parsed.date_to is None

    def test_top_k_token(self):
        parsed = parse_message("протесты top:10")
        assert parsed.top_k == 10
        assert parsed.text == "протесты"

    def test_out_of_range_top_k_stays_in_question(self):
        parsed = parse_message("протесты top:99")
        assert parsed.top_k is None
        assert "top:99" in parsed.text

    def test_lang_override(self):
        parsed = parse_message("протесты lang:en")
        assert parsed.language == "en"
        assert parsed.text == "протесты"

    def test_slack_link_markup_unwrapped(self):
        parsed = parse_message("video: <https://youtu.be/abc|clip>")
        assert parsed.mode == "video"
        assert parsed.text == "https://youtu.be/abc"

    def test_name_mode(self):
        parsed = parse_message("name: вечернее шоу")
        assert parsed.mode == "name"
        assert parsed.text == "вечернее шоу"

    def test_name_mode_russian_prefix(self):
        parsed = parse_message("Название: интервью")
        assert parsed.mode == "name"
        assert parsed.text == "интервью"

    def test_video_mode_russian_prefix(self):
        parsed = parse_message("видео: https://t.me/c/1/2")
        assert parsed.mode == "video"
        assert parsed.text == "https://t.me/c/1/2"

    def test_status_mode(self):
        assert parse_message("status").mode == "status"
        assert parse_message("статус").mode == "status"

    def test_help_mode(self):
        assert parse_message("help").mode == "help"
        assert parse_message("помощь").mode == "help"

    def test_empty_after_mention_is_help(self):
        assert parse_message("<@U0123ABCD>").mode == "help"

    def test_lang_applies_to_video_mode(self):
        parsed = parse_message("video: https://youtu.be/abc lang:en")
        assert parsed.mode == "video"
        assert parsed.language == "en"


class TestDetectLanguage:
    def test_cyrillic_is_russian(self):
        assert detect_language("что случилось в Москве?") == "ru"

    def test_latin_is_english(self):
        assert detect_language("what happened in Moscow?") == "en"

    def test_mixed_prefers_russian(self):
        assert detect_language("what about Навальный?") == "ru"


# ============================================================================
# Event de-duplication and thread registry
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


class TestVideoThreadRegistry:
    def test_bind_and_lookup(self):
        bind_video_thread("C1", "111.222", "sess-1")
        assert video_session_for_thread("C1", "111.222") == "sess-1"

    def test_unknown_thread_returns_none(self):
        assert video_session_for_thread("C1", "999.999") is None

    def test_expired_binding_pruned(self):
        bind_video_thread("C1", "111.222", "sess-1")
        _video_threads[("C1", "111.222")] = ("sess-1", time.time() - 25 * 3600)
        assert video_session_for_thread("C1", "111.222") is None


# ============================================================================
# Media links
# ============================================================================


class TestMediaLinks:
    def test_no_token_without_secret(self):
        assert issue_media_token() == ""

    def test_token_minted_with_secret(self, monkeypatch):
        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "secret")
        token = issue_media_token()
        assert token.startswith("v1.")
        assert len(token.split(".")) == 3

    def test_append_auth_preserves_fragment(self, monkeypatch):
        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "secret")
        url = append_auth_query("https://rag.example/video/ab/cd.mp4#t=42")
        assert "auth=v1." in url
        assert url.endswith("#t=42")

    def test_no_auth_param_when_auth_disabled(self):
        url = append_auth_query("https://rag.example/video/x.mp4")
        assert "auth=" not in url

    def test_public_media_url_requires_asset_base(self):
        assert public_media_url("/video/x.mp4") is None

    def test_public_media_url_joins_base(self, monkeypatch):
        monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.example")
        assert public_media_url("/video/x.mp4") == "https://rag.example/video/x.mp4"


# ============================================================================
# Answer, context, name and status formatting
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


class TestFormatContextBlocks:
    CHUNK = {
        "doc_id": "doc-1",
        "text": "Обсуждение закона " * 40,
        "web_title": "Вечернее шоу",
        "web_date": "2021-05-01",
        "start_time": "00:10:00",
        "end_time": "00:12:30",
        "score": 0.91,
        "video_url": "/video/ab/cd_720p.mp4#t=600",
        "vtt_url": "/vtt/ab/cd.ru.vtt",
        "web_url": "https://tvrain.tv/x",
    }

    def test_empty_context_returns_none(self):
        assert format_context_blocks([], "ru") is None

    def test_excerpt_metadata_and_button(self):
        _, blocks = format_context_blocks([self.CHUNK], "ru")
        section = blocks[1]["text"]["text"]
        assert "Вечернее шоу" in section
        assert "2021-05-01" in section
        assert "00:10:00–00:12:30" in section
        assert "score 0.91" in section
        assert "> Обсуждение" in section
        actions = blocks[-1]
        assert actions["type"] == "actions"
        assert actions["elements"][0]["value"] == "doc-1"
        assert actions["elements"][0]["action_id"] == "related_1"

    def test_media_links_only_with_asset_base(self, monkeypatch):
        _, blocks = format_context_blocks([self.CHUNK], "ru")
        assert "/video/" not in blocks[1]["text"]["text"]

        monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.example")
        _, blocks = format_context_blocks([self.CHUNK], "ru")
        section = blocks[1]["text"]["text"]
        assert "https://rag.example/video/ab/cd_720p.mp4#t=600" in section
        assert "https://rag.example/vtt/ab/cd.ru.vtt" in section

    def test_media_links_carry_expiring_token(self, monkeypatch):
        monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.example")
        monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "secret")
        _, blocks = format_context_blocks([self.CHUNK], "ru")
        section = blocks[1]["text"]["text"]
        assert "auth=v1." in section
        # The standing secret itself must never appear in a Slack message.
        assert "auth=secret" not in section

    def test_chunk_without_doc_id_gets_no_button(self):
        chunk = {**self.CHUNK}
        chunk.pop("doc_id")
        _, blocks = format_context_blocks([chunk], "ru")
        assert all(b.get("type") != "actions" for b in blocks)


class TestFormatNameResults:
    def test_results_with_links(self, monkeypatch):
        monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.example")
        result = {
            "results": [
                {
                    "video_hash": "abc",
                    "name": "Вечернее шоу",
                    "date": "2021-05-01",
                    "web_url": "https://tvrain.tv/x",
                    "teleshow_name": "Вечернее шоу",
                    "languages": {"ru": {"video_url": "/video/ab.mp4", "vtt_url": None}},
                }
            ],
            "query": "вечернее",
        }
        _, blocks = format_name_results(result, "ru")
        rendered = blocks[0]["text"]["text"]
        assert "<https://tvrain.tv/x|Вечернее шоу (2021-05-01)>" in rendered
        assert "<https://rag.example/video/ab.mp4|▶ RU>" in rendered

    def test_empty_results(self):
        text, _ = format_name_results({"results": []}, "en")
        assert text == "No videos match that name."


class TestFormatStatus:
    HEALTH = {
        "status": "healthy",
        "llm_provider": "claude",
        "llm_model": "claude-sonnet-5",
        "embedding_provider": "local",
        "embedding_model": "e5-large",
        "qdrant_collection": "broadcast",
        "hybrid_search_enabled": True,
        "reranker_enabled": False,
    }

    def test_status_russian(self):
        text = format_status(self.HEALTH, "ru")
        assert "Статус: healthy" in text
        assert "Гибридный поиск: вкл" in text
        assert "Реранкер: выкл" in text

    def test_status_english(self):
        text = format_status(self.HEALTH, "en")
        assert "Hybrid search: on" in text

    def test_unreachable(self):
        assert format_status(None, "en") == "The API is unreachable."


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
    def test_message_edit_subtype_ignored(self, mock_process, client):
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "subtype": "message_changed",
        }
        body = event_body(event)
        client.post("/slack/events", content=body, headers=sign(body))
        mock_process.assert_not_called()

    @patch("src.rainrag.slack_connector.process_event_question", new_callable=AsyncMock)
    def test_file_share_subtype_processed(self, mock_process, client):
        # file_share is the one subtype that is a user action we handle.
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "ts": "1.0",
            "subtype": "file_share",
            "files": [{"name": "clip.mp4", "mimetype": "video/mp4"}],
        }
        body = event_body(event)
        client.post("/slack/events", content=body, headers=sign(body))
        mock_process.assert_called_once()

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
        assert "вопрос" in response.json()["text"]
        mock_process.assert_not_called()


# ============================================================================
# Interactivity endpoint
# ============================================================================


class TestInteractiveEndpoint:
    def interactive_body(self, action_id: str = "related_1", value: str = "doc-1") -> bytes:
        payload = {
            "type": "block_actions",
            "actions": [{"action_id": action_id, "value": value}],
            "channel": {"id": "C1"},
            "container": {"message_ts": "1.0", "thread_ts": "0.5", "is_ephemeral": False},
            "response_url": "https://hooks.slack.com/r/act",
        }
        return urlencode({"payload": json.dumps(payload)}).encode()

    @patch("src.rainrag.slack_connector.process_related_request", new_callable=AsyncMock)
    def test_related_button_dispatched(self, mock_related, client):
        body = self.interactive_body()
        response = client.post(
            "/slack/interactive",
            content=body,
            headers={**sign(body), "Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        mock_related.assert_called_once_with(
            "doc-1", "C1", "0.5", "https://hooks.slack.com/r/act", False
        )

    @patch("src.rainrag.slack_connector.process_related_request", new_callable=AsyncMock)
    def test_unknown_action_ignored(self, mock_related, client):
        body = self.interactive_body(action_id="something_else")
        response = client.post(
            "/slack/interactive",
            content=body,
            headers={**sign(body), "Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == 200
        mock_related.assert_not_called()

    def test_unsigned_interactive_rejected(self, client):
        response = client.post("/slack/interactive", content=self.interactive_body())
        assert response.status_code == 401


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
    def test_context_message_follows_answer(self, mock_query, mock_post):
        mock_query.return_value = {
            "answer": "ответ",
            "context": [{"doc_id": "d1", "text": "фрагмент", "filename": "a.vtt"}],
            "num_documents": 1,
        }
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> вопрос"}
        asyncio.run(slack_connector.process_event_question(event))
        assert mock_post.call_count == 2  # answer + context excerpts
        context_blocks = mock_post.call_args_list[1].args[2]
        assert any(b.get("type") == "actions" for b in context_blocks)

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_context_message_disabled_by_env(self, mock_query, mock_post, monkeypatch):
        monkeypatch.setenv("RAINRAG_SLACK_SHOW_CONTEXT", "false")
        mock_query.return_value = {
            "answer": "ответ",
            "context": [{"doc_id": "d1", "text": "фрагмент"}],
            "num_documents": 1,
        }
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> вопрос"}
        asyncio.run(slack_connector.process_event_question(event))
        assert mock_post.call_count == 1

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

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.search_by_name_api", new_callable=AsyncMock)
    def test_name_mode_dispatched(self, mock_search, mock_post):
        mock_search.return_value = {"results": [], "query": "шоу"}
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> name: шоу"}
        asyncio.run(slack_connector.process_event_question(event))
        mock_search.assert_called_once_with("шоу")
        assert "Ничего не найдено" in mock_post.call_args.args[1]

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.api_health", new_callable=AsyncMock)
    def test_status_mode_dispatched(self, mock_health, mock_post):
        mock_health.return_value = None
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> status"}
        asyncio.run(slack_connector.process_event_question(event))
        assert "unreachable" in mock_post.call_args.args[1]

    @patch("src.rainrag.slack_connector._handle_video_file", new_callable=AsyncMock)
    def test_video_file_routed_to_upload(self, mock_upload):
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "ts": "1.0",
            "subtype": "file_share",
            "text": "",
            "files": [{"name": "clip.mp4", "mimetype": "video/mp4", "size": 1024}],
        }
        asyncio.run(slack_connector.process_event_question(event))
        mock_upload.assert_called_once()
        assert mock_upload.call_args.args[0]["name"] == "clip.mp4"
        assert mock_upload.call_args.args[3] == "ru"  # newsroom default for bare uploads

    @patch("src.rainrag.slack_connector._handle_video_file", new_callable=AsyncMock)
    def test_non_video_file_not_uploaded(self, mock_upload):
        with patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock):
            event = {
                "type": "message",
                "channel_type": "im",
                "channel": "D1",
                "ts": "1.0",
                "subtype": "file_share",
                "text": "вопрос",
                "files": [{"name": "notes.pdf", "mimetype": "application/pdf"}],
            }
            with patch(
                "src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock
            ) as mock_query:
                mock_query.return_value = {"answer": "a", "context": [], "num_documents": 0}
                asyncio.run(slack_connector.process_event_question(event))
        mock_upload.assert_not_called()

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_video_session", new_callable=AsyncMock)
    def test_bound_thread_queries_video_session(self, mock_session_query, mock_post):
        bind_video_thread("D1", "100.0", "sess-42")
        mock_session_query.return_value = {"answer": "из видео", "context": []}
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "ts": "101.0",
            "thread_ts": "100.0",
            "text": "о чём это видео?",
        }
        asyncio.run(slack_connector.process_event_question(event))
        assert mock_session_query.call_args.args[0] == "sess-42"
        assert mock_post.call_args_list[0].args[3] == "100.0"  # stays in the session thread

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_unbound_thread_falls_back_to_corpus(self, mock_query, mock_post):
        mock_query.return_value = {"answer": "a", "context": [], "num_documents": 0}
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "ts": "101.0",
            "thread_ts": "100.0",
            "text": "вопрос",
        }
        asyncio.run(slack_connector.process_event_question(event))
        mock_query.assert_called_once()


# ============================================================================
# Inline clips
# ============================================================================


class TestClips:
    CHUNK = {
        "doc_id": "d1",
        "text": "фрагмент",
        "web_title": "Вечернее шоу",
        "video_url": "/video/ab/cd_720p.mp4#t=600",
        "start_time": "00:10:00",
        "end_time": "00:12:30",
    }

    def test_timecode_to_seconds(self):
        assert slack_connector._timecode_to_seconds("01:02:03") == 3723
        assert slack_connector._timecode_to_seconds("02:30") == 150
        assert slack_connector._timecode_to_seconds("bogus") is None
        assert slack_connector._timecode_to_seconds(None) is None

    def test_clip_request_derivation(self):
        rel, start, end = slack_connector._clip_request_for(self.CHUNK)
        assert rel == "ab/cd_720p.mp4"
        assert start == 600
        assert end == 750

    def test_clip_request_requires_video_and_timecodes(self):
        assert slack_connector._clip_request_for({"video_url": None}) is None
        assert (
            slack_connector._clip_request_for(
                {"video_url": "/video/x.mp4", "start_time": "00:05:00", "end_time": "00:04:00"}
            )
            is None
        )

    @patch("src.rainrag.slack_connector.upload_clip_to_slack", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.fetch_video_clip", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_top_chunk_clip_posted_in_thread(self, mock_query, mock_post, mock_fetch, mock_upload):
        mock_query.return_value = {"answer": "ответ", "context": [self.CHUNK], "num_documents": 1}
        mock_fetch.return_value = b"clip-bytes"
        mock_upload.return_value = True
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> вопрос"}
        asyncio.run(slack_connector.process_event_question(event))
        mock_fetch.assert_called_once_with("ab/cd_720p.mp4", 600, 750)
        assert mock_upload.call_args.args[0] == "C1"
        assert mock_upload.call_args.args[1] == "1.0"  # same thread as the answer
        assert "Вечернее шоу" in mock_upload.call_args.args[4]

    @patch("src.rainrag.slack_connector.upload_clip_to_slack", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.fetch_video_clip", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_clip_failure_does_not_break_answer(
        self, mock_query, mock_post, mock_fetch, mock_upload
    ):
        mock_query.return_value = {"answer": "ответ", "context": [self.CHUNK], "num_documents": 1}
        mock_fetch.return_value = None  # clip fetch failed / too large
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> вопрос"}
        asyncio.run(slack_connector.process_event_question(event))
        mock_upload.assert_not_called()
        assert mock_post.call_count == 2  # answer + context still delivered

    @patch("src.rainrag.slack_connector.fetch_video_clip", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_clips_disabled_by_env(self, mock_query, mock_post, mock_fetch, monkeypatch):
        monkeypatch.setattr(slack_connector, "CLIP_CHUNKS", 0)
        mock_query.return_value = {"answer": "a", "context": [self.CHUNK], "num_documents": 1}
        event = {"type": "app_mention", "channel": "C1", "ts": "1.0", "text": "<@U1> вопрос"}
        asyncio.run(slack_connector.process_event_question(event))
        mock_fetch.assert_not_called()

    @patch("src.rainrag.slack_connector.fetch_video_clip", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.post_response_url", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_rainrag", new_callable=AsyncMock)
    def test_slash_command_answers_without_clips(self, mock_query, mock_respond, mock_fetch):
        # Slash commands answer ephemerally; there is no channel message to
        # attach footage to, so the clip step is skipped entirely.
        mock_query.return_value = {"answer": "a", "context": [self.CHUNK], "num_documents": 1}
        asyncio.run(
            slack_connector.process_command_question("вопрос", "https://hooks.slack.com/r/1")
        )
        assert mock_respond.call_count >= 1
        mock_fetch.assert_not_called()


# ============================================================================
# Video session flows
# ============================================================================


class TestVideoSessionFlows:
    @patch("src.rainrag.slack_connector._watch_video_session", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.create_video_session_from_url", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    def test_video_url_binds_thread(self, mock_post, mock_create, mock_watch):
        mock_create.return_value = {"id": "sess-9", "status": "queued"}
        event = {
            "type": "app_mention",
            "channel": "C1",
            "ts": "5.0",
            "text": "<@U1> video: https://youtu.be/abc",
        }
        asyncio.run(slack_connector.process_event_question(event))
        mock_create.assert_called_once_with("https://youtu.be/abc")
        assert video_session_for_thread("C1", "5.0") == "sess-9"
        mock_watch.assert_called_once()

    @patch("src.rainrag.slack_connector.create_video_session_from_url", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    def test_import_error_mapped_to_advice(self, mock_post, mock_create):
        request = httpx.Request("POST", "http://api/video-sessions/from-url")
        response = httpx.Response(451, request=request)
        mock_create.side_effect = httpx.HTTPStatusError("", request=request, response=response)
        event = {
            "type": "app_mention",
            "channel": "C1",
            "ts": "5.0",
            "text": "<@U1> video: https://youtu.be/abc",
        }
        asyncio.run(slack_connector.process_event_question(event))
        assert "недоступно из региона" in mock_post.call_args.args[1]
        assert video_session_for_thread("C1", "5.0") is None

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.get_video_session", new_callable=AsyncMock)
    def test_watch_announces_ready(self, mock_get, mock_post):
        mock_get.return_value = {"id": "s1", "status": "ready"}
        asyncio.run(slack_connector._watch_video_session("s1", "C1", "5.0", "ru"))
        assert "готово" in mock_post.call_args.args[1].lower()

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.get_video_session", new_callable=AsyncMock)
    def test_watch_announces_error_and_unbinds(self, mock_get, mock_post):
        bind_video_thread("C1", "5.0", "s1")
        mock_get.return_value = {"id": "s1", "status": "error", "error": "boom"}
        asyncio.run(slack_connector._watch_video_session("s1", "C1", "5.0", "en"))
        assert "failed" in mock_post.call_args.args[1]
        assert video_session_for_thread("C1", "5.0") is None

    @patch("src.rainrag.slack_connector.post_slack_message", new_callable=AsyncMock)
    @patch("src.rainrag.slack_connector.query_video_session", new_callable=AsyncMock)
    def test_expired_session_unbinds_thread(self, mock_query, mock_post):
        bind_video_thread("D1", "100.0", "sess-dead")
        request = httpx.Request("POST", "http://api/video-sessions/sess-dead/query")
        response = httpx.Response(404, request=request)
        mock_query.side_effect = httpx.HTTPStatusError("", request=request, response=response)
        event = {
            "type": "message",
            "channel_type": "im",
            "channel": "D1",
            "ts": "101.0",
            "thread_ts": "100.0",
            "text": "вопрос",
        }
        asyncio.run(slack_connector.process_event_question(event))
        assert video_session_for_thread("D1", "100.0") is None
        assert "истекла" in mock_post.call_args.args[1]
