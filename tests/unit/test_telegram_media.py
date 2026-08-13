"""Tests for Telegram link parsing and the guards around downloading.

The download itself needs real API credentials and a logged-in session, so it is
not covered here. What *is* covered is everything that decides whether a
download is attempted at all, which is where the interesting mistakes live.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rainrag.telegram_media import (
    TelegramNotDownloadableError,
    TelegramRef,
    TelegramUnavailableError,
    _video_size,
    check_downloadable,
    download_telegram_video,
    parse_telegram_url,
)


def _video_message(*, size: int, mime: str = "video/mp4", video_attr: bool = True, **extra):
    """Build a minimal stand-in for a Telethon video message."""
    from telethon.tl.types import DocumentAttributeVideo, MessageMediaDocument

    class Doc:
        mime_type = mime
        attributes = [DocumentAttributeVideo(duration=10, w=640, h=360)] if video_attr else []

    Doc.size = size
    media = MessageMediaDocument.__new__(MessageMediaDocument)
    media.document = Doc()

    class Msg:
        pass

    msg = Msg()
    msg.media = media
    for k, v in extra.items():
        setattr(msg, k, v)
    return msg


class TestParsePublicLinks:
    def test_plain_channel_post(self):
        ref = parse_telegram_url("https://t.me/europa_press/613")
        assert ref == TelegramRef(message_id=613, username="europa_press")
        assert not ref.is_private

    def test_query_string_is_ignored(self):
        # ?single is how Telegram addresses one item of an album.
        assert parse_telegram_url("https://t.me/vorposte/29342?single") == TelegramRef(
            message_id=29342, username="vorposte"
        )

    def test_preview_s_prefix(self):
        assert parse_telegram_url("https://t.me/s/tvrain/109028") == TelegramRef(
            message_id=109028, username="tvrain"
        )

    def test_topic_link_uses_the_last_number(self):
        # t.me/<channel>/<topic>/<message>: the message id is last, not first.
        assert parse_telegram_url("https://t.me/somechan/456/789") == TelegramRef(
            message_id=789, username="somechan"
        )

    def test_alternate_hosts(self):
        for host in ("telegram.me", "telegram.dog", "www.t.me"):
            assert parse_telegram_url(f"https://{host}/chan/12") == TelegramRef(
                message_id=12, username="chan"
            )

    def test_trailing_slash(self):
        assert parse_telegram_url("https://t.me/chan/12/") == TelegramRef(
            message_id=12, username="chan"
        )


class TestParsePrivateLinks:
    def test_private_channel_is_not_mistaken_for_a_username(self):
        # yt-dlp's unanchored pattern reads this as channel "c", message 1234567890.
        ref = parse_telegram_url("https://t.me/c/1234567890/456")
        assert ref == TelegramRef(message_id=456, internal_id=1234567890)
        assert ref.is_private

    def test_private_topic_link(self):
        assert parse_telegram_url("https://t.me/c/1234567890/456/789") == TelegramRef(
            message_id=789, internal_id=1234567890
        )

    def test_invite_link(self):
        ref = parse_telegram_url("https://t.me/+AbCdEf123")
        assert ref.invite_hash == "AbCdEf123"
        assert ref.is_private

    def test_legacy_joinchat_invite(self):
        assert parse_telegram_url("https://t.me/joinchat/AbCdEf123").invite_hash == "AbCdEf123"

    def test_invite_hash_never_appears_in_describe(self):
        # describe() lands in logs, so the hash must not leak there.
        ref = parse_telegram_url("https://t.me/+SuperSecretHash")
        assert "SuperSecretHash" not in ref.describe()


class TestParseRejections:
    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/video.mp4",
            "https://www.youtube.com/watch?v=abc",
            "https://t.me/somechannel",  # channel root, no message
            "https://t.me/",
            "ftp://t.me/chan/1",
            "file:///etc/passwd",
            "not a url at all",
            "https://nott.me/chan/1",  # must not match on a suffix
            "https://t.me.evil.com/chan/1",  # nor on a prefix
        ],
    )
    def test_returns_none(self, url):
        assert parse_telegram_url(url) is None


class TestVideoSize:
    def test_none_when_no_media(self):
        assert _video_size(object()) is None

    def test_none_for_photo_media(self):
        class Msg:
            media = object()  # not a MessageMediaDocument

        assert _video_size(Msg()) is None

    def test_reads_size_from_a_video_document(self):
        msg = _video_message(size=5 * 1024 * 1024)
        assert _video_size(msg) == 5 * 1024 * 1024

    def test_none_for_non_video_document(self):
        msg = _video_message(size=1234, mime="application/pdf", video_attr=False)
        assert _video_size(msg) is None


class TestSessionGuard:
    def test_missing_session_is_reported_not_attempted(self, tmp_path: Path):
        """Without a session we must fail clearly, never try an interactive login."""
        ref = TelegramRef(message_id=1, username="chan")
        with pytest.raises(TelegramUnavailableError, match="No Telegram session"):
            asyncio.run(
                download_telegram_video(
                    ref,
                    tmp_path,
                    api_id=1,
                    api_hash="x",
                    session_path=str(tmp_path / "absent"),
                    max_bytes=1024,
                )
            )


class TestNotDownloadableIsDistinctFromFailure:
    def test_exception_types_are_separate(self):
        # The API maps these to different statuses: 422 vs 503.
        assert not issubclass(TelegramNotDownloadableError, TelegramUnavailableError)
        assert not issubclass(TelegramUnavailableError, TelegramNotDownloadableError)


class TestCheckDownloadable:
    """The refusal rules, which decide whether bytes are fetched at all."""

    def test_accepts_an_ordinary_video(self):
        msg = _video_message(size=5 * 1024 * 1024)
        assert check_downloadable(msg, max_bytes=512 * 1024 * 1024) == 5 * 1024 * 1024

    def test_missing_message(self):
        with pytest.raises(TelegramNotDownloadableError, match="does not exist"):
            check_downloadable(None, max_bytes=1024)

    def test_post_without_video(self):
        msg = _video_message(size=10, mime="image/jpeg", video_attr=False)
        with pytest.raises(TelegramNotDownloadableError, match="no video"):
            check_downloadable(msg, max_bytes=1024)

    def test_over_size_limit(self):
        msg = _video_message(size=600 * 1024 * 1024)
        with pytest.raises(TelegramNotDownloadableError, match="over the"):
            check_downloadable(msg, max_bytes=512 * 1024 * 1024)

    def test_protected_content_fails_closed_on_the_message(self):
        msg = _video_message(size=1024, noforwards=True)
        with pytest.raises(TelegramNotDownloadableError, match="prohibits saving"):
            check_downloadable(msg, max_bytes=512 * 1024 * 1024)

    def test_protected_content_fails_closed_on_the_chat(self):
        class Chat:
            noforwards = True

        msg = _video_message(size=1024, chat=Chat())
        with pytest.raises(TelegramNotDownloadableError, match="prohibits saving"):
            check_downloadable(msg, max_bytes=512 * 1024 * 1024)

    def test_protection_is_checked_before_size(self):
        """A protected video must be refused as protected, not as oversized."""
        msg = _video_message(size=600 * 1024 * 1024, noforwards=True)
        with pytest.raises(TelegramNotDownloadableError, match="prohibits saving"):
            check_downloadable(msg, max_bytes=512 * 1024 * 1024)
