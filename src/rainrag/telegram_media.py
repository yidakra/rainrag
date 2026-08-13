"""Download Telegram videos over MTProto.

yt-dlp reaches Telegram by scraping the ``t.me`` embed page for a ``<video src>``
element.  Telegram frequently omits that element and renders the player as
``not_supported`` instead, so on a live sample only about a third of ordinary
public-channel video posts were retrievable that way.  The gate is server-side:
it is not a size threshold that can be pre-checked, and a browser User-Agent
does not change the response.

MTProto asks Telegram for the message directly, so it does not depend on what
the web embed chooses to inline.  It needs an ``api_id``/``api_hash`` pair and a
**user** session -- a bot session cannot read a channel it was never added to,
and the Bot API caps downloads at 20 MB regardless.

Downloads are serialised on purpose.  Telethon's own guidance is that running
them in parallel only makes ``FloodWait`` arrive sooner.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


# One download at a time per process: concurrency here buys nothing and
# provokes FloodWait.  Created lazily so importing this module does not need a
# running event loop.
_download_lock: asyncio.Lock | None = None


def _get_download_lock() -> asyncio.Lock:
    global _download_lock
    if _download_lock is None:
        _download_lock = asyncio.Lock()
    return _download_lock


class TelegramUnavailableError(RuntimeError):
    """Telegram downloading is not configured or the library is missing."""


class TelegramNotDownloadableError(RuntimeError):
    """The link resolves, but no video can legitimately be taken from it.

    Covers a post with no video, a message that does not exist, and content the
    channel owner marked as protected.
    """


@dataclass(frozen=True)
class TelegramRef:
    """A parsed ``t.me`` link.

    Exactly one of ``username``, ``internal_id`` or ``invite_hash`` identifies
    the peer.  ``message_id`` is the message to fetch.
    """

    message_id: int
    username: str | None = None
    internal_id: int | None = None
    invite_hash: str | None = None

    @property
    def is_private(self) -> bool:
        """True when the peer is not publicly resolvable by username."""
        return self.username is None

    def describe(self) -> str:
        """A log-safe description that never includes an invite hash."""
        if self.username:
            return f"@{self.username}/{self.message_id}"
        if self.internal_id is not None:
            return f"private:{self.internal_id}/{self.message_id}"
        return f"invite:<redacted>/{self.message_id}"


# t.me/c/<internal>/<msg> and t.me/c/<internal>/<topic>/<msg> -- private channels.
_RE_PRIVATE = re.compile(r"^/c/(?P<internal>\d+)(?:/(?P<a>\d+))?(?:/(?P<b>\d+))?/?$")
# t.me/+<hash> or t.me/joinchat/<hash> -- invite links, optionally with a message.
_RE_INVITE = re.compile(r"^/(?:\+|joinchat/)(?P<hash>[A-Za-z0-9_-]+)(?:/(?P<msg>\d+))?/?$")
# t.me/<username>/<msg>, t.me/<username>/<topic>/<msg>, t.me/s/<username>/<msg>.
_RE_PUBLIC = re.compile(
    r"^/(?:s/)?(?P<user>[A-Za-z][A-Za-z0-9_]{3,31})(?:/(?P<a>\d+))?(?:/(?P<b>\d+))?/?$"
)


def parse_telegram_url(url: str) -> TelegramRef | None:
    """Parse a ``t.me`` URL into a :class:`TelegramRef`, or None if not Telegram.

    Deliberately explicit about the private and invite forms.  yt-dlp's pattern
    (``t\\.me/(?P<channel_id>[^/]+)/(?P<id>\\d+)``) is unanchored and silently
    mis-parses ``t.me/c/123/456`` as channel ``c`` message ``123``, then fetches
    a nonsense embed URL.  Topic links carry an extra path segment and the
    *last* number is the message id.
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    if parts.scheme not in ("http", "https"):
        return None
    host = (parts.hostname or "").lower()
    if host not in ("t.me", "telegram.me", "telegram.dog", "www.t.me"):
        return None

    path = parts.path if parts.path.startswith("/") else "/" + parts.path

    if m := _RE_PRIVATE.match(path):
        msg = m.group("b") or m.group("a")
        if not msg:
            return None
        return TelegramRef(message_id=int(msg), internal_id=int(m.group("internal")))

    if m := _RE_INVITE.match(path):
        msg = m.group("msg")
        return TelegramRef(message_id=int(msg) if msg else 0, invite_hash=m.group("hash"))

    if m := _RE_PUBLIC.match(path):
        msg = m.group("b") or m.group("a")
        if not msg:
            return None
        return TelegramRef(message_id=int(msg), username=m.group("user"))

    return None


def _video_size(message: Any) -> int | None:
    """Return the byte size of the message's video, or None if it has no video."""
    from telethon.tl.types import DocumentAttributeVideo, MessageMediaDocument

    media = getattr(message, "media", None)
    if not isinstance(media, MessageMediaDocument):
        return None
    doc = getattr(media, "document", None)
    if doc is None:
        return None
    attrs = getattr(doc, "attributes", None) or []
    mime = (getattr(doc, "mime_type", "") or "").lower()
    is_video = mime.startswith("video/") or any(
        isinstance(a, DocumentAttributeVideo) for a in attrs
    )
    if not is_video:
        return None
    return int(getattr(doc, "size", 0) or 0)


def check_downloadable(message: Any, max_bytes: int) -> int:
    """Return the video's size, or raise :class:`TelegramNotDownloadableError`.

    Split out from the download so the refusal rules can be tested without a
    Telegram session.
    """
    if message is None:
        raise TelegramNotDownloadableError("That Telegram message does not exist.")

    # Honour the channel owner's content protection.  Telegram's spec says
    # receiving clients must disable downloads for these messages; an MTProto
    # library will hand over the bytes regardless, which is exactly why
    # refusing is a policy decision rather than a technical limit.
    if getattr(message, "noforwards", False) or getattr(
        getattr(message, "chat", None), "noforwards", False
    ):
        raise TelegramNotDownloadableError(
            "This channel prohibits saving its content, so it will not be downloaded."
        )

    size = _video_size(message)
    if size is None:
        raise TelegramNotDownloadableError("That Telegram post contains no video.")
    if size > max_bytes:
        raise TelegramNotDownloadableError(
            f"Video is {size // (1024 * 1024)} MB, over the {max_bytes // (1024 * 1024)} MB limit."
        )
    return size


async def _resolve_peer(client: Any, ref: TelegramRef) -> Any:
    """Resolve a :class:`TelegramRef` to a Telethon entity."""
    from telethon.tl.functions.messages import CheckChatInviteRequest

    if ref.username:
        # Usernames resolve server-side with no prior contact with the channel.
        return await client.get_entity(ref.username)

    if ref.invite_hash:
        # An invite link must be checked before the chat can be used, and it
        # only works if this account has already joined.
        invite = await client(CheckChatInviteRequest(ref.invite_hash))
        chat = getattr(invite, "chat", None)
        if chat is None:
            raise TelegramNotDownloadableError("This invite link requires joining the chat first.")
        return chat

    # Private channels use the -100<internal> form and require membership; the
    # access hash cannot be resolved cold.
    return await client.get_entity(int(f"-100{ref.internal_id}"))


async def download_telegram_video(
    ref: TelegramRef,
    dest_dir: Path,
    *,
    api_id: int,
    api_hash: str,
    session_path: str,
    max_bytes: int,
    flood_sleep_threshold: int = 60,
) -> Path:
    """Download the video from ``ref`` into ``dest_dir`` and return its path.

    Raises :class:`TelegramNotDownloadableError` when the message has no video, does
    not exist, or is marked as protected content.
    """
    try:
        from telethon import TelegramClient
        from telethon.errors import FileReferenceExpiredError, FloodWaitError
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise TelegramUnavailableError("telethon is not installed") from exc

    if not Path(session_path + ".session").exists() and not Path(session_path).exists():
        raise TelegramUnavailableError(
            f"No Telegram session at {session_path!r}; run scripts/telegram_login.py first"
        )

    async with _get_download_lock():
        client = TelegramClient(
            session_path, api_id, api_hash, flood_sleep_threshold=flood_sleep_threshold
        )
        await client.start()  # never prompts: the session already exists
        try:
            if not await client.is_user_authorized():
                raise TelegramUnavailableError(
                    "Telegram session exists but is not authorised; re-run the login script"
                )

            peer = await _resolve_peer(client, ref)

            async def fetch_message() -> Any:
                msg = await client.get_messages(peer, ids=ref.message_id)
                if isinstance(msg, list):
                    msg = next((m for m in msg if m is not None), None)
                return msg

            message = await fetch_message()
            size = check_downloadable(message, max_bytes)

            dest = dest_dir / "telegram_video"
            logger.info(
                "Downloading Telegram video {} ({} MB)", ref.describe(), size // (1024 * 1024)
            )
            for attempt in (1, 2):
                try:
                    out = await client.download_media(message, file=str(dest))
                    break
                except FileReferenceExpiredError:
                    # File references are short-lived and must never be cached;
                    # refetching the message mints a fresh one.
                    if attempt == 2:
                        raise
                    logger.info("Telegram file reference expired, refetching message")
                    message = await fetch_message()
                    if message is None:
                        raise TelegramNotDownloadableError(
                            "That Telegram message disappeared mid-download."
                        ) from None
                except FloodWaitError as exc:
                    # flood_sleep_threshold already absorbs short waits, so
                    # reaching here means Telegram asked for a long one.
                    raise TelegramNotDownloadableError(
                        f"Telegram rate-limited this download; retry in {exc.seconds}s."
                    ) from exc

            if not out:
                raise TelegramNotDownloadableError("Telegram returned no file for that message.")
            return Path(out)
        finally:
            await client.disconnect()
