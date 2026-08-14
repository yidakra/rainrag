#!/usr/bin/env python3
"""One-time interactive Telegram login, creating the session RainRAG reuses.

Put the credentials in ``.env`` alongside every other secret this project
uses -- the API service reads that file through systemd's ``EnvironmentFile``,
so shell exports would reach this script but *not* the running server:

    TELEGRAM_API_ID=...      # from https://my.telegram.org
    TELEGRAM_API_HASH=...

Then run this once, by hand, on the machine that will serve downloads:

    .venv/bin/python scripts/telegram_login.py

You will be asked for the phone number of the account to log in as, then the
code Telegram sends, then the two-factor password if the account has one.
Real environment variables still win over ``.env`` if you prefer to export.

The resulting session file grants **full access to that account** -- it is a
credential, not a cache. Keep it off version control and off shared storage,
and prefer a dedicated number over a personal one: server-side automation
carries a real risk of the account being limited under Telegram's terms.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_SESSION = "./data/telegram.session"


def main() -> int:
    try:
        from telethon import TelegramClient
    except ImportError:
        print("telethon is not installed. Install project dependencies first.", file=sys.stderr)
        return 1

    # Read .env so the credentials live in exactly one place -- the same file
    # systemd hands to the API service. Without this, .env would work for the
    # server and shell exports for this script, which is a good way to end up
    # with a session the server cannot use.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        pass

    api_id_raw = os.getenv("TELEGRAM_API_ID", "")
    api_hash = os.getenv("TELEGRAM_API_HASH", "")
    if not api_id_raw or not api_hash:
        print(
            "Set TELEGRAM_API_ID and TELEGRAM_API_HASH in .env "
            "(get them from https://my.telegram.org).",
            file=sys.stderr,
        )
        return 1
    try:
        api_id = int(api_id_raw)
    except ValueError:
        print("TELEGRAM_API_ID must be an integer.", file=sys.stderr)
        return 1

    session_path = os.getenv("TELEGRAM_SESSION_PATH", DEFAULT_SESSION)
    Path(session_path).parent.mkdir(parents=True, exist_ok=True)

    with TelegramClient(session_path, api_id, api_hash) as client:
        me = client.get_me()
        who = getattr(me, "username", None) or getattr(me, "phone", "unknown")
        print(f"Logged in as {who}.")

    # Telethon creates the session world-readable. It authenticates as the full
    # account, so tighten it to owner-only rather than leaving a credential
    # readable by every user on the box.
    for candidate in (Path(session_path), Path(f"{session_path}.session")):
        if candidate.exists():
            candidate.chmod(0o600)
            print(f"Session written to {candidate} (mode 600)")

    print("Set video_upload.telegram_enabled: true in config.yaml to use it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
