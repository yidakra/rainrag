"""Tests for the optional yt-dlp cookie file.

A missing or unreadable cookies file must never take the endpoint down -- it
should degrade the sites that need cookies and leave everything else working.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from rainrag.api import _yt_dlp_cookiefile


def _cfg(path: str = "") -> SimpleNamespace:
    return SimpleNamespace(yt_dlp_cookies_path=path)


class TestCookieFileResolution:
    def test_unset_returns_none(self):
        assert _yt_dlp_cookiefile(_cfg("")) is None

    def test_whitespace_only_is_treated_as_unset(self):
        assert _yt_dlp_cookiefile(_cfg("   ")) is None

    def test_absent_attribute_is_tolerated(self):
        # Config objects predating this field must not raise.
        assert _yt_dlp_cookiefile(SimpleNamespace()) is None

    def test_existing_file_is_returned(self, tmp_path: Path):
        f = tmp_path / "cookies.txt"
        f.write_text("# Netscape HTTP Cookie File\n")
        assert _yt_dlp_cookiefile(_cfg(str(f))) == str(f)

    def test_missing_file_is_ignored_not_fatal(self, tmp_path: Path):
        assert _yt_dlp_cookiefile(_cfg(str(tmp_path / "absent.txt"))) is None

    def test_directory_is_not_accepted(self, tmp_path: Path):
        assert _yt_dlp_cookiefile(_cfg(str(tmp_path))) is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission checks")
    def test_unreadable_file_is_ignored(self, tmp_path: Path):
        f = tmp_path / "cookies.txt"
        f.write_text("# Netscape HTTP Cookie File\n")
        f.chmod(0o000)
        try:
            assert _yt_dlp_cookiefile(_cfg(str(f))) is None
        finally:
            f.chmod(0o600)
