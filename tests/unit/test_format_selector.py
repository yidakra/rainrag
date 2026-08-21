"""Tests for the yt-dlp format selector.

Two things have to hold at once: the cap must actually bite on sites that report
a height, and it must not exclude sites that report none. Coub publishes a
video-only mp4 with no dimensions, so a height filter alone matches nothing and
the download fails outright -- which is how an earlier selector change broke
Coub entirely.
"""

from __future__ import annotations

from rainrag.api import yt_dlp_format_selector


class TestCapApplied:
    def test_capped_preferences_come_first(self):
        sel = yt_dlp_format_selector(720)
        tiers = sel.split("/")
        assert "height<=720" in tiers[0]
        # Every capped tier must precede every uncapped one, or the cap is moot.
        last_capped = max(i for i, t in enumerate(tiers) if "height<=" in t)
        first_uncapped = min(i for i, t in enumerate(tiers) if "height<=" not in t)
        assert last_capped < first_uncapped

    def test_respects_a_different_height(self):
        assert "height<=480" in yt_dlp_format_selector(480)
        assert "height<=720" not in yt_dlp_format_selector(480)

    def test_prefers_mp4_and_m4a_first(self):
        first = yt_dlp_format_selector(720).split("/")[0]
        assert "ext=mp4" in first and "ext=m4a" in first


class TestFallbacksSurvive:
    """A height filter must never be the only option."""

    def test_uncapped_tiers_are_present(self):
        tiers = yt_dlp_format_selector(720).split("/")
        uncapped = [t for t in tiers if "height<=" not in t]
        assert "best" in uncapped, "a plain 'best' fallback must remain"
        assert "bestaudio" in uncapped, "audio-only must remain reachable"

    def test_split_stream_sites_stay_reachable(self):
        """Coub offers video-only mp4 plus mp3 audio and no height."""
        tiers = yt_dlp_format_selector(720).split("/")
        assert "bestvideo*+bestaudio" in tiers
        assert "bestvideo*" in tiers


class TestCapDisabled:
    def test_zero_disables_the_cap(self):
        sel = yt_dlp_format_selector(0)
        assert "height<=" not in sel
        assert sel.split("/")[0] == "bestvideo[ext=mp4]+bestaudio[ext=m4a]"

    def test_none_disables_the_cap(self):
        assert "height<=" not in yt_dlp_format_selector(None)

    def test_disabled_matches_the_previous_behaviour(self):
        """With no cap, the chain is what shipped before this change."""
        assert yt_dlp_format_selector(0) == (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/"
            "bestvideo*+bestaudio/best/bestvideo*/bestaudio"
        )
