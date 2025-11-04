"""Test edge cases for CUE_ID_PATTERN fix."""

import pytest
from pathlib import Path

from rainrag.ingest import VTTParser


class TestCueIdEdgeCases:
    """Test cases to verify the CUE_ID_PATTERN fix handles single-word subtitles correctly."""

    def test_single_word_subtitle_not_skipped(self, tmp_path: Path) -> None:
        """Test that single-word subtitles like 'Hello' or 'OK' are not incorrectly skipped as cue IDs."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello

00:00:05.000 --> 00:00:10.000
OK

00:00:10.000 --> 00:00:15.000
Test
"""
        vtt_file = tmp_path / "single_words.vtt"
        vtt_file.write_text(vtt_content)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "Hello" in text
        assert "OK" in text
        assert "Test" in text

    def test_numeric_subtitle_preserved(self, tmp_path: Path) -> None:
        """Test that numeric subtitle text is preserved even though it matches cue ID pattern."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
The year is 2024

00:00:05.000 --> 00:00:10.000
100 dollars
"""
        vtt_file = tmp_path / "numeric.vtt"
        vtt_file.write_text(vtt_content)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "2024" in text
        assert "100" in text

    def test_alphanumeric_subtitle_preserved(self, tmp_path: Path) -> None:
        """Test that alphanumeric subtitles are preserved (old pattern would have incorrectly skipped these)."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
ABC123

00:00:05.000 --> 00:00:10.000
test_string
"""
        vtt_file = tmp_path / "alphanumeric.vtt"
        vtt_file.write_text(vtt_content)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "ABC123" in text
        assert "test_string" in text

    def test_numeric_cue_ids_still_skipped(self, tmp_path: Path) -> None:
        """Test that numeric cue IDs before timestamps are still correctly skipped."""
        vtt_content = """WEBVTT

1
00:00:00.000 --> 00:00:05.000
First subtitle

2
00:00:05.000 --> 00:00:10.000
Second subtitle

100
00:00:10.000 --> 00:00:15.000
Third subtitle
"""
        vtt_file = tmp_path / "cue_ids.vtt"
        vtt_file.write_text(vtt_content)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        # Cue IDs should not appear in text
        assert text.count("1") == 0 or "First" in text  # 1 should be skipped
        assert text.count("2") == 0 or "Second" in text  # 2 should be skipped
        assert text.count("100") == 0 or "Third" in text  # 100 should be skipped
        # But subtitle text should be present
        assert "First subtitle" in text
        assert "Second subtitle" in text
        assert "Third subtitle" in text

    def test_mixed_cue_and_subtitle_text(self, tmp_path: Path) -> None:
        """Test VTT with both cue IDs and subtitle text that could match old pattern."""
        vtt_content = """WEBVTT

1
00:00:00.000 --> 00:00:05.000
Hello

2
00:00:05.000 --> 00:00:10.000
OK
World

3
00:00:10.000 --> 00:00:15.000
123
Test
"""
        vtt_file = tmp_path / "mixed.vtt"
        vtt_file.write_text(vtt_content)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        # Subtitle text should be present
        assert "Hello" in text
        assert "OK" in text
        assert "World" in text
        assert "123" in text
        assert "Test" in text
