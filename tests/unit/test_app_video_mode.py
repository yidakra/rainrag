"""Tests for the Streamlit helpers behind the single-video upload mode."""

from __future__ import annotations

import pytest

import app


class TestExtractTimecodes:
    def test_extracts_hms_and_ms_forms_in_order(self) -> None:
        text = "Об этом говорится в 00:12:34, затем в 1:02:05."

        assert app.extract_timecodes(text) == [("00:12:34", 754), ("1:02:05", 3725)]

    def test_deduplicates_equivalent_timecodes(self) -> None:
        """12:34 and 00:12:34 point at the same moment; offer one button."""
        assert app.extract_timecodes("00:12:34 ... 12:34") == [("00:12:34", 754)]

    def test_ignores_text_without_timecodes(self) -> None:
        assert app.extract_timecodes("Опубликовано 2026-07-30, без таймкодов.") == []

    def test_respects_the_limit(self) -> None:
        text = " ".join(f"00:0{i}:00" for i in range(1, 8))

        assert len(app.extract_timecodes(text, limit=3)) == 3

    def test_rejects_out_of_range_minutes_and_seconds(self) -> None:
        assert app.extract_timecodes("99:99") == []


class TestLanguageDisplayName:
    @pytest.mark.parametrize(
        ("code", "ui_lang", "expected"),
        [("ru", "ru", "русский"), ("ru", "en", "Russian"), ("uk", "ru", "украинский")],
    )
    def test_known_codes_are_localised(self, code: str, ui_lang: str, expected: str) -> None:
        assert app.language_display_name(code, ui_lang) == expected

    def test_unknown_code_falls_back_to_the_code(self) -> None:
        """Whisper reports ~100 languages; unknown ones must still render."""
        assert app.language_display_name("zz", "en") == "ZZ"
