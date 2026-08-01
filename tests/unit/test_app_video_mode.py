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


class TestSessionContextRendering:
    """Uploaded-video fragments must not reuse the archive media widgets.

    Regression: session transcripts live outside the archive root, so the
    archive renderer produced ``/video/<absolute path>`` and
    ``/vtt/<absolute path>`` URLs that the archive routes reject with 400 --
    surfacing as a dead player and "Не удалось загрузить VTT".
    """

    def test_message_bubble_accepts_a_session_key(self) -> None:
        import inspect

        signature = inspect.signature(app.render_message_bubble)

        assert "video_session_key" in signature.parameters
        assert signature.parameters["video_session_key"].default is None

    def test_fragment_start_time_yields_a_seek_target(self) -> None:
        """The seek button needs seconds parsed out of the chunk's start_time."""
        assert app.extract_timecodes("00:02:01", limit=1) == [("00:02:01", 121)]

    def test_fragment_without_start_time_has_no_seek_target(self) -> None:
        assert app.extract_timecodes("", limit=1) == []


class TestSessionPlayerSizing:
    """The frame must fit any aspect ratio without cropping or overlapping.

    Two regressions here. First a fixed 420px frame clipped the bottom of a
    3420x2224 screen recording, hiding the controls. Then resizing the frame
    from inside the iframe made it overflow a wrapper that did not reflow, so
    the caption, the ready banner and the download button were drawn on top of
    the video. The height is therefore decided server-side, before render.
    """

    def test_tall_video_gets_more_height_than_widescreen(self) -> None:
        widescreen = app.player_frame_height(1920, 1080)
        screen_recording = app.player_frame_height(3420, 2224)

        assert screen_recording > widescreen

    def test_unknown_dimensions_fall_back_to_widescreen(self) -> None:
        assert app.player_frame_height(None, None) == app.player_frame_height(1920, 1080)

    @pytest.mark.parametrize(
        ("width", "height"),
        [(1080, 1920), (640, 480), (3840, 1080), (0, 0), (None, 720), (-4, 8)],
    )
    def test_height_stays_within_bounds(self, width, height) -> None:
        """A portrait phone clip must not push the chat off screen."""
        result = app.player_frame_height(width, height)

        assert app._PLAYER_MIN_HEIGHT <= result <= app._PLAYER_MAX_HEIGHT

    def test_frame_is_not_resized_from_inside(self) -> None:
        """Growing the iframe in JS is what made the page elements overlap."""
        rendered = {}

        def _capture(block, height=None):
            rendered["html"] = block
            rendered["height"] = height

        original = app.components.html
        app.components.html = _capture
        try:
            app.render_video_session_player("/media", "vid1", start_seconds=12.0, height=500)
        finally:
            app.components.html = original

        block = rendered["html"]
        assert "frameElement" not in block
        assert "style.height" not in block
        # The caller's height is honoured, and the video scales to fit inside it.
        assert rendered["height"] == 500
        assert "max-height: 100%" in block
        assert "12.0" in block
