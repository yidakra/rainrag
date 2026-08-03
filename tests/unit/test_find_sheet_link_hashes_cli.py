import sys

import pytest

from scripts import find_sheet_link_hashes as sheet_script


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--openai-chunk-seconds", "0", "must be positive"),
        ("--openai-chunk-seconds", "-1", "must be positive"),
        ("--openai-silence-window-seconds", "-1", "cannot be negative"),
    ],
)
def test_invalid_openai_chunk_settings_fail_before_sheet_work(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "find_sheet_link_hashes.py",
            "https://docs.google.com/spreadsheets/d/test-sheet/edit",
            option,
            value,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        sheet_script.main()

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_partial_translation_failures_have_distinct_exit_status() -> None:
    assert sheet_script._translation_exit_code(0) == 0
    assert sheet_script._translation_exit_code(1) == 2
    assert sheet_script._translation_exit_code(10) == 2
