import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import find_sheet_link_hashes as sheet_script


def _credential_args(**overrides: object) -> SimpleNamespace:
    defaults = {
        "google_access_token": None,
        "google_access_token_env": "GOOGLE_ACCESS_TOKEN",
        "service_account_file": None,
        "service_account_env": "GOOGLE_APPLICATION_CREDENTIALS",
        "upload_copy_folder_to_drive": True,
    }
    return SimpleNamespace(**{**defaults, **overrides})


def test_explicit_access_token_does_not_use_service_account(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/secrets/service-account.json")
    monkeypatch.setattr(
        sheet_script,
        "_mint_access_token_from_service_account",
        lambda *_args, **_kwargs: pytest.fail("service account must not be used"),
    )

    token, using_service_account = sheet_script._resolve_google_credentials(
        _credential_args(google_access_token="oauth-token"),  # type: ignore[arg-type]
        want_write=True,
    )

    assert token == "oauth-token"
    assert using_service_account is False


def test_service_account_token_reports_credential_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GOOGLE_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(
        sheet_script,
        "_mint_access_token_from_service_account",
        lambda path, **_kwargs: f"token-for:{path}",
    )

    token, using_service_account = sheet_script._resolve_google_credentials(
        _credential_args(service_account_file="/secrets/service-account.json"),  # type: ignore[arg-type]
        want_write=True,
    )

    assert token == "token-for:/secrets/service-account.json"
    assert using_service_account is True


def test_empty_drive_upload_is_rejected_before_folder_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sheet_script,
        "_create_drive_folder",
        lambda **_kwargs: pytest.fail("Drive folder must not be created"),
    )

    with pytest.raises(RuntimeError, match="empty Drive folder"):
        sheet_script._upload_folder_to_drive(
            access_token="test-token",
            local_dir=tmp_path,
            folder_name="fast_subtitles_2026-08-03",
        )


def test_unavailable_translation_source_produces_partial_failure(
    tmp_path: Path,
) -> None:
    video_hash = "ab" * 20

    staged, skipped = sheet_script._prepare_translation_staging(
        archive_root=tmp_path / "archive",
        hashes=[video_hash],
        staging_input=tmp_path / "staging",
    )

    assert staged == {}
    assert skipped == 1
    assert sheet_script._translation_exit_code(skipped) == 2


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
