"""Tests for archive media URLs and the untitled-episode media precompute."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

REL = "0c/14/22/9e/fc/ba/43/6a/4c/22/f8/d9/6f/67/e0/cb/93/bc/90/76/abc_720p.mp4"


def test_no_asset_base_means_no_link(monkeypatch):
    """A box without media configured shows plain text, it does not guess a host."""
    from rainrag.media_links import archive_media_url

    monkeypatch.delenv("RAINRAG_ASSET_URL", raising=False)
    assert archive_media_url(REL) is None


def test_url_joins_base_and_kind(monkeypatch):
    from rainrag.media_links import archive_media_url

    monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.tvrain.tv/")
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    assert archive_media_url(REL) == f"https://rag.tvrain.tv/video/{REL}"
    assert archive_media_url("a/b.ru.vtt", kind="vtt") == "https://rag.tvrain.tv/vtt/a/b.ru.vtt"


def test_unknown_kind_is_rejected(monkeypatch):
    """A typo'd kind must not silently produce a 404 URL."""
    from rainrag.media_links import archive_media_url

    monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.tvrain.tv")
    with pytest.raises(ValueError):
        archive_media_url(REL, kind="videos")


def test_path_separators_survive_encoding(monkeypatch):
    """Segments are quoted, the slashes between them are structure and stay."""
    from rainrag.media_links import archive_media_url

    monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.tvrain.tv")
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    url = archive_media_url("aa/bb/файл имя.mp4")
    assert url.startswith("https://rag.tvrain.tv/video/aa/bb/")
    assert "%20" in url and "/aa/bb/" in url


def test_token_is_accepted_by_the_api_that_verifies_it(monkeypatch):
    """The whole feature rests on this: a link we mint must open.

    api.py owns the format and does the checking; a drift here would hand
    editors links that 401 rather than play.
    """
    from rainrag.api import _media_token_is_valid
    from rainrag.media_links import issue_media_token

    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "shared-secret")
    assert _media_token_is_valid(issue_media_token(), "shared-secret")
    assert not _media_token_is_valid(issue_media_token(), "another-secret")


def test_auth_is_appended_without_losing_the_query(monkeypatch):
    from rainrag.media_links import append_auth_query

    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s")
    url = append_auth_query("https://h/video/x.mp4?q=1#t=42")
    assert "q=1" in url and "auth=v1." in url and url.endswith("#t=42")


def test_no_auth_param_when_auth_is_disabled(monkeypatch):
    from rainrag.media_links import append_auth_query, issue_media_token

    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    assert issue_media_token() == ""
    assert append_auth_query("https://h/video/x.mp4") == "https://h/video/x.mp4"


# --- the precompute script -------------------------------------------------

HASH = "a" * 40


def _archive(tmp_path: Path, names: list[str]) -> Path:
    shard = tmp_path.joinpath(*[HASH[i : i + 2] for i in range(0, 40, 2)])
    shard.mkdir(parents=True)
    for n in names:
        (shard / n).write_bytes(b"x")
    return tmp_path


EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov")
VTT = (".vtt", ".en.vtt", ".ru.vtt")


def test_video_extensions_come_from_the_config_the_api_reads(tmp_path: Path):
    """Hardcoding them drifts from `find_video_file` the moment either moves."""
    from library_untitled_media import video_extensions

    # No config to read: the schema default, which carries .avi and .mov.
    fallback = video_extensions(str(tmp_path / "absent.yaml"))
    assert ".avi" in fallback and ".mov" in fallback and ".mp4" in fallback

    # The repo's own config wins when it is there.
    from rainrag.config import load_config

    repo_config = REPO_ROOT / "config.yaml"
    assert video_extensions(str(repo_config)) == tuple(
        load_config(str(repo_config)).video.extensions
    )


def test_pick_video_follows_the_quality_order_the_api_serves():
    from library_untitled_media import pick_video

    files = [Path(f"{HASH}_{q}.mp4") for q in ("360p", "1080p", "720p")]
    assert pick_video(files, HASH, EXT).stem == f"{HASH}_1080p"
    assert pick_video([Path(f"{HASH}_180p.mp4")], HASH, EXT).stem == f"{HASH}_180p"
    assert pick_video([Path("notes.txt")], HASH, EXT) is None


def test_pick_video_serves_an_extension_the_config_allows():
    """A deployment that adds .avi must not leave those episodes dead cards."""
    from library_untitled_media import pick_video

    files = [Path(f"{HASH}_480p.avi")]
    assert pick_video(files, HASH, EXT).name == f"{HASH}_480p.avi"
    assert pick_video(files, HASH, (".mp4", ".mkv", ".webm")) is None


def test_pick_video_never_returns_another_episode(tmp_path: Path):
    """A stray transcode in the shard must not become this episode's link.

    `find_video_file` requires the hash prefix at every stage; linking an
    editor to the wrong video is worse than linking her nowhere.
    """
    from library_untitled_media import pick_video

    stranger = [Path("b" * 40 + "_1080p.mp4")]
    assert pick_video(stranger, HASH, EXT) is None
    both = stranger + [Path(f"{HASH}_360p.mp4")]
    assert pick_video(both, HASH, EXT).stem == f"{HASH}_360p"


def test_pick_vtt_prefers_russian():
    from library_untitled_media import pick_vtt

    files = [Path(f"{HASH}.en.vtt"), Path(f"{HASH}.ru.vtt")]
    assert pick_vtt(files, HASH, VTT).name == f"{HASH}.ru.vtt"
    assert pick_vtt([Path(f"{HASH}.en.vtt")], HASH, VTT).name == f"{HASH}.en.vtt"
    assert pick_vtt([Path("other.mp4")], HASH, VTT) is None


def test_pick_vtt_never_returns_another_episodes_transcript():
    from library_untitled_media import pick_vtt

    assert pick_vtt([Path("b" * 40 + ".ru.vtt")], HASH, VTT) is None


def test_media_for_reports_paths_relative_to_the_archive_root(tmp_path: Path):
    from library_untitled_media import media_for

    root = _archive(tmp_path, [f"{HASH}_720p.mp4", f"{HASH}.ru.vtt"])
    found = media_for(root, HASH)
    assert found["video"].endswith(f"{HASH}_720p.mp4")
    assert found["vtt"].endswith(f"{HASH}.ru.vtt")
    # Relative: joining onto the root must land back on a real file.
    assert (root / found["video"]).exists()
    assert not Path(found["video"]).is_absolute()


def test_media_for_tolerates_transcript_only_and_missing_shards(tmp_path: Path):
    """Ingest is subtitle-driven, so some hashes have a VTT and no video."""
    from library_untitled_media import media_for

    root = _archive(tmp_path, [f"{HASH}.ru.vtt"])
    found = media_for(root, HASH)
    assert "video" not in found and "vtt" in found
    assert media_for(root, "b" * 40) == {}
    assert media_for(root, "not-a-hash") == {}


def test_script_writes_only_episodes_with_media(tmp_path: Path, capsys):
    from library_untitled_media import main

    root = _archive(tmp_path / "arc", [f"{HASH}_480p.mp4"])
    tags = tmp_path / "tags.jsonl"
    tags.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"video_hash": HASH},  # untitled, has media
                {"video_hash": "c" * 40},  # untitled, nothing in archive
                {"video_hash": "d" * 40, "title": "Есть карточка"},  # has a CMS title
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "untitled_media.json"
    assert main(["--tags", str(tags), "--out", str(out), "--archive-root", str(root)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert list(written) == [HASH]
    assert "playable: 1" in capsys.readouterr().out


def test_bad_ttl_env_does_not_take_the_page_down(monkeypatch):
    """A typo in .env should cost the default TTL, not the whole Library."""
    from rainrag.media_links import _DEFAULT_TTL_SECONDS, _ttl_seconds, issue_media_token

    monkeypatch.setenv("RAINRAG_MEDIA_TOKEN_TTL_SECONDS", "twelve hours")
    assert _ttl_seconds() == _DEFAULT_TTL_SECONDS
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s")
    assert issue_media_token().startswith("v1.")

    monkeypatch.setenv("RAINRAG_MEDIA_TOKEN_TTL_SECONDS", "60")
    assert _ttl_seconds() == 60
    monkeypatch.delenv("RAINRAG_MEDIA_TOKEN_TTL_SECONDS")
    assert _ttl_seconds() == _DEFAULT_TTL_SECONDS


def test_pick_vtt_skips_suffixes_the_route_would_reject():
    """`serve_vtt` 400s on anything outside config.video.vtt_extensions."""
    from library_untitled_media import pick_vtt

    files = [Path(f"{HASH}.srt"), Path(f"{HASH}.de.vtt")]
    assert pick_vtt(files, HASH, (".ru.vtt", ".en.vtt")) is None
    assert pick_vtt(files, HASH, (".vtt",)).name == f"{HASH}.de.vtt"


def test_vtt_extensions_come_from_the_same_config(tmp_path: Path):
    from library_untitled_media import vtt_extensions

    assert ".ru.vtt" in vtt_extensions(str(tmp_path / "absent.yaml"))


def test_no_token_in_a_plaintext_link(monkeypatch):
    """A bearer credential in an http URL is readable and replayable.

    Fail closed: no link at all, which degrades to the plain text this
    feature replaced, rather than leaking the token.
    """
    from rainrag.media_links import archive_media_url

    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s")
    monkeypatch.setenv("RAINRAG_ASSET_URL", "http://rag.tvrain.tv")
    assert archive_media_url(REL) is None

    monkeypatch.setenv("RAINRAG_ASSET_URL", "https://rag.tvrain.tv")
    assert archive_media_url(REL).startswith("https://rag.tvrain.tv/video/")


def test_plaintext_is_fine_without_a_token_and_on_loopback(monkeypatch):
    from rainrag.media_links import archive_media_url

    # Auth disabled: nothing secret travels, so http is not a leak.
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("RAINRAG_ASSET_URL", "http://rag.tvrain.tv")
    assert archive_media_url(REL) == f"http://rag.tvrain.tv/video/{REL}"

    # Loopback never leaves the machine, so a dev box with auth still works.
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "s")
    monkeypatch.setenv("RAINRAG_ASSET_URL", "http://localhost:8001")
    assert "auth=v1." in archive_media_url(REL)
