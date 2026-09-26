#!/usr/bin/env python3
"""Locate archive media for tagged episodes that have no CMS card.

`library_untitled_titles.py` gave these episodes a name; this gives them a
destination. An episode with a CMS article links to its page on the site, so
in the Library every card is clickable except these 1,727, which render as
dead text -- an editor can read what the episode is but cannot open it
(86cbgqr9v, Varya 2026-09-11).

There is no page to link to, but the media is there: the archive path falls
out of the video hash alone, no CMS lookup involved. This records the paths
the UI needs, because `ui_library.py` is deliberately file-backed and reaches
neither the API nor the archive itself.

    scripts/library_untitled_media.py            # write data/untitled_media.json
    scripts/library_untitled_media.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# The order `rainrag.api.find_video_file` serves, so the link opens the same
# file the player would. One preference, in one place, or the two drift.
QUALITIES = ("1080p", "720p", "480p", "360p", "180p")
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm")


def pick_video(files: list[Path], video_hash: str) -> Path | None:
    """The file `find_video_file` would serve for this hash, or None."""
    videos = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS]
    if not videos:
        return None
    for quality in QUALITIES:
        for f in videos:
            if f.stem == f"{video_hash}_{quality}":
                return f
    for f in videos:
        if f.stem == video_hash:
            return f
    # A hash-prefixed name with an unknown suffix still beats no link at all.
    named = sorted(f for f in videos if f.name.startswith(video_hash))
    return named[0] if named else sorted(videos)[0]


def pick_vtt(files: list[Path], video_hash: str) -> Path | None:
    """The transcript to offer, Russian first, as the archive routes prefer."""
    for suffix in (".ru.vtt", ".en.vtt"):
        for f in files:
            if f.name == f"{video_hash}{suffix}":
                return f
    vtts = sorted(f for f in files if f.name.endswith(".vtt"))
    return vtts[0] if vtts else None


def media_for(archive_root: Path, video_hash: str) -> dict[str, str]:
    """{"video": rel, "vtt": rel} for one hash; keys absent when the file is not there."""
    from rainrag.ingest import WebMetadataLoader

    try:
        shard = WebMetadataLoader.hash_to_archive_dir(video_hash)
    except ValueError:
        return {}
    directory = archive_root / shard
    if not directory.is_dir():
        return {}
    try:
        files = [f for f in directory.iterdir() if f.is_file()]
    except OSError:
        return {}

    found: dict[str, str] = {}
    video = pick_video(files, video_hash)
    if video is not None:
        found["video"] = str(shard / video.name)
    vtt = pick_vtt(files, video_hash)
    if vtt is not None:
        found["vtt"] = str(shard / vtt.name)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", default=str(REPO_ROOT / "data" / "library_tags.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "untitled_media.json"))
    parser.add_argument("--archive-root", default="/mnt/vod/srv/storage/transcoded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    from library_untitled_titles import untitled_hashes

    untitled = untitled_hashes(Path(args.tags).read_text(encoding="utf-8").splitlines())

    root = Path(args.archive_root)
    out: dict[str, dict[str, str]] = {}
    for h in untitled:
        found = media_for(root, h)
        if found:
            out[h] = found

    playable = sum(1 for v in out.values() if "video" in v)
    transcript_only = sum(1 for v in out.values() if "video" not in v and "vtt" in v)
    print(
        f"untitled tagged episodes: {len(untitled)}, "
        f"playable: {playable}, transcript only: {transcript_only}, "
        f"nothing in archive: {len(untitled) - len(out)}"
    )
    if args.dry_run:
        return 0
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
