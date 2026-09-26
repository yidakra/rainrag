#!/usr/bin/env python3
"""Synthesise display titles for tagged episodes that have no CMS card.

27% of indexed videos (37,660 of 140,586) have no article in the CMS, so no
title and no programme. They are not junk: mostly the 2021-2022 relaunch
period and 2025-2026 output (news digests, итоги года, the ЧГК game), which
is the freshest content the Library could publish. Hiding them would hide
exactly that. Instead the first sentence of the transcript stands in as a
title, and the UI marks the card as having no CMS record.

    scripts/library_untitled_titles.py            # write data/untitled_titles.json
    scripts/library_untitled_titles.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import tempfile
from contextlib import suppress
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MAX_CHARS = 90


def write_json_atomic(path: Path, payload: object) -> None:
    """Publish a map by rename so an interrupted run keeps the previous file.

    ``write_text`` truncates in place, and these maps are now rewritten hourly
    by the incremental updater: a run killed between truncate and flush leaves
    half a map, which the Library rejects wholesale, and the good file it
    replaced is already gone. The temp file sits beside the target so the
    rename stays on one filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=0)
            # mkstemp creates 0600 and rename carries that mode onto the
            # target, so publishing would quietly make a world-readable map
            # private -- and the Library, which may well run as another user,
            # would lose every entry. Keep the mode the map already had; a
            # fresh one gets what an ordinary create would give. Set before the
            # fsync so the sync covers the new mode, not just the bytes.
            tmp.chmod(_publish_mode(path))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        _fsync_dir(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _fsync_dir(directory: Path) -> None:
    """Make the rename durable, not just the bytes it points at.

    Suppressed on failure: by this point the map is already published, and a
    filesystem that will not fsync a directory handle is not a reason to fail.
    """
    with suppress(OSError):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def listable_dir(path: Path) -> bool:
    """A root we can actually read, not merely stat.

    ``is_dir()`` is true for a directory whose contents the updater's user
    cannot list -- and every lookup under it then misses silently, which is
    the empty map the guards exist to prevent. Probe it for real.
    """
    try:
        next(iter(path.iterdir()), None)
    except OSError:
        return False
    return True


def refuse_empty_replacement(out: Path, payload: dict, allow_empty: bool) -> str | None:
    """Why publishing ``payload`` over ``out`` would destroy the map, if it would.

    A readable but *empty* root -- a stale mount, a tree that came up blank --
    passes every directory check and produces a map with nothing in it. That is
    indistinguishable from a legitimate result, so the only thing left to
    compare against is what is already on disk.
    """
    if payload or allow_empty:
        return None
    try:
        previous = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not previous:
        return None
    return (
        f"refusing to replace {len(previous)} entries in {out} with an empty map; "
        "the archive is more likely unreadable than genuinely empty. "
        "Pass --allow-empty if it really is."
    )


def _publish_mode(path: Path) -> int:
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def snippet(text: str, max_chars: int = MAX_CHARS) -> str:
    """The transcript's first sentence, capped, as a stand-in title.

    Openings are usually a greeting or the lead item («Глава СБУ Малюк уходит
    в отставку.»), which is exactly what an editor scanning a list needs.
    """
    text = re.sub(r"\s+", " ", text).strip()
    m = re.match(rf"(.{{1,{max_chars}}}?[.!?])(?:\s|$)", text)
    if m:
        return m.group(1).strip()
    if len(text) <= max_chars:
        return text
    # no sentence boundary within reach: cut and say so
    return text[: max_chars - 1].rstrip() + "…"


def untitled_hashes(lines: list[str]) -> list[str]:
    """Hashes whose *latest* successful row has no title.

    The tag file is append-only and the UI reads it last-row-wins
    (``dedupe_latest``); this must agree, or a re-tagged episode whose newer
    row gained a title would still get a stand-in, and one whose newer row
    lost it would get none.
    """
    latest: dict[str, dict] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("error") or not r.get("video_hash"):
            continue
        latest[r["video_hash"]] = r
    return [h for h, r in latest.items() if not r.get("title")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", default=str(REPO_ROOT / "data" / "library_tags.jsonl"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "untitled_titles.json"))
    parser.add_argument("--archive-root", default="/mnt/vod/srv/storage/transcoded")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="publish even when the result is empty and the previous map was not",
    )
    args = parser.parse_args(argv)

    from library_tag_batch import transcript_path

    from rainrag.library_tagger import read_vtt_text

    untitled = untitled_hashes(Path(args.tags).read_text(encoding="utf-8").splitlines())

    root = Path(args.archive_root)
    # An archive that is not mounted reads as "no transcript anywhere": every
    # lookup misses, the map comes out empty, and the write still succeeds --
    # so the hourly updater would replace a good map with an empty one and
    # report success. Refuse, and the caller keeps what it already has.
    if not root.is_dir():
        parser.error(f"archive root is not a directory: {root}")
    if not listable_dir(root):
        parser.error(f"archive root is not readable: {root}")

    out: dict[str, str] = {}
    for h in untitled:
        p = transcript_path(root, h)
        if not p:
            continue
        text = read_vtt_text(p)
        if len(text) >= 40:
            out[h] = snippet(text)
    print(f"untitled tagged episodes: {len(untitled)}, titles synthesised: {len(out)}")
    if args.dry_run:
        return 0
    problem = refuse_empty_replacement(Path(args.out), out, args.allow_empty)
    if problem:
        parser.error(problem)
    write_json_atomic(Path(args.out), out)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
