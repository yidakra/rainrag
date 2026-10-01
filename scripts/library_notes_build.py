#!/usr/bin/env python3
"""Scan the archive's transcripts for the topics an editor wants flagged.

Varya's ask (86cbhemwp): picking archive material means finding out, by
watching it or reading the whole transcript, that a guest spends four minutes
on Ukraine in the middle of an hour. The card should say so up front.

Reading a transcript is ~230 KB and a couple of thousand cues, so it happens
here, once, and not while a card renders. The output is one small JSON keyed
by video hash, which the Library reads the way it already reads the stand-in
titles and the archive media map.

    scripts/library_notes_build.py                 # the whole tagged pool
    scripts/library_notes_build.py --limit 50      # a sample, for a look
    scripts/library_notes_build.py --force         # ignore the existing file

Nothing here decides what is worth flagging. The topic list is editorial and
lives in data/library_note_topics.csv, seeded with the two Varya named.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEFAULT_TOPICS = REPO_ROOT / "data" / "library_note_topics.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "library_notes.json"
DEFAULT_TAGS = REPO_ROOT / "data" / "library_tags.jsonl"


def pool_hashes(tags_path: Path) -> list[str]:
    """Every tagged episode, newest row wins, failures skipped."""
    seen: dict[str, None] = {}
    for line in tags_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("error"):
            continue
        video_hash = record.get("video_hash")
        if video_hash:
            seen[str(video_hash)] = None
    return list(seen)


def notes_for(vtt_path: Path, topics: list) -> list[dict]:
    """The flagged passages in one transcript, or none when it cannot be read."""
    from rainrag.library_notes import find_spans, parse_vtt

    try:
        text = vtt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [span.as_dict() for span in find_spans(parse_vtt(text), topics)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", default=str(DEFAULT_TOPICS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tags", default=str(DEFAULT_TAGS))
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="stop after N episodes")
    parser.add_argument("--force", action="store_true", help="rescan episodes already done")
    args = parser.parse_args(argv)

    from library_untitled_media import archive_roots, media_for, vtt_extensions

    from rainrag.library_notes import load_topics

    topics = load_topics(Path(args.topics))
    if not topics:
        print(f"no topics in {args.topics}; nothing to look for")
        return 1
    print(f"topics: {', '.join(t.name for t in topics)}")

    archive_root, video_root = archive_roots(args.config)
    if archive_root is None:
        print("no archive root configured; transcripts are not reachable from here")
        return 1

    out_path = Path(args.out)
    done: dict[str, list[dict]] = {}
    if out_path.exists() and not args.force:
        try:
            done = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError:
            done = {}

    hashes = pool_hashes(Path(args.tags))
    todo = [h for h in hashes if h not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"pool {len(hashes)} | already scanned {len(done)} | scanning {len(todo)}")

    started = time.monotonic()
    scanned = 0
    flagged = 0
    exts = vtt_extensions()
    for scanned, video_hash in enumerate(todo, 1):
        media = media_for(archive_root, video_hash, vtt_exts=exts, video_root=video_root)
        relative = media.get("vtt")
        # Recorded even when empty, so the next run does not read the same
        # transcript again to learn the same nothing.
        spans = notes_for(archive_root / relative, topics) if relative else []
        done[video_hash] = spans
        if spans:
            flagged += 1
        if scanned % 200 == 0:
            rate = scanned / max(time.monotonic() - started, 1e-6)
            print(f"  {scanned}/{len(todo)}, {flagged} with notes, {rate:.0f}/s")

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)
    with_notes = sum(1 for spans in done.values() if spans)
    print(f"written {out_path}: {len(done)} episodes, {with_notes} with something flagged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
