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
import hashlib
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEFAULT_TOPICS = REPO_ROOT / "data" / "library_note_topics.csv"
# data/ is not in the repository, so a fresh checkout has no topic list and
# the default run would exit having found nothing to look for (Copilot on
# #101). The seed ships here and is copied on first use; the editable copy
# is the one under data/.
SEED_TOPICS = REPO_ROOT / "deploy" / "library_note_topics.seed.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "library_notes.json"
DEFAULT_TAGS = REPO_ROOT / "data" / "library_tags.jsonl"

# Stored beside the episodes so a changed topic list invalidates the scan.
# Not a hash, so it cannot collide with one.
FINGERPRINT_KEY = "__topics__"


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


def notes_for(vtt_path: Path, topics: list) -> list[dict] | None:
    """The flagged passages in one transcript, or None when it cannot be read.

    None and [] are different answers and the caller needs both: nothing
    found is settled, unreadable is something to try again next run.
    """
    from rainrag.library_notes import find_spans, parse_vtt

    try:
        text = vtt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
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

    # A limited rescan must not become the whole file. `--force --limit 50`
    # started from an empty `done` and then replaced a complete
    # library_notes.json with fifty episodes, so every other episode lost
    # its notes in the UI until a full scan finished (CodeRabbit on #101).
    if args.force and args.limit and Path(args.out) == DEFAULT_OUT:
        print(
            "refusing: --force with --limit would replace the complete notes file "
            "with a partial one. Pass --out to write the sample somewhere else."
        )
        return 1

    topics_path = Path(args.topics)
    if not topics_path.exists() and SEED_TOPICS.exists():
        topics_path.parent.mkdir(parents=True, exist_ok=True)
        topics_path.write_text(SEED_TOPICS.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"seeded {topics_path} from {SEED_TOPICS.name}; edit it to change what is flagged")
    topics = load_topics(topics_path)
    if not topics:
        print(f"no topics in {topics_path}; nothing to look for")
        return 1
    print(f"topics: {', '.join(t.name for t in topics)}")

    archive_root, video_root = archive_roots(args.config)
    if archive_root is None:
        print("no archive root configured; transcripts are not reachable from here")
        return 1

    # What was looked for last time. Editing the topic list and rerunning
    # used to scan nothing, because every episode was already in `done` and
    # the operator had no way to know `--force` was needed (Copilot on #101).
    fingerprint = hashlib.sha256(
        "\u0000".join(f"{t.name}:{';'.join(t.patterns)}" for t in topics).encode("utf-8")
    ).hexdigest()[:16]

    out_path = Path(args.out)
    done: dict[str, list[dict]] = {}
    if out_path.exists() and not args.force:
        try:
            stored = json.loads(out_path.read_text(encoding="utf-8"))
        except ValueError:
            stored = {}
        if isinstance(stored, dict):
            if stored.get(FINGERPRINT_KEY) == fingerprint:
                done = {k: v for k, v in stored.items() if k != FINGERPRINT_KEY}
            else:
                print("the topic list changed since the last run; rescanning everything")

    hashes = pool_hashes(Path(args.tags))
    todo = [h for h in hashes if h not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"pool {len(hashes)} | already scanned {len(done)} | scanning {len(todo)}")

    started = time.monotonic()
    scanned = 0
    flagged = 0
    missing = 0
    unreadable = 0
    exts = vtt_extensions(args.config)
    for scanned, video_hash in enumerate(todo, 1):
        media = media_for(archive_root, video_hash, vtt_exts=exts, video_root=video_root)
        relative = media.get("vtt")
        if not relative:
            # No transcript *yet*. Recording an empty result would retire the
            # episode from every future run, including the one after the
            # transcript lands (CodeRabbit on #101).
            missing += 1
            continue
        spans = notes_for(archive_root / relative, topics)
        if spans is None:
            unreadable += 1
            continue
        # An empty list here is a real answer: read, scanned, nothing found.
        done[video_hash] = spans
        if spans:
            flagged += 1
        if scanned % 200 == 0:
            rate = scanned / max(time.monotonic() - started, 1e-6)
            print(f"  {scanned}/{len(todo)}, {flagged} with notes, {rate:.0f}/s")

    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({FINGERPRINT_KEY: fingerprint, **done}, ensure_ascii=False), encoding="utf-8"
    )
    tmp.replace(out_path)
    with_notes = sum(1 for spans in done.values() if spans)
    print(f"written {out_path}: {len(done)} episodes, {with_notes} with something flagged")
    if missing or unreadable:
        print(
            f"  {missing} without a transcript and {unreadable} unreadable, left for the next run"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
