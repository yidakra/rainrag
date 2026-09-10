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
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MAX_CHARS = 90


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
    args = parser.parse_args(argv)

    from library_tag_batch import transcript_path

    from rainrag.library_tagger import read_vtt_text

    untitled = untitled_hashes(Path(args.tags).read_text(encoding="utf-8").splitlines())

    out: dict[str, str] = {}
    root = Path(args.archive_root)
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
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
