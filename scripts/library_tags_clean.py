#!/usr/bin/env python3
"""Apply the subject scope rule to already-tagged episodes.

The editor's scope says subject holds abstract topics only; people, places
and organisations have their own fields. The tagger now enforces that on
the way in (strip_entities_from_subjects). This applies the same rule to
cards tagged before the fix: 16% of all subject tags on the first 10,178
episodes duplicated an entity already on the card.

Measured before running it: Varya's reference query keeps recall@10 4/6
with identical ranks, and subject recall/precision against her hand-tagged
cards are unchanged. The original file is kept as *.pre-scope.bak.

    scripts/library_tags_clean.py            # rewrite data/library_tags.jsonl
    scripts/library_tags_clean.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def clean_record(record: dict) -> tuple[dict, int]:
    """One tag row with entities removed from subject; returns (row, removed)."""
    from rainrag.library_tagger import strip_entities_from_subjects

    if record.get("error") or not record.get("subject"):
        return record, 0
    before = len(record["subject"])
    parsed = {
        k: list(record.get(k) or [])
        for k in ("guest", "subject", "place", "organization", "genre", "mentioned_extra")
    }
    cleaned = strip_entities_from_subjects(
        parsed, record.get("presenter_cms") or [], record.get("mentioned_cms") or []
    )
    out = dict(record)
    out["subject"] = cleaned["subject"]
    return out, before - len(out["subject"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", default=str(REPO_ROOT / "data" / "library_tags.jsonl"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.tags)
    rows: list[str] = []
    removed = touched = total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            rows.append(line)  # a torn line from a live run: keep as is
            continue
        cleaned, n = clean_record(record)
        total += len(record.get("subject") or [])
        if n:
            touched += 1
            removed += n
        rows.append(json.dumps(cleaned, ensure_ascii=False))
    pct = 100 * removed / total if total else 0
    print(f"subject tags: {total}, removed as entities: {removed} ({pct:.1f}%) on {touched} cards")
    if args.dry_run:
        return 0
    backup = path.with_suffix(".jsonl.pre-scope.bak")
    shutil.copy2(path, backup)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"rewritten; original kept at {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
