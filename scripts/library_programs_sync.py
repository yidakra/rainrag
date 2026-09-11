#!/usr/bin/env python3
"""Refresh the programme table from Varya's sheet export and report coverage.

The Programs tab of the "Tags taxonomy" sheet is the editorial source of truth
for programme genre, and the speaker rule in `rainrag.library_programs` reads
it. The sheet is shared with named people rather than published, so there is no
API key path to it: export the tab as CSV (File, Download, Comma-separated
values) and point this script at the download.

    scripts/library_programs_sync.py ~/Downloads/Tags\\ taxonomy\\ -\\ Programs.csv

It writes `data/library_programs.csv` and prints how much of the tagged archive
the table now covers, which is the number worth watching: a programme title
that drifts between the sheet and the catalogue silently takes its whole
back catalogue out of the genre rule.

Usage:
    scripts/library_programs_sync.py <export.csv>
    scripts/library_programs_sync.py --check      # report coverage, write nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_TABLE = REPO_ROOT / "data" / "library_programs.csv"
DEFAULT_TAGS = REPO_ROOT / "data" / "library_tags.jsonl"

REQUIRED_COLUMNS = {"title", "genre"}


def validate_export(path: Path) -> list[dict[str, str]]:
    """Read the export, failing loudly if it is not the Programs tab."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    missing = REQUIRED_COLUMNS - {(name or "").strip() for name in rows[0]}
    if missing:
        raise SystemExit(
            f"{path} is missing the column(s) {sorted(missing)}. "
            "Export the Programs tab, not the whole workbook."
        )
    return rows


def latest_tag_rows(path: Path) -> list[dict[str, object]]:
    """Last good row wins, matching how the ranker reads the tagging output.

    Errors are dropped *before* the dedup, not after. The ranker does the same,
    and the difference is not cosmetic: an episode that was tagged, then
    re-tagged into a failure, still has a usable earlier row. Filtering last
    would discard it here while the interface keeps serving it, so the coverage
    report would blame the programme table for a gap that does not exist.
    """
    if not path.exists():
        return []
    latest: dict[str, dict[str, object]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or record.get("error"):
                continue
            if record.get("video_hash"):
                latest[str(record["video_hash"])] = record
    return list(latest.values())


def report_coverage(table: Path, tags: Path) -> int:
    """Print how many tagged episodes the table reaches. Returns episodes missed."""
    from rainrag.library_programs import load_programmes, normalise_title

    programmes = load_programmes(table)
    episodes = latest_tag_rows(tags)
    if not episodes:
        print(f"no tagged episodes at {tags}; coverage not checked")
        return 0
    with_genre = 0
    listed_no_genre: Counter[str] = Counter()
    unlisted: Counter[str] = Counter()
    for record in episodes:
        name = str(record.get("program") or "")
        programme = programmes.get(normalise_title(name)) if name else None
        if programme is None:
            unlisted[name or "(без программы)"] += 1
        elif programme.genres:
            with_genre += 1
        else:
            listed_no_genre[programme.title] += 1
    total = len(episodes)
    print(f"programmes in table: {len(programmes)}")
    print(f"tagged episodes: {total}")
    print(f"  programme has a genre: {with_genre} ({100 * with_genre / total:.0f}%)")
    print(f"  listed without a genre: {sum(listed_no_genre.values())}")
    print(f"  programme not in table: {sum(unlisted.values())}")
    for name, count in unlisted.most_common(10):
        print(f"      {count:>5}  {name}")
    return sum(unlisted.values()) + sum(listed_no_genre.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", help="CSV export of the Programs tab")
    parser.add_argument("--table", default=str(DEFAULT_TABLE))
    parser.add_argument("--tags", default=str(DEFAULT_TAGS))
    parser.add_argument(
        "--check", action="store_true", help="report coverage of the current table only"
    )
    args = parser.parse_args(argv)

    table = Path(args.table)
    if not args.check:
        if not args.export:
            parser.error("give a CSV export, or --check to only report coverage")
        from rainrag.library_programs import parse_genres

        source = Path(args.export)
        rows = validate_export(source)
        genres = sum(1 for row in rows if parse_genres(row.get("genre") or ""))
        table.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, table)
        print(f"wrote {table}: {len(rows)} programmes, {genres} with a genre")

    report_coverage(table, Path(args.tags))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
