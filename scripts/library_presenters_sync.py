#!/usr/bin/env python3
"""Refresh the presenter override table from Varya's sheet and report coverage.

The CMS has no presenter for the lecture programmes, so those episodes carry
no speaker at all and sit out the heaviest axis in the ranking. Varya filled
them in by hand («Заполнить ведущих», 86cbhq9q8). The sheet is shared with
named people rather than published, so there is no API key path to it: export
it (File, Download) and point this script at the download.

    scripts/library_presenters_sync.py ~/Downloads/lectures_with_presenter.xlsx
    scripts/library_presenters_sync.py --check

Only the `presenter` column is read. `кто_похож_на_лектора` beside it is a
generated guess, in the genitive and sometimes carrying a job title
(«Екатерины Михайловой», «Психотерапевт Алена»), and feeding that to a name
matcher would be worse than the gap it fills.

The table is keyed on content_id and lives beside the programme table rather
than being written back into `library_tags.jsonl`, which the tagging batch
regenerates: hand work a machine run can destroy is hand work that will be
lost.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_TABLE = REPO_ROOT / "data" / "library_presenters.csv"
DEFAULT_TAGS = REPO_ROOT / "data" / "library_tags.jsonl"

COLUMNS = ("content_id", "presenter")


def rows_from_export(path: Path) -> list[dict[str, str]]:
    """(content_id, presenter) pairs from an .xlsx or .csv export."""
    from rainrag.library_presenters import split_presenters

    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        import openpyxl

        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = book[book.sheetnames[0]]
        raw = sheet.iter_rows(values_only=True)
        header = [str(c or "").strip() for c in next(raw, ())]
        records = [dict(zip(header, row, strict=False)) for row in raw]
    else:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                reader.fieldnames = [(name or "").strip() for name in reader.fieldnames]
            records = list(reader)

    out: list[dict[str, str]] = []
    for record in records:
        # Excel hands back a float for a numeric cell, so 412707 arrives as
        # 412707.0 and would never match a content_id from the tag file.
        content_id = str(record.get("content_id") or "").strip()
        if content_id.endswith(".0"):
            content_id = content_id[:-2]
        names = split_presenters(record.get("presenter") or "")
        if content_id and names:
            out.append({"content_id": content_id, "presenter": ", ".join(names)})
    return out


def write_table(rows: list[dict[str, str]], path: Path) -> None:
    """Publish the table by rename, so a torn file is never read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def coverage(table: Path, tags: Path) -> tuple[int, int, int]:
    """(overrides, of those in the pool, of those with no speaker without it).

    The third number is the one that matters: it is how many episodes gain a
    speaker they did not have, which is what the table exists for.
    """
    from rainrag.library_presenters import load_presenters

    overrides = load_presenters(table)
    # Distinct episodes, not rows: the tag file is append-only, so a re-tagged
    # episode has more than one row and counting rows overstates the reach.
    in_pool: set[str] = set()
    rescued: set[str] = set()
    if tags.exists():
        for line in tags.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            content_id = str(record.get("content_id") or "")
            if content_id not in overrides:
                continue
            in_pool.add(content_id)
            if not (record.get("presenter_cms") or []) and not (record.get("guest") or []):
                rescued.add(content_id)
    return len(overrides), len(in_pool), len(rescued)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", help="xlsx or csv export of the sheet")
    parser.add_argument("--table", default=str(DEFAULT_TABLE))
    parser.add_argument("--tags", default=str(DEFAULT_TAGS))
    parser.add_argument("--check", action="store_true", help="report coverage only")
    args = parser.parse_args(argv)

    table = Path(args.table)
    if args.export and not args.check:
        rows = rows_from_export(Path(args.export))
        if not rows:
            print("no usable rows: expected columns content_id and presenter")
            return 1
        write_table(rows, table)
        print(f"wrote {len(rows)} presenter override(s) to {table}")
    elif not args.check:
        parser.error("give an export to sync, or --check to report coverage")

    total, in_pool, rescued = coverage(table, Path(args.tags))
    print(f"overrides: {total} | in the tagged pool: {in_pool} | gaining a speaker: {rescued}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
