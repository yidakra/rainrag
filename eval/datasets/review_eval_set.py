"""Interactive terminal tool for human review of generated eval sets.

Opens each unreviewed record one at a time and lets the reviewer:

  [a] Accept  – mark valid, keep as-is
  [e] Edit    – correct the reference_answer inline
  [s] Skip    – leave for later (not marked reviewed)
  [d] Delete  – mark invalid (excluded from experiments)
  [q] Quit    – save progress and exit

Progress is written back to the file after every decision so the session
can be interrupted and resumed safely.

Usage
-----
    python -m eval.datasets.review_eval_set eval/datasets/eval_set_en.jsonl

    # Or via the CLI:
    python -m eval.run_eval review --input eval/datasets/eval_set_en.jsonl

After review, run experiments only on accepted records:

    python -m eval.run_eval ablation \\
        --dataset eval/datasets/eval_set_en_reviewed.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any


# simple alias for the JSONL records so we can give them a proper type
Record = dict[str, Any]

# (typing import added above)


# ---------------------------------------------------------------------------
# Optional rich formatting
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel
    # from rich.text import Text  # not used

    _console = Console()
    _rich = True
except ImportError:
    _console = None  # type: ignore[assignment]
    _rich = False
    Panel = None  # type: ignore[name-defined]  # ensure Panel is always defined


def _print_record(record: Record, index: int, total: int, accepted: int, deleted: int) -> None:
    """Render one record to the terminal."""
    if _rich:
        # _console is only None when rich wasn't imported; the _rich flag
        # guards all usage, but help static checkers understand that.
        assert _console is not None
        assert Panel is not None  # ensure Panel is callable when rich is available

        header = (
            f"[bold cyan]{index}/{total}[/]  "
            f"[green]{accepted} accepted[/]  [red]{deleted} deleted[/]  "
            f"remaining: {total - index + 1}"
        )
        body = (
            f"[bold]ID:[/]       {record.get('query_id', '—')}\n"
            f"[bold]Lang:[/]     {record.get('language', '—')}  "
            f"[bold]Category:[/] {record.get('category', '—')}  "
            f"[bold]Temporal:[/] {record.get('temporal', '—')}\n\n"
            f"[bold yellow]Query:[/]\n{record.get('query', '')}\n\n"
            f"[bold yellow]Reference answer:[/]\n{record.get('reference_answer', '—') or '(empty)'}\n\n"
            f"[bold]Relevant doc IDs:[/] {', '.join(record.get('relevant_doc_ids', []))}\n"
        )
        if record.get("source_path"):
            body += f"[bold]Source:[/] {record['source_path']}\n"
        if record.get("beir_dataset"):
            body += (
                f"[bold]BEIR:[/] {record['beir_dataset']}  query_id={record.get('beir_query_id')}\n"
            )
        _console.print()
        _console.rule(header)
        _console.print(Panel(body.strip(), expand=False))
    else:
        print(f"\n{'─' * 60}")
        print(f"{index}/{total}  accepted={accepted}  deleted={deleted}")
        print(f"ID:       {record.get('query_id', '—')}")
        print(
            f"Lang:     {record.get('language', '—')}  Category: {record.get('category', '—')}  Temporal: {record.get('temporal', '—')}"
        )
        print(f"\nQuery:\n{record.get('query', '')}")
        print(f"\nReference answer:\n{record.get('reference_answer', '') or '(empty)'}")
        print(f"\nRelevant doc IDs: {', '.join(record.get('relevant_doc_ids', []))}")
        if record.get("source_path"):
            print(f"Source: {record['source_path']}")


def _prompt(msg: str = "") -> str:
    """Read a line of input; handle EOF gracefully."""
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _save(records: list[Record], path: Path) -> None:
    """Write records atomically to *path*.

    A temporary file is created in the same directory, each record is written
    (flushed and fsync'd to ensure durability), and then the temp file is
    atomically moved into place via ``os.replace``/``Path.replace``.  On any
    exception we attempt to remove the temporary file so we don't leave
    cruft behind.
    """

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            # ensure all data is on disk before renaming
            f.flush()
            os.fsync(f.fileno())
        # atomic replace
        tmp_path.replace(path)
    except Exception:
        # clean up temp file if something went wrong
        with suppress(OSError):
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# Main review loop
# ---------------------------------------------------------------------------


def review_eval_set(
    input_path: str,
    output_path: str | None = None,
    only_unreviewed: bool = True,
) -> dict[str, int]:
    """Run the interactive review session.

    Args:
        input_path: Path to the eval JSONL file to review.
        output_path: Where to save reviewed records (defaults to overwriting input).
        only_unreviewed: If True, skip records already marked ``"reviewed": true``.

    Returns:
        Summary dict with keys: total, accepted, deleted, skipped.
    """
    inp = Path(input_path)
    if not inp.exists():
        # allow callers to handle the error programmatically instead of
        # terminating the process directly
        raise FileNotFoundError(f"file not found: {inp}")

    out = Path(output_path) if output_path else inp

    # Load all records
    records: list[Record] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    if total == 0:
        print("No records found.")
        return {"total": 0, "accepted": 0, "deleted": 0, "skipped": 0}

    # Build list of indices to review
    pending = [i for i, r in enumerate(records) if not (only_unreviewed and r.get("reviewed"))]

    n_pending = len(pending)
    accepted = sum(1 for r in records if r.get("valid") is True)
    deleted = sum(1 for r in records if r.get("valid") is False)

    if _rich:
        # rich is available, so _console must not be None – help the type checker.
        assert _console is not None
        _console.print(
            f"\n[bold]Reviewing [cyan]{inp.name}[/] — {n_pending} records pending, {total - n_pending} already reviewed[/]\n"
        )
        _console.print("[bold]Keys:[/]  [a] Accept  [e] Edit  [s] Skip  [d] Delete  [q] Quit\n")
    else:
        print(f"\nReviewing {inp.name} — {n_pending} pending of {total} total")
        print("Keys:  [a] Accept  [e] Edit  [s] Skip  [d] Delete  [q] Quit\n")

    reviewed_count = 0
    for seq, idx in enumerate(pending, 1):
        record = records[idx]
        _print_record(record, seq, n_pending, accepted, deleted)

        while True:
            key = _prompt("\n  [a/e/s/d/q] → ").lower()
            if key == "":
                continue

            if key == "q":
                _save(records, out)
                print(
                    f"\nSaved. Session ended: {accepted} accepted, {deleted} deleted, {seq - 1} reviewed this session."
                )
                return {
                    "total": total,
                    "accepted": accepted,
                    "deleted": deleted,
                    "skipped": n_pending - seq + 1,
                }

            if key == "a":
                if record.get("valid") is not True:
                    accepted += 1
                elif record.get("valid") is False:
                    deleted -= 1
                    accepted += 1
                record["valid"] = True
                record["reviewed"] = True
                reviewed_count += 1
                break

            if key == "d":
                if record.get("valid") is not False:
                    deleted += 1
                elif record.get("valid") is True:
                    accepted -= 1
                    deleted += 1
                record["valid"] = False
                record["reviewed"] = True
                reviewed_count += 1
                break

            if key == "s":
                # Skip without marking reviewed
                break

            if key == "e":
                print("  Enter corrected reference answer (blank line to finish):")
                lines: list[str] = []
                while True:
                    ln = _prompt("  > ")
                    if ln == "":
                        break
                    if ln.lower() == "q":
                        print("  Edit cancelled")
                        lines = []
                        break
                    lines.append(ln)
                if lines:
                    record["reference_answer"] = " ".join(lines)
                    print(f"  Updated: {record['reference_answer'][:80]}")
                    record["valid"] = True
                    record["reviewed"] = True
                    accepted += 1
                    reviewed_count += 1
                break

            print("  Unknown key. Use a, e, s, d, or q.")

        # Save after every decision (except skip) to preserve progress
        if key in ("a", "d", "e"):
            _save(records, out)

    # All pending records reviewed
    _save(records, out)
    summary = {"total": total, "accepted": accepted, "deleted": deleted, "skipped": 0}

    if _rich:
        assert _console is not None
        _console.print(
            f"\n[bold green]Review complete![/]  {accepted} accepted · {deleted} deleted · {reviewed_count} reviewed this session\n"
            + f"Saved to [cyan]{out}[/]\n"
        )
    else:
        print(f"\nReview complete: {accepted} accepted, {deleted} deleted.")
        print(f"Saved to {out}")

    return summary


def filter_valid(input_path: str, output_path: str) -> int:
    """Write only records marked ``valid=True`` to *output_path*.

    Call this after review to get a clean dataset for experiments.

    Returns:
        Number of records written.
    """
    inp = Path(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    records: list[Record] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    valid = [r for r in records if r.get("valid") is True]
    with open(out, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Written {len(valid)}/{len(records)} valid records to {out}")
    return len(valid)


def review_stats(input_path: str) -> dict[str, int]:
    """Print a quick summary of review progress without starting a session."""
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    reviewed = sum(1 for r in records if r.get("reviewed"))
    accepted = sum(1 for r in records if r.get("valid") is True)
    deleted = sum(1 for r in records if r.get("valid") is False)
    pending = total - reviewed

    stats = {
        "total": total,
        "reviewed": reviewed,
        "accepted": accepted,
        "deleted": deleted,
        "pending": pending,
    }

    if _rich:
        assert _console is not None
        _console.print(
            # avoid implicit literal concatenation by joining strings explicitly
            f"[bold]{Path(input_path).name}[/]\n"
            + f"  total={total}  reviewed={reviewed}  "
            + f"[green]accepted={accepted}[/]  [red]deleted={deleted}[/]  pending={pending}"
        )
    else:
        print(
            f"{input_path}: total={total} reviewed={reviewed} accepted={accepted} deleted={deleted} pending={pending}"
        )

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Human review of eval JSONL files.")
    sub = parser.add_subparsers(dest="cmd")

    review_p = sub.add_parser("review", help="Start interactive review session.")
    review_p.add_argument("input")
    review_p.add_argument("--output", "-o")
    review_p.add_argument(
        "--all", action="store_true", dest="all_records", help="Re-review already reviewed records"
    )

    filter_p = sub.add_parser("filter", help="Export only valid=True records.")
    filter_p.add_argument("input")
    filter_p.add_argument("output")

    stats_p = sub.add_parser("stats", help="Show review progress stats.")
    stats_p.add_argument("input")

    args = parser.parse_args()

    if args.cmd == "review" or args.cmd is None:
        inp = args.input if hasattr(args, "input") else (sys.argv[1] if len(sys.argv) > 1 else None)
        if not inp:
            parser.print_help()
            sys.exit(1)
        try:
            review_eval_set(
                inp,
                output_path=getattr(args, "output", None),
                only_unreviewed=not getattr(args, "all_records", False),
            )
        except FileNotFoundError as e:
            # mirror previous behavior by printing an error message
            print(f"ERROR: {e}")
            sys.exit(1)
    elif args.cmd == "filter":
        filter_valid(args.input, args.output)
    elif args.cmd == "stats":
        review_stats(args.input)
