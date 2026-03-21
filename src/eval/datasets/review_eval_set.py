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
import logging
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, cast


logger = logging.getLogger(__name__)


# simple alias for the JSONL records so we can give them a proper type
Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Optional rich formatting
# ---------------------------------------------------------------------------

try:
    from rich.console import Console
    from rich.panel import Panel as RichPanel

    _console: Console | None = Console()
    _panel_cls: type[RichPanel] | None = RichPanel
    _rich = True
except ImportError:
    _console = None
    _panel_cls = None
    _rich = False


def _print_record(record: Record, index: int, total: int, accepted: int, deleted: int) -> None:
    """Render one record to the terminal."""
    if _rich:
        # _console is only None when rich wasn't imported; the _rich flag
        # guards all usage, but help static checkers understand that.
        assert _console is not None
        assert _panel_cls is not None  # ensure Panel is callable when rich is available

        header = (
            f"[bold cyan]{index}/{total}[/]  "
            f"[green]{accepted} accepted[/]  [red]{deleted} deleted[/]  "
            f"remaining: {total - index}"
        )
        body = (
            f"[bold]ID:[/]       {record.get('query_id', '—')}\n"
            f"[bold]Lang:[/]     {record.get('language', '—')}  "
            f"[bold]Category:[/] {record.get('category', '—')}  "
            f"[bold]Temporal:[/] {record.get('temporal', '—')}\n\n"
            f"[bold yellow]Query:[/]\n{record.get('query', '')}\n\n"
            f"[bold yellow]Reference answer:[/]\n{record.get('reference_answer', '—') or '(empty)'}\n\n"
            f"[bold]Relevant doc IDs:[/] {', '.join(str(x) for x in record.get('relevant_doc_ids', []))}\n"
        )
        if record.get("source_path"):
            body += f"[bold]Source:[/] {record['source_path']}\n"
        beir_dataset = record.get("beir_dataset")
        beir_query_id = record.get("beir_query_id")
        if beir_dataset or beir_query_id:
            body += f"[bold]BEIR:[/] {beir_dataset or '—'}  query_id={beir_query_id or '—'}\n"
        _console.print()
        _console.rule(header)
        _console.print(_panel_cls(body.strip(), expand=False))
    else:
        print(f"\n{'─' * 60}")
        print(f"{index}/{total}  accepted={accepted}  deleted={deleted}")
        print(f"ID:       {record.get('query_id', '—')}")
        print(
            f"Lang:     {record.get('language', '—')}  Category: {record.get('category', '—')}  Temporal: {record.get('temporal', '—')}"
        )
        print(f"\nQuery:\n{record.get('query', '')}")
        print(f"\nReference answer:\n{record.get('reference_answer', '') or '(empty)'}")
        print(
            f"\nRelevant doc IDs: {', '.join(str(x) for x in record.get('relevant_doc_ids', []))}"
        )
        if record.get("source_path"):
            print(f"Source: {record['source_path']}")
        beir_dataset = record.get("beir_dataset")
        beir_query_id = record.get("beir_query_id")
        if beir_dataset or beir_query_id:
            print(f"BEIR: {beir_dataset or '—'}  Query ID: {beir_query_id or '—'}")


def _prompt(msg: str = "", *, raise_on_interrupt: bool = False) -> str:
    """Read a line of input; handle EOF gracefully.

    If :paramref:`raise_on_interrupt` is True, a KeyboardInterrupt is
    propagated to the caller (useful for cancelling nested input loops).
    Otherwise, it is treated as a quit signal ("q").
    """
    try:
        return input(msg).strip()
    except (EOFError, KeyboardInterrupt):
        if raise_on_interrupt:
            raise
        return "q"


def _save(records: list[Record], path: Path) -> None:
    """Write records atomically to *path*.

    A temporary file is created in the same directory, each record is written
    (flushed and fsync'd to ensure durability), and then the temp file is
    atomically moved into place via ``os.replace``/``Path.replace``.  On any
    exception we attempt to remove the temporary file so we don't leave
    cruft behind.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # ensure all data is on disk before renaming
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    except Exception:
        # clean up temp file if something went wrong
        with suppress(OSError):
            tmp_path.unlink()
        raise


def _load_records(path: Path) -> list[Record]:
    """Load JSONL records from *path*, skipping blank lines and non-dict entries."""
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not path.is_file():
        raise ValueError(f"path is not a file: {path}")

    records: list[Record] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSON line at %s:%d: %r (%s)",
                    path,
                    idx,
                    line,
                    exc,
                )
                continue

            if isinstance(loaded, dict):
                records.append(cast(Record, loaded))
            else:
                logger.warning(
                    "Skipping non-dict JSONL record at %s:%d: %r",
                    path,
                    idx,
                    loaded,
                )
    return records


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
    records: list[Record] = _load_records(inp)

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

    def _update_counters_for_valid(
        record: Record,
        new_valid: bool,
        accepted: int,
        deleted: int,
    ) -> tuple[int, int]:
        prev = record.get("valid")
        if new_valid is True:
            if prev is True:
                pass
            elif prev is False:
                deleted -= 1
                accepted += 1
            else:
                accepted += 1
        else:
            if prev is False:
                pass
            elif prev is True:
                accepted -= 1
                deleted += 1
            else:
                deleted += 1
        return accepted, deleted

    reviewed_count = 0
    skipped = 0
    for seq, idx in enumerate(pending, 1):
        record = records[idx]
        _print_record(record, seq, n_pending, accepted, deleted)

        edited = False
        while True:
            key = _prompt("\n  [a/e/s/d/q] → ").lower()
            if key == "":
                continue

            if key == "q":
                _save(records, out)
                print(
                    f"\nSaved. Session ended: {accepted} accepted, {deleted} deleted, {skipped} skipped, {reviewed_count} reviewed this session."
                )
                return {
                    "total": total,
                    "accepted": accepted,
                    "deleted": deleted,
                    "skipped": skipped,
                }

            if key == "a":
                accepted, deleted = _update_counters_for_valid(record, True, accepted, deleted)
                record["valid"] = True
                record["reviewed"] = True
                reviewed_count += 1
                break

            if key == "d":
                accepted, deleted = _update_counters_for_valid(record, False, accepted, deleted)
                record["valid"] = False
                record["reviewed"] = True
                reviewed_count += 1
                break

            if key == "s":
                # Skip without marking reviewed
                skipped += 1
                break

            if key == "e":
                print("  Enter corrected reference answer (blank line to finish; :q to cancel):")
                lines: list[str] = []
                edit_cancelled = False
                while True:
                    try:
                        ln = _prompt("  > ", raise_on_interrupt=True)
                    except KeyboardInterrupt:
                        print("  Edit cancelled")
                        edit_cancelled = True
                        break

                    if ln == "":
                        if not lines:
                            print("  Edit cancelled")
                            edit_cancelled = True
                        break
                    if ln == ":q":
                        print("  Edit cancelled")
                        edit_cancelled = True
                        break
                    lines.append(ln)

                if not edit_cancelled and lines:
                    record["reference_answer"] = "\n".join(lines)
                    edited = True
                    print(f"  Updated: {record['reference_answer'][:80]}")
                    accepted, deleted = _update_counters_for_valid(record, True, accepted, deleted)
                    record["valid"] = True
                    record["reviewed"] = True
                    reviewed_count += 1
                    break
                # If edit was cancelled or no lines were entered, stay on the same record.
                continue

            print("  Unknown key. Use a, e, s, d, or q.")

        # Save after every decision (except skip) to preserve progress.
        # Only write to disk for accept/delete, or when an edit actually changed the record.
        if key in ("a", "d") or edited:
            _save(records, out)

    # All pending records reviewed
    _save(records, out)
    summary = {"total": total, "accepted": accepted, "deleted": deleted, "skipped": skipped}

    if _rich:
        assert _console is not None
        _console.print(
            f"\n[bold green]Review complete![/]  {accepted} accepted · {deleted} deleted · {skipped} skipped · {reviewed_count} reviewed this session\n"
            + f"Saved to [cyan]{out}[/]\n"
        )
    else:
        print(f"\nReview complete: {accepted} accepted, {deleted} deleted, {skipped} skipped.")
        print(f"Saved to {out}")

    return summary


def filter_valid(input_path: str, output_path: str) -> int:
    """Write only records marked ``valid=True`` to *output_path*.

    Call this after review to get a clean dataset for experiments.

    Returns:
        Number of records written.
    """
    inp = Path(input_path)
    records = _load_records(inp)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    valid = [r for r in records if r.get("valid") is True]
    _save(valid, out)

    print(f"Written {len(valid)}/{len(records)} valid records to {out}")
    return len(valid)


def review_stats(input_path: str) -> dict[str, int]:
    """Print a quick summary of review progress without starting a session."""
    inp = Path(input_path)
    records = _load_records(inp)

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

    parser = argparse.ArgumentParser(
        description=(
            "Human review of eval JSONL files. "
            "If no subcommand is given, the first unknown argument is treated as the input path "
            "and the 'review' subcommand is used by default."
        )
    )
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

    args, unknown = parser.parse_known_args()

    # Fallback behavior: when args.cmd is None, the CLI defaults to the
    # review subcommand. `unknown` is parsed with review_p, and then args.cmd
    # is set to 'review' so downstream logic uses review behavior.
    if args.cmd is None:
        try:
            review_args = review_p.parse_args(unknown)
        except SystemExit:
            # argparse will call sys.exit() on parse errors; show help instead.
            review_p.print_help()
            sys.exit(1)
        args = review_args
        args.cmd = "review"

    if args.cmd == "review":
        if not getattr(args, "input", None):
            parser.print_help()
            sys.exit(1)
        try:
            review_eval_set(
                args.input,
                output_path=getattr(args, "output", None),
                only_unreviewed=not getattr(args, "all_records", False),
            )
        except FileNotFoundError as e:
            # mirror previous behavior by printing an error message
            print(f"ERROR: {e}")
            sys.exit(1)
    elif args.cmd == "filter":
        try:
            filter_valid(args.input, args.output)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    elif args.cmd == "stats":
        try:
            review_stats(args.input)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
