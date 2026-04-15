#!/usr/bin/env python3
"""Fast manifest rebuild from current archive files.

This recovery mode prioritizes getting incremental ingestion running again.
It does not backfill per-file doc_ids; entries start with empty doc_ids.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild manifest from archive tree")
    parser.add_argument(
        "--archive-root", required=True, help="Root directory containing .vtt files"
    )
    parser.add_argument("--manifest", required=True, help="Path to output manifest JSON")
    parser.add_argument(
        "--min-entries",
        type=int,
        default=100,
        help="Refuse to write manifest if scanned entries are below this threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.archive_root).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Archive root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Archive root must be a directory: {root}")

    manifest: dict[str, dict[str, object]] = {}
    count = 0

    for p in root.rglob("*.vtt"):
        try:
            if not p.is_file():
                continue
            st = p.stat()
        except FileNotFoundError:
            continue

        manifest[str(p.resolve())] = {
            "mtime": st.st_mtime,
            "size": st.st_size,
            "file_hash": "",
            "doc_ids": [],
        }
        count += 1

    if count < args.min_entries:
        raise RuntimeError(
            f"Refusing to write manifest: only {count} entries found (< min-entries={args.min_entries})."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    temp_path.replace(manifest_path)

    print(f"Rebuilt manifest at {manifest_path} with {count} archive entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
