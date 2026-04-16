#!/usr/bin/env python3
"""Rebuild incremental manifest from existing docs JSONL.

This is intended for recovery when docs/embeddings/Qdrant exist but
manifest.json is missing or stale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild RainRAG manifest from docs JSONL")
    parser.add_argument("--docs", required=True, help="Path to docs JSONL")
    parser.add_argument("--manifest", required=True, help="Path to output manifest JSON")
    parser.add_argument(
        "--min-entries",
        type=int,
        default=100,
        help="Refuse to write manifest if rebuilt entries are below this threshold",
    )
    parser.add_argument(
        "--archive-root",
        default="",
        help="Current archive root used for path remapping (e.g. /mnt/vod/.../transcoded)",
    )
    return parser.parse_args()


def normalize_doc_path(raw_path: str, archive_root: Path | None) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p.resolve()

    if archive_root is not None:
        # Common legacy prefixes observed in older docs caches.
        for legacy_prefix in ("/data/archive/", "/archive/"):
            if raw_path.startswith(legacy_prefix):
                rel = raw_path[len(legacy_prefix) :]
                candidate = (archive_root / rel).resolve()
                if candidate.exists():
                    return candidate

        # Generic fallback: if path contains '/archive/', keep suffix under archive_root.
        marker = "/archive/"
        if marker in raw_path:
            rel = raw_path.split(marker, 1)[1]
            candidate = (archive_root / rel).resolve()
            if candidate.exists():
                return candidate

        # Relative paths are assumed relative to archive_root.
        if not p.is_absolute():
            candidate = (archive_root / p).resolve()
            if candidate.exists():
                return candidate

    return p.resolve()


def main() -> int:
    args = parse_args()
    docs_path = Path(args.docs).resolve()
    manifest_path = Path(args.manifest).resolve()
    archive_root = Path(args.archive_root).resolve() if args.archive_root else None

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs file not found: {docs_path}")

    raw_file_to_doc_ids: dict[str, list[str]] = defaultdict(list)

    with docs_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {docs_path} at line {line_no}: {exc}") from exc

            path_val = obj.get("path")
            doc_id = obj.get("id")
            if not isinstance(path_val, str) or not path_val:
                print(
                    f"Skipping invalid doc entry at line {line_no}: missing or invalid path: {obj!r}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(doc_id, str) or not doc_id:
                print(
                    f"Skipping invalid doc entry at line {line_no}: missing or invalid id: {obj!r}",
                    file=sys.stderr,
                )
                continue

            raw_file_to_doc_ids[path_val].append(doc_id)

    file_to_doc_ids: dict[str, list[str]] = defaultdict(list)
    for raw_path, doc_ids in raw_file_to_doc_ids.items():
        abs_path = str(normalize_doc_path(raw_path, archive_root))
        file_to_doc_ids[abs_path].extend(doc_ids)

    manifest: dict[str, dict[str, object]] = {}
    existing_count = 0
    missing_count = 0

    for file_path, doc_ids in file_to_doc_ids.items():
        p = Path(file_path)
        if p.exists():
            st = p.stat()
            existing_count += 1
            manifest[file_path] = {
                "mtime": st.st_mtime,
                "size": st.st_size,
                # Leave hash empty in recovery mode; mtime+size fast path still works.
                "file_hash": "",
                "doc_ids": doc_ids,
            }
        else:
            # Keep missing entries so next incremental run can clean stale docs.
            missing_count += 1
            manifest[file_path] = {
                "mtime": 0.0,
                "size": 0,
                "file_hash": "",
                "doc_ids": doc_ids,
            }

    total = len(file_to_doc_ids)
    if total < args.min_entries:
        raise RuntimeError(
            f"Refusing to write manifest: rebuilt {total} entries (< min-entries={args.min_entries})."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    temp_path.replace(manifest_path)

    print(
        f"Rebuilt manifest at {manifest_path} with {total} entries "
        f"({existing_count} existing, {missing_count} missing)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
