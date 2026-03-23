#!/usr/bin/env python3
"""Backfill manifest doc_ids from existing docs.jsonl without re-embedding/re-indexing.

Designed for recovery when manifest exists with correct file metadata but empty doc_ids.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill manifest doc_ids from docs.jsonl")
    parser.add_argument("--docs", required=True, help="Path to docs.jsonl")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument(
        "--archive-root",
        required=True,
        help="Current archive root path (e.g. /mnt/vod/.../transcoded)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output manifest path. Defaults to in-place atomic replace.",
    )
    parser.add_argument(
        "--min-manifest-entries",
        type=int,
        default=100,
        help="Refuse to run if manifest has fewer entries than this threshold",
    )
    return parser.parse_args()


def map_doc_path(raw_path: str, archive_root: str) -> str:
    if raw_path.startswith("/data/archive/"):
        return str(Path(archive_root) / raw_path[len("/data/archive/") :])
    if raw_path.startswith("/archive/"):
        return str(Path(archive_root) / raw_path[len("/archive/") :])
    return raw_path


def main() -> int:
    args = parse_args()
    docs_path = Path(args.docs).resolve()
    manifest_path = Path(args.manifest).resolve()
    archive_root = str(Path(args.archive_root).resolve())

    if not docs_path.exists():
        raise FileNotFoundError(f"docs file not found: {docs_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest is not a dict: {manifest_path}")
    if len(manifest) < args.min_manifest_entries:
        raise ValueError(
            f"manifest has only {len(manifest)} entries (< min-manifest-entries={args.min_manifest_entries}); "
            + "refusing backfill to avoid persisting an empty/stale manifest"
        )

    path_to_doc_ids: dict[str, list[str]] = defaultdict(list)
    total_lines = 0
    mapped_lines = 0
    unmatched_lines = 0

    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("id")
            raw_path = obj.get("path")
            if not isinstance(doc_id, str) or not isinstance(raw_path, str):
                continue
            mapped_path = map_doc_path(raw_path, archive_root)
            if mapped_path in manifest:
                path_to_doc_ids[mapped_path].append(doc_id)
                mapped_lines += 1
            else:
                unmatched_lines += 1

            if total_lines % 200000 == 0:
                print(
                    f"progress lines={total_lines} mapped={mapped_lines} unmatched={unmatched_lines}",
                    flush=True,
                )

    for path, entry in manifest.items():
        if isinstance(entry, dict):
            entry["doc_ids"] = path_to_doc_ids.get(path, [])

    output_path = Path(args.output).resolve() if args.output else manifest_path
    if output_path == manifest_path:
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
        tmp_path.replace(manifest_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)

    non_empty = sum(
        1
        for v in manifest.values()
        if isinstance(v, dict) and isinstance(v.get("doc_ids"), list) and len(v["doc_ids"]) > 0
    )
    print(
        f"done lines={total_lines} mapped={mapped_lines} unmatched={unmatched_lines} "
        f"manifest_entries={len(manifest)} entries_with_doc_ids={non_empty}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
