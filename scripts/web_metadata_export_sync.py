#!/usr/bin/env python3
"""Recover CMS articles the per-hash lookup cannot see.

`GET /video/{hash}/article` is the only source the metadata backfill uses, and
for some videos it answers 404 while the article exists and its page is live on
tvrain.tv. Six were verified by hand on 2026-09-29: 404 from the API, 200 from
the site. Sergei described the cause in May, the video service having stored
different storage ids so the hash mapping drifted.

`GET /article/export` does not go through that mapping. It returns articles in
time windows, each already carrying `video_hash` and `url`, in exactly the
shape the per-hash cache stores. So the export can fill the gap: whatever it
knows and the cache does not is written into `web_metadata/`, and any recovered
hash is dropped from the misses file so the backfill stops skipping it.

This does not conjure articles that were never written. Across the export, 11
of the 1,748 uncarded episodes in the Library pool gain a site link and 26 of
36,534 uncarded videos overall. The rest genuinely have no article, which holds
in 2025 and 2026 where the export is complete. The wider value is the 1,454
articles absent from the cache for videos that do have titles.

    scripts/web_metadata_export_sync.py --dry-run
    scripts/web_metadata_export_sync.py --since 2023-01-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The API caps a window at 180 days. Long windows are cheaper but a single
# failure loses the whole span, so a failed one is retried in short slices
# before being given up on.
WINDOW_DAYS = 179
RETRY_SLICE_DAYS = 59


def safe_hash(video_hash: Any) -> str | None:
    """The hash as a filename, or None when it cannot be trusted as one.

    The same rule `WebMetadataLoader._safe_metadata_hash` applies: 40 hex
    characters, lowercased, no path separators. The export is remote input and
    this value becomes a filename, so accepting it raw would let a malformed
    or hostile `video_hash` write outside the cache (Tenki on #93).
    """
    candidate = str(video_hash or "").strip()
    if not candidate or Path(candidate).name != candidate:
        return None
    if not re.fullmatch(r"[a-fA-F0-9]{40}", candidate):
        return None
    return candidate.lower()


def windows(
    since: dt.datetime, until: dt.datetime, span_days: int
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Consecutive [start, end) spans covering the period."""
    out: list[tuple[dt.datetime, dt.datetime]] = []
    start = since
    while start < until:
        end = min(start + dt.timedelta(days=span_days), until)
        out.append((start, end))
        start = end
    return out


def fetch_window(client: Any, start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]]:
    """Articles published in one window, or raise."""
    return client.export_batch(start_time=int(start.timestamp()), end_time=int(end.timestamp()))


def _collect(batch: list[dict[str, Any]], found: dict[str, dict[str, Any]]) -> int:
    """Take the articles with a usable hash; returns how many were rejected."""
    rejected = 0
    for article in batch:
        key = safe_hash(article.get("video_hash"))
        if key is None:
            rejected += 1
            continue
        found[key] = article
    return rejected


def articles_in_period(
    client: Any, since: dt.datetime, until: dt.datetime
) -> tuple[dict[str, dict[str, Any]], list[tuple[dt.datetime, dt.datetime]]]:
    """(hash -> article, windows that failed even when sliced).

    A window that fails is not fatal: the export 500s on parts of 2019-2022 and
    refusing everything because of that would recover nothing at all. The
    failures are returned so the caller can report exactly which periods went
    unchecked rather than implying full coverage.

    An authentication failure is different and is raised, not absorbed. A stale
    LIBRARY_API_TOKEN would otherwise look identical to the API's known
    flakiness: every window "fails", nothing is recovered, and the operator is
    told the export is empty (Tenki on #93).
    """
    found: dict[str, dict[str, Any]] = {}
    failed: list[tuple[dt.datetime, dt.datetime]] = []
    rejected = 0

    def attempt(start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]] | None:
        try:
            return fetch_window(client, start, end)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                raise
            print(f"  window {start.date()} to {end.date()}: {exc.response.status_code}")
            return None
        except Exception as exc:  # network, timeout, malformed zip
            print(f"  window {start.date()} to {end.date()}: {type(exc).__name__}: {exc}"[:160])
            return None

    for start, end in windows(since, until, WINDOW_DAYS):
        batch = attempt(start, end)
        if batch is not None:
            rejected += _collect(batch, found)
            continue
        for slice_start, slice_end in windows(start, end, RETRY_SLICE_DAYS):
            sliced = attempt(slice_start, slice_end)
            if sliced is None:
                failed.append((slice_start, slice_end))
                continue
            rejected += _collect(sliced, found)
    if rejected:
        print(f"  {rejected} article(s) skipped: unusable video_hash")
    return found, failed


def write_article(directory: Path, video_hash: str, article: dict[str, Any]) -> None:
    """Publish one article into the cache by rename.

    Raises on a hash that cannot be a filename rather than writing it: callers
    get their keys from `safe_hash`, so reaching here with a bad one is a bug.
    """
    key = safe_hash(video_hash)
    if key is None:
        raise ValueError(f"unsafe video_hash: {video_hash!r}")
    video_hash = key
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{video_hash}.", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(article, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp).replace(directory / f"{video_hash}.json")
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def drop_from_misses(path: Path, recovered: set[str]) -> int:
    """Remove recovered hashes from the misses file; returns how many went.

    Leaving them there would keep the backfill skipping the very videos this
    script just found an article for.
    """
    if not path.exists() or not recovered:
        return 0
    kept = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    remaining = [h for h in kept if h not in recovered]
    if len(remaining) == len(kept):
        return 0
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(remaining) + ("\n" if remaining else ""))
        handle.flush()
        os.fsync(handle.fileno())
    Path(tmp).replace(path)
    return len(kept) - len(remaining)


def _as_utc(value: dt.datetime) -> dt.datetime:
    """Read a naive datetime as UTC, keep an aware one as given."""
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2019-01-01")
    parser.add_argument("--until", default=None, help="default: now")
    parser.add_argument("--metadata-dir", default=str(REPO_ROOT / "web_metadata"))
    parser.add_argument("--misses-file", default=str(REPO_ROOT / "data/web_metadata_misses.txt"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    from rainrag.web_metadata_api import WebMetadataAPIClient

    # UTC throughout: naive datetimes take the host's offset in timestamp(),
    # which shifts every window boundary and can drop articles that sit near an
    # edge out of all of them (Tenki on #93).
    since = _as_utc(dt.datetime.fromisoformat(args.since))
    until = (
        _as_utc(dt.datetime.fromisoformat(args.until))
        if args.until
        else dt.datetime.now(dt.timezone.utc)
    )
    directory = Path(args.metadata_dir)

    client = WebMetadataAPIClient.from_env()
    found, failed = articles_in_period(client, since, until)
    print(f"export returned {len(found)} articles between {since.date()} and {until.date()}")
    if failed:
        print(f"  {len(failed)} window(s) failed and were not checked:")
        for start, end in failed[:5]:
            print(f"    {start.date()} to {end.date()}")
        if len(failed) > 5:
            print(f"    and {len(failed) - 5} more")

    new = {h: a for h, a in found.items() if not (directory / f"{h}.json").exists()}
    print(f"  not in the local cache: {len(new)}")
    if args.dry_run:
        print("dry run, nothing written")
        return 0

    for video_hash, article in new.items():
        write_article(directory, video_hash, article)
    # Every hash the export returned, not just the newly written ones. A run
    # that crashed between writing an article and clearing the misses file
    # leaves the hash in both places, and the backfill checks the misses file
    # before the cache -- so that video would be skipped forever with its
    # article sitting right there (Tenki on #93).
    cleared = drop_from_misses(Path(args.misses_file), set(found))
    print(f"  written: {len(new)}")
    print(f"  cleared from the misses file: {cleared}")
    print("\nNext: scripts/backfill_web_metadata.py, then library_catalogue.py --refresh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
