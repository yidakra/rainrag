#!/usr/bin/env python3
"""Backfill web metadata and library taxonomy onto existing Qdrant points.

The archive collection was indexed before the web-metadata pipeline ever ran:
its points carry ``web_title``/``web_tags``/... keys with null values. Doing
this properly through the ingest pipeline means re-embedding most of the corpus
(metadata is appended to chunk text, which changes every content hash) -- a
day-plus of CPU. This script takes the cheap half: it fetches each video's
article from the library CMS and writes the payload fields onto the video's
existing points with ``set_payload``, leaving vectors and ``content_hash``
untouched. A later full re-embed will simply overwrite these payloads with
identical values plus new text.

Field extraction is deliberately not reimplemented: the script calls the same
``extract_clean_metadata`` / ``document_web_fields`` the ingest pipeline uses,
so a backfilled point is byte-identical (payload-wise) to what ingest would
have written.

Restartable by construction:

* metadata hits are cached to ``web_metadata/`` by the loader itself;
* 404s are appended to a misses file (the loader does not cache misses, and
  most of the archive predates the CMS, so re-asking would be the bulk of a
  re-run);
* videos whose points were written are appended to a done file.

    scripts/backfill_web_metadata.py --limit 5 --dry-run   # pilot, no writes
    scripts/backfill_web_metadata.py --limit 100           # small real run
    scripts/backfill_web_metadata.py                       # the full archive
"""

from __future__ import annotations

import argparse
import functools
import re
import sys
import time
from pathlib import Path
from typing import Any


# Allow running both as `scripts/backfill_web_metadata.py` and via pytest import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_HASH_RE = re.compile(r"^[a-f0-9]{40}$")

# Mirrors web_metadata.max_description_chars; the loader here is constructed
# directly rather than from the full config, so the default is repeated.
DESCRIPTION_CHAR_LIMIT = 600

# The run is hours long and usually redirected to a log file, where stdout is
# block-buffered -- an operator tailing the log would see nothing until exit.
print = functools.partial(print, flush=True)


def load_line_set(path: Path) -> set[str]:
    """Read a newline-delimited state file into a set; absent file is empty."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_line(path: Path, value: str) -> None:
    """Append one line, creating parents. Append-per-item so an interrupt loses
    at most the video being processed, never the whole run's progress."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(value + "\n")


def video_hash_from_path(vtt_path: str) -> str | None:
    """Extract the 40-hex video hash from a point's ``path`` payload field.

    Returns None for anything that is not a real archive hash (the corpus
    contains at least one ``test.en.vtt``), so junk paths are skipped rather
    than sent to the CMS API.
    """
    stem = Path(vtt_path).name
    candidate = stem.split(".", 1)[0].strip().lower()
    return candidate if _HASH_RE.fullmatch(candidate) else None


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into consecutive chunks of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]


class ThrottledAPIClient:
    """Wrap the CMS client so remote calls are paced and 404s observable.

    The loader treats a None from ``fetch_by_hash`` as "no article"; recording
    which hashes returned it lets the caller skip them on the next run. Local
    cache hits never reach this wrapper, so pacing costs nothing on resume.
    """

    def __init__(self, inner: Any, sleep_seconds: float) -> None:
        self.inner = inner
        self.sleep_seconds = sleep_seconds
        self.remote_calls = 0
        # The loader swallows fetch exceptions and returns None -- during a CMS
        # outage that is indistinguishable from a 404, and recording an outage
        # as permanent misses would silently exclude those videos forever.
        # The exception passes through this wrapper first, so remember it here
        # for backfill_video to re-raise.
        self.last_error: Exception | None = None

    def fetch_by_hash(self, video_hash: str) -> dict[str, Any] | None:
        if self.remote_calls and self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        self.remote_calls += 1
        self.last_error = None
        try:
            return self.inner.fetch_by_hash(video_hash)
        except Exception as exc:
            self.last_error = exc
            raise


def collect_points_by_video(
    client: Any, collection: str, scroll_batch: int = 2000
) -> tuple[dict[str, list[Any]], int]:
    """Scroll the whole collection and group point ids by video hash.

    Returns (hash -> point ids, count of points whose path had no usable hash).
    """
    by_video: dict[str, list[Any]] = {}
    skipped = 0
    seen = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=scroll_batch,
            offset=offset,
            with_payload=["path"],
            with_vectors=False,
        )
        for point in points:
            seen += 1
            payload = point.payload or {}
            video_hash = video_hash_from_path(str(payload.get("path") or ""))
            if video_hash is None:
                skipped += 1
                continue
            by_video.setdefault(video_hash, []).append(point.id)
        if seen and seen % 200_000 < scroll_batch:
            print(f"  scrolled {seen} points, {len(by_video)} videos so far")
        if offset is None:
            break
    return by_video, skipped


def backfill_video(
    *,
    qdrant: Any,
    collection: str,
    loader: Any,
    video_hash: str,
    point_ids: list[Any],
    dry_run: bool,
    payload_batch: int = 500,
) -> tuple[str, int]:
    """Fetch one video's metadata and write it onto its points.

    Returns (outcome, points written) where outcome is one of
    ``"written"``, ``"no_article"``, ``"empty"``.
    """
    api = getattr(loader, "api_client", None)
    if api is not None and hasattr(api, "last_error"):
        # Clear any error left over from a previous video: a local-cache hit
        # makes no API call, so a stale flag would otherwise misclassify it.
        api.last_error = None

    raw = loader.load_metadata(video_hash)
    if raw is None:
        pending_error = getattr(api, "last_error", None)
        if pending_error is not None:
            # The fetch failed rather than 404ing; surface it so the caller
            # counts an error (retried next run) instead of a permanent miss.
            raise pending_error
        return "no_article", 0

    cleaned = loader.extract_clean_metadata(raw)
    if not cleaned:
        # An article exists but carries neither title nor description. The
        # ingest pipeline indexes such videos without web fields; match it.
        return "empty", 0

    from rainrag.ingest import document_web_fields

    # Same cap as ingest: otherwise a backfill re-run would put the full
    # article body back onto payloads the reindex had just trimmed.
    max_chars = (
        getattr(getattr(loader, "config", None), "max_description_chars", None)
        or DESCRIPTION_CHAR_LIMIT
    )
    fields = document_web_fields(cleaned, max_chars)
    if dry_run:
        return "written", 0

    written = 0
    for batch in chunked(point_ids, payload_batch):
        qdrant.set_payload(
            collection_name=collection,
            payload=fields,
            points=batch,
            wait=True,
        )
        written += len(batch)
    return "written", written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="process at most N videos (0 = all)")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="seconds between CMS API calls (cache hits are not paced)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and extract, but write nothing to Qdrant and no state files",
    )
    parser.add_argument("--misses-file", default="data/web_metadata_misses.txt")
    parser.add_argument("--done-file", default="data/web_metadata_backfill_done.txt")
    args = parser.parse_args(argv)

    # .env is where the deployment keeps LIBRARY_API_TOKEN; systemd units load
    # it via EnvironmentFile but a manual script run would miss it otherwise.
    try:
        from dotenv import find_dotenv, load_dotenv

        # usecwd: resolve .env from the deployment root the script is run in,
        # not from wherever the script file happens to live (e.g. a worktree).
        load_dotenv(find_dotenv(usecwd=True))
    except ImportError:
        pass

    from qdrant_client import QdrantClient

    from rainrag.config import load_config
    from rainrag.ingest import WebMetadataLoader
    from rainrag.web_metadata_api import WebMetadataAPIClient

    config = load_config(args.config)
    collection = config.qdrant.collection_name

    api_client = ThrottledAPIClient(
        WebMetadataAPIClient.from_env(
            base_url=config.web_metadata.api_url,
            token_env=config.web_metadata.api_token_env,
        ),
        sleep_seconds=args.sleep,
    )
    loader = WebMetadataLoader(
        Path(config.web_metadata.path), source="hybrid", api_client=api_client
    )

    qdrant = QdrantClient(host=config.qdrant.host, port=config.qdrant.port, timeout=60)

    misses_path = Path(args.misses_file)
    done_path = Path(args.done_file)
    known_misses = load_line_set(misses_path)
    done = load_line_set(done_path)

    print(f"Scrolling {collection} to group points by video...")
    by_video, junk_points = collect_points_by_video(qdrant, collection)
    print(
        f"Found {len(by_video)} videos across the collection"
        + (f" ({junk_points} points had no usable hash and were skipped)" if junk_points else "")
    )

    todo = [h for h in sorted(by_video) if h not in done and h not in known_misses]
    print(f"{len(done)} already done, {len(known_misses)} known misses, {len(todo)} to process")
    if args.limit:
        todo = todo[: args.limit]
        print(f"Limited to {len(todo)} videos for this run")

    counts = {"written": 0, "no_article": 0, "empty": 0, "error": 0}
    points_written = 0
    started = time.monotonic()

    for i, video_hash in enumerate(todo, 1):
        try:
            outcome, written = backfill_video(
                qdrant=qdrant,
                collection=collection,
                loader=loader,
                video_hash=video_hash,
                point_ids=by_video[video_hash],
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            print(f"\nInterrupted at {i}/{len(todo)}; state files are current, rerun to resume.")
            break
        except Exception as exc:
            # Transient failures (network blip, CMS hiccup) are not recorded as
            # misses -- the next run retries them.
            counts["error"] += 1
            print(f"  [error] {video_hash}: {type(exc).__name__}: {exc}")
            continue

        counts[outcome] += 1
        points_written += written
        if not args.dry_run:
            if outcome == "no_article":
                append_line(misses_path, video_hash)
            else:
                append_line(done_path, video_hash)

        if i % 200 == 0 or i == len(todo):
            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            remaining = (len(todo) - i) / rate if rate > 0 else float("inf")
            print(
                f"  {i}/{len(todo)} videos | written {counts['written']}, "
                f"no article {counts['no_article']}, empty {counts['empty']}, "
                f"errors {counts['error']} | {points_written} points | "
                f"{rate:.1f} videos/s, ~{remaining / 3600:.1f}h left"
            )

    print(
        f"\nDone. {counts['written']} videos written ({points_written} points), "
        f"{counts['no_article']} without an article, {counts['empty']} with empty metadata, "
        f"{counts['error']} errors, {api_client.remote_calls} CMS API calls."
    )
    if counts["error"]:
        print("Errors are retried on the next run; nothing was recorded for them.")
    return 1 if counts["error"] and not counts["written"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
