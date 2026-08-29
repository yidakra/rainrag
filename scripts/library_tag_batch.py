#!/usr/bin/env python3
"""Tag the Library-relevant slice of the archive with Varya's schema.

Not the whole archive: 139k videos would be slow and mostly pointless, because
«Библиотека Дождя» republishes long-form lectures and interviews, not two-minute
news hits. The default selection is videos of at least 30 minutes — the same
filter Varya's own query uses, and the one that separates full episodes from
the clips that share a programme with them.

Output is one JSON object per line, appended as each episode finishes, so an
interrupted run resumes instead of restarting. Re-running skips what is already
tagged unless --refresh is given.

    scripts/library_tag_batch.py --programs "Лекции на Дожде" "Синдеева"
    scripts/library_tag_batch.py --min-minutes 30 --limit 50 --dry-run
    scripts/library_tag_batch.py --workers 8

Tagging is network-bound on the LLM API, so threads overlap the waiting; the
worker count is the concurrency against that API, not against the CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEFAULT_OUT = REPO_ROOT / "data" / "library_tags.jsonl"
DEFAULT_RAW = REPO_ROOT / "data" / "library_catalogue.raw.json"
DEFAULT_GOLD = REPO_ROOT / "data" / "library_gold.json"


def load_cms_index(metadata_dir: Path) -> dict[str, dict[str, Any]]:
    """Map video_hash -> cached CMS article, for presenters and mentions."""
    index: dict[str, dict[str, Any]] = {}
    for path in metadata_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        video_hash = data.get("video_hash")
        if video_hash:
            index[str(video_hash)] = data
    return index


def transcript_path(archive_root: Path, video_hash: str) -> Path | None:
    """Locate a video's Russian transcript in the hash-sharded archive."""
    shard = archive_root.joinpath(*[video_hash[i : i + 2] for i in range(0, 40, 2)])
    if not shard.is_dir():
        return None
    for candidate in (f"{video_hash}.ru.vtt", f"{video_hash}.en.vtt"):
        path = shard / candidate
        if path.exists():
            return path
    return next(iter(sorted(shard.glob("*.vtt"))), None)


def cms_people(article: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(presenters, mentioned) as the CMS knows them.

    `mentioned` is category=person tags: people discussed, not speaking.
    """
    presenters = []
    for person in article.get("presentors") or []:
        name = (person.get("name") or "").strip() or " ".join(
            part
            for key in ("firstname", "lastname")
            if (part := str(person.get(key) or "").strip())
        )
        if name:
            presenters.append(name)
    mentioned = [
        str(t.get("name")).strip()
        for t in (article.get("tags") or [])
        if isinstance(t, dict) and t.get("category") == "person" and t.get("name")
    ]
    return presenters, mentioned


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--raw-cache", default=str(DEFAULT_RAW))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--metadata-dir", default="/home/ubuntu/rainrag/web_metadata")
    parser.add_argument("--archive-root", default="/mnt/vod/srv/storage/transcoded")
    parser.add_argument("--min-minutes", type=float, default=30.0)
    parser.add_argument("--programs", nargs="*", default=None, help="limit to these programmes")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--content-ids",
        nargs="*",
        default=None,
        help="tag only these CMS content ids, ignoring the duration filter "
        "(used to tag an evaluation set before the batch reaches it)",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--refresh", action="store_true", help="re-tag already-tagged episodes")
    parser.add_argument("--dry-run", action="store_true", help="select and report, tag nothing")
    args = parser.parse_args(argv)

    from library_catalogue import fold_chunks_to_videos

    from rainrag.library_tagger import read_vtt_text, tag_episode

    payloads = json.loads(Path(args.raw_cache).read_text(encoding="utf-8"))
    videos = fold_chunks_to_videos(payloads)

    # Reading the CMS cache means opening ~139k small JSON files, so it is
    # deferred until something actually needs it -- a --dry-run should not pay
    # a minute of I/O just to print a selection.
    cms_all: dict[str, dict[str, Any]] = {}
    if args.content_ids:
        cms_all = load_cms_index(Path(args.metadata_dir))
        # An explicit list is a deliberate choice, so neither the duration nor
        # the programme filter should quietly drop entries from it.
        wanted_hashes = {h for h, a in cms_all.items() if str(a.get("id")) in set(args.content_ids)}
        wanted = [v for v in videos.values() if v.video_key in wanted_hashes]
    else:
        min_seconds = args.min_minutes * 60
        wanted = [
            v
            for v in videos.values()
            if (v.duration_seconds or 0) >= min_seconds
            and (not args.programs or v.program in set(args.programs))
        ]
    wanted.sort(key=lambda v: (v.program or "", v.date or ""))

    out_path = Path(args.out)
    done: set[str] = set()
    if out_path.exists() and not args.refresh:
        for line in out_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            # Only a successful row counts as done. Treating failures as done
            # made a rate-limited run permanently skip the episodes it lost.
            if row.get("video_hash") and not row.get("error"):
                done.add(row["video_hash"])

    todo = [v for v in wanted if v.video_key not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Selected {len(wanted)} episode(s) >= {args.min_minutes:g} min", flush=True)
    print(f"  already tagged: {len(done)}   to tag now: {len(todo)}", flush=True)
    if args.dry_run:
        for v in todo[:20]:
            print(
                f"    {v.date}  {round((v.duration_seconds or 0) / 60):>4}min  {v.program} — {(v.title or '')[:48]}"
            )
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    gold = (
        json.loads(Path(args.gold).read_text(encoding="utf-8")) if Path(args.gold).exists() else {}
    )
    fewshot = (gold.get("episodes") or [None])[0]

    from rainrag.config import load_config
    from rainrag.query import RAGQueryEngine

    config = load_config(args.config)
    engine = RAGQueryEngine(config)
    cms = cms_all or load_cms_index(Path(args.metadata_dir))
    archive_root = Path(args.archive_root)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    counters = {"ok": 0, "failed": 0, "no_transcript": 0, "tokens_in": 0, "tokens_out": 0}
    started = time.monotonic()

    def process(video: Any) -> None:
        path = transcript_path(archive_root, video.video_key)
        if path is None:
            with write_lock:
                counters["no_transcript"] += 1
            return
        try:
            text = read_vtt_text(path)
        except OSError:
            with write_lock:
                counters["no_transcript"] += 1
            return

        article = cms.get(video.video_key, {})
        presenters, mentioned = cms_people(article)
        result = tag_episode(
            engine,
            video_hash=video.video_key,
            content_id=str(article.get("id")) if article.get("id") else None,
            title=video.title,
            program=video.program,
            date=video.date,
            duration_minutes=(video.duration_seconds or 0) / 60,
            presenters=presenters or video.presenters,
            mentioned=mentioned,
            transcript=text,
            fewshot=fewshot,
        )
        record = result.to_json()
        record.update(
            {
                "title": video.title,
                "program": video.program,
                "date": video.date,
                "duration_seconds": video.duration_seconds,
                "presenter_cms": presenters or video.presenters,
                "mentioned_cms": mentioned,
                "url": video.url,
            }
        )
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if result.error:
                counters["failed"] += 1
            else:
                counters["ok"] += 1
            counters["tokens_in"] += result.tokens_in
            counters["tokens_out"] += result.tokens_out
            total = counters["ok"] + counters["failed"] + counters["no_transcript"]
            if total % 25 == 0:
                rate = total / max(time.monotonic() - started, 1e-9)
                left = (len(todo) - total) / rate if rate else 0
                print(
                    f"  {total}/{len(todo)}  ok={counters['ok']} failed={counters['failed']} "
                    f"no_transcript={counters['no_transcript']}  {rate:.2f}/s  ~{left / 60:.0f}min left",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(process, todo))

    elapsed = time.monotonic() - started
    print(
        f"\nDone in {elapsed / 60:.1f} min: {counters['ok']} tagged, {counters['failed']} failed, "
        f"{counters['no_transcript']} without a transcript",
        flush=True,
    )
    print(
        f"  tokens: {counters['tokens_in']:,} in / {counters['tokens_out']:,} out".replace(",", " ")
    )
    print(f"  output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
