#!/usr/bin/env python3
"""Catalogue the indexed archive by programme (линейка), presenter and theme.

Built for the Библиотека Дождя project. The complaint it answers is that the
archive is a black box: the RAG service finds five-minute fragments by topic,
which is what a journalist chasing a quote needs, but it cannot tell an editor
which *programmes* exist, how many episodes each has, over what years, with
which presenters and about what. Without that you cannot plan a month of
uploads; you can only guess and then watch tape.

This walks the Qdrant collection once, folds ~1.7M chunks up to videos and
videos up to programmes, and writes the result twice: JSON for the next tool
to consume, Markdown for a human to read. (Both, deliberately: the call on
2026-08-25 asked for automatic output to be human-readable too.)

    scripts/library_catalogue.py                      # full catalogue
    scripts/library_catalogue.py --program "Здесь и сейчас"   # one programme's episodes
    scripts/library_catalogue.py --min-episodes 20    # only substantial линейки

Re-runnable and cheap to re-run: the expensive scroll is cached to
data/library_catalogue.raw.json, so shaping the output again is instant.
Pass --refresh after a reindex.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO_ROOT / "data" / "library_catalogue.raw.json"
DEFAULT_JSON = REPO_ROOT / "data" / "library_catalogue.json"
DEFAULT_MARKDOWN = REPO_ROOT / "data" / "library_catalogue.md"

UNTAGGED = "(без программы)"


@dataclass
class Video:
    """One archive video, folded up from its chunks."""

    video_key: str
    title: str | None = None
    date: str | None = None
    program: str | None = None
    url: str | None = None
    duration_seconds: float | None = None
    presenters: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    stories: list[str] = field(default_factory=list)
    chunks: int = 0
    languages: set[str] = field(default_factory=set)


def video_key_from_path(path: str) -> str:
    """Collapse a VTT path to the video it belongs to.

    Language variants of one broadcast (``<hash>.ru.vtt`` / ``<hash>.en.vtt``)
    are the same video and must not be counted as two episodes.
    """
    name = Path(path).name
    return name.split(".", 1)[0] if name else path


def fold_chunks_to_videos(points: Any) -> dict[str, Video]:
    """Fold chunk payloads up into per-video records.

    Metadata repeats on every chunk of a video, so the first non-empty value
    wins; only the chunk count and language set accumulate.
    """
    videos: dict[str, Video] = {}
    for payload in points:
        path = str(payload.get("path") or "")
        if not path:
            continue
        key = video_key_from_path(path)
        video = videos.get(key)
        if video is None:
            video = Video(video_key=key)
            videos[key] = video

        video.chunks += 1
        if payload.get("language"):
            video.languages.add(str(payload["language"]))

        video.title = video.title or payload.get("web_title")
        video.date = video.date or payload.get("web_date") or payload.get("date")
        video.program = video.program or payload.get("web_program")
        video.url = video.url or payload.get("web_url")
        if video.duration_seconds is None and payload.get("duration_seconds"):
            video.duration_seconds = payload["duration_seconds"]
        for attr, key_name in (
            ("presenters", "web_presenters"),
            ("themes", "web_tags_theme"),
            ("persons", "web_tags_person"),
            ("locations", "web_tags_location"),
            ("stories", "web_stories"),
        ):
            if not getattr(video, attr):
                value = payload.get(key_name)
                if isinstance(value, list) and value:
                    setattr(video, attr, [str(v) for v in value])
    return videos


def summarise_programs(videos: dict[str, Video], top_tags: int = 12) -> list[dict[str, Any]]:
    """Group videos by programme, newest-first within each.

    Videos with no programme are kept under a single bucket rather than
    dropped: how much of the archive is unattributed is itself a finding.
    """
    grouped: dict[str, list[Video]] = defaultdict(list)
    for video in videos.values():
        grouped[video.program or UNTAGGED].append(video)

    programs = []
    for name, items in grouped.items():
        dates = sorted(d for d in (v.date for v in items) if d)
        themes: Counter[str] = Counter()
        persons: Counter[str] = Counter()
        presenters: Counter[str] = Counter()
        durations = [v.duration_seconds for v in items if v.duration_seconds]
        for v in items:
            themes.update(v.themes)
            persons.update(v.persons)
            presenters.update(v.presenters)
        programs.append(
            {
                "program": name,
                "episodes": len(items),
                "chunks": sum(v.chunks for v in items),
                "first_date": dates[0] if dates else None,
                "last_date": dates[-1] if dates else None,
                "dated_episodes": len(dates),
                "median_duration_minutes": (
                    round(median(durations) / 60, 1) if durations else None
                ),
                "presenters": [n for n, _ in presenters.most_common(top_tags)],
                "top_themes": [{"tag": t, "episodes": c} for t, c in themes.most_common(top_tags)],
                "top_persons": [
                    {"tag": t, "episodes": c} for t, c in persons.most_common(top_tags)
                ],
            }
        )
    programs.sort(key=lambda p: (-p["episodes"], p["program"]))
    return programs


def episodes_for_program(videos: dict[str, Video], program: str) -> list[dict[str, Any]]:
    """Every episode of one programme, newest first — the shortlist view."""
    items = [v for v in videos.values() if (v.program or UNTAGGED) == program]
    items.sort(key=lambda v: (v.date or "", v.title or ""), reverse=True)
    return [
        {
            "video": v.video_key,
            "date": v.date,
            "title": v.title,
            "duration_minutes": round(v.duration_seconds / 60, 1) if v.duration_seconds else None,
            "presenters": v.presenters,
            "themes": v.themes,
            "persons": v.persons,
            "languages": sorted(v.languages),
            "url": v.url,
        }
        for v in items
    ]


def render_markdown(programs: list[dict[str, Any]], totals: dict[str, Any]) -> str:
    """The human-readable half: a table an editor can skim."""
    lines = [
        "# Каталог архива по программам",
        "",
        f"Видео: **{totals['videos']:,}** · фрагментов: **{totals['chunks']:,}** · "
        f"программ: **{totals['programs']:,}**".replace(",", " "),
        "",
        f"С названием: {totals['with_title_pct']}% · с датой: {totals['with_date_pct']}% · "
        f"с тегами темы: {totals['with_theme_pct']}% · с ведущими: {totals['with_presenter_pct']}%",
        "",
        "| Программа | Выпусков | Период | Медиана, мин | Ведущие | Ключевые темы |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for p in programs:
        period = (
            f"{p['first_date']} — {p['last_date']}" if p["first_date"] and p["last_date"] else "—"
        )
        presenters = ", ".join(p["presenters"][:3]) or "—"
        themes = ", ".join(t["tag"] for t in p["top_themes"][:5]) or "—"
        duration = p["median_duration_minutes"] or "—"
        lines.append(
            f"| {p['program']} | {p['episodes']} | {period} | {duration} | {presenters} | {themes} |"
        )
    return "\n".join(lines) + "\n"


def scroll_payloads(client: Any, collection: str, batch: int = 4000) -> list[dict[str, Any]]:
    """Read every point's metadata (not its vector or text) once."""
    fields = [
        "path",
        "language",
        "duration_seconds",
        "web_title",
        "web_date",
        "date",
        "web_program",
        "web_url",
        "web_presenters",
        "web_tags_theme",
        "web_tags_person",
        "web_tags_location",
        "web_stories",
    ]
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=fields,
            with_vectors=False,
        )
        payloads.extend(p.payload or {} for p in points)
        if len(payloads) % 200_000 < batch:
            print(f"  scrolled {len(payloads):,} chunks".replace(",", " "))
        if offset is None:
            return payloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--raw-cache", default=str(DEFAULT_RAW))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown-out", default=str(DEFAULT_MARKDOWN))
    parser.add_argument(
        "--refresh", action="store_true", help="re-scroll Qdrant instead of using the cache"
    )
    parser.add_argument("--program", help="list every episode of one programme instead")
    parser.add_argument(
        "--min-episodes", type=int, default=1, help="hide programmes with fewer episodes"
    )
    args = parser.parse_args(argv)

    raw_path = Path(args.raw_cache)
    if args.refresh or not raw_path.exists():
        from qdrant_client import QdrantClient

        from rainrag.config import load_config

        config = load_config(args.config)
        client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port, timeout=120)
        print(f"Scrolling {config.qdrant.collection_name}...")
        payloads = scroll_payloads(client, config.qdrant.collection_name)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payloads, ensure_ascii=False), encoding="utf-8")
        print(f"Cached {len(payloads):,} chunk payloads to {raw_path}".replace(",", " "))
    else:
        payloads = json.loads(raw_path.read_text(encoding="utf-8"))
        print(
            f"Using cached scroll ({len(payloads):,} chunks); --refresh to re-read".replace(
                ",", " "
            )
        )

    videos = fold_chunks_to_videos(payloads)

    if args.program:
        episodes = episodes_for_program(videos, args.program)
        print(f"\n{args.program}: {len(episodes)} выпуск(ов)\n")
        for e in episodes[:200]:
            title = (e["title"] or "(без названия)")[:70]
            print(
                f"  {e['date'] or '????-??-??'}  {str(e['duration_minutes'] or '?'):>6}м  {title}"
            )
        out = Path(args.json_out).with_name("library_episodes.json")
        out.write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  full list ({len(episodes)}): {out}")
        return 0

    all_programs = summarise_programs(videos)
    programs = [p for p in all_programs if p["episodes"] >= args.min_episodes]
    n = len(videos)
    totals = {
        "videos": n,
        "chunks": sum(v.chunks for v in videos.values()),
        "programs": len([p for p in programs if p["program"] != UNTAGGED]),
        "with_title_pct": round(100 * sum(1 for v in videos.values() if v.title) / max(n, 1)),
        "with_date_pct": round(100 * sum(1 for v in videos.values() if v.date) / max(n, 1)),
        "with_theme_pct": round(100 * sum(1 for v in videos.values() if v.themes) / max(n, 1)),
        "with_presenter_pct": round(
            100 * sum(1 for v in videos.values() if v.presenters) / max(n, 1)
        ),
    }

    for out in (args.json_out, args.markdown_out):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps({"totals": totals, "programs": programs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.markdown_out).write_text(render_markdown(programs, totals), encoding="utf-8")

    # Genre is a property of the programme, not the episode, so it is answered
    # by a table an editor reviews once -- not by classifying 139k videos.
    from rainrag.library import load_program_genres, write_program_genre_draft

    genre_path = Path(args.json_out).with_name("program_genres.csv")
    reviewed = load_program_genres(genre_path)
    # Deliberately the unfiltered list: --min-episodes narrows the catalogue an
    # editor reads, but rewriting the genre table from a filtered list would
    # drop every smaller programme's row -- including reviewed ones.
    write_program_genre_draft(genre_path, all_programs, reviewed)
    print(f"  Genres:   {genre_path} ({len(reviewed)} already reviewed)")

    print(f"\n{totals['videos']:,} videos · {totals['programs']:,} programmes".replace(",", " "))
    print(f"  JSON:     {args.json_out}")
    print(f"  Markdown: {args.markdown_out}\n")
    for p in programs[:15]:
        period = f"{p['first_date']}—{p['last_date']}" if p["first_date"] else "—"
        print(f"  {p['episodes']:>5} выпусков  {period:<24} {p['program'][:52]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
