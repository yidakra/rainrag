"""Episode records for other TV Rain services: metadata, summary, tags, transcript.

The recommendation system pulls these once per episode and caches them on its
side, so the store favours a complete, stable record over speed. One episode per
CMS ``content_id`` (the key the product side uses):

* the base set is the archive index (``archive_by_content_id.json``, rebuilt
  daily by ``scripts/youtube_map.py``): every video with a CMS card and an
  indexed transcript, short news items included;
* tags come from the tagging run (``library_tags.jsonl``) and summaries from the
  summary run (``episode_summaries.jsonl``) where those exist, else null;
* the transcript is read from the archive on request.

Files are re-read when their modification time changes, so a running API picks
up the nightly index and new tagging or summary rows without a restart.
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from rainrag.library_tagger import read_vtt_text


# Fields a caller can ask for with ``fields=``; content_id is always returned.
EPISODE_FIELDS = (
    "content_id",
    "video_hash",
    "title",
    "program",
    "date",
    "url",
    "duration_seconds",
    "presenters",
    "summary",
    "summary_model",
    "summary_prompt_version",
    "tags",
    "transcript",
    "updated_at",
)


@dataclass
class _Episode:
    """One episode as the store holds it; transcript and CMS people are read on demand."""

    content_id: str
    video_hash: str
    title: str | None
    date: str | None
    url: str | None
    duration_seconds: float | None
    base_updated_at: str
    program: str | None = None
    presenters: list[str] | None = None
    tags: dict[str, list[str]] | None = None
    tags_updated_at: str | None = None
    summary: str | None = None
    summary_model: str | None = None
    summary_prompt_version: str | None = None
    summary_updated_at: str | None = None

    @property
    def updated_at(self) -> str:
        """The latest timestamp among the parts of the record (ISO 8601, UTC)."""
        return max(
            t for t in (self.base_updated_at, self.tags_updated_at, self.summary_updated_at) if t
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        """Newest first, then content_id: a total order, so cursors are stable."""
        return (self.date or "", self.content_id)


def parse_fields(raw: str | None) -> set[str]:
    """``fields=a,b`` as a set of known field names; None or empty means all.

    Unknown names are an error rather than silently ignored, so a typo does not
    quietly drop the transcript a caller meant to keep.
    """
    if not raw or not raw.strip():
        return set(EPISODE_FIELDS)
    wanted = {name.strip() for name in raw.split(",") if name.strip()}
    unknown = wanted - set(EPISODE_FIELDS)
    if unknown:
        raise ValueError(f"Unknown fields: {', '.join(sorted(unknown))}")
    return wanted | {"content_id"}


def encode_cursor(key: tuple[str, str]) -> str:
    """An opaque cursor for the position after ``key``."""
    return base64.urlsafe_b64encode(json.dumps(list(key)).encode()).decode()


def decode_cursor(cursor: str) -> tuple[str, str]:
    """Inverse of encode_cursor; ValueError on anything else."""
    try:
        date, content_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError("Invalid cursor") from exc
    if not isinstance(date, str) or not isinstance(content_id, str):
        raise ValueError("Invalid cursor")
    return (date, content_id)


def transcript_path(archive_root: Path, video_hash: str) -> Path | None:
    """The episode's transcript in the hash-sharded archive, Russian first."""
    shard = archive_root.joinpath(*[video_hash[i : i + 2] for i in range(0, 40, 2)])
    if not shard.is_dir():
        return None
    for candidate in (f"{video_hash}.ru.vtt", f"{video_hash}.en.vtt"):
        path = shard / candidate
        if path.exists():
            return path
    return next(iter(sorted(shard.glob("*.vtt"))), None)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a row mid-write during a run
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _cms_people(metadata_dir: Path, video_hash: str) -> tuple[str | None, list[str]]:
    """(programme, presenters) from the cached CMS article, if there is one."""
    try:
        article = json.loads((metadata_dir / f"{video_hash}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    program = (article.get("teleshow") or {}).get("name")
    presenters = []
    for person in article.get("presentors") or []:
        name = (person.get("name") or "").strip() or " ".join(
            part
            for key in ("firstname", "lastname")
            if (part := str(person.get(key) or "").strip())
        )
        if name:
            presenters.append(name)
    return program, presenters


class EpisodeStore:
    """Thread-safe, mtime-reloading index of episodes by content_id."""

    def __init__(
        self,
        index_path: Path,
        tags_path: Path,
        summaries_path: Path,
        archive_root: Path,
        metadata_dir: Path,
    ) -> None:
        """Remember the sources; nothing is read until the first request."""
        self.index_path = index_path
        self.tags_path = tags_path
        self.summaries_path = summaries_path
        self.archive_root = archive_root
        self.metadata_dir = metadata_dir
        self._lock = threading.Lock()
        self._stamp: tuple[float | None, ...] | None = None
        self._by_id: dict[str, _Episode] = {}
        self._ordered: list[_Episode] = []

    def _reload_if_changed(self) -> None:
        stamp = tuple(_mtime(p) for p in (self.index_path, self.tags_path, self.summaries_path))
        with self._lock:
            if stamp == self._stamp:
                return
            self._by_id = self._build()
            self._ordered = sorted(self._by_id.values(), key=lambda e: e.sort_key, reverse=True)
            self._stamp = stamp
            logger.info(f"Episode store loaded {len(self._by_id)} episodes")

    def _build(self) -> dict[str, _Episode]:
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning(f"Episode store: cannot read {self.index_path}: {exc}")
            return {}
        if not isinstance(index, dict):
            return {}
        base_updated_at = _mtime_iso(self.index_path)
        by_id: dict[str, _Episode] = {}
        for content_id, row in index.items():
            if not isinstance(row, dict) or not row.get("video_hash"):
                continue
            by_id[str(content_id)] = _Episode(
                content_id=str(content_id),
                video_hash=row["video_hash"],
                title=row.get("title"),
                date=row.get("date"),
                url=row.get("url"),
                duration_seconds=row.get("duration_seconds"),
                base_updated_at=base_updated_at,
            )
        self._merge_tags(by_id)
        self._merge_summaries(by_id)
        return by_id

    def _merge_tags(self, by_id: dict[str, _Episode]) -> None:
        if not self.tags_path.exists():
            return
        updated_at = _mtime_iso(self.tags_path)
        latest: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self.tags_path):
            if row.get("video_hash"):
                latest[row["video_hash"]] = row  # append-only: the last row wins
        for row in latest.values():
            episode = by_id.get(str(row.get("content_id") or ""))
            # Tags belong to the file the index chose for this article, not to
            # another part or re-ingest that shares its content_id.
            if row.get("error") or episode is None or episode.video_hash != row["video_hash"]:
                continue
            episode.program = row.get("program")
            episode.presenters = list(row.get("presenter_cms") or [])
            episode.tags = {
                "subject": list(row.get("subject") or []),
                "guest": list(row.get("guest") or []),
                "mentioned": list(
                    dict.fromkeys(
                        [*(row.get("mentioned_cms") or []), *(row.get("mentioned_extra") or [])]
                    )
                ),
                "place": list(row.get("place") or []),
                "organization": list(row.get("organization") or []),
                "genre": list(row.get("genre") or []),
            }
            episode.tags_updated_at = updated_at

    def _merge_summaries(self, by_id: dict[str, _Episode]) -> None:
        if not self.summaries_path.exists():
            return
        fallback = _mtime_iso(self.summaries_path)
        for row in _read_jsonl(self.summaries_path):
            episode = by_id.get(str(row.get("content_id") or ""))
            if episode is None or not row.get("summary"):
                continue
            episode.summary = row["summary"]
            episode.summary_model = row.get("model")
            episode.summary_prompt_version = row.get("prompt_version")
            episode.summary_updated_at = row.get("generated_at") or fallback

    def _transcript(self, episode: _Episode) -> dict[str, str] | None:
        path = transcript_path(self.archive_root, episode.video_hash)
        if path is None:
            return None
        try:
            text = read_vtt_text(path)
        except OSError:
            return None
        language = "en" if path.name.endswith(".en.vtt") else "ru"
        return {"language": language, "text": text}

    def _render(self, episode: _Episode, fields: set[str]) -> dict[str, Any]:
        program, presenters = episode.program, episode.presenters
        if {"program", "presenters"} & fields and (program is None or presenters is None):
            cms_program, cms_presenters = _cms_people(self.metadata_dir, episode.video_hash)
            program = program or cms_program
            presenters = presenters if presenters is not None else cms_presenters
        full: dict[str, Any] = {
            "content_id": episode.content_id,
            "video_hash": episode.video_hash,
            "title": episode.title,
            "program": program,
            "date": episode.date,
            "url": episode.url,
            "duration_seconds": episode.duration_seconds,
            "presenters": presenters or [],
            "summary": episode.summary,
            "summary_model": episode.summary_model,
            "summary_prompt_version": episode.summary_prompt_version,
            "tags": episode.tags,
            "updated_at": episode.updated_at,
        }
        out = {name: value for name, value in full.items() if name in fields}
        if "transcript" in fields:
            out["transcript"] = self._transcript(episode)
        return out

    def get(self, content_id: str, fields: set[str]) -> dict[str, Any] | None:
        """One episode, or None when the content_id is unknown."""
        self._reload_if_changed()
        episode = self._by_id.get(content_id)
        return None if episode is None else self._render(episode, fields)

    def page(
        self,
        fields: set[str],
        *,
        limit: int,
        cursor: str | None = None,
        since: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """A page of episodes, newest first, and the cursor for the next one.

        ``since`` keeps episodes whose ``updated_at`` is later than it, for
        incremental sync. The next cursor is None on the last page.
        """
        self._reload_if_changed()
        after = decode_cursor(cursor) if cursor else None
        items: list[_Episode] = []
        for episode in self._ordered:
            if after is not None and episode.sort_key >= after:
                continue
            if since is not None and episode.updated_at <= since:
                continue
            items.append(episode)
            if len(items) > limit:
                break
        has_more = len(items) > limit
        items = items[:limit]
        next_cursor = encode_cursor(items[-1].sort_key) if has_more and items else None
        return [self._render(e, fields) for e in items], next_cursor
