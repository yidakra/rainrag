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
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
    program: str | None = None
    presenters: list[str] | None = None
    tags: dict[str, list[str]] | None = None
    summary: str | None = None
    summary_model: str | None = None
    summary_prompt_version: str | None = None
    # When this record's content last changed (ISO 8601, UTC), from the state file.
    updated_at: str = ""

    def fingerprint(self) -> str:
        """A digest of everything the feed serves except the transcript."""
        content = [
            self.video_hash,
            self.title,
            self.date,
            self.url,
            self.duration_seconds,
            self.program,
            self.presenters,
            self.tags,
            self.summary,
            self.summary_model,
            self.summary_prompt_version,
        ]
        raw = json.dumps(content, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

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


class EpisodeStoreUnavailableError(RuntimeError):
    """The index cannot be read and there is no earlier snapshot to serve."""


def iso_utc(moment: datetime) -> str:
    """The one timestamp format of the feed: ISO 8601, UTC, microseconds.

    A single fixed-width form keeps string comparison equal to time order, and
    microseconds keep two changes within one second apart.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of an optional JSONL source; a missing file is no rows.

    Any other read error propagates, so the store keeps its previous snapshot
    instead of publishing records stripped of their tags or summaries.
    """
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
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


class EpisodeStore:
    """Thread-safe, mtime-reloading index of episodes by content_id."""

    def __init__(
        self,
        index_path: Path,
        tags_path: Path,
        summaries_path: Path,
        archive_root: Path,
        state_path: Path | None = None,
        clock: Callable[[], datetime] = _now,
    ) -> None:
        """Remember the sources; nothing is read until the first request.

        ``state_path`` keeps each episode's content fingerprint and the time it
        last changed, so ``updated_at`` survives restarts and moves only when an
        episode's own record changes, not when the daily index is rewritten.
        """
        self.index_path = index_path
        self.tags_path = tags_path
        self.summaries_path = summaries_path
        self.archive_root = archive_root
        self.state_path = state_path
        self._clock = clock
        self._state: dict[str, list[str]] | None = None
        self._last_stamp = ""
        # A failed state write is retried on the next request, so a recovered
        # disk gets the stamps before a restart can lose them.
        self._state_pending = False
        self._lock = threading.Lock()
        self._stamp: tuple[float | None, ...] | None = None
        self._by_id: dict[str, _Episode] = {}
        self._ordered: list[_Episode] = []

    def _reload_if_changed(self) -> None:
        stamp = tuple(_mtime(p) for p in (self.index_path, self.tags_path, self.summaries_path))
        with self._lock:
            if stamp == self._stamp:
                if self._state_pending:
                    self._write_state()
                return
            try:
                by_id = self._build()
            except (OSError, ValueError) as exc:
                # Keep serving the last good snapshot and leave the reload
                # pending: an unreadable index must not become an empty feed.
                logger.warning(f"Episode store: cannot read {self.index_path}: {exc}")
                if self._stamp is None:
                    raise EpisodeStoreUnavailableError(str(exc)) from exc
                return
            self._by_id = by_id
            self._ordered = sorted(by_id.values(), key=lambda e: e.sort_key, reverse=True)
            self._stamp = stamp
            logger.info(f"Episode store loaded {len(by_id)} episodes")

    def _build(self) -> dict[str, _Episode]:
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            raise ValueError("the index is not a JSON object")
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
                program=row.get("program"),
                presenters=list(row.get("presenters") or []),
            )
        self._merge_tags(by_id)
        self._merge_summaries(by_id)
        self._stamp_changes(by_id)
        return by_id

    def _load_state(self) -> dict[str, list[str]]:
        if self._state is None:
            self._state = {}
            if self.state_path is not None:
                try:
                    data = json.loads(self.state_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, list) and len(value) == 2:
                                stamp = iso_utc(datetime.fromisoformat(value[1]))
                                self._state[str(key)] = [value[0], stamp]
                                self._last_stamp = max(self._last_stamp, stamp)
                except (OSError, ValueError, TypeError):
                    pass
        return self._state

    def _next_stamp(self) -> str:
        """Now, but always later than any stamp already issued.

        Two loads within one tick, or a clock step back, must still give a
        changed record a later updated_at than the version a caller cached.
        """
        stamp = iso_utc(self._clock())
        if stamp <= self._last_stamp:
            stamp = iso_utc(datetime.fromisoformat(self._last_stamp) + timedelta(microseconds=1))
        self._last_stamp = stamp
        return stamp

    def _stamp_changes(self, by_id: dict[str, _Episode]) -> None:
        """Set updated_at from the state file; new or changed records get now."""
        state = self._load_state()
        now: str | None = None
        changed = False
        for content_id, episode in by_id.items():
            fingerprint = episode.fingerprint()
            known = state.get(content_id)
            if known and known[0] == fingerprint:
                episode.updated_at = known[1]
            else:
                now = now or self._next_stamp()
                episode.updated_at = now
                state[content_id] = [fingerprint, now]
                changed = True
        for gone in set(state) - set(by_id):
            del state[gone]
            changed = True
        if changed or self._state_pending:
            self._write_state()

    def _write_state(self) -> None:
        """Atomically write the state file; on failure, retry on the next request."""
        if self.state_path is None or self._state is None:
            self._state_pending = False
            return
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError as exc:
            self._state_pending = True
            logger.warning(f"Episode store: cannot write {self.state_path}: {exc}")
        else:
            self._state_pending = False

    def _merge_tags(self, by_id: dict[str, _Episode]) -> None:
        if not self.tags_path.exists():
            return
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
            episode.program = episode.program or row.get("program")
            episode.presenters = episode.presenters or list(row.get("presenter_cms") or [])
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

    def _merge_summaries(self, by_id: dict[str, _Episode]) -> None:
        if not self.summaries_path.exists():
            return
        for row in _read_jsonl(self.summaries_path):
            episode = by_id.get(str(row.get("content_id") or ""))
            if episode is None or not row.get("summary"):
                continue
            episode.summary = row["summary"]
            episode.summary_model = row.get("model")
            episode.summary_prompt_version = row.get("prompt_version")

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
        full: dict[str, Any] = {
            "content_id": episode.content_id,
            "video_hash": episode.video_hash,
            "title": episode.title,
            "program": episode.program,
            "date": episode.date,
            "url": episode.url,
            "duration_seconds": episode.duration_seconds,
            "presenters": episode.presenters or [],
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
