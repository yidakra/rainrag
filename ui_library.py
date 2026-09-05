"""«Библиотека Дождя» mode for the Streamlit frontend.

Two jobs, both editorial and neither served by the chat UI:

*Похожие выпуски* answers Varya's first query — given an episode, what else
should the Library publish — over the LLM-tagged pool, split the way she
actually judges results: episodes by the same speaker first, then episodes
whose subjects overlap. One merged list would bury the theme matches: for a
prolific speaker the same-speaker scores (3+) sit above every theme score
(<0.2) by construction.

*YouTube-сопоставление* turns the automatic upload→episode matching, which is
measurably not trustworthy (1 of 3 on held-out pairs, confidently wrong), into
a review queue. Editor verdicts append to a decisions file that regenerating
the map never touches — the same rule as the genre table: machine output is
replaceable, editorial decisions are not.

Pure file-backed: reads the tag file and the map, writes decisions. No Qdrant,
no embedding model, no API calls — the page must stay fast and must not
compete with a reindex for memory.
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading


try:
    import fcntl
except ImportError:  # Windows dev box: no flock.
    fcntl = None
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rainrag.library import GENRES
from rainrag.library_similar import Episode, Scored, dedupe_latest, find_similar


REPO_ROOT = Path(__file__).resolve().parent
TAGS_PATH = REPO_ROOT / "data" / "library_tags.jsonl"
MAP_PATH = REPO_ROOT / "data" / "youtube_map.json"
DECISIONS_PATH = REPO_ROOT / "data" / "youtube_map_decisions.csv"

_T = {
    "ru": {
        "no_tags": "Файл разметки не найден: {path}. Запустите library_tag_batch.py.",
        "tab_similar": "Похожие выпуски",
        "tab_youtube": "YouTube-сопоставление",
        "seed_search": "Найти выпуск-образец",
        "seed_search_help": "Название выпуска или программы, либо ссылка на ролик YouTube. "
        "Ищем среди размеченных выпусков (в разметку попадает всё длиннее 30 минут).",
        "seed_pick": "Выпуск-образец",
        "no_seed_matches": "Ничего не нашлось среди {n} размеченных выпусков (ещё ~3 200 "
        "длинных ждут разметки, короткие в разметку не попадают). Попробуйте другое "
        "слово или вставьте ссылку на ролик YouTube.",
        "yt_resolved_untagged": "Ролик сопоставлен с архивным выпуском {cid}, но тот ещё не "
        "размечен, поэтому подобрать похожие пока нельзя.",
        "yt_unknown": "Этот ролик ещё не сопоставлен с архивом — его можно подтвердить во "
        "вкладке «YouTube-сопоставление».",
        "queue_note": "Сверка всех роликов канала с архивом. Очередь общая и не зависит от "
        "поиска во вкладке «Похожие выпуски».",
        "min_minutes": "Длительность от, мин",
        "genres": "Жанры",
        "same_speaker": "Тот же спикер",
        "same_theme": "Похожие темы",
        "nothing_similar": "Пересечений не нашлось.",
        "map_missing": "Файл сопоставления не найден: {path}. Запустите youtube_map.py.",
        "review_done": "Всё проверено: {n} решений.",
        "review_stats": "Подтверждено: {ok} · Отклонено: {no} · Осталось: {left}",
        "yt_side": "На YouTube",
        "arc_side": "Кандидат в архиве",
        "no_candidate": "Кандидат не найден автоматически",
        "btn_match": "✅ Совпадает",
        "btn_no": "❌ Не то",
        "btn_skip": "Пропустить",
        "views": "просмотров",
        "min_short": "мин",
    },
    "en": {
        "no_tags": "Tag file not found: {path}. Run library_tag_batch.py.",
        "tab_similar": "Similar episodes",
        "tab_youtube": "YouTube matching",
        "seed_search": "Find a seed episode",
        "seed_search_help": "An episode or programme title, or a YouTube link. "
        "Searches tagged episodes (tagging covers everything over 30 minutes).",
        "seed_pick": "Seed episode",
        "no_seed_matches": "No matches among {n} tagged episodes (~3,200 long ones await "
        "tagging; short episodes are not tagged). Try another word or paste a YouTube link.",
        "yt_resolved_untagged": "This upload maps to archive episode {cid}, which is not tagged "
        "yet, so similar episodes cannot be suggested.",
        "yt_unknown": "This upload is not matched to the archive yet — you can confirm it in "
        "the YouTube matching tab.",
        "queue_note": "Reviews every channel upload against the archive. The queue is global "
        "and independent of the search on the other tab.",
        "min_minutes": "Min duration, min",
        "genres": "Genres",
        "same_speaker": "Same speaker",
        "same_theme": "Similar subjects",
        "nothing_similar": "No overlap found.",
        "map_missing": "Map file not found: {path}. Run youtube_map.py.",
        "review_done": "All reviewed: {n} decisions.",
        "review_stats": "Confirmed: {ok} · Rejected: {no} · Remaining: {left}",
        "yt_side": "On YouTube",
        "arc_side": "Archive candidate",
        "no_candidate": "No automatic candidate",
        "btn_match": "✅ Match",
        "btn_no": "❌ Wrong",
        "btn_skip": "Skip",
        "views": "views",
        "min_short": "min",
    },
}


def _t(key: str, lang: str, **kw: object) -> str:
    return _T.get(lang, _T["ru"]).get(key, key).format(**kw)


# ---------------------------------------------------------------- data access


def load_tagged_episodes(path: Path = TAGS_PATH) -> list[Episode]:
    """The tagged pool, deduped, failures dropped — same rules as the eval."""
    episodes: list[Episode] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue  # a row mid-write during a tagging run
        if record.get("error"):
            continue
        episodes.append(Episode.from_record(record))
    return dedupe_latest(episodes)


_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "youtube-nocookie.com"}
_YT_PATH_ID = re.compile(r"(?:^|[?&]v=|/shorts/|/live/|/embed/|^/)([A-Za-z0-9_-]{11})(?:[/?&#]|$)")
_YT_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def youtube_id_from_query(text: str) -> str | None:
    """Pull a YouTube video id out of a search query, if there is one.

    Only YouTube hosts count: a pasted link to some other site with a v=
    parameter is not a YouTube id and must fall through to title search. A
    bare 11-character token is treated as an id only when it also carries a
    digit, "-", "_" or mixed case: a lowercase English word like
    "managements" is a plausible title search, while real ids are
    base64-flavoured and all-lowercase-letters ones are vanishingly rare.
    """
    text = text.strip()
    if "//" in text or text.startswith(
        ("youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com")
    ):
        from urllib.parse import urlparse

        parsed = urlparse(text if "//" in text else f"https://{text}")
        if (parsed.hostname or "").lower() not in _YT_HOSTS:
            return None
        m = _YT_PATH_ID.search(f"{parsed.path}?{parsed.query}")
        return m.group(1) if m else None
    if _YT_BARE_ID.fullmatch(text) and (
        any(c.isdigit() or c in "-_" for c in text)
        or (text != text.lower() and text != text.upper())
    ):
        return text
    return None


def load_map_rows(path: Path = MAP_PATH) -> list[dict]:
    """The upload->archive map, or empty while absent, mid-regeneration or torn."""
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    # a syntactically valid file can still hold garbage rows
    return [row for row in rows if isinstance(row, dict)]


def resolve_youtube_id(yt_id: str) -> str | None:
    """Archive content_id for an upload, from editor truth only.

    Review-tab confirmations win, read from the decisions file, which stores
    the confirmed content_id precisely so that a later map regeneration
    cannot silently repoint the link. After that the map's editor/exact/
    strong rows: audited against 211 hand-made pairs, the confident tier was
    23/23 correct while the review band was right ~40% of the time, so the
    review band is deliberately not used here.
    """
    verdict = None
    verdict_cid = ""
    for row in _locked_read_rows(DECISIONS_PATH):
        if row.get("youtube_id") == yt_id:
            verdict = row.get("verdict")
            verdict_cid = (row.get("content_id") or "").strip()
    if verdict == "match" and verdict_cid:
        return verdict_cid
    if verdict == "no_match":
        # The editor explicitly rejected the map's candidate; resolving it
        # anyway would override a human with string similarity.
        return None
    for m in load_map_rows():
        if m.get("youtube_id") == yt_id:
            if m.get("confidence") in ("editor", "exact", "strong") and m.get("content_id"):
                return str(m["content_id"])
            break
    return None


def search_episodes(episodes: list[Episode], needle: str, limit: int = 50) -> list[Episode]:
    """Substring search over title and programme, newest first.

    A selectbox over 10k episodes is unusable; a search box narrowing to 50 is
    how an editor actually starts — she knows roughly what she is looking for.
    """
    needle = needle.strip().lower().replace("ё", "е")
    if not needle:
        return []
    hits = [
        e
        for e in episodes
        if needle in f"{e.title or ''} {e.program or ''}".lower().replace("ё", "е")
    ]
    hits.sort(key=lambda e: e.date or "", reverse=True)
    return hits[:limit]


def split_by_speaker(results: list[Scored]) -> tuple[list[Scored], list[Scored]]:
    """Same-speaker matches and theme-only matches, as two lists.

    For a prolific speaker every same-speaker score exceeds every theme score
    (SPEAKER_WEIGHT alone is 3.0 against subject scores below 1), so a merged
    top-10 is all one speaker and the theme matches Varya also expects are
    structurally invisible. Two lists is the fix, not a bigger cut-off.
    """
    same = [r for r in results if r.shared_speakers]
    themed = [r for r in results if not r.shared_speakers]
    return same, themed


def _append_csv_row(path: Path, header: list[str], row: list[object]) -> None:
    """Append one row, writing the header exactly once, under a file lock.

    Verdicts arrive from concurrent editor sessions -- and from two separate
    Streamlit *processes* (the public and the IP-restricted service), so an
    in-process lock cannot serialize them. An exclusive flock over the whole
    write does: the size check decides the header and no two rows interleave.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f, _FileLock(f, exclusive=True):
        writer = csv.writer(f)
        if os.fstat(f.fileno()).st_size == 0:
            writer.writerow(header)
        writer.writerow(row)
        f.flush()


# Production runs two Streamlit processes on Linux, where flock serializes
# them. On a platform without fcntl (a Windows dev box running one process),
# a process-wide lock is the honest equivalent of the same guarantee.
_FALLBACK_LOCK = threading.Lock()


class _FileLock:
    """Exclusive or shared flock when available, process-wide lock otherwise."""

    def __init__(self, f: object, exclusive: bool) -> None:
        self._f = f
        self._exclusive = exclusive

    def __enter__(self) -> None:
        if fcntl is not None:
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH)  # type: ignore[attr-defined]
        else:
            _FALLBACK_LOCK.acquire()

    def __exit__(self, *exc: object) -> None:
        if fcntl is not None:
            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        else:
            _FALLBACK_LOCK.release()


def _locked_read_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a verdict CSV, read under a shared lock.

    Writers hold an exclusive flock for the whole append; taking the shared
    counterpart here means a read never parses a torn final line from a write
    in flight in the other Streamlit process.
    """
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f, _FileLock(f, exclusive=False):
        return list(csv.DictReader(f))


def load_decisions(path: Path = DECISIONS_PATH) -> dict[str, str]:
    """youtube_id -> verdict, last decision wins (the file is append-only)."""
    decisions: dict[str, str] = {}
    for row in _locked_read_rows(path):
        if row.get("youtube_id"):
            decisions[row["youtube_id"]] = row.get("verdict", "")
    return decisions


def append_decision(
    youtube_id: str, content_id: str | None, verdict: str, path: Path = DECISIONS_PATH
) -> None:
    """Append one editor verdict. Append-only for the same reason the tag file
    is: regenerating machine output must never be able to destroy these."""
    _append_csv_row(
        path,
        ["youtube_id", "content_id", "verdict", "decided_at"],
        [youtube_id, content_id or "", verdict, datetime.now(timezone.utc).isoformat()],
    )


FEEDBACK_PATH = REPO_ROOT / "data" / "library_feedback.csv"


def load_feedback(path: Path = FEEDBACK_PATH) -> dict[tuple[str, str], str]:
    """(seed, candidate) -> verdict, last one wins.

    Deliberately keyed by the pair, not by (pair, column): the two columns
    partition one result list, so a candidate appears in exactly one of them
    per render, and when re-tagging later moves a pair across columns the
    editor's judgment should follow it -- they rated the suggestion, not the
    column it was displayed in. The CSV keeps ``column`` for analysis.
    """
    marks: dict[tuple[str, str], str] = {}
    for row in _locked_read_rows(path):
        if row.get("seed_content_id") and row.get("candidate_content_id"):
            marks[(row["seed_content_id"], row["candidate_content_id"])] = row.get("verdict", "")
    return marks


def append_feedback(
    seed_content_id: str,
    candidate_content_id: str,
    column: str,
    rank: int,
    verdict: str,
    path: Path = FEEDBACK_PATH,
) -> None:
    """One editor judgment on one suggestion.

    This file is the ground truth the ranking work is starved of: every
    scorer variant tested against Varya's first query hit the same wall --
    two labelled examples cannot distinguish tuning from overfitting (see
    data/exp/THEME_RANKING.md). Judgments recorded here grow that set as a
    side effect of editors doing their normal work.
    """
    _append_csv_row(
        path,
        ["seed_content_id", "candidate_content_id", "column", "rank", "verdict", "at"],
        [
            seed_content_id,
            candidate_content_id,
            column,
            rank,
            verdict,
            datetime.now(timezone.utc).isoformat(),
        ],
    )


# ------------------------------------------------------------------ rendering


def _fmt_minutes(seconds: float | None, lang: str) -> str:
    return f"{seconds / 60:.0f} {_t('min_short', lang)}" if seconds else "—"


def _untitled(lang: str) -> str:
    return "(без названия)" if lang == "ru" else "(untitled)"


def _episode_label(e: Episode) -> str:
    # A raw video hash means nothing to an editor; the date still narrows it.
    bits = [e.date or "????-??-??", (e.title or _untitled("ru"))[:80]]
    if e.program:
        bits.append(f"({e.program})")
    return " — ".join(bits)


def _render_scored(
    rank: int,
    r: Scored,
    lang: str,
    *,
    seed_id: str | None = None,
    column: str = "",
    marks: dict[tuple[str, str], str] | None = None,
) -> None:
    e = r.episode
    title = e.title or _untitled(lang)
    line = f"**{rank}.** [{title}]({e.url})" if e.url else f"**{rank}.** {title}"
    meta = " · ".join(x for x in [e.program, e.date, _fmt_minutes(e.duration_seconds, lang)] if x)
    body, up, down = st.columns([12, 1, 1])
    with body:
        st.markdown(f"{line}  \n{meta}")
        st.caption(r.explain())
    if seed_id and e.content_id:
        mark = (marks or {}).get((seed_id, e.content_id))
        key = f"fb_{column}_{seed_id}_{e.content_id}"
        if up.button("✓" if mark == "good" else "👍", key=f"{key}_g", disabled=mark == "good"):
            append_feedback(seed_id, e.content_id, column, rank, "good")
            st.rerun()
        if down.button("✗" if mark == "bad" else "👎", key=f"{key}_b", disabled=mark == "bad"):
            append_feedback(seed_id, e.content_id, column, rank, "bad")
            st.rerun()


def render_similar_tab(episodes: list[Episode], lang: str) -> None:
    needle = st.text_input(
        _t("seed_search", lang), help=_t("seed_search_help", lang), key="library_seed_search"
    )
    matches = search_episodes(episodes, needle)
    yt_id = youtube_id_from_query(needle) if needle else None
    if not matches and yt_id:
        cid = resolve_youtube_id(yt_id)
        if cid:
            matches = [e for e in episodes if e.content_id == cid]
            if not matches:
                st.info(_t("yt_resolved_untagged", lang, cid=cid))
                return
        else:
            st.info(_t("yt_unknown", lang))
            return
    if needle and not matches:
        st.info(_t("no_seed_matches", lang, n=len(episodes)))
    if not matches:
        return
    seed = st.selectbox(
        _t("seed_pick", lang),
        matches,
        format_func=_episode_label,
        key="library_seed_pick",
    )
    filter_col, genre_col = st.columns([1, 2])
    with filter_col:
        min_minutes = st.number_input(
            _t("min_minutes", lang), min_value=0, value=30, step=5, key="library_min_minutes"
        )
    with genre_col:
        genres = st.multiselect(_t("genres", lang), list(GENRES), default=[], key="library_genres")

    results = find_similar(
        seed,
        episodes,
        min_duration_minutes=min_minutes or None,
        genres=genres or None,
        limit=len(episodes),
    )
    same, themed = split_by_speaker(results)

    try:
        stat = FEEDBACK_PATH.stat()
        # mtime alone has coarse resolution on some filesystems; two rapid
        # verdicts could share it and pin a stale cache. Size breaks the tie
        # (the file is append-only, so it grows on every write).
        cache_key = (stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        cache_key = (0, 0)
    marks = _cached_feedback(cache_key)
    speaker_col, theme_col = st.columns(2)
    with speaker_col:
        st.subheader(_t("same_speaker", lang))
        if not same:
            st.caption(_t("nothing_similar", lang))
        for i, r in enumerate(same[:10], 1):
            _render_scored(i, r, lang, seed_id=seed.content_id, column="speaker", marks=marks)
    with theme_col:
        st.subheader(_t("same_theme", lang))
        if not themed:
            st.caption(_t("nothing_similar", lang))
        for i, r in enumerate(themed[:10], 1):
            _render_scored(i, r, lang, seed_id=seed.content_id, column="theme", marks=marks)


_CONFIDENCE_ORDER = {"exact": 0, "strong": 1, "review": 2, "none": 3, "editor": 4}


def review_queue(matches: list[dict], decisions: dict[str, str]) -> list[dict]:
    """Undecided uploads, most confident first, so the easy confirms go fast."""
    queue = [
        m
        for m in matches
        if m.get("youtube_id") not in decisions and m.get("confidence") != "editor"
    ]
    queue.sort(
        key=lambda m: (
            _CONFIDENCE_ORDER.get(m.get("confidence", "none"), 9),
            -(m.get("score") or 0),
        )
    )
    return queue


def render_youtube_tab(episodes: list[Episode], lang: str) -> None:
    if not MAP_PATH.exists():
        st.warning(_t("map_missing", lang, path=MAP_PATH.name))
        return
    st.caption(_t("queue_note", lang))
    matches = load_map_rows()
    if not matches:
        st.warning(_t("map_missing", lang, path=MAP_PATH.name))
        return
    decisions = load_decisions()
    queue = review_queue(matches, decisions)

    ok = sum(1 for v in decisions.values() if v == "match")
    no = sum(1 for v in decisions.values() if v == "no_match")
    st.caption(_t("review_stats", lang, ok=ok, no=no, left=len(queue)))
    if not queue:
        st.success(_t("review_done", lang, n=len(decisions)))
        return

    by_content = {e.content_id: e for e in episodes if e.content_id}
    item = queue[0]
    yt_id = item["youtube_id"]

    yt_col, arc_col = st.columns(2)
    with yt_col:
        st.subheader(_t("yt_side", lang))
        st.markdown(f"[{item.get('youtube_title', yt_id)}](https://youtu.be/{yt_id})")
        views = item.get("view_count")
        st.caption(
            " · ".join(
                x
                for x in [
                    item.get("published_at"),
                    _fmt_minutes(item.get("duration_seconds"), lang),
                    f"{views:,} {_t('views', lang)}".replace(",", " ") if views else None,
                ]
                if x
            )
        )
    with arc_col:
        st.subheader(_t("arc_side", lang))
        cid = item.get("content_id")
        episode = by_content.get(str(cid)) if cid else None
        if cid:
            title = item.get("archive_title") or cid
            # Most candidates are outside the tagged pool, so the map's own
            # archive fields are the usual source; the tagged episode wins
            # when present because its metadata is fresher.
            url = (episode.url if episode else None) or item.get("archive_url")
            date = (episode.date if episode else None) or item.get("archive_date")
            duration = (episode.duration_seconds if episode else None) or item.get(
                "archive_duration_seconds"
            )
            st.markdown(f"[{title}]({url})" if url else title)
            st.caption(
                " · ".join(
                    x
                    for x in [
                        date,
                        _fmt_minutes(duration, lang),
                        f"score {item.get('score', 0):.2f} ({item.get('confidence')})",
                    ]
                    if x
                )
            )
        else:
            st.caption(_t("no_candidate", lang))

    b_match, b_no, b_skip = st.columns(3)
    if b_match.button(_t("btn_match", lang), key=f"m_{yt_id}", disabled=not cid):
        append_decision(yt_id, str(cid), "match")
        st.rerun()
    if b_no.button(_t("btn_no", lang), key=f"n_{yt_id}"):
        append_decision(yt_id, str(cid) if cid else None, "no_match")
        st.rerun()
    if b_skip.button(_t("btn_skip", lang), key=f"s_{yt_id}"):
        append_decision(yt_id, str(cid) if cid else None, "skip")
        st.rerun()


@st.cache_data(show_spinner=False)
def _cached_feedback(cache_key: tuple[int, int]) -> dict[tuple[str, str], str]:
    """Feedback marks, re-parsed only when the file changes.

    The page reruns on every widget interaction and this file grows without
    bound, so an uncached read is a per-click cost that only ever rises."""
    del cache_key
    return load_feedback()


@st.cache_data(show_spinner=False)
def _cached_episodes(mtime: float) -> list[Episode]:
    """Cache keyed on the tag file's mtime, so a finished tagging run shows up
    on the next interaction without a service restart."""
    del mtime
    return load_tagged_episodes()


def render_library_mode(lang: str) -> None:
    if not TAGS_PATH.exists():
        st.warning(_t("no_tags", lang, path=TAGS_PATH.name))
        return
    episodes = _cached_episodes(TAGS_PATH.stat().st_mtime)
    similar_tab, youtube_tab = st.tabs([_t("tab_similar", lang), _t("tab_youtube", lang)])
    with similar_tab:
        render_similar_tab(episodes, lang)
    with youtube_tab:
        render_youtube_tab(episodes, lang)
