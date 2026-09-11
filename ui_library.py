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
from types import SimpleNamespace
from typing import Any

import streamlit as st

from rainrag.library import GENRES
from rainrag.library_performance import METRIC_COLUMNS, aggregate, build_uploads, load_metrics
from rainrag.library_programs import load_programmes
from rainrag.library_similar import Episode, Scored, dedupe_latest, find_similar


REPO_ROOT = Path(__file__).resolve().parent
TAGS_PATH = REPO_ROOT / "data" / "library_tags.jsonl"
MAP_PATH = REPO_ROOT / "data" / "youtube_map.json"
DECISIONS_PATH = REPO_ROOT / "data" / "youtube_map_decisions.csv"
VIDEOS_CACHE_PATH = REPO_ROOT / "data" / "library_videos.jsonl"
# A YouTube Studio export (Analytics -> Advanced mode -> Export), columns renamed
# to the «YT metrics» sheet's schema. Absent until the channel owner drops one in.
METRICS_PATH = REPO_ROOT / "data" / "youtube_metrics.csv"
# Stand-in titles for tagged episodes with no CMS card (scripts/library_untitled_titles.py)
UNTITLED_TITLES_PATH = REPO_ROOT / "data" / "untitled_titles.json"
PROGRAMS_PATH = REPO_ROOT / "data" / "library_programs.csv"

_T = {
    "ru": {
        "no_tags": "Файл разметки не найден: {path}. Запустите library_tag_batch.py.",
        "tab_similar": "Похожие выпуски",
        "tab_youtube": "YouTube-сопоставление",
        "tab_perf": "Что смотрят",
        "perf_intro": "{n} роликов Библиотеки сопоставлены с архивом. Метрика: {metric}. "
        "Только просмотры пока публичные; CPM и удержание появятся, когда в data/ ляжет "
        "выгрузка из YouTube Studio.",
        "perf_metric": "Метрика",
        "perf_by_speaker": "По спикерам",
        "perf_by_program": "По программам",
        "perf_uploads": "Все ролики",
        "col_speaker": "спикер",
        "col_program": "программа",
        "col_uploads": "роликов",
        "col_total": "всего",
        "col_median": "медиана",
        "col_best": "лучший",
        "col_title": "ролик",
        "col_archive": "архивный выпуск",
        "col_speakers": "спикеры",
        "col_views": "просмотры",
        "col_published": "опубликован",
        "metric_views": "просмотры",
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
        "seed_untagged": "Выпуск найден в архиве, но ещё не размечен, поэтому подобрать похожие "
        "пока нельзя. Короткие выпуски (до 30 минут) в первую разметку не попадали; "
        "доразметка запланирована.",
        "untagged_mark": "(не размечен)",
        "no_cms_mark": "без карточки в CMS",
        "queue_note": "Сверка всех роликов канала с архивом. Очередь общая и не зависит от "
        "поиска во вкладке «Похожие выпуски».",
        "min_minutes": "Длительность от, мин",
        "genres": "Жанры",
        "same_speaker": "Тот же спикер",
        "same_theme": "Похожие темы",
        "nothing_similar": "Пересечений не нашлось.",
        "no_speaker": "У этого выпуска не указан спикер, поэтому подобрать «того же спикера» "
        "не получится. В карточке нет ни ведущего из CMS, ни гостя из расшифровки.",
        "presenter_demoted": "Ведущий не считается спикером в этом жанре: ищем по гостю.",
        "demoted_no_guest": "Ведущий не считается спикером в этом жанре, а гость в расшифровке "
        "не определился, поэтому подбирать не по кому.",
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
        "tab_perf": "What performs",
        "perf_intro": "{n} Library uploads are linked to archive episodes. Metric: {metric}. "
        "Only view counts are public so far; CPM and retention appear once a YouTube Studio "
        "export is placed in data/.",
        "perf_metric": "Metric",
        "perf_by_speaker": "By speaker",
        "perf_by_program": "By programme",
        "perf_uploads": "All uploads",
        "col_speaker": "speaker",
        "col_program": "programme",
        "col_uploads": "uploads",
        "col_total": "total",
        "col_median": "median",
        "col_best": "best",
        "col_title": "upload",
        "col_archive": "archive episode",
        "col_speakers": "speakers",
        "col_views": "views",
        "col_published": "published",
        "metric_views": "views",
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
        "seed_untagged": "This episode is in the archive but not tagged yet, so similar episodes "
        "cannot be suggested. Episodes under 30 minutes were outside the first tagging pass; "
        "a follow-up is planned.",
        "untagged_mark": "(untagged)",
        "no_cms_mark": "no CMS record",
        "queue_note": "Reviews every channel upload against the archive. The queue is global "
        "and independent of the search on the other tab.",
        "min_minutes": "Min duration, min",
        "genres": "Genres",
        "same_speaker": "Same speaker",
        "same_theme": "Similar subjects",
        "nothing_similar": "No overlap found.",
        "no_speaker": "This episode has no speaker recorded, so there is nothing to match on. "
        "Neither a CMS presenter nor a guest from the transcript is set.",
        "presenter_demoted": "The presenter does not count as a speaker in this genre; "
        "matching on the guest instead.",
        "demoted_no_guest": "The presenter does not count as a speaker in this genre and no "
        "guest was extracted from the transcript, so there is nobody to match on.",
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


def load_tagged_episodes(
    path: Path = TAGS_PATH, programs_path: Path = PROGRAMS_PATH
) -> list[Episode]:
    """The tagged pool, deduped, failures dropped — same rules as the eval.

    The programme table decides whether a presenter counts as a speaker. When
    it is absent the ranking falls back to presenter plus guest, as before.
    """
    programmes = load_programmes(programs_path)
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
        episodes.append(Episode.from_record(record, programmes))
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


def search_untagged(
    videos_by_hash: dict[str, Any],
    needle: str,
    limit: int = 50,
    exclude: set[str] | None = None,
) -> list[Episode]:
    """Title search over every indexed video, for seeds the tagger has not reached.

    150 of the 211 episodes the Library has actually published are under 30
    minutes and therefore untagged; a search that only knew the tagged pool
    reported them as nonexistent. They exist. Returned as Episodes with no
    subjects so the caller can show them and explain, not rank them.
    """
    needle = needle.strip().lower().replace("ё", "е")
    if not needle:
        return []
    # Excluded (already tagged) hashes are dropped before the cut, otherwise a
    # title shared by many tagged episodes would crowd the untagged ones out
    # of the top 50 and the feature would report none.
    exclude = exclude or set()
    hits: list[Episode] = []
    for h, v in videos_by_hash.items():
        if h in exclude:
            continue
        title = getattr(v, "title", None) or ""
        program = getattr(v, "program", None) or ""
        if needle in f"{title} {program}".lower().replace("ё", "е"):
            hits.append(
                Episode(
                    video_hash=h,
                    content_id=None,
                    title=title or None,
                    program=program or None,
                    date=getattr(v, "date", None),
                    duration_seconds=getattr(v, "duration_seconds", None),
                    url=getattr(v, "url", None),
                )
            )
    hits.sort(key=lambda e: e.date or "", reverse=True)
    return hits[:limit]


def search_episodes(
    episodes: list[Episode],
    needle: str,
    limit: int = 50,
    synthetic: dict[str, str] | None = None,
) -> list[Episode]:
    """Substring search over title and programme, newest first.

    A selectbox over 10k episodes is unusable; a search box narrowing to 50 is
    how an editor actually starts — she knows roughly what she is looking for.
    """
    needle = needle.strip().lower().replace("ё", "е")
    if not needle:
        return []
    synthetic = synthetic or {}
    hits = [
        e
        for e in episodes
        if needle
        in f"{e.title or synthetic.get(e.video_hash, '')} {e.program or ''}".lower().replace(
            "ё", "е"
        )
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


def load_untitled_titles(path: Path = UNTITLED_TITLES_PATH) -> dict[str, str]:
    """video_hash -> first transcript sentence, for episodes with no CMS card.

    27% of indexed videos have no CMS article and therefore no title; they
    are mostly the freshest content. Their transcript opening stands in so an
    editor can judge them, and the card says explicitly that the CMS has no
    record, so nobody goes looking for a page that does not exist.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        h: t.strip()
        for h, t in data.items()
        if isinstance(h, str) and isinstance(t, str) and t.strip()
    }


@st.cache_data(show_spinner=False)
def _cached_untitled_titles(mtime: float) -> dict[str, str]:
    del mtime
    return load_untitled_titles()


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~])")


def escape_markdown(text: str) -> str:
    """Neutralise markdown syntax in text that will be rendered by st.markdown.

    Stand-in titles come from transcripts, which are untrusted text: a
    sentence containing ``](`` or a backtick would otherwise break out of the
    link and inject markup into the editor's view. CMS titles never went
    through this path before, so it applies to stand-ins only.
    """
    return _MD_SPECIAL.sub(r"\\\1", text)


def empty_speaker_reason(seed: Episode) -> str:
    """Which message explains an empty «Тот же спикер» column.

    Varya reopened 86cbdbuv9 because "Пересечений не нашлось" reads as a
    ranking miss when the real cause is that there is nobody to match against.
    Three causes, and naming the wrong one is its own bug: telling an editor
    the card has no presenter is false when a presenter was set aside by the
    genre rule, which is the case for 270 episodes.
    """
    if seed.speakers:
        return "nothing_similar"
    if seed.presenter_demoted:
        return "demoted_no_guest"
    return "no_speaker"


def display_title(
    e: Episode, lang: str, synthetic: dict[str, str] | None = None
) -> tuple[str, bool]:
    """(title to show, whether it is a stand-in rather than a CMS title).

    Stand-ins are returned markdown-escaped; callers render them as-is.
    """
    if e.title:
        return e.title, False
    stand_in = (synthetic or {}).get(e.video_hash)
    if stand_in:
        return escape_markdown(stand_in), True
    return _untitled(lang), True


def _episode_label(e: Episode, synthetic: dict[str, str] | None = None) -> str:
    # A raw video hash means nothing to an editor; the date still narrows it.
    title, _ = display_title(e, "ru", synthetic)
    bits = [e.date or "????-??-??", title[:80]]
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
    synthetic: dict[str, str] | None = None,
) -> None:
    e = r.episode
    title, stand_in = display_title(e, lang, synthetic)
    line = f"**{rank}.** [{title}]({e.url})" if e.url else f"**{rank}.** {title}"
    meta_bits = [e.program, e.date, _fmt_minutes(e.duration_seconds, lang)]
    if stand_in:
        meta_bits.append(_t("no_cms_mark", lang))
    meta = " · ".join(x for x in meta_bits if x)
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
    synthetic = _cached_untitled_titles(
        UNTITLED_TITLES_PATH.stat().st_mtime if UNTITLED_TITLES_PATH.exists() else 0.0
    )
    matches = search_episodes(episodes, needle, synthetic=synthetic)
    tagged_hashes = {e.video_hash for e in episodes}
    if needle and not youtube_id_from_query(needle):
        videos = _cached_videos_by_hash(
            VIDEOS_CACHE_PATH.stat().st_mtime if VIDEOS_CACHE_PATH.exists() else 0.0
        )
        extra = search_untagged(videos, needle, exclude=tagged_hashes)
        matches = matches + extra[: max(0, 50 - len(matches))]
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
        format_func=lambda e: (
            _episode_label(e, synthetic)
            + ("" if e.video_hash in tagged_hashes else f" {_t('untagged_mark', lang)}")
        ),
        key="library_seed_pick",
    )
    if seed.video_hash not in tagged_hashes:
        st.info(_t("seed_untagged", lang))
        return
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
            st.caption(_t(empty_speaker_reason(seed), lang))
        elif seed.presenter_demoted:
            st.caption(_t("presenter_demoted", lang))
        for i, r in enumerate(same[:10], 1):
            _render_scored(
                i,
                r,
                lang,
                seed_id=seed.content_id,
                column="speaker",
                marks=marks,
                synthetic=synthetic,
            )
    with theme_col:
        st.subheader(_t("same_theme", lang))
        if not themed:
            st.caption(_t("nothing_similar", lang))
        for i, r in enumerate(themed[:10], 1):
            _render_scored(
                i,
                r,
                lang,
                seed_id=seed.content_id,
                column="theme",
                marks=marks,
                synthetic=synthetic,
            )


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


def _videos_by_hash() -> dict[str, Any]:
    """Programme and presenters for every archive video, keyed by hash.

    Most Library uploads are under 30 minutes and therefore outside the
    tagged pool; the videos cache covers all 139k videos, so the performance
    table does not depend on tagging.
    """
    if not VIDEOS_CACHE_PATH.exists():
        return {}
    out: dict[str, Any] = {}
    with open(VIDEOS_CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                out[row.get("video_key", "")] = SimpleNamespace(**row)
    return out


@st.cache_data(show_spinner=False)
def _cached_videos_by_hash(mtime: float) -> dict[str, Any]:
    del mtime
    return _videos_by_hash()


def render_performance_tab(episodes: list[Episode], lang: str) -> None:
    map_rows = load_map_rows()
    if not map_rows:
        st.warning(_t("map_missing", lang, path=MAP_PATH.name))
        return
    videos = _cached_videos_by_hash(
        VIDEOS_CACHE_PATH.stat().st_mtime if VIDEOS_CACHE_PATH.exists() else 0.0
    )
    tags_by_content = {
        e.content_id: {"presenter_cms": e.speakers, "guest": [], "program": e.program}
        for e in episodes
        if e.content_id
    }
    metrics = load_metrics(METRICS_PATH)
    uploads = build_uploads(map_rows, videos, tags_by_content, metrics)

    available = ["views"] + [
        c for c in METRIC_COLUMNS if c != "views" and any(c in u.metrics for u in uploads)
    ]
    metric = st.selectbox(
        _t("perf_metric", lang),
        available,
        format_func=lambda m: _t("metric_views", lang) if m == "views" else m,
        key="perf_metric",
    )
    st.caption(_t("perf_intro", lang, n=len(uploads), metric=metric))

    def _fmt(v: float | None) -> str:
        if v is None:
            return ""
        return f"{v:,.0f}".replace(",", " ") if metric == "views" else f"{v:,.2f}".replace(",", " ")

    def _val(u: Any) -> float | None:
        if metric == "views":
            return float(u.view_count) if u.view_count is not None else None
        return u.metrics.get(metric)

    def _table(rows: list[dict[str, Any]], key: str) -> None:
        st.dataframe(
            [
                {
                    _t(f"col_{key}", lang): r[key],
                    _t("col_uploads", lang): r["uploads"],
                    _t("col_total", lang): _fmt(r["total"]),
                    _t("col_median", lang): _fmt(r["median"]),
                    _t("col_best", lang): _fmt(r["best"]),
                }
                for r in rows[:30]
            ],
            hide_index=True,
            width="stretch",
        )

    sp_col, pr_col = st.columns(2)
    with sp_col:
        st.subheader(_t("perf_by_speaker", lang))
        _table(aggregate(uploads, "speaker", metric), "speaker")
    with pr_col:
        st.subheader(_t("perf_by_program", lang))
        _table(aggregate(uploads, "program", metric), "program")

    st.subheader(_t("perf_uploads", lang))
    # Uploads without the metric sort last and render blank, not as zero.
    ordered = sorted(uploads, key=lambda u: (_val(u) is None, -(_val(u) or 0.0)))
    st.dataframe(
        [
            {
                _t("col_title", lang): u.youtube_title,
                _t("col_archive", lang): u.archive_title or u.content_id,
                _t("col_program", lang): u.program or "",
                _t("col_speakers", lang): ", ".join(u.speakers),
                _t("col_views", lang) if metric == "views" else metric: _fmt(_val(u)),
                _t("col_published", lang): u.published_at or "",
                "youtube": f"https://youtu.be/{u.youtube_id}",
            }
            for u in ordered
        ],
        hide_index=True,
        width="stretch",
        column_config={"youtube": st.column_config.LinkColumn()},
    )


@st.cache_data(show_spinner=False)
def _cached_feedback(cache_key: tuple[int, int]) -> dict[tuple[str, str], str]:
    """Feedback marks, re-parsed only when the file changes.

    The page reruns on every widget interaction and this file grows without
    bound, so an uncached read is a per-click cost that only ever rises."""
    del cache_key
    return load_feedback()


@st.cache_data(show_spinner=False)
def _cached_episodes(mtime: float, programs_mtime: float) -> list[Episode]:
    """Cache keyed on the tag file's mtime, so a finished tagging run shows up
    on the next interaction without a service restart. The programme table is
    in the key too: editing a genre in the sheet changes who counts as a
    speaker, and that must not need a restart either."""
    del mtime, programs_mtime
    return load_tagged_episodes()


def render_library_mode(lang: str) -> None:
    if not TAGS_PATH.exists():
        st.warning(_t("no_tags", lang, path=TAGS_PATH.name))
        return
    programs_mtime = PROGRAMS_PATH.stat().st_mtime if PROGRAMS_PATH.exists() else 0.0
    episodes = _cached_episodes(TAGS_PATH.stat().st_mtime, programs_mtime)
    similar_tab, perf_tab, youtube_tab = st.tabs(
        [_t("tab_similar", lang), _t("tab_perf", lang), _t("tab_youtube", lang)]
    )
    with similar_tab:
        render_similar_tab(episodes, lang)
    with perf_tab:
        render_performance_tab(episodes, lang)
    with youtube_tab:
        render_youtube_tab(episodes, lang)
