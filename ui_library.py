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
        "seed_search_help": "Поиск по названию и программе среди размеченных выпусков",
        "seed_pick": "Выпуск-образец",
        "no_seed_matches": "Ничего не нашлось — попробуйте другое слово.",
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
        "seed_search_help": "Searches titles and programmes of tagged episodes",
        "seed_pick": "Seed episode",
        "no_seed_matches": "No matches — try another word.",
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


def load_decisions(path: Path = DECISIONS_PATH) -> dict[str, str]:
    """youtube_id -> verdict, last decision wins (the file is append-only)."""
    decisions: dict[str, str] = {}
    if not path.exists():
        return decisions
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("youtube_id"):
                decisions[row["youtube_id"]] = row.get("verdict", "")
    return decisions


def append_decision(
    youtube_id: str, content_id: str | None, verdict: str, path: Path = DECISIONS_PATH
) -> None:
    """Append one editor verdict. Append-only for the same reason the tag file
    is: regenerating machine output must never be able to destroy these."""
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["youtube_id", "content_id", "verdict", "decided_at"])
        writer.writerow(
            [youtube_id, content_id or "", verdict, datetime.now(timezone.utc).isoformat()]
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


def _render_scored(rank: int, r: Scored, lang: str) -> None:
    e = r.episode
    title = e.title or _untitled(lang)
    line = f"**{rank}.** [{title}]({e.url})" if e.url else f"**{rank}.** {title}"
    meta = " · ".join(x for x in [e.program, e.date, _fmt_minutes(e.duration_seconds, lang)] if x)
    st.markdown(f"{line}  \n{meta}")
    st.caption(r.explain())


def render_similar_tab(episodes: list[Episode], lang: str) -> None:
    needle = st.text_input(
        _t("seed_search", lang), help=_t("seed_search_help", lang), key="library_seed_search"
    )
    matches = search_episodes(episodes, needle)
    if needle and not matches:
        st.info(_t("no_seed_matches", lang))
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

    speaker_col, theme_col = st.columns(2)
    with speaker_col:
        st.subheader(_t("same_speaker", lang))
        if not same:
            st.caption(_t("nothing_similar", lang))
        for i, r in enumerate(same[:10], 1):
            _render_scored(i, r, lang)
    with theme_col:
        st.subheader(_t("same_theme", lang))
        if not themed:
            st.caption(_t("nothing_similar", lang))
        for i, r in enumerate(themed[:10], 1):
            _render_scored(i, r, lang)


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
    matches = json.loads(MAP_PATH.read_text(encoding="utf-8"))
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
