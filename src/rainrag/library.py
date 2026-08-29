"""The Библиотека Дождя content model: rubric → programme → episode.

The archive is indexed for a journalist hunting a quote: five-minute chunks,
retrieved by meaning. The Library needs the opposite shape — whole episodes and
whole программы, chosen by genre, speaker and subject, so an editor can plan a
month of uploads without watching tape.

This module holds the vocabulary for that: the asset hierarchy Varya specified,
the per-episode fields her tagging scheme uses, and the programme→genre mapping
that lets genre be answered today rather than after a CMS migration.

What the CMS gives us, and what it does not:

    presenter      partly. For a lecture the CMS `presentors` really is the
                   lecturer; for an interview it is the host, and the guest
                   appears nowhere but the title.
    guest          no. Absent from the CMS entirely.
    mentioned      yes -- CMS tags with category=person are people discussed.
    rubric         yes -- CMS tags with category=lite are Дождь Lite sections.
    programme      yes -- CMS `teleshow`.
    genre          no, but it is a property of the programme, not the episode,
                   so a reviewed lookup table covers it.
    subject        no. This is the bulk of the tagging work.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# Varya's asset hierarchy. `parent_id` points an episode at its programme and a
# programme at its rubric, so a query can start anywhere in the tree.
ASSET_EPISODE = "episode"
ASSET_PROGRAM = "program"
ASSET_RUBRIC = "rubric"

# Genres as they appear in her sheet. Free text would drift immediately.
GENRES = (
    "лекция",
    "мастер-класс",
    "интервью",
    "ток-шоу",
    "новости",
    "репортаж",
    "документальный фильм",
    "дискуссия",
    "обзор",
)

# Stated explicitly in the Content samples tab; these are editorial calls, not
# inferences, so they are not overridden by the keyword heuristic below.
CONFIRMED_PROGRAM_GENRES: dict[str, tuple[str, ...]] = {
    "Лекции на Дожде": ("лекция", "мастер-класс"),
    "Сто лекций с Дмитрием Быковым": ("лекция",),
    "Синдеева": ("интервью",),
}

# Weak signals from the programme's own name, used only to draft rows for
# review. Order matters: the first match wins.
_GENRE_HINTS: tuple[tuple[str, str], ...] = (
    ("лекци", "лекция"),
    ("мастер-класс", "мастер-класс"),
    ("интервью", "интервью"),
    ("новости", "новости"),
    ("диалог", "дискуссия"),
    ("круглый стол", "дискуссия"),
    ("прямая линия", "дискуссия"),
    ("разбор", "обзор"),
    ("итоги", "обзор"),
)


class LibraryAsset(BaseModel):
    """One row of Varya's tagging sheet.

    Deliberately mirrors her column names rather than the CMS field names: the
    sheet is the contract the editors work against, and a rename in translation
    is a bug waiting to be argued about.
    """

    content_id: str | None = None
    asset_type: str = ASSET_EPISODE
    title: str | None = None
    parent_id: str | None = None
    duration_seconds: float | None = None
    released_date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    youtube_id: str | None = None

    presenter: list[str] = Field(default_factory=list)
    correspondent: list[str] = Field(default_factory=list)
    guest: list[str] = Field(default_factory=list)
    mentioned: list[str] = Field(default_factory=list)
    organization: list[str] = Field(default_factory=list)
    subject: list[str] = Field(default_factory=list)
    place: list[str] = Field(default_factory=list)
    genre: list[str] = Field(default_factory=list)

    # Provenance, so a reviewer can tell a machine guess from an editorial fact.
    video_hash: str | None = None
    source: str = "cms"


def draft_genre(program: str) -> tuple[str, ...]:
    """Best guess at a programme's genre from its name, for review.

    A guess, explicitly: the output belongs in a CSV an editor corrects, never
    straight into the index. Programmes whose name says nothing return ().
    """
    confirmed = CONFIRMED_PROGRAM_GENRES.get(program)
    if confirmed:
        return confirmed
    lowered = program.lower()
    for needle, genre in _GENRE_HINTS:
        if needle in lowered:
            return (genre,)
    return ()


def load_program_genres(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read the reviewed programme→genre table.

    Missing file means "nobody has reviewed it yet", which is a normal state and
    returns an empty mapping rather than raising: genre simply goes unfiltered.
    """
    table: dict[str, tuple[str, ...]] = {}
    file_path = Path(path)
    if not file_path.exists():
        return table
    with open(file_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            program = (row.get("program") or "").strip()
            if not program:
                continue
            genres = tuple(g.strip() for g in (row.get("genre") or "").split(";") if g.strip())
            if genres:
                table[program] = genres
    return table


def write_program_genre_draft(
    path: str | Path, programs: list[dict[str, Any]], reviewed: dict[str, tuple[str, ...]]
) -> int:
    """Write the programme→genre table for an editor to complete.

    Existing reviewed rows are preserved verbatim -- regenerating this after a
    reindex must never silently discard someone's editorial decisions. Rows are
    ordered by episode count so the programmes worth deciding about come first.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["program", "episodes", "median_minutes", "genre", "source"])
        for p in sorted(programs, key=lambda x: -x["episodes"]):
            name = p["program"]
            if name in reviewed:
                genres, source = reviewed[name], "reviewed"
            else:
                genres = draft_genre(name)
                source = "confirmed" if name in CONFIRMED_PROGRAM_GENRES else "draft"
                if not genres:
                    source = "unknown"
            writer.writerow(
                [
                    name,
                    p["episodes"],
                    p.get("median_duration_minutes") or "",
                    ";".join(genres),
                    source,
                ]
            )
    return len(programs)
