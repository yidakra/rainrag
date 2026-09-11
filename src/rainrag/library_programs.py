"""Varya's programme table: authoritative genres, and who counts as a speaker.

Two genre vocabularies exist in this project and they are not interchangeable.
The tagging model emits a free-form genre per episode ("дискуссия" is a common
one) while the editorial taxonomy is per *programme* and draws on a closed list
of 36 terms that has no "дискуссия" in it. Varya's speaker rule is written
against the editorial list, so it has to read the programme table rather than
the model's output.

The rule (ClickUp 86cbgy6tv, in her words): a guest is always a speaker; a
presenter or correspondent is a speaker too, except in дебаты, документальный
фильм, игровое шоу, интервью, мини-док, ток-шоу and реалити-шоу. The point is
that «Синдеева» should surface other episodes featuring her *guest*, not other
interviews she happens to conduct.

Applied to the 13,808 tagged episodes this demotes the presenter on 3,304 of
them. On 2,642 a guest remains and the match improves, which is the intended
effect. On 270 no guest was extracted and the episode is left with no speaker
at all; that is preferred to keeping a presenter whose other episodes are
exactly the wrong answer, and the interface says so rather than reporting an
empty result.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PRESENTER_NOT_SPEAKER_GENRES = frozenset(
    {
        "дебаты",
        "документальный фильм",
        "игровое шоу",
        "интервью",
        "мини-док",
        "ток-шоу",
        "реалити-шоу",
    }
)

# Sheets renders a typed apostrophe as U+2019, so "Hard Day’s Night" in the
# programme table never matched "Hard Day's Night" in the catalogue and took
# 250 episodes' worth of genre with it.
_APOSTROPHES = dict.fromkeys(map(ord, "’‘ʼ´`"), "'")
_WHITESPACE = re.compile(r"\s+")


def normalise_title(title: str) -> str:
    """Key for joining programme names across the sheet and the catalogue.

    Folds ё to е like `normalise_tag` and `normalise_person` do, because the
    sheet is typed by hand and «Как всё начиналось» is one keystroke from
    «Как все начиналось». A miss here is silent: the programme reads as absent
    and its episodes quietly keep the presenter the rule meant to demote.

    The fold runs *after* casefold, not before. Replacing ё first leaves an
    uppercase Ё untouched, which casefold then turns into ё, and the key still
    fails to match.
    """
    folded = unicodedata.normalize("NFKC", title).translate(_APOSTROPHES)
    return _WHITESPACE.sub(" ", folded).strip().casefold().replace("ё", "е")


@dataclass(frozen=True)
class Programme:
    """One row of the Programs tab, only the parts the ranker needs."""

    title: str
    genres: tuple[str, ...] = ()
    presenter: str | None = None

    @property
    def presenter_is_speaker(self) -> bool:
        """False when any of the programme's genres demotes the presenter.

        Any, not all: a programme tagged "аналитика, ток-шоу" is still a show
        where the guest is the draw.
        """
        return not any(genre in PRESENTER_NOT_SPEAKER_GENRES for genre in self.genres)


def parse_genres(raw: str) -> tuple[str, ...]:
    """Split the sheet's multi-value genre cell, comma separated, lowercased."""
    return tuple(part.strip().casefold() for part in (raw or "").split(",") if part.strip())


def load_programmes(path: Path) -> dict[str, Programme]:
    """Read the Programs tab export into a lookup keyed by normalised title.

    A missing file means "no programme table yet", which callers must treat as
    "fall back to the old behaviour" rather than as an error: the table is
    maintained by hand in a Google Sheet and can lag a fresh checkout.
    """
    if not path.exists():
        return {}
    programmes: dict[str, Programme] = {}
    # utf-8-sig, not utf-8: a Sheets CSV export starts with a byte-order mark
    # and the sync script copies the file through byte for byte. Read as plain
    # utf-8 the mark stays glued to the first header name, so that column is
    # unreachable by name. With "title" first that empties the whole table and
    # the speaker rule silently reverts to presenter-plus-guest.
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            presenter = (row.get("presenter") or "").strip() or None
            programmes[normalise_title(title)] = Programme(
                title=title,
                genres=parse_genres(row.get("genre") or ""),
                presenter=presenter,
            )
    return programmes


def programme_for(program_name: str | None, programmes: dict[str, Programme]) -> Programme | None:
    """Look up an episode's programme, tolerating apostrophe and case drift."""
    if not program_name:
        return None
    return programmes.get(normalise_title(program_name))


@dataclass
class SpeakerResolution:
    """Who speaks in an episode, and whether a presenter was set aside."""

    speakers: list[str] = field(default_factory=list)
    presenter_demoted: bool = False


def resolve_speakers(
    record: dict[str, Any], programmes: dict[str, Programme] | None = None
) -> SpeakerResolution:
    """Apply the editorial speaker rule to one tagging-run row.

    Without a programme table the old behaviour is kept verbatim, presenter
    then guest, so an out-of-date checkout degrades to the previous ranking
    instead of silently dropping every presenter.
    """
    presenters = [str(name) for name in (record.get("presenter_cms") or []) if name]
    guests = [str(name) for name in (record.get("guest") or []) if name]
    if not programmes:
        return SpeakerResolution(speakers=presenters + guests)
    programme = programme_for(record.get("program"), programmes)
    if programme is not None and not programme.presenter_is_speaker:
        return SpeakerResolution(speakers=guests, presenter_demoted=bool(presenters))
    return SpeakerResolution(speakers=presenters + guests)
