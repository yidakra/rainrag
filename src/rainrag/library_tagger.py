"""Generate Varya's episode tagging from transcripts and CMS metadata.

Four of the fields her scheme needs have no source in the CMS: `subject` (the
topic vocabulary the whole similarity idea rests on), `guest` (who is actually
speaking in an interview), `place`, and `organization`. They exist only in the
video itself — which we have, as transcripts.

So an LLM reads the transcript and emits them. Two rules shape the design:

*Never invent provenance.* `presenter` and `mentioned` come from the CMS where
the CMS has them, and the model is told what they are rather than asked to
guess. What it returns for those is treated as a supplement, not a correction.

*Prefer recall on `subject`.* The gold episodes carry 26-34 subject tags each,
mixing broad ("политика") with specific ("теория пустоты"), because retrieval
matches on overlap: a tag nobody searches costs nothing, a missing tag loses the
episode. Precision is the editor's job at review time, not the model's here.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


# A transcript of a 40-minute lecture is far longer than the tagging needs.
# Head and tail together carry the framing ("сегодня у нас в гостях...") and
# the wrap-up, which is where speakers and topics are actually named.
TRANSCRIPT_HEAD_CHARS = 10_000
TRANSCRIPT_MIDDLE_CHARS = 8_000
TRANSCRIPT_TAIL_CHARS = 4_000

SYSTEM_PROMPT = """Ты помогаешь редакции телеканала Дождь размечать архив для «Библиотеки Дождя».

По расшифровке эфира и метаданным ты заполняешь карточку выпуска. Отвечай ТОЛЬКО \
валидным JSON без markdown-обёртки, по схеме:

{
  "guest": ["имя гостя, если это интервью или беседа"],
  "subject": ["тема", "тема", ...],
  "place": ["город", "страна"],
  "organization": ["организация"],
  "genre": ["лекция" | "мастер-класс" | "интервью" | "ток-шоу" | "новости" | \
"репортаж" | "документальный фильм" | "дискуссия" | "обзор"],
  "mentioned_extra": ["человек, о котором говорят, но которого нет в метаданных"]
}

Правила:
- guest — те, кого пригласили говорить: гость интервью, лектор. НЕ ведущий. \
Если это лекция и лектор уже указан как ведущий в метаданных — оставь guest пустым.
- subject — 20-35 тегов. Смешивай широкие темы («политика», «психология») и \
конкретные («теория пустоты», «муниципальные выборы»). Включай: предметную область, \
ключевые понятия, профессию говорящего, если она по теме. Это теги для поиска \
похожего контента, поэтому лучше добавить лишний тег, чем упустить нужный.
- place — где происходит действие или о чём речь: город и страна.
- organization — упомянутые организации, компании, институции.
- genre — один или два из перечисленных вариантов, ничего другого.
- mentioned_extra — люди, о которых говорят (не говорящие). Только те, кого нет \
в списке «Уже известные упоминания».
- Всё на русском языке, в именительном падеже, строчными буквами кроме имён \
собственных.
- Если поле нечем заполнить — пустой список. Не выдумывай."""


@dataclass
class TaggingResult:
    """One episode's generated tags, plus what it cost and whether it worked."""

    video_hash: str
    content_id: str | None = None
    guest: list[str] = field(default_factory=list)
    subject: list[str] = field(default_factory=list)
    place: list[str] = field(default_factory=list)
    organization: list[str] = field(default_factory=list)
    genre: list[str] = field(default_factory=list)
    mentioned_extra: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "video_hash": self.video_hash,
            "content_id": self.content_id,
            "guest": self.guest,
            "subject": self.subject,
            "place": self.place,
            "organization": self.organization,
            "genre": self.genre,
            "mentioned_extra": self.mentioned_extra,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "error": self.error,
        }


def read_vtt_text(path: Path) -> str:
    """Strip a VTT down to spoken text.

    Cue numbers, timestamps and the WEBVTT header carry no topical signal and
    would eat the token budget.
    """
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line == "WEBVTT" or "-->" in line or line.isdigit():
            continue
        if line.startswith(("NOTE", "STYLE", "REGION")):
            continue
        lines.append(line)
    return " ".join(lines)


def condense_transcript(
    text: str,
    head: int = TRANSCRIPT_HEAD_CHARS,
    middle: int = TRANSCRIPT_MIDDLE_CHARS,
    tail: int = TRANSCRIPT_TAIL_CHARS,
) -> str:
    """Sample the opening, the middle and the close of a long transcript.

    The opening states who is speaking and about what; the close names the
    takeaway. Head and tail alone scored worst on the longest gold episode
    (29% subject recall on a 58k transcript against 44% on a 32k one), because
    an hour-long interview changes subject repeatedly in the body -- so the
    middle is sampled too. Sending the whole thing would multiply cost for
    diminishing signal.
    """
    if len(text) <= head + middle + tail:
        return text
    centre = len(text) // 2
    mid = text[centre - middle // 2 : centre + middle // 2]
    return f"{text[:head].rstrip()}\n\n[...]\n\n{mid.strip()}\n\n[...]\n\n{text[-tail:].lstrip()}"


def build_user_prompt(
    *,
    title: str | None,
    program: str | None,
    date: str | None,
    duration_minutes: float | None,
    presenters: list[str],
    mentioned: list[str],
    transcript: str,
) -> str:
    """Assemble the per-episode prompt.

    The CMS facts are stated as facts so the model supplements rather than
    re-derives them -- and so a wrong guess about the presenter cannot
    overwrite a field the CMS actually knows.
    """
    parts = [
        f"Название: {title or '(нет)'}",
        f"Программа: {program or '(нет)'}",
        f"Дата: {date or '(нет)'}",
        f"Длительность: {round(duration_minutes)} мин"
        if duration_minutes
        else "Длительность: (нет)",
        f"Ведущие по данным CMS: {', '.join(presenters) if presenters else '(нет)'}",
        f"Уже известные упоминания: {', '.join(mentioned) if mentioned else '(нет)'}",
        "",
        "Расшифровка:",
        transcript,
    ]
    return "\n".join(parts)


def build_fewshot_block(example: dict[str, Any]) -> str:
    """Render one editor-tagged episode as a worked example.

    Tag *style* is what the model gets wrong unprompted -- it produces
    reasonable topics that simply are not the ones an editor writes. The gold
    cards mix breadth with specificity and include the speaker's profession,
    which is hard to convey as a rule and obvious as an example.
    """
    card = {
        "guest": example.get("guest", []),
        "subject": example.get("subject", []),
        "place": example.get("place", []),
        "organization": example.get("organization", []),
        "genre": example.get("genre", []),
        "mentioned_extra": [],
    }
    return (
        "Пример правильно заполненной карточки.\n"
        f"Название: {example.get('title')}\n"
        f"Программа: {example.get('parent_program')}\n"
        f"Ответ:\n{json.dumps(card, ensure_ascii=False, indent=2)}"
    )


_RETRYABLE = ("rate limit", "429", "timeout", "timed out", "503", "502", "overloaded")


def _is_retryable(exc: Exception) -> bool:
    """True for transient provider conditions, false for a bad transcript."""
    return any(marker in str(exc).lower() for marker in _RETRYABLE)


def _first_json_object(raw: str) -> Any:
    """Decode the first complete JSON object in a string.

    A greedy `{.*}` spans from the first brace to the *last* one, so a reply
    with any trailing object -- "{...}\n\nКомментарий: {...}" -- produced
    invalid JSON and lost an otherwise good response. raw_decode stops at the
    end of the first complete value instead.
    """
    decoder = json.JSONDecoder()
    for start in range(len(raw)):
        if raw[start] != "{":
            continue
        try:
            value, _end = decoder.raw_decode(raw[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object in response")


def parse_tagging_response(raw: str) -> dict[str, list[str]]:
    """Pull the tag lists out of a model response.

    Models wrap JSON in prose or code fences despite instructions, so the first
    balanced-looking object is extracted rather than trusting the whole string.
    Unknown keys are dropped and every value is coerced to a list of clean
    strings, because one malformed field must not lose the other five.
    """
    data = _first_json_object(raw or "")
    if not isinstance(data, dict):
        raise ValueError("response JSON is not an object")

    out: dict[str, list[str]] = {}
    for key in ("guest", "subject", "place", "organization", "genre", "mentioned_extra"):
        value = data.get(key)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            out[key] = []
            continue
        seen: list[str] = []
        for item in value:
            text = str(item).strip().strip(".,;")
            if text and text.lower() not in {s.lower() for s in seen}:
                seen.append(text)
        out[key] = seen
    return out


def tag_episode(
    engine: Any,
    *,
    video_hash: str,
    content_id: str | None,
    title: str | None,
    program: str | None,
    date: str | None,
    duration_minutes: float | None,
    presenters: list[str],
    mentioned: list[str],
    transcript: str,
    fewshot: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_attempts: int = 5,
) -> TaggingResult:
    """Tag one episode. Never raises: a failure is one unusable row.

    A tagging run covers thousands of episodes; one bad transcript or one
    provider hiccup must cost that episode and nothing else.
    """
    result = TaggingResult(video_hash=video_hash, content_id=content_id)
    system = SYSTEM_PROMPT
    if fewshot:
        system = f"{SYSTEM_PROMPT}\n\n{build_fewshot_block(fewshot)}"
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": build_user_prompt(
                title=title,
                program=program,
                date=date,
                duration_minutes=duration_minutes,
                presenters=presenters,
                mentioned=mentioned,
                transcript=condense_transcript(transcript),
            ),
        },
    ]
    usage: dict[str, int] = {}
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            raw = engine.generate_answer(messages, temperature=temperature, usage_sink=usage)
            parsed = parse_tagging_response(raw)
            break
        except Exception as exc:  # noqa: BLE001 - recorded per episode, never fatal
            last_error = exc
            # A rate limit means the provider wants us slower, not that this
            # episode is untaggable. Back off exponentially with jitter so a
            # pool of workers does not retry in lockstep and re-trigger it.
            if attempt < max_attempts - 1 and _is_retryable(exc):
                delay = min(60.0, 2.0 * (2**attempt)) * (0.5 + random.random())
                logger.info(
                    "Tagging {} hit {}; retry {}/{} in {:.0f}s",
                    video_hash,
                    type(exc).__name__,
                    attempt + 1,
                    max_attempts - 1,
                    delay,
                )
                time.sleep(delay)
                continue
            logger.warning("Tagging failed for {}: {}: {}", video_hash, type(exc).__name__, exc)
            result.error = f"{type(exc).__name__}: {exc}"
            return result
    else:
        result.error = f"{type(last_error).__name__}: {last_error}"
        return result

    result.guest = parsed["guest"]
    result.subject = parsed["subject"]
    result.place = parsed["place"]
    result.organization = parsed["organization"]
    result.genre = parsed["genre"]
    result.mentioned_extra = parsed["mentioned_extra"]
    result.tokens_in = usage.get("tokens_in", 0)
    result.tokens_out = usage.get("tokens_out", 0)
    return result


def tag_overlap_score(predicted: list[str], gold: list[str]) -> dict[str, float]:
    """Score generated tags against an editor's, leniently but honestly.

    Exact string equality is the wrong bar for free-text tags: «женщины-лидеры»
    and «женское лидерство» are the same idea. Matching is case-insensitive on
    a normalised form, and a tag also counts when one contains the other, so
    «российская политика» matches «политика». Recall is reported separately
    from precision because they mean different things here: recall is whether
    the episode can be found at all, precision is how much noise the editor
    has to strike out.
    """

    def norm(tag: str) -> str:
        return re.sub(r"[^\w\s-]", "", tag.lower()).strip()

    pred = [norm(t) for t in predicted if norm(t)]
    want = [norm(t) for t in gold if norm(t)]
    if not want:
        return {"recall": 0.0, "precision": 0.0, "matched": 0.0, "gold": 0.0}

    matched = sum(1 for g in want if any(g == p or g in p or p in g for p in pred))
    hit = sum(1 for p in pred if any(g == p or g in p or p in g for g in want))
    return {
        "recall": matched / len(want),
        "precision": hit / len(pred) if pred else 0.0,
        "matched": float(matched),
        "gold": float(len(want)),
    }
