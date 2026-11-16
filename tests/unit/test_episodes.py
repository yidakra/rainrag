"""The episode feed: store behaviour and the /episodes routes."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag import api as api_module  # noqa: E402
from src.rainrag.api import app  # noqa: E402
from src.rainrag.episodes import (  # noqa: E402
    EpisodeStore,
    EpisodeStoreUnavailableError,
    decode_cursor,
    encode_cursor,
    parse_fields,
)


H1, H2, H3, H_PART = "1" * 40, "2" * 40, "3" * 40, "4" * 40
VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nДобрый вечер, это Дождь.\n"


def _write_vtt(root: Path, video_hash: str, lang: str = "ru") -> None:
    shard = root.joinpath(*[video_hash[i : i + 2] for i in range(0, 40, 2)])
    shard.mkdir(parents=True)
    (shard / f"{video_hash}.{lang}.vtt").write_text(VTT, encoding="utf-8")


@pytest.fixture
def store(tmp_path: Path) -> EpisodeStore:
    """Three articles in the index; one tagged (plus a second file of it), one summarised."""
    index = {
        "100": {
            "title": "Old",
            "date": "2024-01-01",
            "url": "u100",
            "video_hash": H1,
            "program": "Здесь и сейчас",
            "presenters": ["Анна Монгайт"],
        },
        "200": {
            "title": "Mid",
            "date": "2025-06-01",
            "url": "u200",
            "video_hash": H2,
            "duration_seconds": 3000.0,
        },
        "300": {"title": "New", "date": "2026-03-01", "url": "u300", "video_hash": H3},
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    tags = [
        {
            "video_hash": H2,
            "content_id": "200",
            "program": "Нюансы",
            "presenter_cms": ["Таратута"],
            "subject": ["выборы"],
            "guest": ["Шульман"],
            "mentioned_cms": ["Путин"],
            "mentioned_extra": ["Путин", "Орбан"],
            "place": ["Венгрия"],
            "organization": [],
            "genre": ["интервью"],
            "error": None,
        },
        # Another file of the same article: its tags must not replace the indexed file's.
        {
            "video_hash": H_PART,
            "content_id": "200",
            "program": "Нюансы",
            "subject": ["чужое"],
            "error": None,
        },
        {"video_hash": H3, "content_id": "300", "subject": ["сбой"], "error": "parse failed"},
    ]
    (tmp_path / "tags.jsonl").write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tags) + "\n{broken\n",
        encoding="utf-8",
    )
    summaries = [
        {
            "content_id": "300",
            "summary": "Описание.",
            "model": "mistral-medium-3.5-128b",
            "prompt_version": "summary-v1",
            "generated_at": "2099-01-01T00:00:00+00:00",
        },
        {"content_id": "999", "summary": "Не из индекса."},
    ]
    (tmp_path / "summaries.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in summaries), encoding="utf-8"
    )
    archive = tmp_path / "archive"
    _write_vtt(archive, H2)
    _write_vtt(archive, H3, lang="en")
    return _make_store(tmp_path, archive)


T0 = "2026-11-16T10:00:00.000000+00:00"
T1 = "2026-11-17T04:00:00.000000+00:00"


class _Clock:
    """A settable clock for updated_at."""

    def __init__(self) -> None:
        """Start at a fixed time."""
        self.now = datetime.fromisoformat(T0)

    def __call__(self) -> datetime:
        """The current fake time."""
        return self.now


def _make_store(tmp_path: Path, archive: Path, clock=None) -> EpisodeStore:
    return EpisodeStore(
        index_path=tmp_path / "index.json",
        tags_path=tmp_path / "tags.jsonl",
        summaries_path=tmp_path / "summaries.jsonl",
        archive_root=archive,
        state_path=tmp_path / "state.json",
        clock=clock or _Clock(),
    )


def _touch_later(path: Path) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 5))


ALL = parse_fields(None)


def test_every_indexed_article_is_an_episode_newest_first(store):
    """The base set is the index, not only tagged episodes, newest first."""
    items, cursor, _ = store.page(ALL, limit=10)
    assert [i["content_id"] for i in items] == ["300", "200", "100"]
    assert cursor is None


def test_tags_merge_only_from_the_indexed_file(store):
    """Tags come from the file the index chose; another file of the article is ignored."""
    episode = store.get("200", ALL)
    assert episode["tags"]["subject"] == ["выборы"]
    assert episode["tags"]["mentioned"] == ["Путин", "Орбан"]
    assert episode["program"] == "Нюансы"
    assert episode["presenters"] == ["Таратута"]


def test_failed_tagging_rows_leave_tags_null(store):
    """A tagging failure is no tags, not a broken record."""
    assert store.get("300", ALL)["tags"] is None


def test_untagged_episode_takes_program_and_presenters_from_the_index(store):
    """Without tags, programme and presenters come from the CMS card via the index."""
    episode = store.get("100", ALL)
    assert episode["program"] == "Здесь и сейчас"
    assert episode["presenters"] == ["Анна Монгайт"]


def test_summary_carries_its_model_and_prompt_version(store):
    """Summaries say which model and prompt made them; unsummarised is null."""
    episode = store.get("300", ALL)
    assert episode["summary"] == "Описание."
    assert episode["summary_model"] == "mistral-medium-3.5-128b"
    assert episode["summary_prompt_version"] == "summary-v1"
    assert store.get("200", ALL)["summary"] is None


def test_transcript_is_plain_text_with_its_language(store):
    """The transcript is VTT text without cues; missing is null."""
    assert store.get("200", ALL)["transcript"] == {
        "language": "ru",
        "text": "Добрый вечер, это Дождь.",
    }
    assert store.get("300", ALL)["transcript"]["language"] == "en"
    assert store.get("100", ALL)["transcript"] is None


def test_unknown_content_id_is_none(store):
    """A content_id outside the index is not an episode."""
    assert store.get("999", ALL) is None


def test_fields_trim_the_record_and_keep_content_id(store):
    """fields= returns only what was asked, plus content_id."""
    episode = store.get("200", parse_fields("title,summary"))
    assert set(episode) == {"content_id", "title", "summary"}


def test_unknown_fields_are_an_error():
    """A typo in fields= is an error, not a silently dropped field."""
    with pytest.raises(ValueError, match="nope"):
        parse_fields("title,nope")


def test_cursor_pages_without_gaps_or_repeats(store):
    """Two pages of two cover all three episodes once."""
    first, cursor, _ = store.page(ALL, limit=2)
    second, last, _ = store.page(ALL, limit=2, cursor=cursor)
    assert [i["content_id"] for i in first + second] == ["300", "200", "100"]
    assert last is None


def _add_summary(tmp_path: Path, content_id: str, text: str) -> None:
    path = tmp_path / "summaries.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"content_id": content_id, "summary": text}, ensure_ascii=False))
    _touch_later(path)


def test_a_change_during_a_pull_is_returned_by_the_next_pull(tmp_path, store):
    """The watermark, not the largest updated_at seen, is the safe `since` (CodeRabbit on #131)."""
    page1, cursor, watermark = store.page(ALL, limit=1)
    assert [i["content_id"] for i in page1] == ["300"]
    store._clock.now = datetime.fromisoformat(T1)
    _add_summary(tmp_path, "300", "Изменено во время выгрузки.")  # already passed by the cursor
    page2, cursor, again = store.page(ALL, limit=1, cursor=cursor)
    store._clock.now = datetime.fromisoformat("2026-11-17T00:00:00+00:00")
    _add_summary(tmp_path, "100", "Изменено позже.")
    page3, last, third = store.page(ALL, limit=1, cursor=cursor)
    assert [i["content_id"] for i in page2 + page3] == ["200", "100"]
    assert last is None
    assert watermark == again == third
    seen = max(i["updated_at"] for i in page1 + page2 + page3)
    assert store.get("300", ALL)["updated_at"] < seen  # saving `seen` would lose episode 300
    items, _, _ = store.page(ALL, limit=10, since=watermark)
    assert [i["content_id"] for i in items] == ["300", "100"]


def test_first_load_stamps_every_episode_once(store):
    """With no state file every episode is new at the first load."""
    items, _, _ = store.page(parse_fields("updated_at"), limit=10)
    assert {i["updated_at"] for i in items} == {T0}


def test_rewriting_the_index_unchanged_moves_nothing(tmp_path, store):
    """The daily rebuild of the index must not make every episode look changed."""
    store.get("100", ALL)
    store._clock.now = datetime.fromisoformat(T1)
    index = tmp_path / "index.json"
    index.write_text(index.read_text(encoding="utf-8"), encoding="utf-8")
    _touch_later(index)
    items, _, _ = store.page(ALL, limit=10, since=T0)
    assert items == []


def test_only_the_changed_episode_moves(tmp_path, store):
    """A new summary moves its episode's updated_at and no other."""
    store.get("100", ALL)
    store._clock.now = datetime.fromisoformat(T1)
    path = tmp_path / "summaries.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"content_id": "100", "summary": "Новое."}, ensure_ascii=False))
    _touch_later(path)
    items, _, _ = store.page(ALL, limit=10, since=T0)
    assert [i["content_id"] for i in items] == ["100"]
    assert items[0]["updated_at"] == T1


def test_updated_at_survives_a_restart(tmp_path, store):
    """The state file keeps the times, so a new process does not restamp everything."""
    store.get("100", ALL)
    clock = _Clock()
    clock.now = datetime.fromisoformat("2026-12-01T00:00:00+00:00")
    again = _make_store(tmp_path, store.archive_root, clock)
    assert again.get("100", ALL)["updated_at"] == T0


def test_a_failed_state_write_is_retried_without_a_new_change(tmp_path, store, monkeypatch):
    """A disk error on the state file is retried on the next request (CodeRabbit on #131)."""
    real_write = Path.write_text

    def failing_write(self, *args, **kwargs):
        if self.name.startswith("state.json"):
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write)
    store.get("100", ALL)
    assert not (tmp_path / "state.json").exists()
    monkeypatch.setattr(Path, "write_text", real_write)
    store.get("100", ALL)  # sources unchanged: no rebuild, but the write is retried
    assert (tmp_path / "state.json").exists()
    clock = _Clock()
    clock.now = datetime.fromisoformat("2026-12-01T00:00:00+00:00")
    again = _make_store(tmp_path, store.archive_root, clock)
    assert again.get("100", ALL)["updated_at"] == T0


def test_an_unreadable_state_file_is_not_restamped_or_overwritten(tmp_path, store, monkeypatch):
    """A state file that exists but cannot be read stops the load (CodeRabbit on #131)."""
    store.get("100", ALL)
    state_path = tmp_path / "state.json"
    saved = state_path.read_text(encoding="utf-8")
    real_read = Path.read_text

    def failing_read(self, *args, **kwargs):
        if self.name == "state.json":
            raise PermissionError("denied")
        return real_read(self, *args, **kwargs)

    clock = _Clock()
    clock.now = datetime.fromisoformat("2026-12-01T00:00:00+00:00")
    again = _make_store(tmp_path, store.archive_root, clock)
    monkeypatch.setattr(Path, "read_text", failing_read)
    with pytest.raises(EpisodeStoreUnavailableError):
        again.get("100", ALL)
    monkeypatch.setattr(Path, "read_text", real_read)
    assert state_path.read_text(encoding="utf-8") == saved
    assert again.get("100", ALL)["updated_at"] == T0  # the load was retried, not restamped


def test_a_cms_edit_in_the_index_moves_updated_at(tmp_path, store):
    """Programme and presenters ride in the index, so a CMS edit is a change (CodeRabbit on #131)."""
    store.get("100", ALL)
    store._clock.now = datetime.fromisoformat(T1)
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["100"]["presenters"] = ["Анна Монгайт", "Тихон Дзядко"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    _touch_later(index_path)
    items, _, _ = store.page(ALL, limit=10, since=T0)
    assert [i["content_id"] for i in items] == ["100"]


def test_changes_in_the_same_instant_get_later_stamps(tmp_path, store):
    """Two loads within one clock tick still give the second change a later updated_at."""
    first = store.get("100", ALL)["updated_at"]
    path = tmp_path / "summaries.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"content_id": "100", "summary": "Новое."}, ensure_ascii=False))
    _touch_later(path)
    second = store.get("100", ALL)["updated_at"]
    assert second > first
    items, _, _ = store.page(ALL, limit=10, since=first)
    assert [i["content_id"] for i in items] == ["100"]


def test_an_unreadable_index_keeps_the_last_snapshot(tmp_path, store):
    """A failed read must not publish an empty feed, and the reload stays pending."""
    assert store.get("100", ALL) is not None
    index_path = tmp_path / "index.json"
    good = index_path.read_text(encoding="utf-8")
    index_path.write_text("{broken", encoding="utf-8")
    _touch_later(index_path)
    assert store.get("100", ALL)["title"] == "Old"
    data = json.loads(good)
    data["100"]["title"] = "Fixed"
    index_path.write_text(json.dumps(data), encoding="utf-8")
    assert store.get("100", ALL)["title"] == "Fixed"


def test_no_snapshot_and_no_index_is_unavailable(tmp_path):
    """With nothing to serve, the store says so instead of returning an empty feed."""
    empty = EpisodeStore(
        index_path=tmp_path / "missing.json",
        tags_path=tmp_path / "tags.jsonl",
        summaries_path=tmp_path / "summaries.jsonl",
        archive_root=tmp_path,
    )
    with pytest.raises(EpisodeStoreUnavailableError):
        empty.page(ALL, limit=10)


def test_cursor_round_trip_and_garbage():
    """Cursors decode back; anything else is a ValueError."""
    cursor = encode_cursor(("2026-01-01", "7"), T0)
    assert decode_cursor(cursor) == (("2026-01-01", "7"), T0)
    with pytest.raises(ValueError):
        decode_cursor("not-a-cursor")


def test_store_reloads_when_a_source_file_changes(store, tmp_path):
    """A new summary row shows up without a restart."""
    assert store.get("200", ALL)["summary"] is None
    path = tmp_path / "summaries.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"content_id": "200", "summary": "Новое."}, ensure_ascii=False))
    _touch_later(path)
    assert store.get("200", ALL)["summary"] == "Новое."


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client(store, monkeypatch):
    """A TestClient over the app with the fixture store and no tokens set."""
    monkeypatch.setattr(
        api_module,
        "config",
        SimpleNamespace(
            episodes=SimpleNamespace(
                enabled=True,
                default_page_size=2,
                max_page_size=2,
                token_env="RAINRAG_EPISODES_TOKEN",
            )
        ),
    )
    monkeypatch.setattr(api_module, "_EPISODE_STORE", store)
    monkeypatch.delenv("RAINRAG_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("RAINRAG_EPISODES_TOKEN", raising=False)
    with TestClient(app) as c:
        yield c


def test_list_route_pages_and_omits_trimmed_fields(client):
    """Trimmed fields are absent, not null, and the cursor reaches the last page."""
    first = client.get("/episodes", params={"fields": "title"}).json()
    assert first["items"] == [
        {"content_id": "300", "title": "New"},
        {"content_id": "200", "title": "Mid"},
    ]
    rest = client.get("/episodes", params={"fields": "title", "cursor": first["next_cursor"]})
    rest = rest.json()
    assert [i["content_id"] for i in rest["items"]] == ["100"]
    assert rest.get("next_cursor") is None
    assert first["sync_watermark"] == rest["sync_watermark"] == T0


def test_page_size_is_capped(client):
    """limit above max_page_size is cut to the cap."""
    body = client.get("/episodes", params={"limit": 50, "fields": "title"}).json()
    assert len(body["items"]) == 2


def test_single_route_and_404(client):
    """One episode with everything; an unknown id is 404."""
    body = client.get("/episodes/200").json()
    assert body["tags"]["guest"] == ["Шульман"]
    assert body["transcript"]["text"] == "Добрый вечер, это Дождь."
    assert client.get("/episodes/999").status_code == 404


def test_bad_fields_cursor_and_since_are_400(client):
    """Malformed parameters are client errors."""
    assert client.get("/episodes", params={"fields": "nope"}).status_code == 400
    assert client.get("/episodes", params={"cursor": "garbage"}).status_code == 400
    assert client.get("/episodes", params={"since": "yesterday"}).status_code == 400


def test_since_accepts_z_suffix(client):
    """since accepts the usual Z form and compares in UTC."""
    before = client.get("/episodes", params={"since": "2026-11-16T09:00:00Z", "fields": "title"})
    after = client.get("/episodes", params={"since": "2026-11-16T10:00:00Z", "fields": "title"})
    assert (
        client.get("/episodes", params={"since": "2026-11-16T12:00:00+02:00"}).json()["items"] == []
    )
    assert len(before.json()["items"]) == 2
    assert after.json()["items"] == []


def test_episodes_token_opens_only_the_episode_routes(client, monkeypatch):
    """The read-only token opens /episodes and nothing else; the API token still works."""
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "api-secret")
    monkeypatch.setenv("RAINRAG_EPISODES_TOKEN", "read-only")
    ok = {"Authorization": "Bearer read-only"}
    assert client.get("/episodes/200", headers=ok).status_code == 200
    api = {"Authorization": "Bearer api-secret"}
    assert client.get("/episodes/200", headers=api).status_code == 200
    wrong = {"Authorization": "Bearer wrong"}
    assert client.get("/episodes/200", headers=wrong).status_code == 401
    assert client.get("/episodes/200").status_code == 401
    assert client.post("/query", json={"question": "x"}, headers=ok).status_code == 401


def test_episodes_token_closes_the_routes_without_an_api_token(client, monkeypatch):
    """A configured episodes token protects the routes even with no API token."""
    monkeypatch.setenv("RAINRAG_EPISODES_TOKEN", "read-only")
    assert client.get("/episodes/200").status_code == 401
    ok = {"Authorization": "Bearer read-only"}
    assert client.get("/episodes/200", headers=ok).status_code == 200


def test_disabled_feed_is_503(client):
    """episodes.enabled=false closes the feed."""
    api_module.config.episodes.enabled = False
    assert client.get("/episodes").status_code == 503


def test_an_unreadable_summaries_file_keeps_the_last_snapshot(tmp_path, store):
    """A read error on an existing optional source must not strip metadata (CodeRabbit on #131)."""
    assert store.get("300", ALL)["summary"] == "Описание."
    path = tmp_path / "summaries.jsonl"
    saved = path.read_text(encoding="utf-8")
    path.unlink()
    path.mkdir()  # reading a directory raises IsADirectoryError, an OSError
    assert store.get("300", ALL)["summary"] == "Описание."
    path.rmdir()
    path.write_text(saved.replace("Описание.", "Обновлено."), encoding="utf-8")
    assert store.get("300", ALL)["summary"] == "Обновлено."


def test_a_missing_optional_source_is_simply_no_data(tmp_path, store):
    """Deleting the summaries file is a real change: summaries go, the feed stays."""
    assert store.get("300", ALL)["summary"] == "Описание."
    (tmp_path / "summaries.jsonl").unlink()
    assert store.get("300", ALL)["summary"] is None
