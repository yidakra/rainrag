"""Video-level lookups: /similar-videos and /transcript/{video_id}."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag import api as api_module  # noqa: E402
from src.rainrag.api import _resolve_video_ref, app  # noqa: E402
from src.rainrag.ingest import VTTParser, WebMetadataLoader  # noqa: E402
from src.rainrag.query import RAGQueryEngine  # noqa: E402
from src.rainrag.subtitles import vtt_to_srt  # noqa: E402


HASH_A = "a" * 40
HASH_B = "b" * 40
HASH_C = "c" * 40

VTT = """WEBVTT

NOTE made by a test

1
00:00:00.000 --> 00:00:03.500
Рада вас видеть.

00:03.780 --> 00:05.980 align:start
Спасибо, Катя.
Рада быть с вами.

STYLE
::cue { color: red }
"""


def test_vtt_to_srt_renumbers_and_drops_header_blocks():
    srt = vtt_to_srt(VTT)
    assert srt == (
        "1\n00:00:00,000 --> 00:00:03,500\nРада вас видеть.\n\n"
        "2\n00:00:03,780 --> 00:00:05,980\nСпасибо, Катя.\nРада быть с вами.\n"
    )


def test_vtt_to_srt_empty_input_is_empty():
    assert vtt_to_srt("WEBVTT\n") == ""


def _write_record(cache: Path, video_hash: str, library_id: int, url: str, name: str = "x") -> None:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{video_hash}.json").write_text(
        json.dumps(
            {
                "id": library_id,
                "video_hash": video_hash,
                "url": url,
                "name": name,
                "teleshow": {"name": "Newsroom"},
                "date_active_start": "2024-02-16",
            }
        ),
        encoding="utf-8",
    )


def test_resolve_video_ref_accepts_hash_id_and_url(tmp_path, monkeypatch):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 357040, "https://tvrain.tv/teleshow/x/a-357040/")
    loader = WebMetadataLoader(cache, source="local")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    assert _resolve_video_ref(HASH_A.upper(), loader) == HASH_A
    assert _resolve_video_ref("357040", loader) == HASH_A
    assert _resolve_video_ref("https://tvrain.tv/teleshow/x/a-357040/", loader) == HASH_A
    with pytest.raises(Exception) as missing:
        _resolve_video_ref("999", loader)
    assert missing.value.status_code == 404
    with pytest.raises(Exception) as bad:
        _resolve_video_ref("not-a-ref", loader)
    assert bad.value.status_code == 400


def test_library_id_index_skips_records_without_an_id(tmp_path):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 1, "u1")
    (cache / f"{HASH_B}.json").write_text(json.dumps({"video_hash": HASH_B}), encoding="utf-8")
    (cache / "broken.json").write_text("{", encoding="utf-8")
    assert WebMetadataLoader(cache, source="local").library_id_index() == {1: HASH_A}


class _Points:
    def __init__(self, points):
        self.points = points


class _FakeQdrant:
    """Three seed chunks; hits spread over two other videos and the seed itself."""

    def __init__(self, seed_video_id: str, archive_root: Path):
        self.seed_video_id = seed_video_id
        self.root = archive_root
        self.scroll_filters = []
        self.query_filters = []

    def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
        self.scroll_filters.append(scroll_filter)
        return (
            [
                SimpleNamespace(vector=[1.0, 0.0]),
                SimpleNamespace(vector=[0.0, 1.0]),
                SimpleNamespace(vector=[0.5, 0.5]),
            ],
            None,
        )

    def _hit(self, video_hash, score, lang="ru", **web):
        path = str(
            self.root
            / WebMetadataLoader.hash_to_archive_dir(video_hash)
            / f"{video_hash}.{lang}.vtt"
        )
        payload = {"path": path, "video_id": video_hash[:16], "language": lang, **web}
        return SimpleNamespace(score=score, payload=payload)

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        self.query_filters.append(query_filter)
        if query == [1.0, 0.0]:
            return _Points(
                [
                    self._hit(HASH_B, 0.9, web_title="B title", web_url="https://tvrain.tv/b"),
                    self._hit(HASH_C, 0.4),
                ]
            )
        if query == [0.0, 1.0]:
            return _Points(
                [self._hit(HASH_B, 0.7, lang="en"), self._hit(HASH_C, 0.6, web_title="C title")]
            )
        return _Points([self._hit(HASH_C, 0.5)])


def _engine(tmp_path) -> tuple[RAGQueryEngine, _FakeQdrant]:
    config = SimpleNamespace(
        paths=SimpleNamespace(archive_root=str(tmp_path / "archive")),
        qdrant=SimpleNamespace(collection_name="broadcast_transcripts"),
    )
    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.config = config  # type: ignore[attr-defined]
    seed_file = (
        tmp_path / "archive" / WebMetadataLoader.hash_to_archive_dir(HASH_A) / f"{HASH_A}.ru.vtt"
    )
    fake = _FakeQdrant(VTTParser.generate_video_id(seed_file), tmp_path / "archive")
    engine.qdrant_client = fake  # type: ignore[attr-defined]
    return engine, fake


def test_find_similar_videos_groups_hits_by_video_and_sums_scores(tmp_path):
    engine, fake = _engine(tmp_path)
    ranked = engine.find_similar_videos(HASH_A, limit=10)
    assert [r["video_hash"] for r in ranked] == [HASH_B, HASH_C]
    b, c = ranked
    assert b["score"] == pytest.approx(1.6) and b["best"] == 0.9 and b["matching_chunks"] == 2
    assert c["score"] == pytest.approx(1.5) and c["matching_chunks"] == 3
    assert b["title"] == "B title" and b["url"] == "https://tvrain.tv/b"
    assert c["title"] == "C title"
    # the seed video is excluded through its payload video_id, in every language
    seed_filter = fake.scroll_filters[0]
    assert seed_filter.must[0].match.value == fake.seed_video_id
    assert fake.query_filters[0].must_not[0].match.value == fake.seed_video_id


def test_seed_video_id_comes_from_the_index_when_the_corpus_is_in_memory(tmp_path):
    """The archive root at query time differs from the one at ingest; the index wins."""
    engine, fake = _engine(tmp_path)
    ingest_time_path = f"/mnt/other-root/aa/{HASH_A}.ru.vtt"
    engine.bm25_corpus = [
        {"path": "/x/" + HASH_B + ".ru.vtt", "doc_id": "other"},  # type: ignore[attr-defined]
        {"path": ingest_time_path, "doc_id": "seed-doc"},
    ]
    real_query_points = fake.query_points

    def query_points(collection_name, query=None, query_filter=None, limit=10, with_payload=True):
        if query is None:
            assert query_filter.must[0].match.value == "seed-doc"
            return _Points([SimpleNamespace(score=1.0, payload={"video_id": "payload-id-123"})])
        return real_query_points(collection_name, query, query_filter, limit, with_payload)

    fake.query_points = query_points  # type: ignore[method-assign]
    engine.find_similar_videos(HASH_A)
    assert fake.scroll_filters[0].must[0].match.value == "payload-id-123"
    assert fake.query_filters[-1].must_not[0].match.value == "payload-id-123"


def test_seed_video_id_uses_the_corpus_video_id_when_present(tmp_path):
    engine, fake = _engine(tmp_path)
    engine.bm25_corpus = [
        {"path": f"/any/{HASH_A}.en.vtt", "doc_id": "d", "video_id": "from-corpus"}
    ]  # type: ignore[attr-defined]
    engine.find_similar_videos(HASH_A)
    assert fake.scroll_filters[0].must[0].match.value == "from-corpus"


def test_find_similar_videos_limit_and_language(tmp_path):
    engine, fake = _engine(tmp_path)
    ranked = engine.find_similar_videos(HASH_A, limit=1, language="en")
    assert len(ranked) == 1
    assert fake.scroll_filters[0].must[1].match.value == "en"


def test_find_similar_videos_without_seed_chunks_is_empty(tmp_path):
    engine, fake = _engine(tmp_path)
    fake.scroll = lambda **kw: ([], None)  # type: ignore[method-assign]
    assert engine.find_similar_videos(HASH_A) == []


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _config(tmp_path):
    return SimpleNamespace(
        paths=SimpleNamespace(archive_root=str(tmp_path / "archive")),
        web_metadata=SimpleNamespace(
            path=str(tmp_path / "web_metadata"),
            source="local",
            api_url="https://library.tvrain.tv",
            api_token_env="LIBRARY_API_TOKEN",
        ),
    )


def test_similar_videos_endpoint_enriches_from_the_metadata_cache(client, tmp_path, monkeypatch):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 1, "https://tvrain.tv/a", name="A")
    _write_record(cache, HASH_B, 2, "https://tvrain.tv/b", name="B title from cache")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    ranked = [
        {
            "video_hash": HASH_B,
            "score": 1.6,
            "best": 0.9,
            "matching_chunks": 2,
            "title": "payload title",
            "url": None,
            "program": None,
            "date": None,
        },
        {
            "video_hash": HASH_C,
            "score": 0.5,
            "best": 0.5,
            "matching_chunks": 1,
            "title": "C title",
            "url": None,
            "program": None,
            "date": "2023-01-01",
        },
    ]
    with (
        patch("src.rainrag.api.query_engine") as engine,
        patch("src.rainrag.api.config", _config(tmp_path)),
    ):
        engine.find_similar_videos.return_value = ranked
        r = client.get("/similar-videos", params={"video_id": "1", "limit": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["video_hash"] == HASH_A and body["video_id"] == "1" and body["num_results"] == 2
        first, second = body["results"]
        assert first["library_id"] == 2 and first["title"] == "B title from cache"
        assert first["url"] == "https://tvrain.tv/b" and first["program"] == "Newsroom"
        assert (
            second["library_id"] is None
            and second["title"] == "C title"
            and second["date"] == "2023-01-01"
        )
        engine.find_similar_videos.assert_called_once_with(HASH_A, 5, None)


def test_similar_videos_endpoint_rejects_a_bad_reference(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    with patch("src.rainrag.api.query_engine"), patch("src.rainrag.api.config", _config(tmp_path)):
        assert client.get("/similar-videos", params={"video_id": "nope"}).status_code == 400
        assert client.get("/similar-videos", params={"video_id": "42"}).status_code == 404


def test_transcript_endpoint_serves_vtt_and_srt(client, tmp_path):
    track_dir = tmp_path / "archive" / WebMetadataLoader.hash_to_archive_dir(HASH_A)
    track_dir.mkdir(parents=True)
    (track_dir / f"{HASH_A}.ru.vtt").write_text(VTT, encoding="utf-8")
    with patch("src.rainrag.api.config", _config(tmp_path)):
        r = client.get(f"/transcript/{HASH_A}")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/vtt")
        assert "Рада вас видеть." in r.text
        r = client.get(f"/transcript/{HASH_A}", params={"format": "srt"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-subrip")
        assert r.text.startswith("1\n00:00:00,000 --> 00:00:03,500\n")
        assert client.get(f"/transcript/{HASH_A}", params={"language": "en"}).status_code == 404
        assert client.get("/transcript/zzz").status_code == 400


def test_transcript_endpoint_accepts_a_tvrain_url_as_reference(client, tmp_path, monkeypatch):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 7, "https://tvrain.tv/teleshow/x/a-7/")
    track_dir = tmp_path / "archive" / WebMetadataLoader.hash_to_archive_dir(HASH_A)
    track_dir.mkdir(parents=True)
    (track_dir / f"{HASH_A}.ru.vtt").write_text(VTT, encoding="utf-8")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    with patch("src.rainrag.api.config", _config(tmp_path)):
        r = client.get("/transcript/https://tvrain.tv/teleshow/x/a-7/")
        assert r.status_code == 200, r.text
        assert "Рада вас видеть." in r.text
        r = client.get("/transcript/7")
        assert r.status_code == 200


def test_resolve_video_ref_rejects_bad_cached_hashes_and_odd_digits(tmp_path, monkeypatch):
    cache = tmp_path / "web_metadata"
    cache.mkdir()
    (cache / f"{HASH_A}.json").write_text(
        json.dumps({"id": 5, "video_hash": None, "url": "https://tvrain.tv/n"}), encoding="utf-8"
    )
    (cache / f"{HASH_B}.json").write_text(
        json.dumps({"id": 6, "video_hash": "short", "url": "https://tvrain.tv/s"}), encoding="utf-8"
    )
    loader = WebMetadataLoader(cache, source="local")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    for ref in ("https://tvrain.tv/n", "https://tvrain.tv/s", "6", "\u00b2"):
        with pytest.raises(Exception) as err:
            _resolve_video_ref(ref, loader)
        assert err.value.status_code in (400, 404), ref


def test_library_id_index_is_not_cached_when_empty(tmp_path, monkeypatch):
    cache = tmp_path / "web_metadata"
    loader = WebMetadataLoader(cache, source="local")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    assert api_module._library_ids(loader) == {}
    _write_record(cache, HASH_A, 9, "https://tvrain.tv/a")
    assert api_module._library_ids(loader) == {9: HASH_A}
