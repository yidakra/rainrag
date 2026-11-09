"""Video-level lookups: /similar-videos and /transcript/{video_id}."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rainrag import api as api_module  # noqa: E402
from src.rainrag.api import _resolve_video_ref, app  # noqa: E402
from src.rainrag.index import _build_payload  # noqa: E402
from src.rainrag.ingest import Document, VTTParser, WebMetadataLoader  # noqa: E402
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


def test_vtt_to_srt_keeps_cues_whose_identifier_starts_with_a_block_name():
    vtt = (
        "WEBVTT\n\nNOTE_1\n00:00:01.000 --> 00:00:02.000\nkept\n\n"
        "STYLE-1\n00:00:03.000 --> 00:00:04.000\nalso kept\n\n"
        "NOTE\nthis is dropped\n\nSTYLE\n::cue {}\n\n"
        "100:00:00.000 --> 100:00:01.000\nhour 100\n"
    )
    assert vtt_to_srt(vtt) == (
        "1\n00:00:01,000 --> 00:00:02,000\nkept\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nalso kept\n\n"
        "3\n100:00:00,000 --> 100:00:01,000\nhour 100\n"
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


def test_library_id_index_survives_a_truncated_file_and_a_non_string_hash(tmp_path):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 1, "u1")
    # a record caught mid-write, cut inside a multibyte character
    (cache / f"{HASH_B}.json").write_bytes(
        json.dumps({"id": 2, "name": "Дождь"}, ensure_ascii=False).encode("utf-8")[:-3]
    )
    (cache / f"{HASH_C}.json").write_text(
        json.dumps({"id": 3, "video_hash": 12345}), encoding="utf-8"
    )
    assert WebMetadataLoader(cache, source="local").library_id_index() == {1: HASH_A, 3: "12345"}


def test_library_id_index_skips_records_without_an_id(tmp_path):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 1, "u1")
    (cache / f"{HASH_B}.json").write_text(json.dumps({"video_hash": HASH_B}), encoding="utf-8")
    (cache / "broken.json").write_text("{", encoding="utf-8")
    assert WebMetadataLoader(cache, source="local").library_id_index() == {1: HASH_A}


class _Points:
    def __init__(self, points):
        self.points = points


def _matches(flt, payload: dict) -> bool:
    """Qdrant filter semantics: every must, any should, no must_not; nested filters recurse."""

    def hit(cond) -> bool:
        if hasattr(cond, "key"):
            return payload.get(cond.key) == cond.match.value
        return _matches(cond, payload)

    return (
        all(hit(c) for c in (flt.must or []))
        and (not flt.should or any(hit(c) for c in flt.should))
        and not any(hit(c) for c in (flt.must_not or []))
    )


def _seed_points(archive_root: Path, *, with_video_id: bool = True) -> list[SimpleNamespace]:
    """Three chunks of HASH_A under the payload the indexer writes (``_build_payload``)."""
    track_dir = archive_root / WebMetadataLoader.hash_to_archive_dir(HASH_A)
    points = []
    for i, (vector, lang) in enumerate(
        [([1.0, 0.0], "ru"), ([0.0, 1.0], "ru"), ([0.5, 0.5], "en")]
    ):
        path = track_dir / f"{HASH_A}.{lang}.vtt"
        doc = Document(
            id=f"{HASH_A}-{i}",
            path=str(path),
            language=lang,
            text="t",
            length=1,
            video_id=VTTParser.generate_video_id(path),
        )
        payload = _build_payload(doc)
        if not with_video_id:
            del payload["video_id"]  # an index built before video_id was stored
        points.append(SimpleNamespace(vector=vector, payload=payload))
    return points


class _FakeQdrant:
    """Seed chunks filtered the way Qdrant would; hits spread over two other videos."""

    def __init__(self, archive_root: Path, points: list[SimpleNamespace]):
        self.root = archive_root
        self.points = points
        self.scroll_filters = []
        self.query_filters = []
        self.scroll_limits = []
        self.timeouts = []
        self.batches = []

    def scroll(
        self, collection_name, scroll_filter, limit, with_payload, with_vectors, timeout=None
    ):
        self.scroll_filters.append(scroll_filter)
        self.scroll_limits.append(limit)
        self.timeouts.append(("scroll", timeout))
        return [p for p in self.points if _matches(scroll_filter, p.payload)][:limit], None

    def _hit(self, video_hash, score, lang="ru", **web):
        path = str(
            self.root
            / WebMetadataLoader.hash_to_archive_dir(video_hash)
            / f"{video_hash}.{lang}.vtt"
        )
        payload = {"path": path, "video_id": video_hash[:16], "language": lang, **web}
        return SimpleNamespace(score=score, payload=payload)

    def query_batch_points(self, collection_name, requests, timeout=None):
        self.timeouts.append(("query_batch_points", timeout))
        self.batches.append(requests)
        return [
            self.query_points(collection_name, list(r.query), r.filter, r.limit, r.with_payload)
            for r in requests
        ]

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


def _engine(tmp_path, *, with_video_id: bool = True) -> tuple[RAGQueryEngine, _FakeQdrant]:
    root = tmp_path / "archive"
    config = SimpleNamespace(
        paths=SimpleNamespace(archive_root=str(root)),
        qdrant=SimpleNamespace(collection_name="broadcast_transcripts", distance="Cosine"),
    )
    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.config = config  # type: ignore[attr-defined]
    fake = _FakeQdrant(root, _seed_points(root, with_video_id=with_video_id))
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
    # the seed video is selected, then excluded, by its video_id and its file paths
    seed_filter = fake.scroll_filters[0].must[0]
    seed_video_id = fake.points[0].payload["video_id"]
    assert seed_filter.should[0].match.value == seed_video_id
    assert {c.key for c in seed_filter.should} == {"video_id", "path"}
    assert fake.query_filters[0].must_not[0].match.value == seed_video_id


def test_seed_paths_come_from_the_corpus_when_the_archive_root_moved(tmp_path):
    """Indexed under another root, without video_id: the corpus paths still find the seed."""
    engine, fake = _engine(tmp_path, with_video_id=False)
    ingest_time_path = f"/mnt/other-root/aa/{HASH_A}.ru.vtt"
    engine.bm25_corpus = [
        {"path": "/x/" + HASH_B + ".ru.vtt", "doc_id": "other"},  # type: ignore[attr-defined]
        {"path": ingest_time_path, "doc_id": "seed-doc"},
    ]
    for point in fake.points:
        point.payload["path"] = ingest_time_path
    ranked = engine.find_similar_videos(HASH_A)
    assert [r["video_hash"] for r in ranked] == [HASH_B, HASH_C]
    should = fake.scroll_filters[0].must[0].should
    assert [c.key for c in should] == ["video_id", "path"]
    assert should[1].match.value == ingest_time_path
    # no per-chunk Qdrant lookup: the only queries are the three seed vectors, in one batch
    assert len(fake.query_filters) == 3
    assert fake.query_filters[-1].must_not[1].match.value == ingest_time_path


def test_seed_video_id_uses_the_corpus_video_id_when_present(tmp_path):
    engine, fake = _engine(tmp_path)
    engine.bm25_corpus = [
        {"path": f"/any/{HASH_A}.en.vtt", "doc_id": "d", "video_id": "from-corpus"}
    ]  # type: ignore[attr-defined]
    engine.find_similar_videos(HASH_A)
    assert fake.scroll_filters[0].must[0].should[0].match.value == "from-corpus"
    assert fake.scroll_filters[0].must[0].should[1].match.value == f"/any/{HASH_A}.en.vtt"


def test_find_similar_videos_drops_seed_chunks_indexed_under_another_id(tmp_path):
    engine, fake = _engine(tmp_path)
    real_query_points = fake.query_points

    def query_points(collection_name, query=None, query_filter=None, limit=10, with_payload=True):
        points = real_query_points(collection_name, query, query_filter, limit, with_payload)
        points.points.append(fake._hit(HASH_A, 0.99))  # the seed under a stray video_id
        return points

    fake.query_points = query_points  # type: ignore[method-assign]
    ranked = engine.find_similar_videos(HASH_A)
    assert HASH_A not in [r["video_hash"] for r in ranked]


def test_as_int_bounds_the_digit_string():
    from src.rainrag.api import _as_int

    assert _as_int("357040") == 357040 and _as_int(7) == 7
    assert _as_int("9" * 18) == int("9" * 18)
    assert _as_int("9" * 19) is None and _as_int("1" * 5000) is None
    assert _as_int(True) is None and _as_int("\u00b2") is None and _as_int("12a") is None


def test_find_similar_videos_limit_and_language(tmp_path):
    engine, fake = _engine(tmp_path)
    ranked = engine.find_similar_videos(HASH_A, limit=1, language="en")
    assert len(ranked) == 1
    assert fake.scroll_filters[0].must[1].match.value == "en"
    # the exclusion still names both tracks: on an index without video_id the
    # ru chunks of the seed must not take the nearest-neighbour slots
    excluded = {c.match.value for c in fake.query_filters[0].must_not if c.key == "path"}
    assert excluded == {p.payload["path"] for p in fake.points}


def test_seed_chunks_of_an_index_without_video_id_are_found_by_path(tmp_path):
    engine, fake = _engine(tmp_path, with_video_id=False)
    ranked = engine.find_similar_videos(HASH_A)
    assert [r["video_hash"] for r in ranked] == [HASH_B, HASH_C]
    paths = {c.match.value for c in fake.scroll_filters[0].must[0].should if c.key == "path"}
    assert paths == {p.payload["path"] for p in fake.points}


def test_find_similar_videos_folds_euclidean_distances_to_similarities(tmp_path):
    engine, fake = _engine(tmp_path)
    engine.config.qdrant.distance = "Euclidean"  # type: ignore[attr-defined]
    ranked = engine.find_similar_videos(HASH_A)
    # the fake's scores are now distances: C (0.4, 0.6, 0.5) is nearer than B (0.9, 0.7)
    assert [r["video_hash"] for r in ranked] == [HASH_C, HASH_B]
    c, b = ranked
    assert c["best"] == pytest.approx(1 / 1.4, abs=1e-4)
    assert b["best"] == pytest.approx(1 / 1.7, abs=1e-4)
    assert b["score"] == pytest.approx(1 / 1.9 + 1 / 1.7, abs=1e-4)


def test_seed_scroll_with_a_language_never_takes_other_videos(tmp_path):
    """A top-level should turns optional next to a must; the seed group must stay mandatory."""
    engine, fake = _engine(tmp_path)
    other = fake._hit(HASH_B, 0.0, lang="en").payload
    fake.points.append(SimpleNamespace(vector=[0.9, 0.1], payload=other))
    engine.find_similar_videos(HASH_A, language="en")
    scrolled = [p for p in fake.points if _matches(fake.scroll_filters[0], p.payload)]
    assert [p.payload["language"] for p in scrolled] == ["en"]
    assert all(HASH_A in p.payload["path"] for p in scrolled)


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


def test_similar_videos_endpoint_fetches_missing_metadata_within_a_deadline(
    client, tmp_path, monkeypatch
):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 1, "https://tvrain.tv/a")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    monkeypatch.setattr(api_module, "_ENRICH_DEADLINE_SECONDS", 0.2)
    monkeypatch.delenv("LIBRARY_API_TOKEN", raising=False)
    cfg = _config(tmp_path)
    cfg.web_metadata.source = "hybrid"
    fetched = []

    def fetch_api(self, video_hash):
        fetched.append(video_hash)
        if video_hash == HASH_B:
            return {"id": 2, "name": "B from the API"}
        time.sleep(1.0)  # a stalled metadata service
        return {"id": 3, "name": "too late"}

    monkeypatch.setattr(api_module.WebMetadataLoader, "_fetch_api", fetch_api)
    entry = {"score": 0.5, "best": 0.5, "matching_chunks": 1, "url": None, "program": None}
    ranked = [
        {"video_hash": HASH_B, "title": "B payload", "date": None, **entry},
        {"video_hash": HASH_C, "title": "C payload", "date": None, **entry},
    ]
    with patch("src.rainrag.api.query_engine") as engine, patch("src.rainrag.api.config", cfg):
        engine.find_similar_videos.return_value = ranked
        started = time.monotonic()
        r = client.get("/similar-videos", params={"video_id": "1"})
        assert r.status_code == 200, r.text
        assert time.monotonic() - started < 0.9
        first, second = r.json()["results"]
        assert first["title"] == "B from the API" and first["library_id"] == 2
        assert second["title"] == "C payload" and second["library_id"] is None
    assert sorted(fetched) == [HASH_B, HASH_C]
    # the stalled fetch ran on the enrichment executor, not the default one
    assert any(t.name.startswith("enrich") for t in threading.enumerate())


def test_metadata_loader_fails_fast_without_a_token_in_api_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBRARY_API_TOKEN", raising=False)
    cfg = _config(tmp_path)
    cfg.web_metadata.source = "api"
    with patch("src.rainrag.api.config", cfg):
        with pytest.raises(HTTPException) as err:
            api_module._metadata_loader()
        assert err.value.status_code == 503
        cfg.web_metadata.source = "hybrid"
        assert api_module._metadata_loader().api_client is None


def test_metadata_cache_writes_leave_no_partial_file(tmp_path):
    cache = tmp_path / "web_metadata"
    cache.mkdir()
    target = cache / f"{HASH_A}.json"
    WebMetadataLoader._write_cache_record(target, {"id": 1, "name": "Дождь"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"id": 1, "name": "Дождь"}
    assert list(cache.iterdir()) == [target]
    # a failed write leaves no temp file behind either
    with pytest.raises(OSError):
        WebMetadataLoader._write_cache_record(cache / "missing" / "x.json", {"id": 2})
    assert list(cache.iterdir()) == [target]


def test_local_records_sweeps_stale_temp_files_only(tmp_path):
    cache = tmp_path / "web_metadata"
    _write_record(cache, HASH_A, 1, "u")
    stale = cache / f"{HASH_B}.json.deadbeef.tmp"
    stale.write_text("{", encoding="utf-8")
    hour_ago = time.time() - 3600
    os.utime(stale, (hour_ago, hour_ago))
    fresh = cache / f"{HASH_C}.json.cafe.tmp"  # a write in progress right now
    fresh.write_text("{", encoding="utf-8")
    loader = WebMetadataLoader(cache, source="local")
    assert [h for h, _ in loader.local_records()] == [HASH_A]
    assert not stale.exists() and fresh.exists()


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


def test_transcript_endpoint_takes_only_a_signed_token_in_the_query(client, tmp_path, monkeypatch):
    monkeypatch.setenv("RAINRAG_AUTH_TOKEN", "standing-secret")
    track_dir = tmp_path / "archive" / WebMetadataLoader.hash_to_archive_dir(HASH_A)
    track_dir.mkdir(parents=True)
    (track_dir / f"{HASH_A}.ru.vtt").write_text(VTT, encoding="utf-8")
    with patch("src.rainrag.api.config", _config(tmp_path)):
        assert client.get(f"/transcript/{HASH_A}").status_code == 401
        # the standing secret in a URL would land in logs and Referer headers
        assert (
            client.get(f"/transcript/{HASH_A}", params={"auth": "standing-secret"}).status_code
            == 401
        )
        signed = api_module.issue_media_token(60, secret="standing-secret")
        assert client.get(f"/transcript/{HASH_A}", params={"auth": signed}).status_code == 200
        headers = {"Authorization": "Bearer standing-secret"}
        assert client.get(f"/transcript/{HASH_A}", headers=headers).status_code == 200


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


def test_find_similar_videos_uses_16_seed_chunks_quantized_search_and_a_timeout(tmp_path):
    """The search takes at most 16 seed chunks, searches the int8 copy with rescoring, and sets a time limit."""
    from src.rainrag.query import SIMILAR_SEARCH_TIMEOUT_S

    engine, fake = _engine(tmp_path)
    engine.find_similar_videos(HASH_A, limit=10)
    assert fake.scroll_limits == [16]
    assert fake.timeouts == [
        ("scroll", SIMILAR_SEARCH_TIMEOUT_S),
        ("query_batch_points", SIMILAR_SEARCH_TIMEOUT_S),
    ]
    for request in fake.batches[0]:
        assert request.params.quantization.rescore is True
        assert request.params.quantization.oversampling == 2.0


def _qdrant_error(message: str):
    """Return the error the Qdrant client raises for a 500 with this message."""
    import httpx
    from qdrant_client.http.exceptions import UnexpectedResponse

    body = json.dumps({"status": {"error": message}}).encode()
    return UnexpectedResponse(500, "Internal Server Error", body, httpx.Headers())


def test_a_qdrant_timeout_becomes_similar_videos_timeout(tmp_path):
    """A Qdrant 'timed out' 500 from either call raises SimilarVideosTimeoutError."""
    from src.rainrag.query import SimilarVideosTimeoutError

    engine, fake = _engine(tmp_path)
    error = _qdrant_error("Timeout error: Operation 'Search' timed out after 30s")

    def batch_timeout(**_kw):
        raise error

    fake.query_batch_points = batch_timeout  # type: ignore[method-assign]
    with pytest.raises(SimilarVideosTimeoutError):
        engine.find_similar_videos(HASH_A, limit=10)

    def scroll_timeout(**_kw):
        raise error

    fake.scroll = scroll_timeout  # type: ignore[method-assign]
    with pytest.raises(SimilarVideosTimeoutError):
        engine.find_similar_videos(HASH_A, limit=10)


def test_another_qdrant_error_is_not_called_a_timeout(tmp_path):
    """A Qdrant 500 that is not a timeout propagates as it is."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    engine, fake = _engine(tmp_path)

    def broken(**_kw):
        raise _qdrant_error("Wrong input: bad vector")

    fake.query_batch_points = broken  # type: ignore[method-assign]
    with pytest.raises(UnexpectedResponse):
        engine.find_similar_videos(HASH_A, limit=10)


def test_similar_videos_endpoint_returns_504_on_a_search_timeout(client, tmp_path, monkeypatch):
    """The route answers 504 with a clear message instead of hanging or a 500."""
    from rainrag.query import SimilarVideosTimeoutError  # the class the route catches

    _write_record(tmp_path / "web_metadata", HASH_A, 1, "https://tvrain.tv/a", name="A")
    monkeypatch.setattr(api_module, "_LIBRARY_IDS", None)
    with (
        patch("src.rainrag.api.query_engine") as engine,
        patch("src.rainrag.api.config", _config(tmp_path)),
    ):
        engine.find_similar_videos.side_effect = SimilarVideosTimeoutError("slow")
        r = client.get("/similar-videos", params={"video_id": "1"})
    assert r.status_code == 504
    assert "timed out" in r.json()["detail"]
