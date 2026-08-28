"""Tests for scripts/backfill_web_metadata.py.

The script writes to the production collection, so what matters is that it can
never do the wrong kind of write: junk paths must not become CMS requests,
transient errors must not be recorded as permanent misses, dry-run must touch
nothing, and the payload it writes must be exactly what the ingest pipeline
would have written.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts import backfill_web_metadata as bf


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestVideoHashFromPath:
    def test_archive_path(self):
        p = "/data/archive/65/23/6523d51cfe439643ca4fa0df3779873f69f21d73.ru.vtt"
        assert bf.video_hash_from_path(p) == "6523d51cfe439643ca4fa0df3779873f69f21d73"

    def test_uppercase_is_normalized(self):
        p = "/x/" + "A" * 40 + ".en.vtt"
        assert bf.video_hash_from_path(p) == "a" * 40

    def test_non_hash_filename_is_rejected(self):
        """The corpus really does contain a test.en.vtt."""
        assert bf.video_hash_from_path("/mnt/vod/te/st/test.en.vtt") is None

    def test_wrong_length_is_rejected(self):
        assert bf.video_hash_from_path("/x/abc123.ru.vtt") is None

    def test_empty_path_is_rejected(self):
        assert bf.video_hash_from_path("") is None


class TestChunked:
    def test_exact_multiple(self):
        assert bf.chunked([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_remainder(self):
        assert bf.chunked([1, 2, 3], 2) == [[1, 2], [3]]

    def test_empty(self):
        assert bf.chunked([], 5) == []


class TestStateFiles:
    def test_round_trip(self, tmp_path: Path):
        f = tmp_path / "state" / "misses.txt"
        assert bf.load_line_set(f) == set()
        bf.append_line(f, "aaa")
        bf.append_line(f, "bbb")
        assert bf.load_line_set(f) == {"aaa", "bbb"}

    def test_blank_lines_are_ignored(self, tmp_path: Path):
        f = tmp_path / "misses.txt"
        f.write_text("aaa\n\n  \nbbb\n")
        assert bf.load_line_set(f) == {"aaa", "bbb"}


# ---------------------------------------------------------------------------
# Throttled API wrapper
# ---------------------------------------------------------------------------


class TestThrottledAPIClient:
    def test_first_call_is_not_paced(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(bf.time, "sleep", sleeps.append)
        inner = MagicMock()
        inner.fetch_by_hash.return_value = {"name": "x"}
        client = bf.ThrottledAPIClient(inner, sleep_seconds=0.5)
        client.fetch_by_hash("a" * 40)
        assert sleeps == []

    def test_subsequent_calls_are_paced(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(bf.time, "sleep", sleeps.append)
        inner = MagicMock()
        inner.fetch_by_hash.return_value = None
        client = bf.ThrottledAPIClient(inner, sleep_seconds=0.5)
        client.fetch_by_hash("a" * 40)
        client.fetch_by_hash("b" * 40)
        assert sleeps == [0.5]
        assert client.remote_calls == 2

    def test_zero_sleep_never_sleeps(self, monkeypatch):
        sleeps: list[float] = []
        monkeypatch.setattr(bf.time, "sleep", sleeps.append)
        inner = MagicMock()
        inner.fetch_by_hash.return_value = None
        client = bf.ThrottledAPIClient(inner, sleep_seconds=0)
        client.fetch_by_hash("a" * 40)
        client.fetch_by_hash("b" * 40)
        assert sleeps == []


# ---------------------------------------------------------------------------
# Grouping the scroll
# ---------------------------------------------------------------------------


def _point(pid: int, path: str):
    p = MagicMock()
    p.id = pid
    p.payload = {"path": path}
    return p


class TestCollectPointsByVideo:
    def test_groups_and_skips_junk(self):
        h1, h2 = "a" * 40, "b" * 40
        pages = [
            ([_point(1, f"/x/{h1}.ru.vtt"), _point(2, f"/x/{h1}.en.vtt")], "cursor"),
            ([_point(3, f"/x/{h2}.ru.vtt"), _point(4, "/x/test.en.vtt")], None),
        ]
        client = MagicMock()
        client.scroll.side_effect = pages
        by_video, skipped = bf.collect_points_by_video(client, "c")
        assert by_video == {h1: [1, 2], h2: [3]}
        assert skipped == 1

    def test_missing_path_payload_is_skipped(self):
        p = MagicMock()
        p.id = 1
        p.payload = None
        client = MagicMock()
        client.scroll.side_effect = [([p], None)]
        by_video, skipped = bf.collect_points_by_video(client, "c")
        assert by_video == {}
        assert skipped == 1


# ---------------------------------------------------------------------------
# Per-video backfill
# ---------------------------------------------------------------------------


RAW_ARTICLE = {
    "name": "Вечерние новости",
    "date_active_start": "2024-05-01T18:00:00Z",
    "url": "https://tvrain.tv/x",
    "preview_text": "Коротко о главном.",
    "detail_text": "",
    "tags": [
        {"id": 1, "name": "Выборы", "category": "theme"},
        {"id": 2, "name": "Москва", "category": "location"},
    ],
    "teleshow": {"name": "Вечер"},
    "presentors": [{"firstname": "Анна", "lastname": "Иванова"}],
    "stories": [],
}


def _real_loader(tmp_path: Path, api_result):
    """A real WebMetadataLoader over a temp dir with a stubbed API client.

    Using the real loader keeps the test honest about the contract the script
    relies on: extraction goes through the same code as the ingest pipeline.
    """
    from rainrag.ingest import WebMetadataLoader

    api = MagicMock()
    api.fetch_by_hash.return_value = api_result
    return WebMetadataLoader(tmp_path / "meta", source="hybrid", api_client=api)


class TestBackfillVideo:
    def test_writes_the_same_payload_ingest_would(self, tmp_path: Path):
        from rainrag.ingest import document_web_fields

        loader = _real_loader(tmp_path, RAW_ARTICLE)
        qdrant = MagicMock()
        outcome, written = bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=[1, 2, 3],
            dry_run=False,
        )
        assert outcome == "written"
        assert written == 3
        payload = qdrant.set_payload.call_args.kwargs["payload"]
        assert payload == document_web_fields(loader.extract_clean_metadata(RAW_ARTICLE))
        assert payload["web_title"] == "Вечерние новости"
        assert payload["web_program"] == "Вечер"
        assert payload["web_tags_theme"] == ["Выборы"]
        assert payload["web_tags_location"] == ["Москва"]
        assert payload["web_presenters"] == ["Анна Иванова"]

    def test_point_ids_are_batched(self, tmp_path: Path):
        loader = _real_loader(tmp_path, RAW_ARTICLE)
        qdrant = MagicMock()
        outcome, written = bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=list(range(1200)),
            dry_run=False,
            payload_batch=500,
        )
        assert outcome == "written"
        assert written == 1200
        assert qdrant.set_payload.call_count == 3

    def test_404_reports_no_article_and_writes_nothing(self, tmp_path: Path):
        loader = _real_loader(tmp_path, None)
        qdrant = MagicMock()
        outcome, written = bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=[1],
            dry_run=False,
        )
        assert outcome == "no_article"
        assert written == 0
        qdrant.set_payload.assert_not_called()

    def test_article_without_title_or_description_is_empty(self, tmp_path: Path):
        bare = {"name": "", "preview_text": "", "detail_text": "", "tags": []}
        loader = _real_loader(tmp_path, bare)
        qdrant = MagicMock()
        outcome, written = bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=[1],
            dry_run=False,
        )
        assert outcome == "empty"
        qdrant.set_payload.assert_not_called()

    def test_dry_run_never_touches_qdrant(self, tmp_path: Path):
        loader = _real_loader(tmp_path, RAW_ARTICLE)
        qdrant = MagicMock()
        outcome, written = bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=[1, 2],
            dry_run=True,
        )
        assert outcome == "written"
        assert written == 0
        qdrant.set_payload.assert_not_called()

    def test_api_error_raises_instead_of_recording_a_miss(self, tmp_path: Path):
        """A CMS outage must not be recorded as thousands of permanent 404s.

        The loader swallows fetch exceptions and returns None ("stale metadata
        beats none" is right at ingest time), which makes an outage look
        identical to a missing article. The ThrottledAPIClient remembers the
        exception as it passes through, and backfill_video re-raises it so the
        main loop counts an error and retries next run.
        """
        from rainrag.ingest import WebMetadataLoader

        inner = MagicMock()
        inner.fetch_by_hash.side_effect = RuntimeError("connection reset")
        api = bf.ThrottledAPIClient(inner, sleep_seconds=0)
        loader = WebMetadataLoader(tmp_path / "meta", source="hybrid", api_client=api)
        with pytest.raises(RuntimeError, match="connection reset"):
            bf.backfill_video(
                qdrant=MagicMock(),
                collection="c",
                loader=loader,
                video_hash="a" * 40,
                point_ids=[1],
                dry_run=False,
            )

    def test_stale_error_flag_does_not_poison_a_cache_hit(self, tmp_path: Path):
        """After an error, a video served from the local cache must still work."""
        import json

        from rainrag.ingest import WebMetadataLoader

        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / ("b" * 40 + ".json")).write_text(
            json.dumps(RAW_ARTICLE, ensure_ascii=False), encoding="utf-8"
        )
        inner = MagicMock()
        inner.fetch_by_hash.side_effect = RuntimeError("boom")
        api = bf.ThrottledAPIClient(inner, sleep_seconds=0)
        loader = WebMetadataLoader(meta_dir, source="hybrid", api_client=api)

        with pytest.raises(RuntimeError):
            bf.backfill_video(
                qdrant=MagicMock(),
                collection="c",
                loader=loader,
                video_hash="a" * 40,
                point_ids=[1],
                dry_run=False,
            )
        # Second video is on disk: no API call, and the old error must not leak.
        outcome, written = bf.backfill_video(
            qdrant=MagicMock(),
            collection="c",
            loader=loader,
            video_hash="b" * 40,
            point_ids=[1],
            dry_run=False,
        )
        assert outcome == "written"


class TestDescriptionCapIsHonoured:
    """The cap must come from config, not be hardcoded at the call site.

    A getattr chain onto the loader silently resolved to None and always used
    the default, so a custom limit was ignored and `0` (disable) could not work.
    """

    LONG = "Заместитель председателя Совета предпринимателей рассказал о теме. " * 200

    def _article(self):
        return {**RAW_ARTICLE, "preview_text": self.LONG, "detail_text": ""}

    def test_configured_limit_is_applied(self, tmp_path: Path):
        loader = _real_loader(tmp_path, self._article())
        qdrant = MagicMock()
        bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=[1],
            dry_run=False,
            max_description_chars=120,
        )
        desc = qdrant.set_payload.call_args.kwargs["payload"]["web_description"]
        assert len(desc) <= 121

    def test_zero_disables_truncation(self, tmp_path: Path):
        loader = _real_loader(tmp_path, self._article())
        qdrant = MagicMock()
        bf.backfill_video(
            qdrant=qdrant,
            collection="c",
            loader=loader,
            video_hash="a" * 40,
            point_ids=[1],
            dry_run=False,
            max_description_chars=0,
        )
        desc = qdrant.set_payload.call_args.kwargs["payload"]["web_description"]
        assert len(desc) > 1000
