"""Tests for parallel VTT parsing during ingestion.

Parsing 287k files over NFS took 22 hours sequentially at 3.6 files/s with the
CPU ~2% busy: the wall-clock cost is waiting on the network, not computing.
Threads overlap the waits. The contract that matters is that they change
nothing else — same documents, same order, same counters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rainrag.config import Config
from rainrag.ingest import Ingester


@pytest.fixture
def archive(tmp_path: Path, sample_vtt_ru: str, sample_vtt_en: str) -> Path:
    """A small archive laid out like the real one (hash-sharded directories)."""
    root = tmp_path / "archive"
    for i in range(24):
        d = root / f"{i:02d}" / "ab" / "cd"
        d.mkdir(parents=True)
        (d / f"{i:040x}.ru.vtt").write_text(sample_vtt_ru, encoding="utf-8")
        (d / f"{i:040x}.en.vtt").write_text(sample_vtt_en, encoding="utf-8")
    return root


@pytest.fixture
def config(tmp_path: Path, archive: Path, test_config: Config) -> Config:
    """The shared test config, pointed at the sharded archive above."""
    cfg = test_config
    cfg.paths.archive_root = str(archive)
    cfg.paths.docs_output = str(tmp_path / "docs.jsonl")
    cfg.web_metadata.enabled = False
    return cfg


def _files(cfg: Config) -> list[Path]:
    return sorted(Path(cfg.paths.archive_root).rglob("*.vtt"))


class TestParallelMatchesSequential:
    def test_same_documents_in_the_same_order(self, config: Config) -> None:
        files = _files(config)

        config.processing.ingest_workers = 1
        seq = [
            (str(p), [d.model_dump_json() for d in docs])
            for p, docs in Ingester(config)._parsed_documents(files)
        ]

        config.processing.ingest_workers = 8
        par = [
            (str(p), [d.model_dump_json() for d in docs])
            for p, docs in Ingester(config)._parsed_documents(files)
        ]

        assert par == seq
        assert sum(len(d) for _, d in seq) > 0, "fixture produced no documents"

    def test_every_input_file_is_yielded_once(self, config: Config) -> None:
        files = _files(config)
        config.processing.ingest_workers = 8
        yielded = [p for p, _ in Ingester(config)._parsed_documents(files)]
        assert yielded == files

    def test_batching_covers_files_beyond_one_batch(self, config: Config) -> None:
        """batch_size is workers*32; with 2 workers the 48 files span batches."""
        files = _files(config)
        config.processing.ingest_workers = 2
        yielded = [p for p, _ in Ingester(config)._parsed_documents(files)]
        assert len(yielded) == len(files)
        assert yielded == files

    def test_counters_match_between_modes(self, config: Config) -> None:
        files = _files(config)

        config.processing.ingest_workers = 1
        a = Ingester(config)
        list(a._parsed_documents(files))

        config.processing.ingest_workers = 8
        b = Ingester(config)
        list(b._parsed_documents(files))

        assert (a.invalid_vtt_count, a.speech_free_count) == (
            b.invalid_vtt_count,
            b.speech_free_count,
        )


class TestWorkerFailureIsolation:
    def test_one_bad_file_does_not_abort_the_run(self, config: Config, monkeypatch) -> None:
        """A parse failure must cost one file, not the whole 287k-file ingest."""
        files = _files(config)
        config.processing.ingest_workers = 4
        ing = Ingester(config)
        real = ing.process_file
        bad = files[3]

        def flaky(path: Path):
            if path == bad:
                raise RuntimeError("corrupt VTT")
            return real(path)

        monkeypatch.setattr(ing, "process_file", flaky)
        out = list(ing._parsed_documents(files))

        assert len(out) == len(files)
        assert dict(out)[bad] == []
        assert ing.invalid_vtt_count >= 1
        # Everything else still parsed.
        assert sum(len(d) for p, d in out if p != bad) > 0


class TestWorkerConfig:
    def test_single_worker_uses_the_sequential_path(self, config: Config) -> None:
        config.processing.ingest_workers = 1
        files = _files(config)
        assert len(list(Ingester(config)._parsed_documents(files))) == len(files)

    def test_worker_count_is_clamped_to_at_least_one(self, config: Config) -> None:
        """A stale config object without the field must not crash the ingest."""
        from types import SimpleNamespace

        ing = Ingester(config)
        ing.config.processing = SimpleNamespace(
            max_file_size=config.processing.max_file_size, min_text_length=50
        )
        assert len(list(ing._parsed_documents(_files(config)))) == len(_files(config))

    def test_default_is_parallel(self) -> None:
        from rainrag.config import ProcessingConfig

        assert ProcessingConfig().ingest_workers > 1


class TestSizeGuardMovedIntoTheWorker:
    def test_oversized_file_is_skipped(self, config: Config) -> None:
        """The check used to be a pre-pass of 287k sequential stats."""
        files = _files(config)
        config.processing.max_file_size = 10  # bytes
        config.processing.ingest_workers = 4
        out = list(Ingester(config)._parsed_documents(files))
        assert all(docs == [] for _, docs in out)

    def test_find_vtt_files_no_longer_stats(self, config: Config, monkeypatch) -> None:
        """Regression: the pre-stat pass cost 4.4 hours before the first document."""
        calls = {"n": 0}
        real_stat = Path.stat

        def counting_stat(self, *a, **k):
            calls["n"] += 1
            return real_stat(self, *a, **k)

        monkeypatch.setattr(Path, "stat", counting_stat)
        ing = Ingester(config)
        files = list(ing.find_vtt_files(Path(config.paths.archive_root)))

        # Directory-walk stats (is_dir) are fine and scale with directories.
        # What must not come back is a stat PER FILE: that was 287k sequential
        # NFS round trips at 18/s before a single document was produced.
        assert calls["n"] < len(files), (
            f"find_vtt_files made {calls['n']} stat calls for {len(files)} files; "
            "the per-file size check must live in process_file"
        )
