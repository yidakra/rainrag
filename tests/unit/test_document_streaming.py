"""Tests for lazy document loading and the description cap.

Both exist because the embed stage was OOM-killed on 2026-08-28: 2.64M
documents from a 27 GB corpus needed 11.9 GB of RSS on a 15 GB host. The
corpus only grows, so the pipeline must not scale its memory with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rainrag.embed import DocumentFile
from rainrag.ingest import Document, truncate_description


def _write_docs(path: Path, n: int) -> list[dict]:
    records = []
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "id": f"doc{i}",
                "path": f"/archive/{i}.ru.vtt",
                "language": "ru",
                "text": f"фрагмент номер {i} с некоторым содержанием",
                "length": 40,
                "content_hash": f"hash{i}",
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


@pytest.fixture
def docs_file(tmp_path: Path) -> Path:
    path = tmp_path / "docs.jsonl"
    _write_docs(path, 50)
    return path


class TestDocumentFile:
    def test_len_matches_line_count(self, docs_file: Path) -> None:
        assert len(DocumentFile(docs_file)) == 50

    def test_iteration_yields_documents_in_order(self, docs_file: Path) -> None:
        docs = list(DocumentFile(docs_file))
        assert len(docs) == 50
        assert all(isinstance(d, Document) for d in docs)
        assert [d.id for d in docs] == [f"doc{i}" for i in range(50)]

    def test_indexing_matches_iteration(self, docs_file: Path) -> None:
        df = DocumentFile(docs_file)
        as_list = list(df)
        assert df[0].id == as_list[0].id
        assert df[17].id == as_list[17].id
        assert df[-1].id == as_list[-1].id

    def test_slicing_returns_documents(self, docs_file: Path) -> None:
        """generate_embeddings slices the sequence into chunks."""
        chunk = DocumentFile(docs_file)[10:15]
        assert [d.id for d in chunk] == [f"doc{i}" for i in range(10, 15)]

    def test_out_of_range_raises(self, docs_file: Path) -> None:
        with pytest.raises(IndexError):
            DocumentFile(docs_file)[999]

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "docs.jsonl"
        _write_docs(path, 3)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n")
        df = DocumentFile(path)
        assert len(df) == 3
        assert len(list(df)) == 3

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert len(DocumentFile(path)) == 0
        assert list(DocumentFile(path)) == []

    def test_for_loop_does_not_build_the_offset_table(self, docs_file: Path) -> None:
        """The hot path is a plain for loop and must stay purely sequential.

        (list(df) does build it, because list() asks for a length hint first;
        the pipeline iterates rather than materialising, which is the point.)
        """
        df = DocumentFile(docs_file)
        seen = 0
        for _ in df:
            seen += 1
        assert seen == 50
        assert df._offsets is None

    def test_reiterable(self, docs_file: Path) -> None:
        """Unlike a generator, this can be walked more than once."""
        df = DocumentFile(docs_file)
        assert [d.id for d in df] == [d.id for d in df]


class TestTruncateDescription:
    LONG = "Заместитель председателя Совета предпринимателей рассказал о перспективах. " * 200

    def test_long_text_is_capped(self) -> None:
        out = truncate_description(self.LONG, 600)
        assert len(out) <= 601  # 600 plus the ellipsis
        assert out.endswith("…")

    def test_cut_lands_on_a_word_boundary(self) -> None:
        out = truncate_description(self.LONG, 600)
        body = out[:-1]  # drop the ellipsis
        assert not body.endswith(" ")
        # Everything kept is a verbatim prefix of the original, so no word was
        # severed mid-way (trailing punctuation is stripped before the ellipsis).
        assert self.LONG.startswith(body)

    def test_short_text_untouched(self) -> None:
        assert truncate_description("короткий текст", 600) == "короткий текст"

    def test_zero_disables(self) -> None:
        assert truncate_description(self.LONG, 0) == self.LONG

    def test_none_and_non_strings_pass_through(self) -> None:
        assert truncate_description(None, 600) is None
        assert truncate_description(123, 600) == 123

    def test_no_word_break_falls_back_to_a_hard_cut(self) -> None:
        out = truncate_description("а" * 5000, 600)
        assert len(out) == 601
        assert out.endswith("…")


class TestDocumentWebFieldsApplyTheCap:
    def test_description_is_capped(self) -> None:
        from rainrag.ingest import document_web_fields

        meta = {"web_title": "Эфир", "web_description": TestTruncateDescription.LONG}
        fields = document_web_fields(meta, 600)
        assert len(fields["web_description"]) <= 601
        assert fields["web_title"] == "Эфир"

    def test_default_is_no_cap_for_callers_that_do_not_pass_one(self) -> None:
        """Back-compatible: the limit is opt-in at the call site."""
        from rainrag.ingest import document_web_fields

        meta = {"web_description": TestTruncateDescription.LONG}
        assert document_web_fields(meta)["web_description"] == TestTruncateDescription.LONG
