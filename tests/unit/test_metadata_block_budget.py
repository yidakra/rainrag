"""Tests for the cap on appended web-metadata blocks.

Regression cover for 2026-08-26: `description` (a full article body, ~5900
tokens) was appended to every chunk against a 462-token budget. The caller
computes `max(1, budget - metadata_tokens)`, so every chunk collapsed to a
single token carrying a copy of the article — 295 GB written before the disk
filled, and an index that would have been useless for retrieval.
"""

from __future__ import annotations

from types import SimpleNamespace

from rainrag.ingest import Ingester


def _ingester(*, fields: list[str], share: float = 0.25, budget: int | None = 462):
    ing = Ingester.__new__(Ingester)
    ing.config = SimpleNamespace(
        web_metadata=SimpleNamespace(
            fields=fields,
            append_label="[Web]",
            append_to_each_chunk=True,
            include_in_text=True,
            max_block_token_share=share,
        ),
        chunking=SimpleNamespace(max_chunk_tokens=budget),
        get_max_chunk_tokens=lambda: budget,
    )
    return ing


ARTICLE = "Заместитель председателя Совета предпринимателей рассказал о перспективах. " * 200
META = {
    "web_title": "Вечернее шоу",
    "web_date": "2021-12-09",
    "web_description": ARTICLE,
}


class TestBlockCap:
    def test_oversized_description_is_truncated(self):
        ing = _ingester(fields=["title", "date", "description"])
        block = ing._build_web_metadata_block(META)
        assert block.endswith("…")
        # Must leave the transcript the bulk of the budget.
        from rainrag.ingest import VTTParser

        assert VTTParser.estimate_tokens(block, "ru") <= int(462 * 0.25) + 5

    def test_short_block_is_untouched(self):
        ing = _ingester(fields=["title", "date"])
        block = ing._build_web_metadata_block(META)
        assert not block.endswith("…")
        assert "Вечернее шоу" in block
        assert "2021-12-09" in block

    def test_zero_share_disables_the_cap(self):
        ing = _ingester(fields=["title", "date", "description"], share=0)
        block = ing._build_web_metadata_block(META)
        assert not block.endswith("…")

    def test_unknown_budget_disables_the_cap(self):
        ing = _ingester(fields=["title", "date", "description"], budget=None)
        block = ing._build_web_metadata_block(META)
        assert not block.endswith("…")

    def test_no_metadata_returns_none(self):
        assert _ingester(fields=["title"])._build_web_metadata_block({}) is None

    def test_capped_block_still_leaves_a_usable_transcript_budget(self):
        """The bug's real signature: adjusted_max_tokens collapsing to 1."""
        from rainrag.ingest import VTTParser

        ing = _ingester(fields=["title", "date", "description"])
        block = ing._build_web_metadata_block(META)
        adjusted = max(1, 462 - VTTParser.estimate_tokens(block, "ru"))
        assert adjusted > 300
