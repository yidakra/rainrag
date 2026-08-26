"""Tests for near-duplicate candidate suppression.

The decisive cases are real: today's live queries returned the same broadcast
segment twice (once cased/punctuated, once raw) in two of five answer slots.
Those exact strings are the fixtures here.
"""

from __future__ import annotations

from types import SimpleNamespace

from rainrag.query import RAGQueryEngine


# Captured verbatim from a live query's rank-1 and rank-3 results: the same
# segment, transcribed twice (different casing/punctuation, different dates).
CASED = (
    "до 2000. Надо как минимум три до конца эфира. Поэтому, пожалуйста, и "
    "подписывайтесь на Дождь tvrain.ru слэш подписка вот этот адрес. Ну, а мы идем "
    "дальше. Владимир Путин выступил против введения ограничений в виде QR-кодов "
    "под Новый год. Президент заметил, что это сложная история."
)
RAW = (
    "подписывайтесь на дождь а тв рейн ру слэш подписка вот этот адрес ну а мы идем "
    "дальше владимир путин выступил против введения ограничений в виде qr-кодов под "
    "новый год президент заметил что это сложная история"
)


def _engine(threshold: float = 0.6) -> RAGQueryEngine:
    engine = RAGQueryEngine.__new__(RAGQueryEngine)
    engine.config = SimpleNamespace(reranker=SimpleNamespace(near_duplicate_threshold=threshold))
    return engine


def _doc(text: str, score: float) -> dict:
    return {"text": text, "score": score, "doc_id": text[:10]}


class TestSuppression:
    def test_live_rebroadcast_pair_collapses_to_the_higher_score(self):
        docs = [_doc(CASED, 0.845), _doc(RAW, 0.829), _doc("совсем другой текст о погоде", 0.5)]
        kept = _engine()._suppress_near_duplicates(docs)
        assert [d["score"] for d in kept] == [0.845, 0.5]

    def test_distinct_texts_survive(self):
        docs = [
            _doc("закон об иностранных агентах приняли в 2012 году", 0.9),
            _doc("закон о домашнем насилии обсуждается из года в год", 0.8),
        ]
        assert len(_engine()._suppress_near_duplicates(docs)) == 2

    def test_shared_phrases_are_not_duplicates(self):
        """Different segments quoting the same person must both survive."""
        docs = [
            _doc(
                "Владимир Путин выступил на пресс-конференции и говорил о экономике, "
                "инфляции и ценах на продукты в регионах страны",
                0.9,
            ),
            _doc(
                "Владимир Путин выступил перед Федеральным собранием с посланием "
                "о внешней политике и отношениях с западными странами",
                0.8,
            ),
        ]
        assert len(_engine()._suppress_near_duplicates(docs)) == 2

    def test_zero_threshold_disables(self):
        docs = [_doc(CASED, 0.9), _doc(RAW, 0.8)]
        assert len(_engine(threshold=0)._suppress_near_duplicates(docs)) == 2

    def test_missing_config_field_disables(self):
        engine = RAGQueryEngine.__new__(RAGQueryEngine)
        engine.config = SimpleNamespace(reranker=SimpleNamespace())
        docs = [_doc(CASED, 0.9), _doc(RAW, 0.8)]
        assert len(engine._suppress_near_duplicates(docs)) == 2

    def test_empty_and_single_are_untouched(self):
        assert _engine()._suppress_near_duplicates([]) == []
        one = [_doc("x", 0.1)]
        assert _engine()._suppress_near_duplicates(one) == one

    def test_empty_text_chunks_do_not_crash_or_collapse_together(self):
        docs = [_doc("", 0.9), _doc("", 0.8), _doc("нормальный текст тут есть", 0.7)]
        kept = _engine()._suppress_near_duplicates(docs)
        # Empty shingle sets have union 0; they must not be treated as
        # duplicates of each other by a 0/0 comparison.
        assert len(kept) == 3


class TestShingles:
    def test_normalization_erases_case_and_punctuation(self):
        a = RAGQueryEngine._text_shingles("Путин, выступил ПРОТИВ ограничений!")
        b = RAGQueryEngine._text_shingles("путин выступил против ограничений")
        assert a == b

    def test_short_text_yields_single_shingle(self):
        assert RAGQueryEngine._text_shingles("два слова") == {("два", "слова")}
