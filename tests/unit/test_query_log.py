"""Tests for the pilot answer log.

It exists to capture everything during the pilot, so what matters is that a
record is complete, that disabling it is honored, and that a logging failure
can never fail the query it describes.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rainrag import api


RESULT = {
    "question": "вопрос",
    "answer": "ответ",
    "retrieved_documents": [
        {
            "doc_id": "d1",
            "path": "/a.ru.vtt",
            "rank": 1,
            "score": 0.9,
            "rerank_score": 0.8,
            "date": "2021-01-01",
            "web_title": "Эфир",
            "start_time": "00:01:00",
            "end_time": "00:02:00",
            "text": "excerpt that must NOT be duplicated into the log",
        }
    ],
    "cost.llm_tokens_in": 100,
    "cost.llm_tokens_out": 20,
}


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "log" / "query_log.jsonl"
    monkeypatch.setattr(api, "QUERY_LOG_PATH", str(path))
    return path


class TestAppendQueryLog:
    def test_record_is_complete(self, log_path):
        api.append_query_log(RESULT, mode="corpus", transport="stream", lang="ru", top_k=5)
        record = json.loads(log_path.read_text().strip())
        assert record["question"] == "вопрос"
        assert record["answer"] == "ответ"
        assert record["mode"] == "corpus"
        assert record["transport"] == "stream"
        assert record["tokens_in"] == 100
        assert record["ts"] > 0
        doc = record["docs"][0]
        assert doc["doc_id"] == "d1"
        assert doc["rerank_score"] == 0.8
        assert doc["web_title"] == "Эфир"
        # The transcript excerpt lives in the archive; duplicating it per
        # query would bloat the log for nothing.
        assert "text" not in doc

    def test_appends_not_overwrites(self, log_path):
        api.append_query_log(RESULT, mode="corpus")
        api.append_query_log(RESULT, mode="corpus")
        assert len(log_path.read_text().strip().splitlines()) == 2

    def test_empty_path_disables(self, monkeypatch, tmp_path):
        monkeypatch.setattr(api, "QUERY_LOG_PATH", "")
        api.append_query_log(RESULT, mode="corpus")  # must simply do nothing

    def test_failure_never_raises(self, monkeypatch):
        monkeypatch.setattr(api, "QUERY_LOG_PATH", "/proc/definitely/not/writable/x.jsonl")
        api.append_query_log(RESULT, mode="corpus")  # swallowed, logged

    def test_mock_result_never_raises(self, log_path):
        api.append_query_log(MagicMock(), mode="corpus")
