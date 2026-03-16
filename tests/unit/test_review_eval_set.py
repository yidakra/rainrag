"""Tests for the interactive review tool.

These tests run entirely locally using a temporary JSONL file and a mocked
`input()` to simulate a reviewer session.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

import pytest

from eval.datasets.review_eval_set import review_eval_set


def test_review_eval_set_editing_deleted_record_does_not_decrement_below_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing a previously deleted record must not make the deleted counter go negative."""

    record = {
        "query_id": "q1",
        "reference_answer": "old",
        "valid": False,
        # mark not reviewed so the record is in the pending list
        "reviewed": False,
    }

    input_path = tmp_path / "eval.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Simulate: edit the record, supply a new answer, finish edit.
    inputs = iter(["e", "new answer", ""])

    def _fake_input(prompt: str = "") -> str:
        return next(inputs)

    monkeypatch.setattr(builtins, "input", _fake_input)

    summary = review_eval_set(str(input_path))

    assert summary["accepted"] == 1
    assert summary["deleted"] == 0
    assert summary["skipped"] == 0

    # Verify the record on disk was updated and marked valid.
    with open(input_path, encoding="utf-8") as f:
        updated = json.loads(f.read().strip())

    assert updated["valid"] is True
    assert updated["reference_answer"] == "new answer"
