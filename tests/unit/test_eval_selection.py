"""Grouped, cached-ranking evaluation with no provider or corpus-service calls."""

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from rainrag.selection import RetrievalAlternative, RidgeSelector
from rainrag_eval.selection import (
    CorpusStatistics,
    JudgedCase,
    app,
    evaluate,
    grouped_folds,
    load_cases,
    signflip_p,
)


def records():
    """Eight information needs in four independent groups; not archive effectiveness data."""
    result = []
    for i in range(8):
        good = i % 2 == 0
        result.append(
            {
                "query_id": f"q{i}",
                "group_id": f"g{i // 2}",
                "relevant_doc_ids": [f"relevant{i}"],
                "retrieval_selection": {
                    "stage": "first_stage_before_merge",
                    "configuration_signature": "fixture",
                    "corpus_revision": "fixture-v1",
                    "alternatives": [
                        {
                            "id": "original",
                            "query": "original",
                            "documents": [
                                {
                                    "doc_id": f"relevant{i}" if good else f"wrong{i}",
                                    "score": 0.9 if good else 0.1,
                                },
                            ],
                        },
                        {
                            "id": "rewrite_1",
                            "query": "rare rewrite",
                            "documents": [
                                {
                                    "doc_id": f"wrong{i}" if good else f"relevant{i}",
                                    "score": 0.1 if good else 0.9,
                                },
                            ],
                        },
                    ],
                },
            }
        )
    return result


def write_dataset(tmp_path, rows=None):
    """Write only synthetic fixtures inside the pytest temporary directory."""
    path = tmp_path / "judged.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in (rows if rows is not None else records())) + "\n"
    )
    return path


def test_folds_never_split_a_group(tmp_path):
    cases = load_cases(write_dataset(tmp_path))
    for train, test in grouped_folds(cases, 4):
        assert {cases[i].group for i in train}.isdisjoint({cases[i].group for i in test})
    with pytest.raises(ValueError, match="group"):
        grouped_folds(cases, 5)


def test_matched_learner_evaluation_and_training_fixed_baseline(tmp_path):
    cases = load_cases(write_dataset(tmp_path))
    result = evaluate(cases, folds=4, corpus=CorpusStatistics(["original original", "rewrite"]))
    assert result["post_retrieval"] == 1.0
    assert result["original"] == 0.5
    assert result["training_chosen_fixed"] == 0.5
    assert "pre_retrieval" in result
    assert result["stage"] == "first_stage_before_merge"
    assert result["post_holm_p_vs_fixed"] >= result["post_p_vs_fixed"]
    assert "cost savings" in result["caveat"]


def test_group_signflip_weights_queries_not_groups():
    cases = [JudgedCase(str(i), "a" if i < 3 else "b", frozenset(), [], "s") for i in range(4)]
    assert signflip_p(np.array([1, 1, 1, -1]), cases) < 0.6
    assert signflip_p(np.zeros(4), cases) == 1.0


def test_corpus_statistics_do_not_consult_candidate_scores():
    corpus = CorpusStatistics(["common", "common", "rare"])
    options = [
        RetrievalAlternative("original", "common", []),
        RetrievalAlternative("rewrite_1", "rare", []),
    ]
    matrix = corpus.matrix(options)
    assert matrix[1, 3] > matrix[0, 3]


@pytest.mark.parametrize(
    "change", ["group", "original", "configuration", "revision", "duplicates", "labels", "scores"]
)
def test_bad_datasets_are_rejected(tmp_path, change):
    rows = records()
    if change == "group":
        del rows[0]["group_id"]
    elif change == "original":
        rows[0]["retrieval_selection"]["alternatives"][0]["id"] = "missing"
    elif change == "configuration":
        rows[0]["retrieval_selection"]["configuration_signature"] = "other"
    elif change == "revision":
        rows[0]["retrieval_selection"]["corpus_revision"] = ""
    elif change == "duplicates":
        rows[1]["query_id"] = rows[0]["query_id"]
    elif change == "labels":
        rows[0]["relevant_doc_ids"] = []
    else:
        rows[0]["retrieval_selection"]["alternatives"][0]["documents"][0]["score"] = float("nan")
    with pytest.raises(ValueError):
        load_cases(write_dataset(tmp_path, rows))


def test_cli_writes_evaluation_and_portable_model_without_providers(tmp_path):
    dataset = write_dataset(tmp_path)
    report = tmp_path / "report.json"
    model = tmp_path / "model.json"
    result = CliRunner().invoke(
        app,
        ["--dataset", str(dataset), "--report", str(report), "--model", str(model), "--folds", "4"],
    )
    assert result.exit_code == 0, result.output
    assert "single-positive" in result.output
    assert json.loads(report.read_text())["post_retrieval"] == 1.0
    RidgeSelector.load(model, "fixture")
    again = CliRunner().invoke(app, ["--dataset", str(dataset), "--report", str(report)])
    assert again.exit_code != 0
    assert "overwrite" in again.output


def test_cli_rejects_shared_output_paths(tmp_path):
    dataset = write_dataset(tmp_path)
    target = tmp_path / "both.json"
    result = CliRunner().invoke(
        app, ["--dataset", str(dataset), "--report", str(target), "--model", str(target)]
    )
    assert result.exit_code != 0
    assert not target.exists()


def test_relevant_documents_cannot_cross_groups(tmp_path):
    rows = records()
    rows[2]["relevant_doc_ids"] = rows[0]["relevant_doc_ids"]
    with pytest.raises(ValueError, match="one group"):
        load_cases(write_dataset(tmp_path, rows))


def test_relevant_video_translations_cannot_cross_groups(tmp_path):
    rows = records()
    for i in (0, 2):
        rows[i]["retrieval_selection"]["alternatives"][0]["documents"][0]["video_id"] = "same-video"
    with pytest.raises(ValueError, match="one group"):
        load_cases(write_dataset(tmp_path, rows))


def test_corpus_revisions_cannot_be_mixed(tmp_path):
    rows = records()
    rows[0]["retrieval_selection"]["corpus_revision"] = "other"
    with pytest.raises(ValueError, match="corpus revisions"):
        load_cases(write_dataset(tmp_path, rows))
