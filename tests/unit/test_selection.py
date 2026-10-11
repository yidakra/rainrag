"""Selector contracts without models, network clients or archive data."""

import json

import numpy as np
import pytest

from rainrag.selection import (
    FEATURE_NAMES,
    RetrievalAlternative,
    RidgeSelector,
    alternative_snapshot,
    configuration_signature,
    feature_matrix,
    fit_ridge,
    select_alternative,
)


def alternative(name, scores):
    """Build a ranked, text-bearing option for numeric and privacy checks."""
    return RetrievalAlternative(
        name,
        name,
        [
            {
                "doc_id": f"{name}-{i}",
                "score": score,
                "text": "private transcript",
                "score_type": "dense",
            }
            for i, score in enumerate(scores)
        ],
    )


def top_score_model(signature="fixture"):
    """Choose using a known coefficient, not an empirical quality assertion."""
    coefficients = np.zeros(len(FEATURE_NAMES))
    coefficients[0] = 1.0
    return RidgeSelector(
        np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), coefficients, signature
    )


@pytest.mark.parametrize("scores", [[], [0.9], [0.9, 0.8], [0, 0, 0], [-4, -5, -6]])
def test_features_support_empty_short_flat_and_negative_scores(scores):
    matrix = feature_matrix([alternative("original", scores)])
    assert matrix.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_features_use_score_order_but_overlap_uses_ranking_order():
    original = alternative("original", [0.4, 0.9, 0.2])
    rewrite = RetrievalAlternative("rewrite_1", "rewrite", [original.documents[0]])
    matrix = feature_matrix([original, rewrite])
    assert matrix[0, 0] == pytest.approx(0.9)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[0, -2] == pytest.approx(1 / 3)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_scores(score):
    with pytest.raises(ValueError, match="finite"):
        feature_matrix([alternative("original", [score])])


def test_rejects_duplicate_options_and_documents():
    original = alternative("original", [1])
    with pytest.raises(ValueError, match="unique IDs"):
        feature_matrix([original, original])
    original.documents.append(original.documents[0])
    with pytest.raises(ValueError, match="document IDs"):
        feature_matrix([original])


def test_selection_ignores_empty_options_and_keeps_original_on_ties():
    original = alternative("original", [0.9])
    choices = [original, alternative("rewrite_1", [0.9]), alternative("rewrite_2", [])]
    selected, trace = select_alternative(choices, top_score_model())
    assert selected.id == "original"
    assert trace["predicted_gain"] == 0


def test_minimum_gain_and_nonempty_recovery():
    choices = [alternative("original", [0.8]), alternative("rewrite_1", [0.9])]
    assert select_alternative(choices, top_score_model())[0].id == "rewrite_1"
    assert select_alternative(choices, top_score_model(), min_gain=0.2)[0].id == "original"
    choices[0] = alternative("original", [])
    assert select_alternative(choices, top_score_model(), min_gain=1.0)[0].id == "rewrite_1"


def test_all_empty_candidates_fall_back_to_original():
    choices = [alternative("original", []), alternative("rewrite_1", [])]
    assert select_alternative(choices, top_score_model())[0].id == "original"


def test_snapshot_does_not_export_candidate_transcripts():
    snapshot = alternative_snapshot([alternative("original", [0.9])])
    assert "private transcript" not in json.dumps(snapshot)
    assert snapshot[0]["documents"][0]["score_type"] == "dense"


def test_fit_centres_within_query_and_does_not_mutate_inputs():
    choices = [alternative("original", [0.2]), alternative("rewrite_1", [0.8])]
    features = feature_matrix(choices)
    untouched = features.copy()
    model = fit_ridge([features, features + 100], [np.array([0, 1]), np.array([3, 4])], "fixture")
    assert np.array_equal(features, untouched)
    assert select_alternative(choices, model)[0].id == "rewrite_1"


def test_artifact_roundtrip_and_configuration_binding(tmp_path):
    path = tmp_path / "model.json"
    top_score_model().save(path)
    model = RidgeSelector.load(path, "fixture")
    assert np.array_equal(model.coefficients, top_score_model().coefficients)
    with pytest.raises(ValueError, match="configuration"):
        RidgeSelector.load(path, "changed-archive")


@pytest.mark.parametrize(
    "field,value",
    [
        ("feature_names", ["wrong"]),
        ("schema_version", 999),
        ("scale", [0] * len(FEATURE_NAMES)),
        ("coefficients", [float("nan")] * len(FEATURE_NAMES)),
    ],
)
def test_invalid_artifacts_are_rejected(tmp_path, field, value):
    path = tmp_path / "model.json"
    top_score_model().save(path)
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        RidgeSelector.load(path, "fixture")


def test_configuration_signature_ignores_secrets_but_binds_corpus(test_config):
    initial = configuration_signature(test_config)
    test_config.openai.api_key = "changed-secret"
    assert configuration_signature(test_config) == initial
    test_config.adaptive_retrieval.corpus_revision = "new-index"
    assert configuration_signature(test_config) != initial


@pytest.mark.parametrize("alpha", [0, -1, float("nan"), float("inf")])
def test_fit_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit_ridge([np.zeros((2, 3))], [np.zeros(2)], "fixture", alpha)
