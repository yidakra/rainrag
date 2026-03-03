"""Unit tests for eval.mlflow_tracking helpers."""

from __future__ import annotations

import eval.mlflow_tracking as mlflow_tracking


class _FakeMlflow:
    def __init__(self) -> None:
        self.logged: dict[str, float] | None = None
        self.step: int | None = None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged = metrics
        self.step = step


class TestMlflowTracking:
    def test_sanitize_metric_name_maps_at_symbol(self) -> None:
        assert mlflow_tracking._sanitize_metric_name("recall@5") == "recall_at_5"

    def test_log_metrics_sanitizes_invalid_names(self, monkeypatch) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", True)

        mlflow_tracking.log_metrics(
            {
                "recall@5": 0.8,
                "ndcg@5_p10": 0.4,
                "metric*bad#chars": 1,
            },
            step=7,
        )

        assert fake.logged is not None
        assert fake.logged["recall_at_5"] == 0.8
        assert fake.logged["ndcg_at_5_p10"] == 0.4
        assert fake.logged["metric_bad_chars"] == 1.0
        assert fake.step == 7

    def test_log_metrics_skips_none_and_nan(self, monkeypatch) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", True)

        mlflow_tracking.log_metrics({"ok@1": 1.0, "none": None, "nan": float("nan")})

        assert fake.logged == {"ok_at_1": 1.0}
