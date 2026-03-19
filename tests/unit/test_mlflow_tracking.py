"""Unit tests for eval.mlflow_tracking helpers."""

from __future__ import annotations

from collections import OrderedDict

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

    def test_log_metrics_no_mlflow_does_not_call_mlflow(self, monkeypatch) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", False)

        # Should be a no-op when MLflow is unavailable.
        mlflow_tracking.log_metrics({"ok@1": 1.0}, step=1)

        assert fake.logged is None
        assert fake.step is None

    def test_log_metrics_skips_none_and_nan(self, monkeypatch) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", True)

        mlflow_tracking.log_metrics({"ok@1": 1.0, "none": None, "nan": float("nan")})

        assert fake.logged == {"ok_at_1": 1.0}
        assert fake.step is None

    def test_log_metrics_empty_metrics_logs_nothing(self, monkeypatch) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", True)

        mlflow_tracking.log_metrics({})

        assert fake.logged == {}
        assert fake.step is None

    def test_log_metrics_skips_inf_values(self, monkeypatch) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", True)

        mlflow_tracking.log_metrics(
            {
                "ok@1": 1.0,
                "inf": float("inf"),
                "ninf": float("-inf"),
                "nan": float("nan"),
            },
            step=15,
        )

        assert fake.logged == {"ok_at_1": 1.0}
        assert fake.step == 15

    def test_log_metrics_collision_warns_once_with_final_key(self, monkeypatch, caplog) -> None:
        fake = _FakeMlflow()
        monkeypatch.setattr(mlflow_tracking, "mlflow", fake)
        monkeypatch.setattr(mlflow_tracking, "_MLFLOW_AVAILABLE", True)

        with caplog.at_level("WARNING"):
            # This test relies on deterministic insertion order for collision handling;
            # Python dict preserves insertion order, but we make it explicit here.
            mlflow_tracking.log_metrics(
                OrderedDict(
                    [
                        ("a#b", 1.0),
                        ("a!b", 2.0),
                        ("a?b", 3.0),
                    ]
                )
            )

        assert fake.logged == {"a_b": 1.0, "a_b_2": 2.0, "a_b_3": 3.0}
        assert len(caplog.records) == 2
        assert "metric name collision" in caplog.records[0].message
        assert "original='a!b'" in caplog.records[0].message
        assert "sanitized='a_b'" in caplog.records[0].message
        assert "renamed='a_b_2'" in caplog.records[0].message
        assert "original='a?b'" in caplog.records[1].message
        assert "renamed='a_b_3'" in caplog.records[1].message
