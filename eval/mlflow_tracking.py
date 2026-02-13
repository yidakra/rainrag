"""MLflow tracking helpers for the RainRAG eval suite.

MLflow is an optional dependency. All functions degrade gracefully to no-ops
when it is not installed, so experiments still run and print results locally.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def is_available() -> bool:
    return _MLFLOW_AVAILABLE


def setup(tracking_uri: str = "./mlruns", experiment_name: str = "rainrag_eval") -> None:
    """Configure MLflow tracking URI and active experiment."""
    if not _MLFLOW_AVAILABLE:
        return
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextlib.contextmanager
def start_run(
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Context manager that wraps mlflow.start_run, or is a no-op if unavailable."""
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    else:
        yield None


def log_params(params: dict[str, Any]) -> None:
    """Log a flat dict of params. Filters NaN/None values."""
    if not _MLFLOW_AVAILABLE:
        return
    clean = {
        k: v
        for k, v in params.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }
    mlflow.log_params(clean)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a flat dict of metrics. Skips NaN values."""
    if not _MLFLOW_AVAILABLE:
        return
    clean = {
        k: v
        for k, v in metrics.items()
        if not (isinstance(v, float) and math.isnan(v))
    }
    mlflow.log_metrics(clean, step=step)


def log_dict_as_artifact(data: Any, filename: str) -> None:
    """Serialise *data* to JSON and log it as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(path)


def log_jsonl_as_artifact(rows: list[Any], filename: str) -> None:
    """Serialise a list of dicts as JSONL and log as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        mlflow.log_artifact(path)


def log_config_snapshot(config: Any, filename: str = "config_snapshot.yaml") -> None:
    """Dump the full Pydantic config to YAML and log as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, filename)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, allow_unicode=True, default_flow_style=False)
        mlflow.log_artifact(path)


def get_run_url() -> str | None:
    """Return the MLflow UI URL for the active run, if available."""
    if not _MLFLOW_AVAILABLE:
        return None
    try:
        run = mlflow.active_run()
        if run is None:
            return None
        uri = mlflow.get_tracking_uri()
        return f"{uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    except Exception:
        return None
