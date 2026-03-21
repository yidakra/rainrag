"""MLflow tracking helpers for the RainRAG eval suite.

MLflow is an optional dependency. All functions degrade gracefully to no-ops
when it is not installed, so experiments still run and print results locally.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast


try:
    import mlflow as _mlflow
except ImportError:
    _mlflow = None

# MLflow may be unavailable in some environments and ships limited type hints.
# Treat it as a dynamic dependency to avoid strict "unknown member" diagnostics.
mlflow: Any | None = cast(Any | None, _mlflow)

_MLFLOW_AVAILABLE = mlflow is not None


def _sanitize_metric_name(name: str) -> str:
    """Return an MLflow-safe metric name.

    MLflow metric names allow alphanumerics, underscore, dash, period,
    spaces, colon and slash.  RainRAG metrics historically include `@`
    (e.g. ``recall@5``), so we map that to ``_at_`` and replace any
    remaining invalid characters with ``_``.
    """
    mapped = name.strip().replace("@", "_at_")
    safe = re.sub(r"[^A-Za-z0-9_\-\. /:]", "_", mapped)
    safe = re.sub(r"_+", "_", safe).strip(" _")
    return safe or "metric"


def is_available() -> bool:
    return _MLFLOW_AVAILABLE


def default_tracking_uri() -> str:
    """Return the default MLflow tracking URI for RainRAG evals.

    Resolution order:
    1) ``RAINRAG_MLFLOW_URI`` (project-specific override)
    2) ``MLFLOW_TRACKING_URI`` (MLflow standard override)
    3) ``$XDG_STATE_HOME/rainrag/mlruns`` (or ``~/.local/state/rainrag/mlruns``)

    Using a user-state directory avoids polluting the git working tree.
    """
    env_override = os.getenv("RAINRAG_MLFLOW_URI") or os.getenv("MLFLOW_TRACKING_URI")
    if env_override:
        return env_override

    xdg_state_home = Path(os.getenv("XDG_STATE_HOME") or str(Path.home() / ".local" / "state"))
    state_dir = xdg_state_home / "rainrag" / "mlruns"

    # Only create the directory when MLflow is actually available.
    # This avoids creating state directories in environments where MLflow is
    # not installed but the helper is still imported.
    if _MLFLOW_AVAILABLE:
        state_dir.mkdir(parents=True, exist_ok=True)

    return str(state_dir)


def setup(tracking_uri: str | None = None, experiment_name: str = "rainrag_eval") -> None:
    """Configure MLflow tracking URI and active experiment."""
    if not _MLFLOW_AVAILABLE:
        return
    uri = tracking_uri or default_tracking_uri()
    # mypy/pyright can't see the guard above, so make the contract explicit
    assert mlflow is not None
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


@contextlib.contextmanager
def start_run(
    run_name: str,
    tags: dict[str, str] | None = None,
) -> Iterator[Any]:
    """Context manager that wraps mlflow.start_run, or is a no-op if unavailable."""
    if _MLFLOW_AVAILABLE:
        # mypy/pyright can't see the guard above, so make the contract explicit
        assert mlflow is not None
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    else:
        yield None


def log_params(params: dict[str, Any]) -> None:
    """Log a flat dict of params. Filters NaN/None/Infinity values."""
    if not _MLFLOW_AVAILABLE:
        return
    assert mlflow is not None
    clean = {
        k: v
        for k, v in params.items()
        if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    }
    try:
        mlflow.log_params(clean)
    except Exception as exc:
        _logger.warning("Failed to log params to MLflow: %s", exc, exc_info=True)


# module-level logger for warnings
_logger = logging.getLogger(__name__)


def log_metrics(metrics: dict[str, float | int | None], step: int | None = None) -> None:
    """Log a flat dict of metrics.

    Skips NaN and None values and casts integer metrics to float so that the
    dict passed to ``mlflow.log_metrics`` has the required ``float`` value
    type.

    Metric name collisions are resolved by appending a suffix ("_2", "_3",
    etc.) to the sanitized name.  Previously this happened silently which made
    debugging confusing; we now emit a warning when a collision occurs so that
    the caller can trace which original key led to the renamed metric.
    """
    if not _MLFLOW_AVAILABLE:
        return
    assert mlflow is not None
    # filter out null/NaN/Infinity, sanitize metric names, and make all values floats
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            continue

        base = _sanitize_metric_name(key)
        safe_key = base
        suffix = 2
        while safe_key in clean:
            safe_key = f"{base}_{suffix}"
            suffix += 1

        if safe_key != base:
            _logger.warning(
                "metric name collision: original=%r sanitized=%r renamed=%r",
                key,
                base,
                safe_key,
            )

        clean[safe_key] = float(value)

    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:
        _logger.warning("Failed to log metrics to MLflow: %s", exc, exc_info=True)


def log_dict_as_artifact(data: Any, filename: str) -> None:
    """Serialise *data* to JSON and log it as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    assert mlflow is not None

    if Path(filename).is_absolute():
        raise ValueError("`filename` must be a relative path")

    safe_name = Path(filename).name
    if not safe_name or safe_name in {"", ".", ".."}:
        raise ValueError("`filename` must include a valid basename")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / safe_name
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:
                _logger.warning(
                    "Failed to log artifact %r to MLflow: %s",
                    safe_name,
                    exc,
                    exc_info=True,
                )
    except (TypeError, ValueError) as exc:
        _logger.warning(
            "Failed to serialize data in log_dict_as_artifact(%r): %s",
            filename,
            exc,
        )
        return


def log_jsonl_as_artifact(rows: list[Any], filename: str) -> None:
    """Serialise a list of dicts as JSONL and log as an MLflow artifact."""
    if not _MLFLOW_AVAILABLE:
        return
    assert mlflow is not None

    if Path(filename).is_absolute():
        raise ValueError("`filename` must be a relative path")

    safe_name = Path(filename).name
    if not safe_name or safe_name in {"", ".", ".."}:
        raise ValueError("`filename` must include a valid basename")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / safe_name
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:
                _logger.warning(
                    "Failed to log artifact %r to MLflow: %s",
                    safe_name,
                    exc,
                    exc_info=True,
                )
    except (TypeError, ValueError) as exc:
        _logger.warning(
            "Failed to serialize rows in log_jsonl_as_artifact(%r): %s",
            filename,
            exc,
        )
        return


def log_config_snapshot(config: Any, filename: str = "config_snapshot.yaml") -> None:
    """Dump the full Pydantic config to YAML and log as an MLflow artifact.

    The function attempts to serialize *config* in a sensible way:
    - call ``model_dump()`` (Pydantic v2)
    - fall back to ``dict()`` (Pydantic v1)
    - if those methods are missing, use the config object directly if it's a
      ``dict`` or otherwise usable by ``yaml.dump``.
    Any ``AttributeError`` from missing methods is caught so the script never
    crashes when the caller passes a plain dict or v1 model.
    """
    if not _MLFLOW_AVAILABLE:
        return
    assert mlflow is not None
    try:
        import yaml
    except ImportError as exc:
        _logger.warning("YAML package not installed, config snapshot will not be logged: %s", exc)
        return
    # prepare serializable data
    data: Any
    if isinstance(config, dict):
        data = cast(dict[str, Any], config)
    else:
        try:
            data = config.model_dump()
        except AttributeError:
            try:
                data = config.dict()
            except AttributeError:
                # last resort: try to use __dict__ or just the object itself
                data = getattr(config, "__dict__", config)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
            mlflow.log_artifact(str(path))
        except Exception as exc:
            _logger.exception(
                "Failed to log config snapshot: %s",
                exc,
                exc_info=True,
            )


def get_run_url() -> str | None:
    """Return the MLflow UI URL for the active run, if available."""
    if not _MLFLOW_AVAILABLE:
        return None
    assert mlflow is not None
    try:
        run = mlflow.active_run()
        if run is None:
            return None
        uri = str(mlflow.get_tracking_uri())
        # File-based tracking has no web UI
        if uri.startswith("file://") or not uri.startswith(("http://", "https://")):
            return None
        uri = uri.rstrip("/")
        return f"{uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
    except Exception:
        return None
