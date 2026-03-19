"""Results dashboard — reads MLflow runs and produces comparison charts.

Generates four publication-ready plots from completed MLflow experiment runs:

  1. **Recall & NDCG bar chart** – recall@5 and ndcg@5 grouped by condition/label
  2. **Robustness bar chart** – mean vs worst-decile (p10) for recall@5 and ndcg@5,
     revealing conditions that improve the average while hurting hard queries
  3. **Latency breakdown stacked bar** – p50 latency per pipeline stage
  4. **Cost vs quality scatter** – total USD/query on x-axis, recall@5 on y-axis

Charts are saved as PNG files (and optionally displayed interactively).

Requires
--------
    pip install matplotlib mlflow pandas

Usage
-----
    # From the project root:
    python -m eval.plot_results --experiment ablation --output plots/

    # Two-stage sweep with interactive display:
    python -m eval.plot_results --experiment two_stage_sweep --show

    # Multiple experiments overlaid on the scatter plot:
    python -m eval.plot_results \\
        --experiment ablation \\
        --experiment two_stage_sweep \\
        --output plots/combined/

CLI reference
-------------
    --mlflow-uri   MLflow tracking URI (default: user state dir)
    --experiment   Experiment name to load (repeatable for multi-experiment scatter)
    --filter-axis  Only include runs whose sweep_axis tag equals this value
    --top-k        Retrieval depth filter; 0 = all (default: 5)
    --output       Directory for PNG output (default: plots/)
    --show         Display charts interactively with matplotlib
    --dpi          Output PNG resolution (default: 150)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, cast

import typer as _typer

# relative import so the module can be resolved when this file is run as
# `python -m eval.plot_results`
from .mlflow_tracking import (
    default_tracking_uri as _default_tracking_uri,
)


typer: Any = cast(Any, _typer)
default_tracking_uri: Callable[[], str] = _default_tracking_uri


app = typer.Typer(
    name="plot-results",
    help="Generate comparison charts from MLflow eval runs.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_runs(
    mlflow_uri: str,
    experiment_names: list[str],
    top_k_filter: int,
    sweep_axis_filter: str | None,
) -> Any:
    """Load MLflow runs for the given experiments, returning a pandas DataFrame."""
    try:
        import mlflow as _mlflow
        import pandas as _pd
    except ImportError as exc:
        raise SystemExit(f"ERROR: {exc}. Install with: pip install mlflow pandas") from exc

    mlflow: Any = cast(Any, _mlflow)
    pd: Any = cast(Any, _pd)

    mlflow.set_tracking_uri(mlflow_uri)
    frames: list[Any] = []
    for name in experiment_names:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            typer.echo(f"[warn] Experiment '{name}' not found in {mlflow_uri}", err=True)
            continue
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
        df["_experiment"] = name
        frames.append(df)

    if not frames:
        raise SystemExit("No runs found.  Check --mlflow-uri and --experiment names.")

    runs = pd.concat(frames, ignore_index=True)

    # Filter by top_k parameter
    if top_k_filter > 0 and "params.top_k" in runs.columns:
        runs = runs[runs["params.top_k"].astype(str) == str(top_k_filter)]

    # Filter by sweep axis tag
    if sweep_axis_filter and "tags.sweep_axis" in runs.columns:
        runs = runs[runs["tags.sweep_axis"] == sweep_axis_filter]

    if runs.empty:
        raise SystemExit("No runs match the filters.  Adjust --top-k or --filter-axis.")

    return runs


def load_runs(
    mlflow_uri: str,
    experiment_names: list[str],
    top_k_filter: int,
    sweep_axis_filter: str | None,
) -> Any:
    """Public wrapper around `_load_runs`.

    Maintains backwards compatibility while allowing callers to use a public API.
    """
    return _load_runs(
        mlflow_uri=mlflow_uri,
        experiment_names=experiment_names,
        top_k_filter=top_k_filter,
        sweep_axis_filter=sweep_axis_filter,
    )


def _condition_label(row: Any) -> str:
    """Best human-readable label for a run row."""
    for col in ("params.condition_label", "tags.mlflow.runName", "run_id"):
        val = row.get(col)
        if val and not (isinstance(val, float) and math.isnan(val)):
            return str(val)
    return "unknown"


def _safe_float(val: Any) -> float | None:
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _metric_col_variants(name: str) -> tuple[str, str]:
    """Return candidate MLflow DataFrame column names for a metric key.

    Supports both legacy keys (e.g. ``metrics.recall@5``) and sanitized keys
    logged via mlflow_tracking (e.g. ``metrics.recall_at_5``).
    """
    return (f"metrics.{name}", f"metrics.{name.replace('@', '_at_')}")


def _metric_from_row(row: Any, name: str) -> float | None:
    """Read a metric value from a run row using compatible column variants."""
    for col in _metric_col_variants(name):
        value = _safe_float(row.get(col))
        if value is not None:
            return value
    return None


def _cost_from_row(row: Any) -> float | None:
    """Read estimated per-query cost from a run row.

    Supports the current key ``metrics.cost.mean_usd_est_per_query`` and a
    fallback to ``metrics.cost.aggregate_usd_est`` if per-query mean is missing.
    """
    for col in (
        "metrics.cost.mean_usd_est_per_query",
        "metrics.cost.aggregate_usd_est",
        "metrics.cost.total_mean_usd_est_per_query",
        "metrics.cost.total_usd_est_per_query",
        "metrics.cost_total_usd_est_per_query",
    ):
        value = _safe_float(row.get(col))
        if value is not None:
            return value
    return None


# ---------------------------------------------------------------------------
# Chart 1: Recall & NDCG bar chart
# ---------------------------------------------------------------------------


def plot_retrieval_bars(runs: Any, output_dir: Path, show: bool, dpi: int) -> None:
    """Grouped bar chart: recall@5 and ndcg@5 per condition."""
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ImportError as exc:
        typer.echo(f"[warn] Skipping retrieval bar chart: {exc}", err=True)
        return

    plt: Any = cast(Any, _plt)
    np: Any = cast(Any, _np)

    labels: list[str] = []
    recall5: list[float] = []
    ndcg5: list[float] = []
    for _, row in runs.iterrows():
        lbl = _condition_label(row)
        r5 = _metric_from_row(row, "recall@5")
        n5 = _metric_from_row(row, "ndcg@5")
        # Only include runs that have both recall and ndcg available.
        # This avoids pretending missing metrics are actual zeros.
        if r5 is not None and n5 is not None:
            labels.append(lbl)
            recall5.append(r5)
            ndcg5.append(n5)

    if not labels:
        typer.echo("[warn] No recall@5 / ndcg@5 metrics found — skipping bar chart.", err=True)
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    ax.bar(x - width / 2, recall5, width, label="recall@5", color="#4C72B0")
    ax.bar(x + width / 2, ndcg5, width, label="ndcg@5", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Quality by Condition")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "retrieval_bars.png"
    fig.savefig(out, dpi=dpi)
    typer.echo(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Latency breakdown stacked bar
# ---------------------------------------------------------------------------


def plot_latency_breakdown(runs: Any, output_dir: Path, show: bool, dpi: int) -> None:
    """Stacked bar chart: p50 latency broken down by pipeline stage."""
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ImportError as exc:
        typer.echo(f"[warn] Skipping latency chart: {exc}", err=True)
        return

    plt: Any = cast(Any, _plt)
    np: Any = cast(Any, _np)

    stages = ["embed", "retrieve", "rerank", "generate"]
    stage_cols = [f"metrics.{s}_p50_ms" for s in stages]

    # Only use runs that have at least one latency stage column
    has_latency = any(col in runs.columns for col in stage_cols)
    # Fallback to total latency from base experiment if stage breakdowns absent
    has_total = "metrics.latency_p50_ms" in runs.columns

    if not has_latency and not has_total:
        typer.echo("[warn] No latency metrics found — skipping latency chart.", err=True)
        return

    labels: list[str] = []
    stage_values: dict[str, list[float]] = {s: [] for s in stages}

    for _, row in runs.iterrows():
        lbl = _condition_label(row)
        row_stages = {s: _safe_float(row.get(f"metrics.{s}_p50_ms")) or 0.0 for s in stages}
        row_total = sum(row_stages.values())
        if row_total == 0.0:
            # Fallback: use total p50 as a single block
            total = _safe_float(row.get("metrics.latency_p50_ms")) or 0.0
            if total == 0.0:
                continue
            row_stages = {s: (total if s == "generate" else 0.0) for s in stages}
        labels.append(lbl)
        for s in stages:
            stage_values[s].append(row_stages[s])

    if not labels:
        typer.echo("[warn] No valid latency data to plot.", err=True)
        return

    x = np.arange(len(labels))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5))
    bottom = np.zeros(len(labels))
    for s, color in zip(stages, colors, strict=False):
        vals = np.array(stage_values[s])
        ax.bar(x, vals, bottom=bottom, label=s, color=color)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Latency p50 (ms)")
    ax.set_title("Per-Stage Latency Breakdown (p50)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "latency_breakdown.png"
    fig.savefig(out, dpi=dpi)
    typer.echo(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 3: Robustness bar chart (mean vs p10 percentile)
# ---------------------------------------------------------------------------


def plot_robustness_bars(runs: Any, output_dir: Path, show: bool, dpi: int) -> None:
    """Side-by-side bars: mean vs worst-decile (p10) for recall@5 and ndcg@5.

    Exposes whether a condition improves average performance while leaving hard
    queries behind — the pattern that VRisk-style analysis targets.  A condition
    is strictly better only if *both* the mean and the p10 improve.
    """
    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
    except ImportError as exc:
        typer.echo(f"[warn] Skipping robustness chart: {exc}", err=True)
        return

    plt: Any = cast(Any, _plt)
    np: Any = cast(Any, _np)

    metrics = [
        ("recall@5", "recall@5_p10", "Recall@5"),
        ("ndcg@5", "ndcg@5_p10", "NDCG@5"),
    ]

    labels: list[str] = []
    data: dict[str, list[float]] = {key: [] for pair in metrics for key in (pair[0], pair[1])}

    for _, row in runs.iterrows():
        lbl = _condition_label(row)
        mean_r5 = _metric_from_row(row, "recall@5")
        if mean_r5 is None:
            continue
        # Skip rows where p10 is missing to keep mean/p10 alignment correct.
        row_missing_p10 = False
        for _, p10_col, _ in metrics:
            if _metric_from_row(row, p10_col) is None:
                row_missing_p10 = True
                break
        if row_missing_p10:
            continue

        labels.append(lbl)
        for mean_col, p10_col, _ in metrics:
            data[mean_col].append(_metric_from_row(row, mean_col) or 0.0)
            data[p10_col].append(_metric_from_row(row, p10_col) or 0.0)

    if not labels:
        typer.echo("[warn] No recall@5 metrics found — skipping robustness chart.", err=True)
        return

    n = len(labels)
    x = np.arange(n)
    # 4 bars per condition: recall mean, recall p10, ndcg mean, ndcg p10
    slot_w = 0.20
    offsets = [-1.5 * slot_w, -0.5 * slot_w, 0.5 * slot_w, 1.5 * slot_w]
    colors = ["#4C72B0", "#9DB8D2", "#DD8452", "#F0C89A"]
    bar_labels = ["recall@5 (mean)", "recall@5 (p10)", "ndcg@5 (mean)", "ndcg@5 (p10)"]
    bar_data = [
        data["recall@5"],
        data["recall@5_p10"],
        data["ndcg@5"],
        data["ndcg@5_p10"],
    ]

    fig, ax = plt.subplots(figsize=(max(9, n * 1.2), 5))
    for offset, vals, color, blabel in zip(offsets, bar_data, colors, bar_labels, strict=False):
        ax.bar(x + offset, vals, slot_w, label=blabel, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Robustness: Mean vs Worst-Decile (p10) by Condition")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "robustness_bars.png"
    fig.savefig(out, dpi=dpi)
    typer.echo(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Cost vs quality scatter
# ---------------------------------------------------------------------------


def plot_cost_vs_quality(runs: Any, output_dir: Path, show: bool, dpi: int) -> None:
    """Scatter plot: total USD/query (x) vs recall@5 (y), coloured by experiment."""
    try:
        import matplotlib.pyplot as _plt
    except ImportError as exc:
        typer.echo(f"[warn] Skipping scatter chart: {exc}", err=True)
        return

    plt: Any = cast(Any, _plt)

    experiment_names = runs["_experiment"].unique() if "_experiment" in runs.columns else ["runs"]
    cmap = plt.get_cmap("tab10")
    color_map = {name: cmap(i) for i, name in enumerate(experiment_names)}

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0

    for _, row in runs.iterrows():
        cost = _cost_from_row(row)
        recall = _metric_from_row(row, "recall@5")
        if cost is None or recall is None:
            continue
        lbl = _condition_label(row)
        exp_name = row.get("_experiment", "runs")
        color = color_map.get(exp_name, "steelblue")
        ax.scatter(cost, recall, color=color, s=80, zorder=3)
        ax.annotate(
            lbl,
            (cost, recall),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            alpha=0.85,
        )
        plotted += 1

    if plotted == 0:
        typer.echo("[warn] No cost + recall@5 pairs found — skipping scatter chart.", err=True)
        plt.close(fig)
        return

    # Legend for experiments
    for name, color in color_map.items():
        ax.scatter([], [], color=color, label=name, s=80)

    ax.set_xlabel("Estimated cost (USD / query)")
    ax.set_ylabel("recall@5")
    ax.set_title("Cost vs Retrieval Quality")
    ax.legend(title="Experiment", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = output_dir / "cost_vs_quality.png"
    fig.savefig(out, dpi=dpi)
    typer.echo(f"Saved: {out}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    mlflow_uri: Annotated[
        str | None, typer.Option("--mlflow-uri", help="MLflow tracking URI")
    ] = None,
    experiment: Annotated[
        list[str] | None, typer.Option("--experiment", "-e", help="Experiment name (repeatable)")
    ] = None,
    filter_axis: Annotated[
        str | None,
        typer.Option("--filter-axis", help="Only include runs with this sweep_axis tag"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Retrieval depth filter; 0 = all")] = 5,
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output directory for PNGs")
    ] = "plots",
    show: Annotated[
        bool, typer.Option("--show/--no-show", help="Display charts interactively")
    ] = False,
    dpi: Annotated[int, typer.Option("--dpi", help="PNG resolution")] = 150,
) -> None:
    """Generate comparison charts from MLflow eval runs.

    Examples::

        # Ablation overview
        python -m eval.plot_results -e ablation --output plots/

        # Two-stage sweep, only HyDE-alpha axis
        python -m eval.plot_results -e two_stage_sweep --filter-axis hyde_alpha

        # Combine ablation + sweep on one scatter
        python -m eval.plot_results -e ablation -e two_stage_sweep --top-k 5
    """
    if experiment is None:
        experiment = ["ablation"]
    resolved_mlflow_uri = mlflow_uri or default_tracking_uri()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading runs from: {resolved_mlflow_uri}")
    runs = load_runs(
        mlflow_uri=resolved_mlflow_uri,
        experiment_names=list(experiment),
        top_k_filter=top_k,
        sweep_axis_filter=filter_axis,
    )
    typer.echo(f"Loaded {len(runs)} run(s) across {list(experiment)}")

    plot_retrieval_bars(runs, output_dir, show=show, dpi=dpi)
    plot_robustness_bars(runs, output_dir, show=show, dpi=dpi)
    plot_latency_breakdown(runs, output_dir, show=show, dpi=dpi)
    plot_cost_vs_quality(runs, output_dir, show=show, dpi=dpi)

    typer.echo(f"\nAll charts written to: {output_dir}/")


if __name__ == "__main__":
    app()
