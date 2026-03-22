"""RainRAG evaluation suite CLI.

Usage
-----
    # 1. Generate a synthetic eval dataset from your live Qdrant collection
    python -m eval.run_eval create-dataset \\
        --lang en --n 50 --output eval/datasets/eval_set_en.jsonl

    # 2. Run the feature ablation experiment
    python -m eval.run_eval ablation \\
        --dataset eval/datasets/eval_set_en.jsonl

    # 3. Run the provider comparison experiment
    python -m eval.run_eval providers \\
        --dataset eval/datasets/eval_set_en.jsonl

    # 4. Profile per-stage latency
    python -m eval.run_eval latency \\
        --dataset eval/datasets/eval_set_en.jsonl \\
        --conditions 01,06,08

    # 5. Human-review the generated dataset
    python -m eval.run_eval review eval/datasets/eval_set_en.jsonl

    # 6. Export only accepted records
    python -m eval.run_eval review eval/datasets/eval_set_en.jsonl --filter-output eval/datasets/eval_set_en_clean.jsonl

    # 7. Open the MLflow UI
    python -m eval.run_eval ui

    # 8. Sweep two-stage retrieval hyper-parameters (all six axes)
    python -m eval.run_eval two-stage \\
        --dataset eval/datasets/eval_set_en.jsonl

    # 9. Two-stage sweep — axes D+E only, save CSV
    python -m eval.run_eval two-stage \\
        --dataset eval/datasets/eval_set_en.jsonl \\
        --axes merge_strategy,merge_rrf_k \\
        --merge-strategies coverage,diverse_rrf \\
        --merge-rrf-ks 20,40,60 \\
        --csv two_stage_merge.csv

    # 10. Two-stage sweep — Axis F: prompt document ordering
    python -m eval.run_eval two-stage \\
        --dataset eval/datasets/eval_set_en.jsonl \\
        --axes doc_order \\
        --doc-orders rank,reversed,book_end
"""

import importlib
import subprocess
import time
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from eval.mlflow_tracking import default_tracking_uri


app = typer.Typer(
    name="eval",
    help="RainRAG evaluation suite.",
    add_completion=False,
)


def _parse_ints(value: str, name: str) -> list[int]:
    """Parse a comma-separated list of integers, exiting with a friendly error on failure."""
    try:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError:
        typer.echo(
            f"ERROR: Invalid {name!r}; expected comma-separated integers, got: {value!r}",
            err=True,
        )
        raise typer.Exit(1)


def _parse_floats(value: str, name: str) -> list[float]:
    """Parse a comma-separated list of floats, exiting with a friendly error on failure."""
    try:
        return [float(v.strip()) for v in value.split(",") if v.strip()]
    except ValueError:
        typer.echo(
            f"ERROR: Invalid {name!r}; expected comma-separated numbers, got: {value!r}",
            err=True,
        )
        raise typer.Exit(1)


def _parse_strs(value: str, name: str) -> list[str]:
    """Parse a comma-separated list of strings with optional leading zeros."""
    values = [v.strip() for v in value.split(",") if v.strip()]
    if not values:
        typer.echo(
            f"ERROR: Invalid {name!r}; expected comma-separated IDs, got: {value!r}",
            err=True,
        )
        raise typer.Exit(1)

    if not all(val.isdigit() for val in values):
        typer.echo(
            f"ERROR: Invalid {name!r}; expected numeric IDs, got: {value!r}",
            err=True,
        )
        raise typer.Exit(1)

    return values


# ── Shared options ──────────────────────────────────────────────────────────

_CONFIG = Annotated[str, typer.Option("--config", "-c", help="Path to config.yaml")]
_MLFLOW = Annotated[str, typer.Option("--mlflow-uri", help="MLflow tracking URI")]
_DATASET = Annotated[str | None, typer.Option("--dataset", "-d", help="Path to eval JSONL dataset")]
_OUTPUT = Annotated[str, typer.Option("--output", "-o", help="Output file path")]

_DEFAULT_MLFLOW_URI: str = default_tracking_uri()


# ── create-dataset ──────────────────────────────────────────────────────────


@app.command("create-dataset")
def create_dataset(
    config: _CONFIG = "config.yaml",
    lang: Annotated[str, typer.Option("--lang", help="Language code: en or ru")] = "en",
    n: Annotated[int, typer.Option("--n", help="Number of eval pairs to generate")] = 50,
    output: _OUTPUT = "eval/datasets/eval_set.jsonl",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    skip_filter: Annotated[
        bool, typer.Option("--skip-filter", help="Skip LLM quality-filter pass")
    ] = False,
) -> None:
    """Generate a synthetic eval dataset from the live Qdrant collection."""
    from eval.datasets.create_eval_set import create_eval_set

    create_eval_set_fn = cast(Any, create_eval_set)

    typer.echo(f"Creating {n} eval pairs (lang={lang}) → {output}")
    create_eval_set_fn(
        config_path=config,
        lang=lang,
        n=n,
        output=output,
        seed=seed,
        skip_filter=skip_filter,
    )


# ── ablation ─────────────────────────────────────────────────────────────────


@app.command("ablation")
def ablation(
    config: _CONFIG = "config.yaml",
    dataset: _DATASET = None,
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    top_ks: Annotated[
        str, typer.Option("--top-ks", help="Comma-separated retrieval depths, e.g. 5,10")
    ] = "5,10",
    conditions: Annotated[
        str | None,
        typer.Option("--conditions", help="Comma-separated condition IDs, e.g. 01,08"),
    ] = None,
    csv_output: Annotated[
        str | None, typer.Option("--csv", help="Write CSV summary to this path")
    ] = None,
    with_ragas: Annotated[
        bool, typer.Option("--ragas/--no-ragas", help="Run RAGAS answer-quality metrics")
    ] = False,
) -> None:
    """Run the 8-condition feature ablation experiment."""
    import eval.mlflow_tracking as mlflow_tracking
    from eval.experiments.ablation import AblationExperiment
    from eval.metrics.answer_quality import compute_ragas_metrics

    mlflow_tracking = cast(Any, mlflow_tracking)
    ablation_experiment_cls = cast(Any, AblationExperiment)
    compute_ragas_metrics_fn = cast(Any, compute_ragas_metrics)

    if dataset is None:
        typer.echo("ERROR: --dataset is required for the ablation experiment.", err=True)
        raise typer.Exit(1)

    ks = tuple(_parse_ints(top_ks, "top_ks"))
    cids = None
    if conditions:
        # Parse conditions as strings to preserve leading zeros (e.g. "01").
        cids = _parse_strs(conditions, "conditions")

    exp = ablation_experiment_cls(
        config_path=config,
        dataset_path=dataset,
        mlflow_uri=mlflow_uri,
        top_ks=ks,
        condition_ids=cids,
    )

    typer.echo(f"Running ablation experiment ({len(exp.conditions())} conditions × {ks} top_k) ...")
    results = exp.run()

    if with_ragas:
        typer.echo("Running RAGAS answer-quality metrics ...")
        mlflow_tracking.setup(mlflow_uri, "ablation")
        for result in results:
            ragas_records = [
                {
                    "question": r["query"],
                    "answer": r.get("answer", ""),
                    "contexts": r.get("contexts", []),
                    "ground_truth": r.get("reference_answer", ""),
                }
                for r in result["per_query"]
                if "answer" in r
            ]
            if ragas_records:
                ragas_metrics = compute_ragas_metrics_fn(ragas_records)
                run_name = f"{result['condition_label']}_k{result['top_k']}_ragas"
                with mlflow_tracking.start_run(run_name=run_name):
                    mlflow_tracking.log_params(result["params"])
                    # cast to satisfy log_metrics signature (float | int | None)
                    mlflow_tracking.log_metrics(cast(dict[str, float | int | None], ragas_metrics))

    if csv_output:
        exp.results_to_csv(results, csv_output)

    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── providers ────────────────────────────────────────────────────────────────


@app.command("providers")
def providers(
    config: _CONFIG = "config.yaml",
    dataset: _DATASET = None,
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    llm_providers: Annotated[
        str | None, typer.Option("--llm", help="Comma-separated LLM providers")
    ] = None,
    embed_providers: Annotated[
        str | None, typer.Option("--embed", help="Comma-separated embedding providers")
    ] = None,
    csv_output: Annotated[
        str | None, typer.Option("--csv", help="Write CSV summary to this path")
    ] = None,
) -> None:
    """Run the LLM × embedding provider comparison experiment."""
    from eval.experiments.provider_comparison import (
        ProviderComparisonExperiment,
    )

    provider_comparison_experiment_cls = cast(Any, ProviderComparisonExperiment)

    if dataset is None:
        typer.echo("ERROR: --dataset is required for the providers experiment.", err=True)
        raise typer.Exit(1)

    llm_list = [p.strip() for p in llm_providers.split(",")] if llm_providers else None
    emb_list = [p.strip() for p in embed_providers.split(",")] if embed_providers else None

    exp = provider_comparison_experiment_cls(
        config_path=config,
        dataset_path=dataset,
        mlflow_uri=mlflow_uri,
        llm_providers=llm_list,
        embed_providers=emb_list,
    )

    typer.echo(f"Running provider comparison ({len(exp.conditions())} conditions) ...")
    results = exp.run()

    if csv_output:
        exp.results_to_csv(results, csv_output)

    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── latency ───────────────────────────────────────────────────────────────────


@app.command("latency")
def latency(
    config: _CONFIG = "config.yaml",
    dataset: _DATASET = None,
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    conditions: Annotated[
        str, typer.Option("--conditions", help="Comma-separated ablation condition IDs")
    ] = "01,06,08",
    n_queries: Annotated[int, typer.Option("--n-queries")] = 10,
    n_repeats: Annotated[int, typer.Option("--n-repeats")] = 3,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
) -> None:
    """Profile per-stage query latency for selected ablation conditions."""
    from eval.experiments.latency import LatencyExperiment

    latency_experiment_cls = cast(Any, LatencyExperiment)

    if dataset is None:
        typer.echo("ERROR: --dataset is required for latency profiling.", err=True)
        raise typer.Exit(1)

    cids = _parse_ints(conditions, "--conditions")

    if not cids:
        typer.echo("ERROR: --conditions must include at least one integer condition ID", err=True)
        raise typer.Exit(1)

    exp = latency_experiment_cls(
        config_path=config,
        dataset_path=dataset,
        mlflow_uri=mlflow_uri,
        condition_ids=cids,
        n_queries=n_queries,
        n_repeats=n_repeats,
        top_k=top_k,
    )

    typer.echo(
        f"Profiling latency for conditions {cids} ({n_queries} queries × {n_repeats} repeats) ..."
    )
    exp.run()
    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── review ────────────────────────────────────────────────────────────────────


@app.command("review")
def review(
    input_path: Annotated[str, typer.Argument(help="Eval JSONL file to review")],
    output: Annotated[
        str | None,
        typer.Option(
            "--output", "-o", help="Save reviewed file here (defaults to overwriting input)"
        ),
    ] = None,
    filter_output: Annotated[
        str | None,
        typer.Option(
            "--filter-output", "-f", help="Also write a clean file with only valid=True records"
        ),
    ] = None,
    all_records: Annotated[
        bool, typer.Option("--all", help="Re-review already reviewed records")
    ] = False,
    stats_only: Annotated[
        bool, typer.Option("--stats", help="Just show review progress stats, don't start a session")
    ] = False,
) -> None:
    """Interactively review a generated eval JSONL dataset.

    For each unreviewed record, you can:

    \b
      [a] Accept  – mark valid
      [e] Edit    – correct the reference_answer
      [s] Skip    – leave for later
      [d] Delete  – mark invalid
      [q] Quit    – save and exit

    Progress is persisted after every decision; the session can be safely interrupted.
    """
    import eval.datasets.review_eval_set as _review_eval_set

    review_mod: Any = cast(Any, _review_eval_set)
    filter_valid = review_mod.filter_valid
    review_eval_set = review_mod.review_eval_set
    review_stats = review_mod.review_stats

    if stats_only:
        review_stats(input_path)
        return

    review_eval_set(
        input_path,
        output_path=output,
        only_unreviewed=not all_records,
    )

    if filter_output:
        src = output or input_path
        n = filter_valid(src, filter_output)
        typer.echo(f"Exported {n} valid records → {filter_output}")


# ── beir ─────────────────────────────────────────────────────────────────────


@app.command("beir")
def beir(
    config: _CONFIG = "config.yaml",
    dataset_name: Annotated[
        str, typer.Option("--dataset", "-d", help="BEIR dataset name, e.g. scifact, nfcorpus, fiqa")
    ] = "scifact",
    qrels_split: Annotated[
        str, typer.Option("--split", help="QRels split to evaluate on")
    ] = "test",
    max_corpus_docs: Annotated[
        int | None, typer.Option("--max-corpus", help="Cap corpus size (None = full)")
    ] = None,
    max_queries: Annotated[
        int | None, typer.Option("--max-queries", help="Cap number of queries")
    ] = None,
    output: _OUTPUT = "eval/datasets/beir_{dataset}.jsonl",
    bm25_baseline: Annotated[
        bool,
        typer.Option(
            "--bm25-baseline/--no-bm25-baseline", help="Run in-memory BM25 baseline before indexing"
        ),
    ] = True,
    skip_index: Annotated[
        bool, typer.Option("--skip-index", help="Skip Qdrant indexing (useful if already indexed)")
    ] = False,
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Embedding batch size for local models")
    ] = 64,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code/--no-trust-remote-code",
            help="Allow execution of remote dataset code when loading BEIR (security risk)",
        ),
    ] = False,
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    run_ablation: Annotated[
        bool,
        typer.Option(
            "--ablation/--no-ablation", help="Run ablation experiment on the generated JSONL"
        ),
    ] = False,
    ablation_conditions: Annotated[
        str | None,
        typer.Option("--conditions", help="Ablation condition IDs to run, e.g. 01,02,08"),
    ] = None,
) -> None:
    """Load a BEIR dataset, index it into Qdrant, and generate an eval JSONL.

    Optionally run the BM25 baseline and/or the ablation experiment on the
    generated dataset.

    ``--trust-remote-code`` controls whether HF's ``load_dataset`` is called
    with ``trust_remote_code=True`` when fetching BEIR. Enabling it may
    execute arbitrary code from the dataset repository; leave off unless you
    trust the source.

    Recommended quick sanity check::

        python -m eval.run_eval beir --dataset scifact --max-corpus 5000 --max-queries 50
    """
    from eval.datasets.beir_adapter import BEIRAdapter
    from eval.experiments.ablation import AblationExperiment
    from rainrag.config import load_config
    from rainrag.query import RAGQueryEngine

    beir_adapter_cls = cast(Any, BEIRAdapter)
    ablation_experiment_cls = cast(Any, AblationExperiment)
    load_config_fn = cast(Any, load_config)
    rag_query_engine_cls = cast(Any, RAGQueryEngine)

    # Resolve output path (substitute {dataset} placeholder)
    resolved_output = output.replace("{dataset}", dataset_name)

    typer.echo(f"Loading BEIR dataset: {dataset_name} (split={qrels_split}) ...")
    adapter = beir_adapter_cls(dataset_name, qrels_split=qrels_split)
    adapter.load(
        max_corpus_docs=max_corpus_docs,
        max_queries=max_queries,
        trust_remote_code=trust_remote_code,
    )
    typer.echo(adapter.summary())

    # Optional: quick in-memory BM25 baseline
    if bm25_baseline:
        typer.echo("\nRunning in-memory BM25 baseline ...")
        try:
            baseline = adapter.eval_bm25_baseline(top_k=10)
            typer.echo(
                "  BM25 baseline: recall@10={:.3f} ndcg@10={:.3f} mrr={:.3f}".format(
                    baseline.get("recall@10", float("nan")),
                    baseline.get("ndcg@10", float("nan")),
                    baseline.get("mrr", float("nan")),
                )
            )
        except Exception as exc:
            typer.echo(f"  [warn] BM25 baseline failed: {exc}")

    # Load config + engine for indexing (only initialize when needed)
    base_config = load_config_fn(config)

    engine = None
    if not skip_index:
        typer.echo(f"\nIndexing corpus into Qdrant collection '{adapter.collection_name}' ...")
        engine = rag_query_engine_cls(base_config)
        engine.initialize()
        adapter.index_corpus(engine, batch_size=batch_size, recreate=True)
    else:
        if run_ablation:
            typer.echo(
                f"\nSkipping index (assuming '{adapter.collection_name}' already exists), but initializing engine for ablation checks."
            )
            engine = rag_query_engine_cls(base_config)
            engine.initialize()
        else:
            typer.echo(f"\nSkipping index (assuming '{adapter.collection_name}' already exists).")

    # Generate eval JSONL
    typer.echo(f"\nGenerating eval JSONL → {resolved_output} ...")
    records = adapter.to_eval_jsonl(resolved_output)
    typer.echo(f"  Written {len(records)} records.")

    # Optional: run ablation against the BEIR eval set
    if run_ablation and records:
        typer.echo("\nRunning ablation experiment on BEIR eval set ...")

        cids = None
        if ablation_conditions:
            cids = _parse_strs(ablation_conditions, "--ablation-conditions")
            # Note: The option is --conditions but we label it --ablation-conditions
            # for clarity in the error message context
            if not cids:
                typer.echo(
                    "ERROR: --conditions must include at least one numeric condition ID",
                    err=True,
                )
                raise typer.Exit(1)

        # Write a temp config with the BEIR collection name override so
        # AblationExperiment can load it
        import tempfile

        import yaml

        cfg_dict = base_config.model_dump()
        cfg_dict["qdrant"]["collection_name"] = adapter.collection_name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(cfg_dict, tmp, allow_unicode=True)
            tmp_config_path = tmp.name

        try:
            exp = ablation_experiment_cls(
                config_path=tmp_config_path,
                dataset_path=resolved_output,
                mlflow_uri=mlflow_uri,
                top_ks=(10,),
                condition_ids=cids,
            )
            results = exp.run()
            csv_path = resolved_output.replace(".jsonl", "_ablation.csv")
            exp.results_to_csv(results, csv_path)
            typer.echo(f"Ablation results written to {csv_path}")
        finally:
            # Clean up the temp config file created above.
            # Retry a few times to handle transient Windows file-lock races.
            tmp_path = Path(tmp_config_path)
            for attempt in range(3):
                try:
                    tmp_path.unlink()
                    break
                except FileNotFoundError:
                    # Already removed, nothing to do.
                    break
                except PermissionError:
                    # May happen on Windows due to file locks; retry a few times.
                    if attempt < 2:
                        time.sleep(0.1)
                        continue
                    raise

    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── two-stage ─────────────────────────────────────────────────────────────────


@app.command("two-stage")
def two_stage(
    config: _CONFIG = "config.yaml",
    dataset: _DATASET = None,
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    top_ks: Annotated[
        str, typer.Option("--top-ks", help="Comma-separated retrieval depths")
    ] = "5,10",
    axes: Annotated[
        str | None,
        typer.Option(
            "--axes",
            help=(
                "Comma-separated axes to sweep. "
                "Valid: hyde_alpha,rewrite_variants,pool_size,merge_strategy,merge_rrf_k,doc_order"
            ),
        ),
    ] = None,
    hyde_alphas: Annotated[
        str | None,
        typer.Option("--hyde-alphas", help="Comma-separated HyDE alpha values, e.g. 0.1,0.5,0.9"),
    ] = None,
    rewrite_variants: Annotated[
        str | None,
        typer.Option("--rewrite-variants", help="Comma-separated variant counts, e.g. 1,2,3,5"),
    ] = None,
    pool_sizes: Annotated[
        str | None,
        typer.Option("--pool-sizes", help="Comma-separated top_k_multiplier values, e.g. 2,3,5"),
    ] = None,
    merge_strategies: Annotated[
        str | None,
        typer.Option(
            "--merge-strategies",
            help="Comma-separated merge strategy names, e.g. coverage,diverse_rrf",
        ),
    ] = None,
    merge_rrf_ks: Annotated[
        str | None,
        typer.Option(
            "--merge-rrf-ks",
            help="Comma-separated RRF k values for diverse_rrf strategy, e.g. 20,40,60",
        ),
    ] = None,
    doc_orders: Annotated[
        str | None,
        typer.Option(
            "--doc-orders",
            help="Comma-separated prompt document ordering strategies, e.g. rank,reversed,book_end",
        ),
    ] = None,
    csv_output: Annotated[
        str | None, typer.Option("--csv", help="Write CSV summary to this path")
    ] = None,
) -> None:
    """Sweep two-stage retrieval hyper-parameters across six independent axes.

    Each axis is swept while the others are held at their defaults, keeping
    wall-clock time proportional to the sum of axis lengths rather than
    their cross-product.  Results are logged to MLflow under the
    ``two_stage_sweep`` experiment.

    Axes
    ----
    A – hyde_alpha      : HyDE blend weight [0.1, 0.3, 0.5, 0.7, 0.9]
    B – rewrite_variants: query-rewrite alternatives [1, 2, 3, 5]
    C – pool_size       : top_k_multiplier before reranking [2, 3, 5]
    D – merge_strategy  : variant-merge algorithm [coverage, diverse_rrf]
    E – merge_rrf_k     : RRF k for diverse_rrf strategy [20, 40, 60]
    F – doc_order       : prompt document ordering [rank, reversed, book_end]

    Examples::

        # Full sweep (all six axes)
        python -m eval.run_eval two-stage --dataset eval/datasets/eval_set_en.jsonl

        # Axis F only — test positional effects
        python -m eval.run_eval two-stage --dataset ... --axes doc_order

        # Axes D+E only — compare merge strategies and tune RRF k
        python -m eval.run_eval two-stage --dataset ... --axes merge_strategy,merge_rrf_k

        # HyDE alpha axis only with custom values
        python -m eval.run_eval two-stage --dataset ... --axes hyde_alpha --hyde-alphas 0.1,0.3,0.7
    """
    from eval.experiments.two_stage_sweep import TwoStageSweepExperiment

    two_stage_sweep_experiment_cls = cast(Any, TwoStageSweepExperiment)

    if dataset is None:
        typer.echo("ERROR: --dataset is required for the two-stage sweep.", err=True)
        raise typer.Exit(1)

    ks = tuple(_parse_ints(top_ks, "top_ks"))
    axes_list = [a.strip() for a in axes.split(",")] if axes else None
    alphas = _parse_floats(hyde_alphas, "hyde_alphas") if hyde_alphas else None
    variants = _parse_ints(rewrite_variants, "rewrite_variants") if rewrite_variants else None
    pools = _parse_ints(pool_sizes, "pool_sizes") if pool_sizes else None
    strategies = [s.strip() for s in merge_strategies.split(",")] if merge_strategies else None
    rrf_ks = _parse_ints(merge_rrf_ks, "merge_rrf_ks") if merge_rrf_ks else None
    orders = [o.strip() for o in doc_orders.split(",")] if doc_orders else None

    exp = two_stage_sweep_experiment_cls(
        config_path=config,
        dataset_path=dataset,
        mlflow_uri=mlflow_uri,
        top_ks=ks,
        axes=axes_list,
        hyde_alphas=alphas,
        rewrite_variants=variants,
        pool_sizes=pools,
        merge_strategies=strategies,
        merge_rrf_ks=rrf_ks,
        doc_orders=orders,
    )

    typer.echo(f"Running two-stage sweep ({len(exp.conditions())} conditions × {ks} top_k) ...")
    results = exp.run()

    if csv_output:
        exp.results_to_csv(results, csv_output)

    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── plot ───────────────────────────────────────────────────────────────────────


@app.command("plot")
def plot(
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    experiment: Annotated[
        str | None,
        typer.Option("--experiment", "-e", help="Comma-separated experiment names"),
    ] = "ablation",
    filter_axis: Annotated[
        str | None,
        typer.Option("--filter-axis", help="Only include runs with this sweep_axis tag"),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Retrieval depth filter; 0 = all")] = 5,
    output: _OUTPUT = "plots",
    show: Annotated[
        bool, typer.Option("--show/--no-show", help="Display charts interactively")
    ] = False,
    dpi: Annotated[int, typer.Option("--dpi", help="Output PNG resolution")] = 150,
) -> None:
    """Generate comparison charts (retrieval bars, latency breakdown, cost scatter).

    Charts are saved as PNG files in the output directory and optionally
    displayed interactively.  Requires matplotlib and mlflow.

    Examples::

        # Ablation overview
        python -m eval.run_eval plot -e ablation --output plots/

        # Two-stage sweep, only HyDE-alpha axis
        python -m eval.run_eval plot -e two_stage_sweep --filter-axis hyde_alpha

        # Combine experiments
        python -m eval.run_eval plot -e ablation,two_stage_sweep
    """

    exp_list = [e.strip() for e in (experiment or "ablation").split(",")]
    # Invoke the plot CLI function directly to avoid sys.argv manipulation
    try:
        importlib.import_module("matplotlib")
        importlib.import_module("mlflow")
    except ImportError as exc:
        typer.echo(f"ERROR: {exc}. Install with: pip install matplotlib mlflow", err=True)
        raise typer.Exit(1)

    import eval.plot_results as _plot_results

    plot_mod: Any = cast(Any, _plot_results)
    load_runs = plot_mod.load_runs
    plot_cost_vs_quality = plot_mod.plot_cost_vs_quality
    plot_latency_breakdown = plot_mod.plot_latency_breakdown
    plot_retrieval_bars = plot_mod.plot_retrieval_bars

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading runs from: {mlflow_uri}")
    runs = load_runs(
        mlflow_uri=mlflow_uri,
        experiment_names=exp_list,
        top_k_filter=top_k,
        sweep_axis_filter=filter_axis,
    )
    typer.echo(f"Loaded {len(runs)} run(s)")

    plot_retrieval_bars(runs, output_dir, show=show, dpi=dpi)
    plot_latency_breakdown(runs, output_dir, show=show, dpi=dpi)
    plot_cost_vs_quality(runs, output_dir, show=show, dpi=dpi)
    typer.echo(f"\nAll charts written to: {output_dir}/")


# ── ui ────────────────────────────────────────────────────────────────────────


@app.command("ui")
def ui(
    mlflow_uri: _MLFLOW = _DEFAULT_MLFLOW_URI,
    port: Annotated[int, typer.Option("--port")] = 5000,
) -> None:
    """Launch the MLflow UI."""
    typer.echo(f"Starting MLflow UI at http://localhost:{port} ...")
    try:
        subprocess.run(
            ["mlflow", "ui", "--backend-store-uri", mlflow_uri, "--port", str(port)],
            check=True,
        )
    except FileNotFoundError:
        typer.echo("ERROR: mlflow not found. Install with: pip install mlflow", err=True)
        raise typer.Exit(1)
    except subprocess.CalledProcessError as exc:
        typer.echo(
            f"ERROR: mlflow ui exited with code {exc.returncode} (cmd: {exc.cmd})",
            err=True,
        )
        raise typer.Exit(exc.returncode)
    except KeyboardInterrupt:
        typer.echo("Interrupted. Stopping MLflow UI.")
        raise typer.Exit(0)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running as `python -m eval.run_eval`
    app()
