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
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="eval",
    help="RainRAG evaluation suite.",
    add_completion=False,
)

# ── Shared options ──────────────────────────────────────────────────────────

_CONFIG = Annotated[str, typer.Option("--config", "-c", help="Path to config.yaml")]
_MLFLOW = Annotated[str, typer.Option("--mlflow-uri", help="MLflow tracking URI")]
_DATASET = Annotated[Optional[str], typer.Option("--dataset", "-d", help="Path to eval JSONL dataset")]
_OUTPUT = Annotated[str, typer.Option("--output", "-o", help="Output file path")]


# ── create-dataset ──────────────────────────────────────────────────────────


@app.command("create-dataset")
def create_dataset(
    config: _CONFIG = "config.yaml",
    lang: Annotated[str, typer.Option("--lang", help="Language code: en or ru")] = "en",
    n: Annotated[int, typer.Option("--n", help="Number of eval pairs to generate")] = 50,
    output: _OUTPUT = "eval/datasets/eval_set.jsonl",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    skip_filter: Annotated[bool, typer.Option("--skip-filter", help="Skip LLM quality-filter pass")] = False,
) -> None:
    """Generate a synthetic eval dataset from the live Qdrant collection."""
    from eval.datasets.create_eval_set import create_eval_set

    typer.echo(f"Creating {n} eval pairs (lang={lang}) → {output}")
    create_eval_set(
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
    mlflow_uri: _MLFLOW = "./mlruns",
    top_ks: Annotated[str, typer.Option("--top-ks", help="Comma-separated retrieval depths, e.g. 5,10")] = "5,10",
    conditions: Annotated[Optional[str], typer.Option("--conditions", help="Comma-separated condition IDs, e.g. 01,08")] = None,
    csv_output: Annotated[Optional[str], typer.Option("--csv", help="Write CSV summary to this path")] = None,
    with_ragas: Annotated[bool, typer.Option("--ragas/--no-ragas", help="Run RAGAS answer-quality metrics")] = False,
) -> None:
    """Run the 8-condition feature ablation experiment."""
    from eval.experiments.ablation import AblationExperiment
    from eval.metrics.answer_quality import compute_ragas_metrics
    import eval.mlflow_tracking as mlflow_tracking

    if dataset is None:
        typer.echo("ERROR: --dataset is required for the ablation experiment.", err=True)
        raise typer.Exit(1)

    ks = tuple(int(k.strip()) for k in top_ks.split(","))
    cids = [c.strip() for c in conditions.split(",")] if conditions else None

    exp = AblationExperiment(
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
                ragas_metrics = compute_ragas_metrics(ragas_records)
                run_name = f"{result['condition_label']}_k{result['top_k']}_ragas"
                with mlflow_tracking.start_run(run_name=run_name):
                    mlflow_tracking.log_params(result["params"])
                    mlflow_tracking.log_metrics(ragas_metrics)

    if csv_output:
        exp.results_to_csv(results, csv_output)

    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── providers ────────────────────────────────────────────────────────────────


@app.command("providers")
def providers(
    config: _CONFIG = "config.yaml",
    dataset: _DATASET = None,
    mlflow_uri: _MLFLOW = "./mlruns",
    llm_providers: Annotated[Optional[str], typer.Option("--llm", help="Comma-separated LLM providers")] = None,
    embed_providers: Annotated[Optional[str], typer.Option("--embed", help="Comma-separated embedding providers")] = None,
    csv_output: Annotated[Optional[str], typer.Option("--csv", help="Write CSV summary to this path")] = None,
) -> None:
    """Run the LLM × embedding provider comparison experiment."""
    from eval.experiments.provider_comparison import ProviderComparisonExperiment

    if dataset is None:
        typer.echo("ERROR: --dataset is required for the providers experiment.", err=True)
        raise typer.Exit(1)

    llm_list = [p.strip() for p in llm_providers.split(",")] if llm_providers else None
    emb_list = [p.strip() for p in embed_providers.split(",")] if embed_providers else None

    exp = ProviderComparisonExperiment(
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
    mlflow_uri: _MLFLOW = "./mlruns",
    conditions: Annotated[str, typer.Option("--conditions", help="Comma-separated ablation condition IDs")] = "01,06,08",
    n_queries: Annotated[int, typer.Option("--n-queries")] = 10,
    n_repeats: Annotated[int, typer.Option("--n-repeats")] = 3,
    top_k: Annotated[int, typer.Option("--top-k")] = 5,
) -> None:
    """Profile per-stage query latency for selected ablation conditions."""
    from eval.experiments.latency import LatencyExperiment

    if dataset is None:
        typer.echo("ERROR: --dataset is required for latency profiling.", err=True)
        raise typer.Exit(1)

    cids = [c.strip() for c in conditions.split(",")]

    exp = LatencyExperiment(
        config_path=config,
        dataset_path=dataset,
        mlflow_uri=mlflow_uri,
        condition_ids=cids,
        n_queries=n_queries,
        n_repeats=n_repeats,
        top_k=top_k,
    )

    typer.echo(f"Profiling latency for conditions {cids} ({n_queries} queries × {n_repeats} repeats) ...")
    exp.run()
    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── review ────────────────────────────────────────────────────────────────────


@app.command("review")
def review(
    input_path: Annotated[str, typer.Argument(help="Eval JSONL file to review")],
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Save reviewed file here (defaults to overwriting input)")] = None,
    filter_output: Annotated[Optional[str], typer.Option("--filter-output", "-f", help="Also write a clean file with only valid=True records")] = None,
    all_records: Annotated[bool, typer.Option("--all", help="Re-review already reviewed records")] = False,
    stats_only: Annotated[bool, typer.Option("--stats", help="Just show review progress stats, don't start a session")] = False,
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
    from eval.datasets.review_eval_set import filter_valid, review_eval_set, review_stats

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
    dataset_name: Annotated[str, typer.Option("--dataset", "-d", help="BEIR dataset name, e.g. scifact, nfcorpus, fiqa")] = "scifact",
    qrels_split: Annotated[str, typer.Option("--split", help="QRels split to evaluate on")] = "test",
    max_corpus_docs: Annotated[Optional[int], typer.Option("--max-corpus", help="Cap corpus size (None = full)")] = None,
    max_queries: Annotated[Optional[int], typer.Option("--max-queries", help="Cap number of queries")] = None,
    output: _OUTPUT = "eval/datasets/beir_{dataset}.jsonl",
    bm25_baseline: Annotated[bool, typer.Option("--bm25-baseline/--no-bm25-baseline", help="Run in-memory BM25 baseline before indexing")] = True,
    skip_index: Annotated[bool, typer.Option("--skip-index", help="Skip Qdrant indexing (useful if already indexed)")] = False,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Embedding batch size for local models")] = 64,
    mlflow_uri: _MLFLOW = "./mlruns",
    run_ablation: Annotated[bool, typer.Option("--ablation/--no-ablation", help="Run ablation experiment on the generated JSONL")] = False,
    ablation_conditions: Annotated[Optional[str], typer.Option("--conditions", help="Ablation condition IDs to run, e.g. 01,02,08")] = None,
) -> None:
    """Load a BEIR dataset, index it into Qdrant, and generate an eval JSONL.

    Optionally run the BM25 baseline and/or the ablation experiment on the
    generated dataset.

    Recommended quick sanity check::

        python -m eval.run_eval beir --dataset scifact --max-corpus 5000 --max-queries 50
    """
    from eval.datasets.beir_adapter import BEIRAdapter
    from eval.experiments.ablation import AblationExperiment
    from eval.experiments.base import apply_overrides
    from rainrag.config import load_config
    from rainrag.query import RAGQueryEngine

    # Resolve output path (substitute {dataset} placeholder)
    resolved_output = output.replace("{dataset}", dataset_name)

    typer.echo(f"Loading BEIR dataset: {dataset_name} (split={qrels_split}) ...")
    adapter = BEIRAdapter(dataset_name, qrels_split=qrels_split)
    adapter.load(max_corpus_docs=max_corpus_docs, max_queries=max_queries)
    typer.echo(adapter.summary())

    # Optional: quick in-memory BM25 baseline
    if bm25_baseline:
        typer.echo("\nRunning in-memory BM25 baseline ...")
        try:
            baseline = adapter.eval_bm25_baseline(top_k=10)
            typer.echo(
                f"  BM25 baseline: recall@10={baseline.get('recall@10', float('nan')):.3f} "
                f"ndcg@10={baseline.get('ndcg@10', float('nan')):.3f} "
                f"mrr={baseline.get('mrr', float('nan')):.3f}"
            )
        except Exception as exc:
            typer.echo(f"  [warn] BM25 baseline failed: {exc}")

    # Load config + engine for indexing
    base_config = load_config(config)

    if not skip_index:
        typer.echo(f"\nIndexing corpus into Qdrant collection '{adapter.collection_name}' ...")
        engine = RAGQueryEngine(base_config)
        engine.initialize()
        adapter.index_corpus(engine, batch_size=batch_size, recreate=True)
    else:
        typer.echo(f"\nSkipping index (assuming '{adapter.collection_name}' already exists).")
        engine = RAGQueryEngine(base_config)
        engine.initialize()

    # Generate eval JSONL
    typer.echo(f"\nGenerating eval JSONL → {resolved_output} ...")
    records = adapter.to_eval_jsonl(resolved_output)
    typer.echo(f"  Written {len(records)} records.")

    # Optional: run ablation against the BEIR eval set
    if run_ablation and records:
        typer.echo("\nRunning ablation experiment on BEIR eval set ...")
        cids = [c.strip() for c in ablation_conditions.split(",")] if ablation_conditions else None

        # Override collection name to the BEIR collection
        overridden_config = apply_overrides(
            base_config, {"qdrant.collection_name": adapter.collection_name}
        )
        _ = overridden_config  # Used inside AblationExperiment via config_path override

        # Write a temp config with the collection name override so AblationExperiment can load it
        import tempfile, yaml  # type: ignore[import]
        cfg_dict = base_config.model_dump()
        cfg_dict["qdrant"]["collection_name"] = adapter.collection_name
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as tmp:
            yaml.dump(cfg_dict, tmp, allow_unicode=True)
            tmp_config_path = tmp.name

        exp = AblationExperiment(
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

    typer.echo(f"\nDone. Open MLflow UI with: mlflow ui --backend-store-uri {mlflow_uri}")


# ── ui ────────────────────────────────────────────────────────────────────────


@app.command("ui")
def ui(
    mlflow_uri: _MLFLOW = "./mlruns",
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


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow running as `python -m eval.run_eval`
    app()
