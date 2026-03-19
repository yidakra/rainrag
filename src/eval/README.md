# RainRAG Evaluation Suite

End-to-end evaluation of every retrieval and generation feature in RainRAG,
tracked with **MLflow** and measured with **RAGAS** + classical IR metrics.

---

## Recommended Setup (Single-Server Production)

Use this profile if your evals run on the same machine as the live API/Qdrant
and you want to avoid overload or accidental full reindex.

### 1. Keep tracking local and explicit

```bash
# Start local MLflow server (persisted in repo-local mlruns/)
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri /home/ubuntu/rainrag/mlruns \
  --default-artifact-root file:///home/ubuntu/rainrag/mlruns

# Optional: make eval commands use this by default
export RAINRAG_MLFLOW_URI=http://127.0.0.1:5000
```

### 2. Run eval only (no ingest/embed/index)

```bash
# Safe baseline check (low risk)
python -m eval.run_eval ablation \
  --dataset eval/datasets/eval_set_en.jsonl \
  --conditions 01 \
  --top-ks 5,10 \
  --mlflow-uri http://127.0.0.1:5000 \
  --csv eval/ablation_results_vector_only.csv
```

Then expand to more conditions (`02`, `07`, `08`) once baseline is stable.

### 3. Avoid refresh/indexing commands during eval windows

Do **not** run these on the same host while evaluating unless you explicitly
intend a full data refresh:

- `rainrag ingest`
- `rainrag embed --force`
- `rainrag index --recreate`
- Any wrapper pipeline that includes the above

These are data-refresh operations, not evaluation operations.

### 4. Where results will be saved

- MLflow runs/artifacts: `/home/ubuntu/rainrag/mlruns`
- CSV summaries: paths passed via `--csv` (gitignored by default for eval outputs)
- Console/log output: whatever log file you redirect to (for example `logs/*.log`)

## Full Quick Start

### 1. Install eval dependencies

```bash
poetry install --with eval
# or, without Poetry:
pip install -e ".[eval]"
```

### 2. Generate a synthetic eval dataset

```bash
python -m eval.run_eval create-dataset \
    --lang en \
    --n 50 \
    --output eval/datasets/eval_set_en.jsonl

python -m eval.run_eval create-dataset \
    --lang ru \
    --n 50 \
    --output eval/datasets/eval_set_ru.jsonl
```

**Review the generated JSONL before running experiments.**  Each record looks like:

```json
{
  "query_id": "en_001",
  "query": "What did the mayor say about the bridge?",
  "language": "en",
  "relevant_doc_ids": ["abc123"],
  "reference_answer": "The mayor said the bridge would be repaired by autumn.",
  "category": "factual",
  "temporal": false
}
```

### 3. Review the generated dataset (human quality gate)

```bash
# Start an interactive terminal review session
python -m eval.run_eval review eval/datasets/eval_set_en.jsonl

# Check progress without starting a session
python -m eval.run_eval review eval/datasets/eval_set_en.jsonl --stats

# Review and simultaneously export a clean file with only accepted records
python -m eval.run_eval review eval/datasets/eval_set_en.jsonl \
    --filter-output eval/datasets/eval_set_en_clean.jsonl
```

During review, for each record:

| Key | Action |
|-----|--------|
| `a` | Accept — mark `valid=true` |
| `e` | Edit — correct `reference_answer` inline, then accept |
| `s` | Skip — come back later (not marked reviewed) |
| `d` | Delete — mark `valid=false` |
| `q` | Quit — save progress and exit |

Progress is saved after every decision; safe to interrupt and resume.

### 4. Run the feature ablation experiment

```bash
python -m eval.run_eval ablation \
    --dataset eval/datasets/eval_set_en.jsonl \
    --top-ks 5,10 \
    --csv ablation_results.csv
```

Runs 8 conditions (vector-only → full pipeline) × 2 top-k values = 16 MLflow runs.

To include RAGAS answer-quality metrics (requires API keys):

```bash
python -m eval.run_eval ablation \
    --dataset eval/datasets/eval_set_en.jsonl \
    --ragas
```

### 5. Run the provider comparison

```bash
python -m eval.run_eval providers \
    --dataset eval/datasets/eval_set_en.jsonl \
    --csv providers_results.csv
```

Sweep all LLM × embedding provider combos with the full retrieval pipeline.

### 6. Profile per-stage latency

```bash
python -m eval.run_eval latency \
    --dataset eval/datasets/eval_set_en.jsonl \
    --conditions 01,06,08 \
    --n-queries 10 \
    --n-repeats 3
```

### 7. Open the MLflow UI

```bash
# wrapper honors the standard MLflow variable `MLFLOW_TRACKING_URI` (see
# `python -m eval.run_eval ui` below), but when you invoke mlflow directly we
# prefer our custom name while still falling back to the standard one.
python -m eval.run_eval ui
# or directly; check RAINRAG_MLFLOW_URI first, then MLFLOW_TRACKING_URI, then
# the default path:
mlflow ui --backend-store-uri "${XDG_STATE_HOME:-$HOME/.local/state}/rainrag/mlruns"
# If using a tracking server, open its URL directly:
# ${RAINRAG_MLFLOW_URI:-${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}}

Then open http://localhost:5000 in your browser.

By default, eval runs are stored outside the repo at:
`$XDG_STATE_HOME/rainrag/mlruns` (fallback: `~/.local/state/rainrag/mlruns`).

To override this location you can set **either** environment variable. The
`python -m eval.run_eval ui` wrapper primarily honors the standard
`MLFLOW_TRACKING_URI`, whereas our direct command uses `RAINRAG_MLFLOW_URI` if
present and otherwise falls back to `MLFLOW_TRACKING_URI`.

---

## Metrics reference

### Retrieval metrics (no LLM required)

| Metric | Description |
|---|---|
| `recall@k` | Fraction of relevant docs found in top-k |
| `precision@k` | Fraction of top-k that are relevant |
| `ndcg@k` | Normalized Discounted Cumulative Gain |
| `mrr` | Mean Reciprocal Rank |
| `map` | Mean Average Precision |

Reported at k ∈ {3, 5, 10}.

### Answer quality metrics (LLM-as-judge via RAGAS)

| Metric | Description |
|---|---|
| `ragas.faithfulness` | Is the answer grounded in retrieved context? |
| `ragas.answer_relevancy` | Does the answer address the question? (canonical name; also logged as `ragas.answer_relevance` for backward compatibility) |
| `ragas.answer_relevance` | Backward-compatible alias for `ragas.answer_relevancy` |
| `ragas.context_precision` | Are retrieved docs relevant to the query? |
| `ragas.context_recall` | Does context cover all aspects of ground truth? |
| `rouge_l` | ROUGE-L F1 vs. reference answer (no LLM) |

### Latency metrics

| Metric | Description |
|---|---|
| `*_p50_ms` | Median latency |
| `*_p95_ms` | 95th-percentile latency |
| `*_mean_ms` | Mean latency |

Stages: `embed`, `retrieve`, `rerank`, `generate`, `total`.

---

## Cost & usage metrics

The evaluation suite reports an **estimated** cost per query based on:

- The main LLM generation call (prompt + output)
- The embedding call
- Estimated cost for any additional LLM calls (query rewrite, HyDE)
- Estimated cost for reranker calls (currently modeled as zero)

Because query rewriting, HyDE, and reranking are optional, we log both a
**breakdown** and a **total** so you can compare ablation conditions directly.

### Logged per-query cost/usage metrics

| Metric | Description |
|---|---|
| `cost.mean_usd_est_per_query` | Estimated mean USD cost per query across the evaluated queries |
| `cost.aggregate_usd_est` | Run-wide total estimated cost across all queries |
| `cost.total_usd_est` | Per-query total estimated cost (for the current query)
| `cost.llm_base_usd_est` | Estimated USD cost of the main answer-generation LLM call |
| `cost.llm_rewrite_usd_est` | Estimated USD cost of query rewrite LLM calls |
| `cost.llm_hyde_usd_est` | Estimated USD cost of HyDE LLM calls |
| `cost.reranker_usd_est` | Estimated USD cost of reranker calls (zero by default) |
| `cost.llm_calls_count` | Total number of LLM calls (incl. rewrite, HyDE, answer) |
| `cost.llm_query_rewrite_calls` | Query rewrite LLM calls |
| `cost.llm_hyde_calls` | HyDE generation LLM calls |
| `cost.reranker_calls_count` | Number of reranker API calls (e.g. Cohere) |

Use the call counts and per-call USD estimates to compare conditions and
understand where costs are coming from.

---

## Ablation conditions

| ID | Label | hybrid | rewrite | HyDE | reranker |
|----|-------|--------|---------|------|----------|
| 01 | vector_only | ✗ | ✗ | ✗ | ✗ |
| 02 | hybrid_rrf | RRF | ✗ | ✗ | ✗ |
| 03 | hybrid_weighted | WTD | ✗ | ✗ | ✗ |
| 04 | hybrid_rrf+rewrite | RRF | ✓ | ✗ | ✗ |
| 05 | hybrid_rrf+hyde | RRF | ✗ | ✓ | ✗ |
| 06 | hybrid_rrf+rewrite+hyde | RRF | ✓ | ✓ | ✗ |
| 07 | hybrid_rrf+reranker | RRF | ✗ | ✗ | ✓ |
| 08 | full_pipeline | RRF | ✓ | ✓ | ✓ |

---

## BEIR integration (retrieval sanity check)

Use BEIR to verify the retrieval pipeline against a public benchmark before
running expensive experiments on your proprietary archive.

### Recommended datasets

| Dataset | Queries | Corpus | Domain |
|---|---|---|---|
| `scifact` | 300 | 5,183 | Scientific fact checking |
| `nfcorpus` | 323 | 3,633 | Medical IR |
| `arguana` | 1,406 | 8,674 | Argument retrieval |
| `fiqa` | 648 | 57,638 | Financial QA |

### Quick sanity check (no Qdrant needed)

```bash
# In-memory BM25 baseline — just validates corpus + qrels loaded correctly
python -m eval.run_eval beir --dataset scifact --max-queries 50 --no-ablation
```

### Full workflow

```bash
# 1. Load corpus, index into Qdrant, generate eval JSONL
python -m eval.run_eval beir \
    --dataset scifact \
    --max-queries 100 \
    --output eval/datasets/beir_scifact.jsonl

# 2. Run ablation on the BEIR eval set (reuse existing index)
python -m eval.run_eval beir \
    --dataset scifact \
    --skip-index \
    --ablation \
    --conditions 01,02,07,08
```

### Python API

```python
from eval.datasets.beir_adapter import BEIRAdapter

adapter = BEIRAdapter("scifact")
adapter.load(max_corpus_docs=5000, max_queries=100)

# Sanity check with pure in-memory BM25 (no Qdrant required)
baseline = adapter.eval_bm25_baseline(top_k=10)
# → {"recall@10": 0.82, "ndcg@10": 0.68, "mrr": 0.71, ...}

# Full eval: index into Qdrant + convert to eval JSONL
adapter.index_corpus(engine)
adapter.to_eval_jsonl("eval/datasets/beir_scifact.jsonl")
```

### BEIR vs. synthetic datasets

| | Synthetic (`create_eval_set`) | BEIR |
|---|---|---|
| Retrieval metrics | ✓ (1 relevant doc / query) | ✓ (multi-doc, graded) |
| Answer quality (RAGAS) | ✓ | ✗ (no reference answers) |
| Domain fit | ✓ your archive | ✗ general |
| No LLM needed to create | ✗ | ✓ |
| Reproducible | Partially | ✓ |

BEIR is a **one-time smoke test** to verify the pipeline is wired correctly.
The synthetic dataset remains the primary eval set for domain-specific tuning.

---

## File structure

```
eval/
├── datasets/
│   ├── create_eval_set.py       # Synthetic dataset generation
│   ├── review_eval_set.py       # Interactive human review tool
│   ├── beir_adapter.py          # BEIR public benchmark integration
│   ├── eval_set_en.jsonl        # English eval set (generated)
│   ├── eval_set_ru.jsonl        # Russian eval set (generated)
│   └── beir_scifact.jsonl       # BEIR scifact eval set (example)
├── metrics/
│   ├── retrieval.py             # Recall@k, MRR, NDCG, MAP
│   └── answer_quality.py        # RAGAS wrapper + ROUGE-L
├── experiments/
│   ├── base.py                  # Shared experiment runner
│   ├── ablation.py              # 8-condition feature ablation
│   ├── provider_comparison.py   # LLM × embedding provider sweep
│   └── latency.py               # Per-stage latency profiling
├── mlflow_tracking.py           # MLflow helpers (graceful fallback)
├── run_eval.py                  # CLI entrypoint
└── README.md                    # This file
```
