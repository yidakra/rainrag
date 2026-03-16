# RainRAG Evaluation Suite — Implementation Plan

## Overview

A full eval suite for all retrieval and generation features, tracked with
**MLflow** (experiment tracking, comparison UI, artifact storage) and measured
with **RAGAS** (RAG-specific LLM-as-judge metrics) plus hand-rolled retrieval
metrics (Recall@k, MRR, NDCG) that require no LLM calls.

---

## 1. Should We Use MLflow?

**Yes.** Rationale:

- Local-first (SQLite backend by default, no cloud account needed)
- Params + metrics logged per run → comparison table in the UI is exactly right
  for feature ablation studies
- Artifacts (retrieved docs, generated answers) are inspectable per run
- Tags let us group runs by experiment type (ablation, provider sweep, etc.)
- Lightweight: `pip install mlflow` adds no heavy transitive dependencies

We pair it with **RAGAS** for the RAG-specific metrics rather than hand-rolling
LLM-as-judge logic. RAGAS computes:

| RAGAS metric         | What it measures                                      |
|----------------------|-------------------------------------------------------|
| Context Precision    | Are retrieved chunks relevant to the query?           |
| Context Recall       | Do retrieved chunks cover all aspects of ground truth?|
| Faithfulness         | Is the generated answer grounded in retrieved context?|
| Answer Relevance     | Does the generated answer actually address the query? |

---

## 2. Directory Structure

```
eval/
├── datasets/
│   ├── create_eval_set.py          # Generates synthetic eval JSONL from corpus
│   ├── eval_set_en.jsonl           # English eval queries (created by script)
│   └── eval_set_ru.jsonl           # Russian eval queries (created by script)
│
├── metrics/
│   ├── __init__.py
│   ├── retrieval.py                # Recall@k, Precision@k, MRR, NDCG@k
│   └── answer_quality.py          # RAGAS wrapper (faithfulness, relevance, etc.)
│
├── experiments/
│   ├── __init__.py
│   ├── base.py                     # Abstract BaseExperiment class
│   ├── ablation.py                 # Feature ablation (the main experiment)
│   ├── provider_comparison.py      # LLM + embedding provider sweep
│   └── latency.py                  # Per-stage latency profiling
│
├── run_eval.py                     # CLI entrypoint: `python -m eval.run_eval ...`
├── mlflow_tracking.py              # MLflow helpers (log_config, log_results, etc.)
└── README.md                       # How to run evals, interpret results
```

---

## 3. Eval Dataset

### Format (JSONL, one record per query)

```json
{
  "query_id": "en_001",
  "query": "What did the reporter say about the bridge collapse?",
  "language": "en",
  "relevant_doc_ids": ["abc123", "def456"],
  "reference_answer": "The reporter described structural failure...",
  "category": "factual",
  "temporal": false
}
```

Fields:
- `relevant_doc_ids` — list of Qdrant payload `doc_id` values that are relevant
  (used for retrieval metrics without needing LLM)
- `reference_answer` — ground-truth answer used for RAGAS Context Recall and
  Answer Relevance; can be null if using LLM-as-judge only
- `category` — `factual | temporal | entity | multilingual`
- `temporal` — whether the query uses recency language ("latest", "recent")

### Dataset Creation (`eval/datasets/create_eval_set.py`)

Since there is no labelled corpus yet, we generate a synthetic eval set:

1. **Sample chunks** from the live Qdrant collection (stratified by language,
   date range, video)
2. **Use an LLM** (via `generate_answer()`) to produce a plausible query and
   reference answer for each chunk — the chunk itself becomes the single
   `relevant_doc_id`
3. **Filter out** low-quality synthetic pairs with a second LLM pass
4. **Manual review** of ≥20 queries per category before committing the JSONL

Target: 50 English + 50 Russian queries covering all four categories.

---

## 4. Retrieval Metrics (`eval/metrics/retrieval.py`)

All are computed from `(query, retrieved_doc_ids, relevant_doc_ids)` — no LLM
call required, fast to run.

```python
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def mrr(retrieved: list[str], relevant: set[str]) -> float: ...
def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float: ...
def average_precision(retrieved: list[str], relevant: set[str]) -> float: ...
```

Reported at k ∈ {3, 5, 10} (precision computed only at k=3 and k=5; NDCG computed only at k=5 and k=10).

---

## 5. Answer Quality Metrics (`eval/metrics/answer_quality.py`)

Thin wrapper around RAGAS:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

def evaluate_answers(dataset: list[EvalRecord], llm_provider: str) -> dict[str, float]: ...
```

RAGAS requires: `question`, `answer`, `contexts` (list of retrieved texts),
`ground_truth` (reference answer). We build this from the query result dict.

Also computed without LLM:
- **ROUGE-L** between generated answer and reference answer
- **Answer length** (sanity check — very short answers may indicate failures)

---

## 6. Ablation Experiment (`eval/experiments/ablation.py`)

This is the core experiment. Each "condition" is a distinct set of config
overrides. We run every condition against the full eval set and log to MLflow.

### Conditions (ordered by complexity)

| ID | Label                         | hybrid | two_stage | rewrite | hyde | reranker |
|----|-------------------------------|--------|-----------|---------|------|----------|
| 01 | vector_only                   | off    | off       | —       | —    | off      |
| 02 | hybrid_rrf                    | RRF    | off       | —       | —    | off      |
| 03 | hybrid_weighted               | WTD    | off       | —       | —    | off      |
| 04 | hybrid_rrf + rewrite          | RRF    | on        | on      | off  | off      |
| 05 | hybrid_rrf + hyde             | RRF    | on        | off     | on   | off      |
| 06 | hybrid_rrf + rewrite + hyde   | RRF    | on        | on      | on   | off      |
| 07 | hybrid_rrf + reranker         | RRF    | off       | —       | —    | on       |
| 08 | full_pipeline                 | RRF    | on        | on      | on   | on       |

Each condition × k ∈ {5, 10} × language ∈ {en, ru} = 32 MLflow runs.

### MLflow Params Logged Per Run

```python
mlflow.log_params({
    "condition_id": "06",
    "condition_label": "hybrid_rrf + rewrite + hyde",
    "top_k": 10,
    "language": "en",
    "hybrid_search.enabled": True,
    "hybrid_search.fusion_method": "rrf",
    "hybrid_search.bm25_weight": 0.3,
    "two_stage.enabled": True,
    "two_stage.query_rewrite_enabled": True,
    "two_stage.query_rewrite_variants": 2,
    "two_stage.query_rewrite_temperature": 0.7,
    "two_stage.hyde_enabled": True,
    "two_stage.hyde_alpha": 0.5,
    "two_stage.hyde_temperature": 0.7,
    "reranker.enabled": False,
    "embedding.provider": "mistral",
    "llm.provider": "openai",
    "num_eval_queries": 50,
})
```

### MLflow Metrics Logged Per Run

```python
mlflow.log_metrics({
    "recall@3": 0.72,
    "recall@5": 0.81,
    "recall@10": 0.89,
    "precision@3": 0.61,
    "precision@5": 0.52,
    "mrr": 0.68,
    "ndcg@5": 0.74,
    "ndcg@10": 0.79,
    "ragas.faithfulness": 0.91,
    "ragas.answer_relevance": 0.87,
    "ragas.answer_relevancy": 0.87,
    "ragas.context_precision": 0.76,
    "ragas.context_recall": 0.83,
    "rouge_l": 0.44,
    "latency_p50_ms": 820,
    "latency_p95_ms": 1640,
    "cost_usd_per_query": 0.003,
})
```

### Artifacts Per Run

- `retrieved_docs.jsonl` — every query's retrieved doc IDs + scores
- `answers.jsonl` — every query's generated answer
- `config_snapshot.yaml` — full config used for reproducibility

---

## 7. Provider Comparison Experiment (`eval/experiments/provider_comparison.py`)

Fixes the best-performing ablation condition and sweeps providers.

### Dimensions

| LLM provider | Embedding provider |
|---|---|
| mistral      | local              |
| openai       | mistral            |
| claude       | openai             |
| gemini       | gemini             |

Runs all 4×4 = 16 combinations (minus invalid ones — Claude has no embeddings
so falls back to local). Reports retrieval + answer quality + latency + cost.

MLflow experiment name: `provider_comparison`.

---

## 8. Latency Profiling (`eval/experiments/latency.py`)

Instruments each stage with `time.perf_counter` and logs stage breakdown:

| Stage                    | MLflow key            |
|--------------------------|-----------------------|
| Query embedding          | `t_embed_ms`          |
| BM25 search              | `t_bm25_ms`           |
| Vector search            | `t_vector_ms`         |
| Score fusion             | `t_fusion_ms`         |
| Query rewriting (LLM)    | `t_rewrite_ms`        |
| HyDE generation (LLM)    | `t_hyde_ms`           |
| Time decay boosting      | `t_decay_ms`          |
| Reranking (Cohere)       | `t_rerank_ms`         |
| Answer generation (LLM)  | `t_generate_ms`       |
| **Total end-to-end**     | `t_total_ms`          |

Run 10 queries per condition × 3 repetitions, report p50/p95.

---

## 9. CLI Interface (`eval/run_eval.py`)

```bash
# Generate synthetic eval dataset from live Qdrant collection
python -m eval.run_eval create-dataset --output eval/datasets/eval_set_en.jsonl --lang en --n 50

# Run ablation experiment
python -m eval.run_eval ablation --dataset eval/datasets/eval_set_en.jsonl --mlflow-uri ./mlruns

# Run provider comparison
python -m eval.run_eval providers --dataset eval/datasets/eval_set_en.jsonl

# Run latency profiling
python -m eval.run_eval latency --conditions 01,06,08

# Open MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

---

## 10. New Dependencies

```toml
# pyproject.toml additions (eval extras group)
[project.optional-dependencies]
eval = [
    "mlflow>=2.14",
    "ragas>=0.1",
    "rouge-score>=0.1.2",
    "pandas>=2.0",          # For result aggregation
]
```

Install with: `pip install -e ".[eval]"`

---

## 11. Implementation Order

1. **`eval/metrics/retrieval.py`** — pure Python, no deps, easiest to test
2. **`eval/datasets/create_eval_set.py`** — generates the eval JSONL
3. **`eval/mlflow_tracking.py`** — thin helpers so experiments stay clean
4. **`eval/experiments/base.py`** — abstract base with shared run loop
5. **`eval/experiments/ablation.py`** — the highest-value experiment
6. **`eval/metrics/answer_quality.py`** — RAGAS wrapper (needs API keys)
7. **`eval/experiments/provider_comparison.py`**
8. **`eval/experiments/latency.py`**
9. **`eval/run_eval.py`** — CLI wiring
10. **`eval/README.md`** — usage docs
11. **`pyproject.toml`** — add eval extras group
