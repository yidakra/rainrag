# RainRAG Evaluation Suite

End-to-end evaluation of every retrieval and generation feature in RainRAG,
tracked with **MLflow** and measured with **RAGAS** + classical IR metrics.

---

## Quick start

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

### 3. Run the feature ablation experiment

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

### 4. Run the provider comparison

```bash
python -m eval.run_eval providers \
    --dataset eval/datasets/eval_set_en.jsonl \
    --csv providers_results.csv
```

Sweep all LLM × embedding provider combos with the full retrieval pipeline.

### 5. Profile per-stage latency

```bash
python -m eval.run_eval latency \
    --dataset eval/datasets/eval_set_en.jsonl \
    --conditions 01,06,08 \
    --n-queries 10 \
    --n-repeats 3
```

### 6. Open the MLflow UI

```bash
python -m eval.run_eval ui
# or directly:
mlflow ui --backend-store-uri ./mlruns
```

Then open http://localhost:5000 in your browser.

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
| `ragas.answer_relevance` | Does the answer address the question? |
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

## File structure

```
eval/
├── datasets/
│   ├── create_eval_set.py       # Synthetic dataset generation
│   ├── eval_set_en.jsonl        # English eval set (generated)
│   └── eval_set_ru.jsonl        # Russian eval set (generated)
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
