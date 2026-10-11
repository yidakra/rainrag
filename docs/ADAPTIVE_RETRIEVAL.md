# Adaptive retrieval: archive-specific formulation selection

This implements a testable application of the Project AI A study, not a claim that its
MultiVENT results transfer to TV Rain. The study found that observed ranking signals
often supported selection better than corpus statistics under a matched ridge learner.
It also found exceptions and collections where selection did not help. Consequently,
the existing pipeline remains the default and model weights must be fitted locally.
The [research repository](https://github.com/yidakra/per-query-selection) records the
benchmark results and their limitations. The formulation gain there is evidence for
testing this design, not an expected RainRAG gain.

## Implemented scope

The option pool contains the original query, the existing transcript-register rewrites,
and the coverage and diverse-RRF merges of their first-stage rankings. A small ridge
selects an option using score-distribution features and ranking agreement. The selected
pool then goes through the existing temporal adjustment, duplicate suppression, shared
reranker and answer generation. This does **not** introduce new OCR/frame indexes,
automatic query translation, or dense-versus-BM25 route selection. Those require separate
candidate construction and evaluation. The `language` argument still specifies answer
and rewrite-prompt language, not a retrieval-language policy.

The shared `_prepare_query` path serves both blocking and streaming requests. Uploaded
single videos and override collections are excluded. No benchmark weights are shipped.
Selection does not avoid generating or retrieving the alternatives. Shadow mode adds
no provider calls beyond the already configured pipeline, but does extra local feature
and merge calculations. Do not claim latency, cost, or answer-quality gains from the
first-stage evaluation.

## 1. Capture without changing answers

Use a separate evaluation configuration, not an unreviewed production toggle:

```yaml
two_stage:
  enabled: true
  query_rewrite_enabled: true
  query_rewrite_variants: 2
  hyde_enabled: false

adaptive_retrieval:
  enabled: true
  mode: shadow
  corpus_revision: archive-snapshot-2026-10-11
  capture_candidates: true
```

Shadow mode keeps the configured merge result. Without `model_path`, it only captures;
with a compatible model, it also records the recommendation without applying it.
`capture_candidates` is independently opt-in and can be used without enabling selection.
Enabling rewrites where they were disabled **does** introduce the configured rewrite and
embedding calls. It is not a free consequence of enabling shadow mode.

The engine result and internal pilot log contain `retrieval_selection`. Its alternatives
include query variants and candidate IDs/scores, but not additional transcript copies.
These remain internal archive data: control access and retention, disable capture when
finished, and never commit real records. Disabling `RAINRAG_QUERY_LOG_PATH` also disables
the pilot file writer; capture can instead be collected through the Python engine.

The trace is captured before time boosting, duplicate suppression and reranking. Raw
score types are retained through fusion/reranking. Cosine, RRF and Cohere scores are not
interchangeable probabilities. `retrieval_documents` preserves final retrieval rank,
while `retrieved_documents` retains prompt context order for compatibility. Evaluation
uses retrieval rank even when prompt ordering is `reversed` or `book_end`.

## 2. Judge the alternatives

Build a representative set of real journalist information needs. Pool candidate
segments from all options, judge multiple relevant segments, and review coverage beyond
that pool where practical. Single-source synthetic labels are only a smoke test: other
relevant segments can be incorrectly treated as negatives. Binary nDCG is currently
used; this is not numerically interchangeable with the paper's graded benchmark metric.

Each input JSONL record must have this structure:

```json
{
  "query_id": "internal-example-001",
  "group_id": "event-example-001",
  "relevant_doc_ids": ["segment-a", "segment-b"],
  "retrieval_selection": {
    "stage": "first_stage_before_merge",
    "configuration_signature": "copied-from-the-captured-trace",
    "corpus_revision": "archive-snapshot-2026-10-11",
    "alternatives": [
      {"id": "original", "query": "original question", "documents": [{"doc_id": "segment-a", "score": 0.7}]},
      {"id": "rewrite_1", "query": "rewritten question", "documents": [{"doc_id": "segment-b", "score": 0.8}]}
    ]
  }
}
```

Assign `group_id` through editorial/event review. Related phrasings, overlapping chunks,
translated versions, rebroadcasts and near-duplicate events must not cross folds. The
evaluator requires explicit groups; it cannot infer their correctness. Shared relevant
documents and known relevant video IDs crossing groups are rejected to catch obvious
leakage. Semantic near-duplicates still require review. Single-positive
queries and ties are reported to expose sparse-label limitations, not to estimate an
attainable oracle ceiling.

## 3. Evaluate cached rankings locally

```bash
PYTHONPATH=src python -m rainrag_eval.selection \
  --dataset /path/to/judged_candidates.jsonl \
  --report /path/to/new_selection_report.json \
  --model /path/to/new_archive_selector.json \
  --folds 5 --alpha 1.0
```

This command makes no provider calls and does not ingest, embed or index the archive.
Outputs must be new paths. Model output is optional. Group-held-out evaluation compares
the selector to original-only and to the best fixed option chosen **on training data**.
If a training-chosen option was not produced for a held-out query, that fixed policy
falls back to the original. `fixed_missing_options` reports how often this occurred.
Alpha and the minimum-gain rule are fixed before evaluation, not tuned on held-out
results. If experimenting with these settings, use an additional validation split or
nested protocol; do not select the best test result and call it held-out evidence.

Optionally pass `--corpus /path/to/fixed_docs.jsonl` to compare the same learner with
seven pre-retrieval query-length/IDF features computed from that corpus snapshot. This
is a bounded corpus-statistic control, not a full reproduction of all eleven predictors
in the paper. Pre/post p-values against the fixed baseline receive Holm adjustment
when both are tested. Corpus reading is streaming, but vocabulary counts use memory.
The caller must provide the same corpus revision used by candidate retrieval.

The exported model is refitted on all supplied cases **after** out-of-fold evaluation.
Its in-sample fit is not another validation result. Inspect group splits, label quality,
ties, fixed baselines and selection stability. An end-to-end pilot must separately
evaluate the final reranked evidence, answer grounding, useful source segments,
errors, p50/p95 latency and actual calls/cost. First-stage gains may disappear downstream.

## 4. Shadow a model, then explicitly activate

```yaml
adaptive_retrieval:
  enabled: true
  mode: shadow  # change to select only after archive-specific validation
  model_path: /path/to/new_archive_selector.json
  corpus_revision: archive-snapshot-2026-10-11
  min_predicted_gain: 0.0
  capture_candidates: false
```

Artifact loading checks schema, feature order, finite coefficients and the retrieval
configuration signature. It uses numeric JSON, not pickle. The signature covers the
operator-provided corpus revision, collection, embedding/rewrite models, hybrid and
two-stage settings, reranker configuration and model. It does not automatically detect
reindexing or provider model drift: update the corpus revision and retrain when those
change. Candidate depth is also bound: capture at the serving configuration's default
depth. If a request changes depth or the configuration changes after loading, selection
falls back rather than applying mismatched weights. Keep the evaluation configuration
identical to serving, including candidate
depth and preprocessing. Changing the signature rejects the model at startup.

`select` requires both a model and an explicit corpus revision. Ties and improvements
below the configured predicted-gain threshold keep the original. Empty alternatives
are excluded; when every ranking is empty, the original remains the fallback. Invalid
runtime features keep the original in selection mode and the existing merge in shadow
mode. Predicted contrasts are not calibrated confidence or an abstention guarantee.

Rollback is `adaptive_retrieval.enabled: false`. Candidate capture is a separate flag,
so also set `capture_candidates: false` when ending the experiment. No automatic
production activation, training against live user feedback, or archive rebuild occurs.
