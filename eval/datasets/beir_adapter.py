"""BEIR dataset adapter for retrieval-only sanity checks.

Loads any dataset from the BEIR benchmark via HuggingFace ``datasets``
(already a dependency in the eval group) and converts it to the RainRAG
eval JSONL format so you can run the standard retrieval metrics (Recall@k,
MRR, NDCG, MAP) against a known public benchmark without needing proprietary
archive data.

Recommended datasets for a quick sanity check
---------------------------------------------
- ``scifact``    – 300 queries, 5 183 passages  (scientific fact checking)
- ``nfcorpus``   – 323 queries, 3 633 passages  (medical information retrieval)
- ``arguana``    – 1 406 queries, 8 674 passages (argument retrieval)
- ``fiqa``       – 648 queries, 57 638 passages  (financial QA)

Full list: https://github.com/beir-cellar/beir#beir-benchmark

Usage
-----
Python API::

    from eval.datasets.beir_adapter import BEIRAdapter
    from rainrag.config import load_config
    from rainrag.query import RAGQueryEngine

    config = load_config("config.yaml")
    engine = RAGQueryEngine(config)
    engine.initialize()

    adapter = BEIRAdapter("scifact", collection_suffix="beir_test")
    adapter.load(max_corpus_docs=5000, max_queries=100)
    adapter.index_corpus(engine)
    adapter.to_eval_jsonl("eval/datasets/beir_scifact.jsonl")
    # Then run the ablation experiment on the generated JSONL

CLI (via run_eval.py)::

    python -m eval.run_eval beir \\
        --dataset scifact \\
        --max-queries 100 \\
        --output eval/datasets/beir_scifact.jsonl
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, cast


logger = logging.getLogger(__name__)


def _new_docs_dict() -> dict[str, dict[str, str]]:
    """Typed default factory for BEIR corpus docs."""
    return {}


def _new_queries_dict() -> dict[str, str]:
    """Typed default factory for BEIR queries."""
    return {}


def _new_qrels_dict() -> dict[str, dict[str, int]]:
    """Typed default factory for BEIR qrels."""
    return {}


def _rows_from_dataset(dataset: Any) -> list[dict[str, Any]]:
    """Normalize a HF dataset split into a typed list of row dicts."""
    return [dict(row) for row in dataset]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class BEIRCorpus:
    """Loaded BEIR corpus: doc_id → {title, text}."""

    docs: dict[str, dict[str, str]] = field(default_factory=_new_docs_dict)

    def __len__(self) -> int:
        return len(self.docs)


@dataclass
class BEIRQueries:
    """Loaded BEIR queries: query_id → query_text."""

    queries: dict[str, str] = field(default_factory=_new_queries_dict)

    def __len__(self) -> int:
        return len(self.queries)


@dataclass
class BEIRQRels:
    """BEIR relevance judgements: query_id → {doc_id → relevance_score}."""

    qrels: dict[str, dict[str, int]] = field(default_factory=_new_qrels_dict)

    def relevant_doc_ids(self, query_id: str, min_score: int = 1) -> list[str]:
        """Return doc IDs with relevance >= min_score."""
        return [
            doc_id for doc_id, score in self.qrels.get(query_id, {}).items() if score >= min_score
        ]


# ---------------------------------------------------------------------------
# HuggingFace loader
# ---------------------------------------------------------------------------


def _load_from_hf(
    name: str,
    qrels_split: str = "test",
    max_corpus_docs: int | None = None,
    max_queries: int | None = None,
    trust_remote_code: bool = False,
) -> tuple[BEIRCorpus, BEIRQueries, BEIRQRels]:
    """Load a BEIR dataset from HuggingFace.

    Args:
        name: BEIR dataset name, e.g. "scifact", "nfcorpus", "fiqa".
        qrels_split: Which qrels split to load (usually "test").
        max_corpus_docs: Limit corpus size (useful for quick tests).
        max_queries: Limit number of queries.
        trust_remote_code: If True, pass ``trust_remote_code`` to
            ``datasets.load_dataset``. This allows execution of code
            shipped with the dataset and can be a security risk when
            loading untrusted HF repositories. Defaults to False.

    Returns:
        (corpus, queries, qrels) triple.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for BEIR loading. "
            + "Install with: pip install datasets"
        ) from exc

    hf_name = f"BeIR/{name}"
    hf_qrels_name = f"BeIR/{name}-qrels"

    logger.info(f"Loading BEIR corpus from {hf_name} ...")
    if trust_remote_code:
        logger.warning(
            "trust_remote_code=True will execute remote dataset code; use only with datasets you trust"
        )
    load_dataset_any = cast(Any, load_dataset)

    corpus_ds_raw: Any = load_dataset_any(
        hf_name,
        "corpus",
        split="corpus",
        trust_remote_code=trust_remote_code,
    )
    logger.info(f"Loading BEIR queries from {hf_name} ...")
    queries_ds_raw: Any = load_dataset_any(
        hf_name, "queries", split="queries", trust_remote_code=trust_remote_code
    )
    logger.info(f"Loading BEIR qrels from {hf_qrels_name} (split={qrels_split}) ...")
    qrels_ds_raw: Any = load_dataset_any(
        hf_qrels_name,
        split=qrels_split,
        trust_remote_code=trust_remote_code,
    )

    corpus_ds = _rows_from_dataset(corpus_ds_raw)
    queries_ds = _rows_from_dataset(queries_ds_raw)
    qrels_ds = _rows_from_dataset(qrels_ds_raw)

    # --- Corpus
    corpus = BEIRCorpus()
    for i, raw in enumerate(corpus_ds):
        if max_corpus_docs and i >= max_corpus_docs:
            break
        doc_id = str(raw.get("_id", ""))
        if not doc_id:
            continue
        corpus.docs[doc_id] = {
            "title": str(raw.get("title", "")),
            "text": str(raw.get("text", "")),
        }
    logger.info(f"Loaded {len(corpus)} corpus documents.")

    # --- QRels first (so we can filter queries to only those with qrels)
    qrels = BEIRQRels()
    corpus_doc_ids = set(corpus.docs.keys())
    for row in qrels_ds:
        qid_raw = row.get("query-id")
        if qid_raw is None:
            qid_raw = row.get("query_id")

        did_raw = row.get("corpus-id")
        if did_raw is None:
            did_raw = row.get("corpus_id")

        if qid_raw is None or did_raw is None:
            continue

        qid = str(qid_raw)
        did = str(did_raw)
        # Skip qrels for docs not in the (possibly limited) corpus
        if did not in corpus_doc_ids:
            continue
        score_raw = row.get("score", 0)
        if score_raw is None:
            score_raw = 0
        score = int(score_raw)
        qrels.qrels.setdefault(qid, {})[did] = score

    # --- Queries (only those that appear in qrels)
    queries = BEIRQueries()
    qids_with_qrels = set(qrels.qrels.keys())
    count = 0
    for row in queries_ds:
        qid = str(row.get("_id", ""))
        if not qid:
            continue
        if qid not in qids_with_qrels:
            continue
        if max_queries and count >= max_queries:
            break
        queries.queries[qid] = str(row.get("text", ""))
        count += 1
    logger.info(f"Loaded {len(queries)} queries (with qrels).")

    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _embed_documents_local(
    engine: Any,
    texts: list[str],
    batch_size: int = 64,
) -> list[list[float]]:
    """Batch-encode document texts using the engine's local SentenceTransformer.

    Uses the "passage: " prefix which E5 models expect for indexed documents.
    """

    model = engine.embedding_model
    normalize = engine.config.embedding.normalize_embeddings

    # determine prefix: user-configured or auto-detect for E5-like models
    prefix = engine.config.embedding.prefix or ""
    if not prefix:
        # try to infer from model name if available
        name = ""
        if model is not None and hasattr(model, "name"):
            name = model.name or ""
        else:
            name = engine.config.embedding.model_name or ""
        if "e5" in name.lower():
            prefix = "passage: "

    prefixed = [f"{prefix}{t}" for t in texts] if prefix else texts

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=True,
    )
    return [e.tolist() if hasattr(e, "tolist") else e for e in embeddings]


def _embed_documents_api(
    engine: Any,
    texts: list[str],
    delay_s: float = 0.05,
) -> list[list[float]]:
    """Embed documents via an API provider by calling embed_query() in a loop.

    Note: this is slow for large corpora. Use local embeddings for BEIR eval
    or keep max_corpus_docs small.
    """
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        if i % 50 == 0 and i > 0:
            logger.info(f"  Embedded {i}/{len(texts)} documents ...")
        embeddings.append(cast(list[float], engine.embed_query(text)))
        if delay_s > 0:
            time.sleep(delay_s)
    return embeddings


def _embed_corpus(
    engine: Any,
    docs: dict[str, dict[str, str]],
    batch_size: int = 64,
) -> dict[str, list[float]]:
    """Embed all corpus documents and return doc_id → vector mapping."""
    doc_ids = list(docs.keys())
    texts = [docs[did]["title"] + " " + docs[did]["text"] for did in doc_ids]

    if engine.config.embedding.provider == "local" and engine.embedding_model is not None:
        logger.info(f"Batch-encoding {len(texts)} documents with local model ...")
        vectors = _embed_documents_local(engine, texts, batch_size=batch_size)
    else:
        logger.info(f"Embedding {len(texts)} documents via API (this may be slow) ...")
        vectors = _embed_documents_api(engine, texts)

    # zip with strict=True to raise if lengths differ (mismatch indicates bug)
    return dict(zip(doc_ids, vectors, strict=True))


# ---------------------------------------------------------------------------
# Qdrant indexer
# ---------------------------------------------------------------------------


def _index_corpus_into_qdrant(
    engine: Any,
    corpus: BEIRCorpus,
    collection_name: str,
    batch_size: int = 64,
    recreate: bool = True,
    upsert_batch: int | None = None,
) -> None:
    """Create a Qdrant collection and upsert all corpus documents.

    Args:
        engine: Initialized RAGQueryEngine (for its Qdrant client + embedding model).
        corpus: Loaded BEIR corpus.
        collection_name: Qdrant collection to create/use.
        batch_size: Embedding batch size (for local models).
        recreate: Drop and recreate the collection if it already exists.
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore[import]

    client = engine.qdrant_client
    vector_size = engine.config.qdrant.vector_size

    # Create or recreate collection
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        if recreate:
            logger.info(f"Dropping existing collection '{collection_name}' ...")
            client.delete_collection(collection_name)
        else:
            logger.info(f"Reusing existing collection '{collection_name}'.")
            return

    logger.info(f"Creating Qdrant collection '{collection_name}' (vector_size={vector_size}) ...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,  # pyright: ignore[reportUnknownMemberType]
        ),
    )

    # Embed corpus
    doc_vectors = _embed_corpus(engine, corpus.docs, batch_size=batch_size)

    # Upsert in batches
    doc_ids = list(corpus.docs.keys())
    total = len(doc_ids)
    logger.info(f"Upserting {total} documents into Qdrant ...")

    # default to embedding batch size if not explicitly provided
    if upsert_batch is None:
        upsert_batch = batch_size
    for start in range(0, total, upsert_batch):
        batch_doc_ids = doc_ids[start : start + upsert_batch]
        points: list[Any] = []
        for int_id, doc_id in enumerate(batch_doc_ids, start=start):
            doc = corpus.docs[doc_id]
            text = (doc["title"] + " " + doc["text"]).strip()
            points.append(
                PointStruct(
                    id=int_id,
                    vector=doc_vectors[doc_id],
                    payload={
                        "doc_id": doc_id,
                        "text": text,
                        "title": doc.get("title", ""),
                        "language": "en",
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points, wait=True)
        logger.info(f"  Upserted {min(start + upsert_batch, total)}/{total} documents ...")

    logger.info(f"Corpus indexed into '{collection_name}'.")


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class BEIRAdapter:
    """Load a BEIR dataset and integrate it with the RainRAG eval pipeline.

    Typical workflow::

        adapter = BEIRAdapter("scifact")
        adapter.load(max_corpus_docs=5000, max_queries=100)
        adapter.index_corpus(engine)                          # pushes to Qdrant
        adapter.to_eval_jsonl("eval/datasets/beir_scifact.jsonl")
        # run any BaseExperiment subclass with the generated JSONL

    Args:
        name: BEIR dataset name (e.g. "scifact", "nfcorpus", "fiqa").
        qrels_split: Which qrels split to evaluate on (default: "test").
        collection_suffix: Suffix appended to the Qdrant collection name
            (default: dataset name → collection "beir_scifact").
    """

    def __init__(
        self,
        name: str,
        qrels_split: str = "test",
        collection_suffix: str | None = None,
    ) -> None:
        self.name = name
        self.qrels_split = qrels_split
        self.collection_name = f"beir_{collection_suffix or name}"

        self._corpus: BEIRCorpus | None = None
        self._queries: BEIRQueries | None = None
        self._qrels: BEIRQRels | None = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        max_corpus_docs: int | None = None,
        max_queries: int | None = None,
        trust_remote_code: bool = False,
    ) -> BEIRAdapter:
        """Download and cache the dataset from HuggingFace.

        Args:
            max_corpus_docs: Cap on corpus size for quick experiments.
            max_queries: Cap on number of queries.
            trust_remote_code: See :func:`_load_from_hf` for security implications.
        """
        self._corpus, self._queries, self._qrels = _load_from_hf(
            self.name,
            qrels_split=self.qrels_split,
            max_corpus_docs=max_corpus_docs,
            max_queries=max_queries,
            trust_remote_code=trust_remote_code,
        )
        return self

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def index_corpus(
        self,
        engine: Any,
        batch_size: int = 64,
        recreate: bool = True,
    ) -> BEIRAdapter:
        """Index the BEIR corpus into a dedicated Qdrant collection.

        The collection name is ``beir_{name}`` (or the custom suffix).
        After calling this, temporarily override ``config.qdrant.collection_name``
        to the BEIR collection name when running experiments.

        Args:
            engine: Initialized RAGQueryEngine.
            batch_size: Embedding batch size (only relevant for local models).
            recreate: Drop the collection first if it already exists.
        """
        self._require_loaded()
        _index_corpus_into_qdrant(
            engine,
            self._corpus,  # type: ignore[arg-type]
            collection_name=self.collection_name,
            batch_size=batch_size,
            recreate=recreate,
            upsert_batch=None,
        )
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_eval_jsonl(
        self,
        output_path: str,
        min_relevance: int = 1,
    ) -> list[dict[str, Any]]:
        """Convert the loaded dataset to RainRAG eval JSONL format.

        Each record maps to one query with its ground-truth relevant doc IDs.
        ``reference_answer`` is empty (BEIR is retrieval-only), so RAGAS
        answer-quality metrics won't be meaningful—use only retrieval metrics.

        Args:
            output_path: Where to write the JSONL file.
            min_relevance: Minimum BEIR relevance score to consider a doc relevant.

        Returns:
            The list of record dicts that was written.
        """
        self._require_loaded()
        assert self._queries is not None
        assert self._qrels is not None

        records: list[dict[str, Any]] = []
        for i, (qid, qtext) in enumerate(self._queries.queries.items(), 1):
            rel_ids = self._qrels.relevant_doc_ids(qid, min_score=min_relevance)
            if not rel_ids:
                continue  # Skip queries with no relevant docs in the corpus

            records.append(
                {
                    "query_id": f"beir_{self.name}_{i:04d}",
                    "query": qtext,
                    "language": "en",
                    "relevant_doc_ids": rel_ids,
                    "reference_answer": "",
                    "category": "factual",
                    "temporal": False,
                    "beir_dataset": self.name,
                    "beir_query_id": qid,
                    "beir_collection": self.collection_name,
                }
            )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(
            f"Wrote {len(records)} BEIR eval records to {output_path} "
            + f"(dataset={self.name}, split={self.qrels_split})"
        )
        return records

    # ------------------------------------------------------------------
    # Quick in-memory retrieval evaluation (no Qdrant required)
    # ------------------------------------------------------------------

    def eval_bm25_baseline(self, top_k: int = 10) -> dict[str, float]:
        """Evaluate a BM25-only baseline entirely in-memory (no Qdrant).

        Useful as a sanity check that the corpus + qrels loaded correctly
        before investing time in indexing. Returns macro-averaged metrics.

        Args:
            top_k: Retrieval depth.
        """
        self._require_loaded()
        assert self._corpus is not None
        assert self._queries is not None
        assert self._qrels is not None

        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("rank-bm25 is required: pip install rank-bm25") from exc

        retrieval_mod = cast(Any, import_module("eval.metrics.retrieval"))
        aggregate_metrics = retrieval_mod.aggregate_metrics
        compute_all_metrics = retrieval_mod.compute_all_metrics

        doc_ids = list(self._corpus.docs.keys())
        texts = [
            self._corpus.docs[did]["title"] + " " + self._corpus.docs[did]["text"]
            for did in doc_ids
        ]
        tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
        # directly instantiate BM25 with the tokenized corpus
        bm25_cls = cast(Any, BM25Okapi)
        bm25: Any = bm25_cls(tokenized)

        per_query: list[dict[str, float]] = []
        # Always include the caller-requested cut-off while preserving the
        # standard dashboard cut-offs.
        ks = tuple(sorted({top_k, 3, 5, 10}))
        for qid, qtext in self._queries.queries.items():
            qtokens = re.findall(r"\w+", qtext.lower())
            raw_scores = bm25.get_scores(qtokens)
            scores = [float(s) for s in cast(list[Any], raw_scores)]
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            retrieved = [doc_ids[i] for i in top_indices]
            relevant = self._qrels.relevant_doc_ids(qid)
            if relevant:
                metrics = cast(dict[str, float], compute_all_metrics(retrieved, relevant, ks=ks))
                per_query.append(metrics)

        agg = aggregate_metrics(per_query)
        logger.info(f"BM25 baseline on BEIR/{self.name}: {agg}")
        return agg

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self._corpus is None or self._queries is None or self._qrels is None:
            raise RuntimeError("Call .load() before using this method.")

    @property
    def corpus(self) -> BEIRCorpus:
        self._require_loaded()
        return self._corpus  # type: ignore[return-value]

    @property
    def queries(self) -> BEIRQueries:
        self._require_loaded()
        return self._queries  # type: ignore[return-value]

    @property
    def qrels(self) -> BEIRQRels:
        self._require_loaded()
        return self._qrels  # type: ignore[return-value]

    def summary(self) -> str:
        # only report counts when both corpus and queries are available
        if self._corpus is None or self._queries is None:
            return f"BEIRAdapter({self.name}) [not loaded]"
        q = len(self._queries.queries)
        c = len(self._corpus)
        return f"BEIRAdapter({self.name}): {c} corpus docs, {q} queries"
