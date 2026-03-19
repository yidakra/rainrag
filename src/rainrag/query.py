"""Query interface for RainRAG using Mistral/OpenAI/Claude/Gemini API and Qdrant."""

import importlib
import math as _math
import re
from datetime import datetime, timedelta
from typing import Any, cast

import cohere
import torch
from anthropic import Anthropic
from google import genai
from google.genai import types
from loguru import logger
from mistralai import Mistral
from openai import OpenAI
from qdrant_client import QdrantClient, models


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from rainrag.config import Config


class RAGQueryEngine:
    """
    RAG Query Engine that retrieves relevant documents and generates answers.

    This class handles the complete query pipeline:
    1. Embed the query using the same model as documents
    2. Search Qdrant for relevant chunks
    3. Build a prompt with retrieved context
    4. Send to Mistral API for answer generation
    """

    # Class constants for temporal keyword detection
    _RECENT_SINGLE_KEYWORDS = {
        # English single words
        "recent",
        "recently",
        "latest",
        "last",
        "new",
        "current",
        "today",
        "yesterday",
        # Russian single words
        "недавн",
        "последн",
        "новый",
        "свежий",
        "актуальн",
        "сегодня",
        "вчера",
    }

    _RECENT_PHRASES = {
        # English multi-word phrases
        "this week",
        "this month",
        "this year",
        # Russian multi-word phrases
        "на этой неделе",
        "в этом месяце",
        "в этом году",
    }

    _RECENT_SINGLE_PATTERN = re.compile(
        r"\b(?:" + "|".join(re.escape(keyword) for keyword in _RECENT_SINGLE_KEYWORDS) + r")\b",
        re.IGNORECASE,
    )

    # Pre-compiled specific time patterns
    _SPECIFIC_TIME_PATTERNS = [
        re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", re.IGNORECASE),  # YYYY-MM-DD
        re.compile(r"\b(20\d{2})\b", re.IGNORECASE),  # Just year
        re.compile(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+(20\d{2})\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, config: Config):
        """
        Initialize the query engine.

        Args:
            config: Configuration object containing all settings
        """
        super().__init__()
        self.config = config
        self.embedding_model: Any = None
        self.qdrant_client: QdrantClient | None = None

        # BM25 for hybrid search
        self.bm25: Any = None
        self.bm25_corpus: list[dict[str, Any]] = []  # Store documents for BM25
        self.bm25_tokenized_corpus: list[list[str]] = []  # Tokenized texts for BM25

        # Cohere client for reranking
        self.cohere_client: Any = None  # Can be ClientV2 or Client depending on SDK version

        # Initialize clients based on what's needed for LLM and embeddings
        needs_mistral = config.llm.provider == "mistral" or config.embedding.provider == "mistral"
        needs_openai = config.llm.provider == "openai" or config.embedding.provider == "openai"
        needs_claude = config.llm.provider == "claude"  # Claude only supports LLM, not embeddings
        needs_gemini = config.llm.provider == "gemini" or config.embedding.provider == "gemini"

        # Initialize Mistral client if needed
        if needs_mistral:
            self.mistral_client = Mistral(api_key=config.mistral.api_key)
            logger.info("Initialized Mistral client")
        else:
            self.mistral_client = None

        # Initialize OpenAI client if needed
        if needs_openai:
            self.openai_client = OpenAI(api_key=config.openai.api_key)
            logger.info("Initialized OpenAI client")
        else:
            self.openai_client = None

        # Initialize Claude client if needed
        if needs_claude:
            self.claude_client = Anthropic(api_key=config.claude.api_key)
            logger.info("Initialized Claude client")
        else:
            self.claude_client = None

        # Initialize Gemini client if needed
        if needs_gemini:
            self.genai_client = genai.Client(api_key=config.gemini.api_key)
            logger.info("Initialized Gemini client")

        # Initialize Cohere client if reranker is enabled
        if config.reranker.enabled and config.reranker.provider == "cohere":
            try:
                # Newer SDK
                self.cohere_client = cohere.ClientV2(api_key=config.cohere.api_key)
            except AttributeError:
                # Older SDK fallback
                self.cohere_client = cohere.Client(api_key=config.cohere.api_key)
            logger.info(f"Initialized Cohere reranker: {config.cohere.model_name}")
        else:
            self.cohere_client = None

        # Log which provider is being used for LLM
        if config.llm.provider == "mistral":
            logger.info(f"Using Mistral for LLM: {config.mistral.model_name}")
        elif config.llm.provider == "openai":
            logger.info(f"Using OpenAI for LLM: {config.openai.model_name}")
        elif config.llm.provider == "claude":
            logger.info(f"Using Claude for LLM: {config.claude.model_name}")
        elif config.llm.provider == "gemini":
            logger.info(f"Using Gemini for LLM: {config.gemini.model_name}")
        else:
            raise ValueError(f"Unknown LLM provider: {config.llm.provider}")

    def initialize(self) -> None:
        """Initialize the embedding model and Qdrant client."""
        logger.info("Initializing query engine...")

        def _resolve_device(configured_device: str) -> str:
            if configured_device == "auto":
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
                logger.info(f"Auto-selected device: {device}")
                return device

            if configured_device.startswith("cuda"):
                if torch.cuda.is_available():
                    return configured_device
                logger.warning(
                    f"CUDA not available, configured device '{configured_device}' not usable, falling back to CPU"
                )
                return "cpu"
            if configured_device == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                logger.warning(
                    f"MPS not available, configured device '{configured_device}' not usable, falling back to CPU"
                )
                return "cpu"
            if configured_device == "cpu":
                return "cpu"

            logger.warning(f"Unknown device '{configured_device}', falling back to CPU")
            return "cpu"

        # Load embedding model only if using local provider
        if self.config.embedding.provider == "local":
            logger.info(f"Loading local embedding model: {self.config.embedding.model_name}")
            try:
                device = _resolve_device(self.config.embedding.device)
                model_cls_any = SentenceTransformer
                if model_cls_any is None:
                    from sentence_transformers import SentenceTransformer as _SentenceTransformer

                    model_cls_any = _SentenceTransformer

                try:
                    model_cls = cast(Any, model_cls_any)
                    self.embedding_model = model_cls(
                        self.config.embedding.model_name,
                        device=device,
                        model_kwargs={"dtype": "auto"},  # Prefer new dtype kwarg when supported
                    )
                except TypeError:
                    # Older sentence-transformers versions don't accept model_kwargs
                    self.embedding_model = cast(Any, model_cls_any)(
                        self.config.embedding.model_name,
                        device=device,
                    )
                if self.embedding_model is not None:
                    self.embedding_model.max_seq_length = self.config.embedding.max_seq_length
            except OSError as e:
                # Handle offline mode / model not cached
                error_msg = (
                    f"Failed to load embedding model '{self.config.embedding.model_name}'. "
                    f"The model is not cached locally and cannot be downloaded. "
                    f"\n\nTo fix this:"
                    f"\n1. Connect to the internet"
                    f"\n2. Run: python scripts/download_models.py"
                    f"\n3. Or run: poetry run python scripts/download_models.py"
                    f"\n\nOriginal error: {e}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        elif self.config.embedding.provider == "mistral":
            logger.info("Using Mistral API for embeddings (mistral-embed)")
        elif self.config.embedding.provider == "openai":
            logger.info(f"Using OpenAI API for embeddings ({self.config.openai.embedding_model})")
        elif self.config.embedding.provider == "gemini":
            logger.info(f"Using Gemini API for embeddings ({self.config.gemini.embedding_model})")
        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding.provider}")

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {self.config.qdrant.host}:{self.config.qdrant.port}")
        # Disable version check to avoid warnings when client/server versions differ slightly
        # The HTTP API is stable and compatible across minor versions
        self.qdrant_client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            prefer_grpc=False,
            api_key=None,  # No authentication for local Qdrant
            timeout=self.config.qdrant.timeout,
        )

        # Test connection
        try:
            assert self.qdrant_client is not None
            collections = self.qdrant_client.get_collections()
            logger.info(
                f"Connected to Qdrant. Available collections: {len(collections.collections)}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

        logger.info("Query engine initialized successfully")

        # Build BM25 index if hybrid search is enabled
        if self.config.hybrid_search.enabled:
            logger.info("Hybrid search enabled. Building BM25 index...")
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in Qdrant collection."""
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client not initialized. Call initialize() first.")

        logger.info("Fetching all documents from Qdrant for BM25 indexing...")

        # Scroll through all documents in the collection
        offset = None
        batch_size = 100

        while True:
            try:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.config.qdrant.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # Don't need vectors for BM25
                )

                points, next_offset = scroll_result

                if not points:
                    break

                # Add documents to BM25 corpus
                for point in points:
                    payload = point.payload or {}  # Ensure payload is not None
                    doc = {
                        "id": point.id,
                        "text": payload.get("text", ""),
                        "path": payload.get("path", ""),
                        "language": payload.get("language", ""),
                        "doc_id": payload.get("doc_id", ""),
                        "date": payload.get("date"),
                        "duration_seconds": payload.get("duration_seconds"),
                        "start_time": payload.get("start_time"),
                        "end_time": payload.get("end_time"),
                        "start_time_seconds": payload.get("start_time_seconds"),
                        "end_time_seconds": payload.get("end_time_seconds"),
                        "is_chunk": payload.get("is_chunk", False),
                        "chunk_index": payload.get("chunk_index"),
                        "total_chunks": payload.get("total_chunks"),
                        "video_id": payload.get("video_id"),
                        "web_title": payload.get("web_title"),
                        "web_date": payload.get("web_date"),
                        "web_date_ts": payload.get("web_date_ts"),
                        "web_description": payload.get("web_description"),
                        "web_url": payload.get("web_url"),
                        "is_speech_free": payload.get("is_speech_free", False),
                    }
                    self.bm25_corpus.append(doc)

                    # Tokenize text for BM25 (language-aware regex for Russian + lowercase)
                    tokenized = re.findall(r"[\w\-а-яА-ЯёЁ]+", doc["text"].lower())
                    self.bm25_tokenized_corpus.append(tokenized)

                offset = next_offset
                if offset is None:
                    break

            except Exception as e:
                logger.error(f"Error building BM25 index: {e}")
                raise

        # Build BM25 index
        if self.bm25_tokenized_corpus:
            try:
                bm25_module = importlib.import_module("rank_bm25")
                bm25_okapi = bm25_module.BM25Okapi
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "rank-bm25 is required for BM25 search. Install with `poetry install` or `pip install rank-bm25`."
                ) from exc
            self.bm25 = bm25_okapi(self.bm25_tokenized_corpus)
            logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents")
        else:
            logger.warning("No documents found for BM25 indexing")

    def _search_bm25(
        self, query: str, top_k: int, exclude_speech_free: bool = False
    ) -> list[dict[str, Any]]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            exclude_speech_free: When True, speech-free documents are skipped.

        Returns:
            List of documents with BM25 scores
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Enable hybrid_search and reinitialize.")

        # Tokenize query
        query_tokens = re.findall(r"\b\w+\b", query.lower())

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Sort all indices by descending score; collect top_k non-excluded docs
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        rank = 1
        for idx in sorted_indices:
            if exclude_speech_free and self.bm25_corpus[idx].get("is_speech_free", False):
                continue
            doc = self.bm25_corpus[idx].copy()
            doc["rank"] = rank
            doc["score"] = float(scores[idx])
            results.append(doc)
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def _fuse_scores_rrf(
        self, vector_results: list[dict[str, Any]], bm25_results: list[dict[str, Any]], k: int = 60
    ) -> list[dict[str, Any]]:
        """
        Fuse scores using Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) for each result list

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (default: 60, standard from literature)

        Returns:
            Combined and reranked results
        """
        # Build score map: doc_id -> RRF score
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        # Add vector search scores
        for result in vector_results:
            doc_id = result["doc_id"]
            rank = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            doc_map[doc_id] = result

        # Add BM25 scores
        for result in bm25_results:
            doc_id = result["doc_id"]
            rank = result["rank"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            # Prefer vector result doc if exists, otherwise use BM25
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # Sort by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

        # Build final results
        results = []
        for rank, doc_id in enumerate(sorted_doc_ids, 1):
            doc = doc_map[doc_id].copy()
            doc["rank"] = rank
            doc["score"] = rrf_scores[doc_id]  # Replace with RRF score
            doc["fusion_method"] = "rrf"
            results.append(doc)

        return results

    def _fuse_scores_weighted(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        bm25_weight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        Fuse scores using weighted sum.

        Combined score = (1 - bm25_weight) * vector_score + bm25_weight * bm25_score

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            bm25_weight: Weight for BM25 scores (0.0-1.0)

        Returns:
            Combined and reranked results
        """
        vector_weight = 1.0 - bm25_weight

        # Normalize scores to 0-1 range for fair weighting
        def normalize_scores(results: list[dict[str, Any]]) -> dict[str, float]:
            """Normalize scores to 0-1 range."""
            scores = {r["doc_id"]: r["score"] for r in results}
            if not scores:
                return {}
            max_score = max(scores.values())
            min_score = min(scores.values())
            score_range = max_score - min_score
            if score_range == 0:
                return dict.fromkeys(scores.keys(), 1.0)
            return {doc_id: (score - min_score) / score_range for doc_id, score in scores.items()}

        vector_scores = normalize_scores(vector_results)
        bm25_scores = normalize_scores(bm25_results)

        # Build doc map
        doc_map: dict[str, dict[str, Any]] = {}
        for result in vector_results:
            doc_map[result["doc_id"]] = result
        for result in bm25_results:
            if result["doc_id"] not in doc_map:
                doc_map[result["doc_id"]] = result

        # Compute weighted scores
        baseline_score = 0.1
        combined_scores: dict[str, float] = {}
        for doc_id in doc_map:
            vec_score = vector_scores.get(doc_id, baseline_score)
            bm_score = bm25_scores.get(doc_id, baseline_score)
            combined_scores[doc_id] = vector_weight * vec_score + bm25_weight * bm_score

        # Sort by combined score
        sorted_doc_ids = sorted(
            combined_scores.keys(), key=lambda d: combined_scores[d], reverse=True
        )

        # Build final results
        results = []
        for rank, doc_id in enumerate(sorted_doc_ids, 1):
            doc = doc_map[doc_id].copy()
            doc["rank"] = rank
            doc["score"] = combined_scores[doc_id]
            doc["fusion_method"] = "weighted"
            results.append(doc)

        return results

    def _merge_variants_coverage(
        self,
        all_variant_docs: list[list[dict[str, Any]]],
        retrieval_k: int,
    ) -> list[dict[str, Any]]:
        """Greedy coverage-maximising merge inspired by VRisker (Takehi et al., WSDM 2026).

        Selects documents one at a time, each time preferring the candidate that
        covers the most *new* query variants (i.e. variants not yet represented in
        the selected set), breaking ties by raw retrieval score.  This prevents the
        majority-intent documents from crowding out candidates that uniquely satisfy
        a minority reading of the query.

        Args:
            all_variant_docs: Per-variant retrieval results (one list per variant).
            retrieval_k: Maximum number of documents to return.

        Returns:
            Ordered list of selected documents (best-coverage first).
        """
        # Map each doc_id to the set of variant indices it appears in, and
        # track the highest score seen across all variants for tiebreaking.
        coverage: dict[str, set[int]] = {}
        best_doc: dict[str, dict[str, Any]] = {}

        for i, variant_docs in enumerate(all_variant_docs):
            for doc in variant_docs:
                doc_id = doc["doc_id"]
                coverage.setdefault(doc_id, set()).add(i)
                # Keep the highest-score copy for a given doc_id so duplicate
                # documents surfaced by multiple variants preserve their best
                # retrieval score in the merged output.
                if doc_id not in best_doc or doc["score"] > best_doc[doc_id]["score"]:
                    best_doc[doc_id] = doc

        selected: list[dict[str, Any]] = []
        covered_variants: set[int] = set()
        remaining = set(best_doc.keys())

        while remaining and len(selected) < retrieval_k:
            # Primary key: new variants this doc would cover.
            # Secondary key: raw score (higher is better).
            best_id = max(
                remaining,
                key=lambda did: (
                    len(coverage.get(did, set()) - covered_variants),
                    best_doc[did]["score"],
                ),
            )
            selected.append(best_doc[best_id])
            covered_variants |= coverage.get(best_id, set())
            remaining.remove(best_id)

        return selected

    @staticmethod
    def _merge_variants_diverse_rrf(
        all_variant_docs: list[list[dict[str, Any]]],
        retrieval_k: int,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Multi-source RRF with concave diversity weighting across query variants.

        Each query variant is treated as a separate ranker in RRF.  A document that
        appears in many variants receives a *lower* per-appearance weight
        (diversity_weight = 1 / sqrt(variant_count)), giving diminishing returns to
        consensus documents and upweighting candidates that are unique to an
        under-served variant.  This implements the concave utility function from the
        VRisk framework (Takehi et al., WSDM 2026).

        Args:
            all_variant_docs: Per-variant retrieval results (one list per variant).
            retrieval_k: Maximum number of documents to return.
            rrf_k: RRF constant (default 60, standard from literature).

        Returns:
            Ordered list of documents scored by diversity-weighted RRF.
        """

        # Count how many variants each doc appears in (for diversity weighting).
        variant_count: dict[str, int] = {}
        best_doc: dict[str, dict[str, Any]] = {}

        for variant_docs in all_variant_docs:
            for doc in variant_docs:
                doc_id = doc["doc_id"]
                variant_count[doc_id] = variant_count.get(doc_id, 0) + 1
                # Prefer the first-seen document for a given doc_id to avoid
                # comparing raw scores across variants (different scales).
                if doc_id not in best_doc:
                    best_doc[doc_id] = doc

        # Compute diversity-weighted RRF score.
        # diversity_weight = 1/sqrt(variant_count) → consensus docs penalised,
        # minority-variant docs relatively upweighted.
        rrf_scores: dict[str, float] = {}
        for variant_docs in all_variant_docs:
            for doc in variant_docs:
                doc_id = doc["doc_id"]
                rank = doc["rank"]
                dw = 1.0 / _math.sqrt(variant_count[doc_id])
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + dw / (rrf_k + rank)

        sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, doc_id in enumerate(sorted_ids[:retrieval_k], 1):
            doc = best_doc[doc_id].copy()
            doc["rank"] = rank
            doc["score"] = rrf_scores[doc_id]
            doc["fusion_method"] = "diverse_rrf"
            results.append(doc)

        return results

    def _detect_temporal_keywords(self, query: str) -> dict[str, Any]:
        """
        Detect temporal keywords in query and extract time context.

        Args:
            query: User query text

        Returns:
            Dictionary with temporal context:
            - has_temporal: bool - Whether query contains temporal keywords
            - time_sensitivity: str - "recent", "latest", "specific", or "none"
            - date_from: str | None - Inferred start date
            - date_to: str | None - Inferred end date
        """
        query_lower = query.lower()

        # Check for recent/latest keywords using compiled regex and phrase matching
        has_recent = bool(self._RECENT_SINGLE_PATTERN.search(query)) or any(
            phrase in query_lower for phrase in self._RECENT_PHRASES
        )

        # Check for specific time patterns using pre-compiled regexes
        has_specific_time = any(
            pattern.search(query_lower) for pattern in self._SPECIFIC_TIME_PATTERNS
        )

        # Determine time sensitivity
        if has_recent:
            time_sensitivity = "recent"
            # Default to last 30 days for "recent" queries
            date_to = datetime.now().strftime("%Y-%m-%d")
            date_from = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        elif has_specific_time:
            time_sensitivity = "specific"
            # Try to extract specific dates (simplified - could be enhanced)
            date_from = None
            date_to = None
        else:
            time_sensitivity = "none"
            date_from = None
            date_to = None

        return {
            "has_temporal": has_recent or has_specific_time,
            "time_sensitivity": time_sensitivity,
            "date_from": date_from,
            "date_to": date_to,
        }

    def _apply_time_decay_boost(
        self, documents: list[dict[str, Any]], time_sensitivity: str = "none"
    ) -> list[dict[str, Any]]:
        """
        Apply time-decay boosting to documents based on temporal context.

        Args:
            documents: List of retrieved documents
            time_sensitivity: Time sensitivity from _detect_temporal_keywords

        Returns:
            Documents with adjusted scores based on recency
        """
        if time_sensitivity == "none" or not documents:
            return documents

        # Only apply boosting for "recent" queries
        if time_sensitivity != "recent":
            return documents

        logger.info(f"Applying time-decay boosting for '{time_sensitivity}' query...")

        current_date = datetime.now()
        boosted_docs = []

        for doc in documents:
            doc_copy = doc.copy()
            # Use prioritized date (web_date or date) for consistency
            doc_date_str = doc.get("web_date") or doc.get("date")

            if doc_date_str:
                try:
                    # Parse document date
                    doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d")

                    # Calculate age in days
                    age_days = max(0, (current_date - doc_date).days)

                    # Time decay formula: boost = 1.0 / (1.0 + age_days / decay_factor)
                    # decay_factor controls how quickly boost decreases
                    decay_factor = 30.0  # Half boost after 30 days

                    time_boost = 1.0 / (1.0 + age_days / decay_factor)

                    # Apply boost to score (multiply by time_boost)
                    original_score = doc_copy.get("score", 0.0)
                    doc_copy["score"] = original_score * (
                        0.7 + 0.3 * time_boost
                    )  # 70% original + 30% time-boosted
                    doc_copy["time_boost"] = time_boost

                    logger.debug(
                        f"Document from {doc_date_str} (age: {age_days} days) - Original score: {original_score:.4f}, Boosted score: {doc_copy['score']:.4f}, Time boost: {time_boost:.4f}"
                    )
                except ValueError:
                    logger.warning(f"Could not parse date: {doc_date_str}")

            boosted_docs.append(doc_copy)

        # Re-sort by boosted scores
        boosted_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return boosted_docs

    def find_related_chunks(
        self, chunk_id: str, top_k: int = 5, same_video_only: bool = False
    ) -> list[dict[str, Any]]:
        """
        Find chunks related to a given chunk based on vector similarity.

        Args:
            chunk_id: The ID of the source chunk
            top_k: Number of related chunks to return
            same_video_only: If True, only return chunks from the same video

        Returns:
            List of related chunks with similarity scores
        """
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client not initialized. Call initialize() first.")

        logger.info(f"Finding {top_k} related chunks for chunk_id: {chunk_id}")

        try:
            # Find the source chunk by doc_id in payload
            source_results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant.collection_name,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=chunk_id),
                        )
                    ]
                ),
                limit=1,
                with_vectors=True,
                with_payload=True,
            ).points

            if not source_results:
                logger.warning(f"Chunk not found with doc_id: {chunk_id}")
                return []

            source_point = source_results[0]
            source_vector = source_point.vector
            # Ensure source_vector is a list of floats for Qdrant
            if not isinstance(source_vector, list):
                tolist = getattr(source_vector, "tolist", None)
                if callable(tolist):
                    source_vector = tolist()
                else:
                    if source_vector is not None:
                        source_vector = cast(list[float], list(source_vector))
            # Type cast for mypy - source_vector should be list[float]
            source_vector = cast(list[float], source_vector)
            source_payload = source_point.payload or {}  # Ensure payload is not None
            source_video_id = source_payload.get("video_id")

            # Search for similar chunks
            # Retrieve more if filtering by video_id
            search_limit = top_k * 3 if same_video_only else top_k + 1

            results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant.collection_name,
                query=source_vector,
                limit=search_limit,
            ).points

            related_chunks = []
            for hit in results:
                # Skip the source chunk itself
                hit_payload = hit.payload or {}
                if hit_payload.get("doc_id") == chunk_id:
                    continue

                # Filter by video_id if requested
                if same_video_only and hit_payload.get("video_id") != source_video_id:
                    continue

                chunk_data = {
                    "doc_id": hit_payload.get("doc_id", ""),
                    "score": hit.score,
                    "text": hit_payload.get("text", ""),
                    "path": hit_payload.get("path", ""),
                    "video_id": hit_payload.get("video_id"),
                    "chunk_index": hit_payload.get("chunk_index"),
                    "total_chunks": hit_payload.get("total_chunks"),
                    "start_time": hit_payload.get("start_time"),
                    "end_time": hit_payload.get("end_time"),
                    "start_time_seconds": hit_payload.get("start_time_seconds"),
                    "language": hit_payload.get("language", ""),
                }
                related_chunks.append(chunk_data)

                if len(related_chunks) >= top_k:
                    break

            logger.info(f"Found {len(related_chunks)} related chunks")
            return related_chunks

        except Exception as e:
            logger.error(f"Error finding related chunks: {e}")
            return []

    def _rewrite_query_for_retrieval(self, query: str, language: str = "en") -> list[str]:
        """
        Stage 2a: Rewrite the user query into transcript-register variants.

        Broadcast transcripts use informal spoken language; user queries tend to be
        formal or terse.  This method uses the already-initialised LLM to generate
        paraphrases that close that register gap, improving both vector and BM25 recall.

        Args:
            query: Original user query
            language: Language code ("en" or "ru") for prompt language

        Returns:
            List starting with the original query followed by rewritten variants.
        """
        n = self.config.two_stage.query_rewrite_variants

        if language == "ru":
            prompt = (
                f"Перепиши следующий поисковый запрос {n} разными способами, "
                "чтобы он лучше совпадал с разговорной речью из видеотранскриптов новостей. "
                "Используй простые, разговорные формулировки, как в реальных репортажах. "
                f"Верни ровно {n} варианта — каждый на отдельной строке, без нумерации и пояснений.\n\n"
                f"Запрос: {query}"
            )
        else:
            prompt = (
                f"Rewrite the following search query in {n} different ways "
                "so that it better matches spoken language found in broadcast news transcripts. "
                "Use informal, conversational phrasing similar to how reporters speak on air. "
                f"Return exactly {n} variants — one per line, no numbering or explanation.\n\n"
                f"Query: {query}"
            )

        logger.info(f"[Two-Stage] Rewriting query into {n} transcript-register variants...")
        try:
            messages = [{"role": "user", "content": prompt}]
            raw = self.generate_answer(
                messages,
                temperature=self.config.two_stage.query_rewrite_temperature,
            )
            variants = [line.strip() for line in raw.splitlines() if line.strip()][:n]
            logger.debug(f"[Two-Stage] Query variants: {variants}")
        except Exception as e:
            logger.warning(f"[Two-Stage] Query rewrite failed, using original only: {e}")
            variants = []

        return [query] + variants

    def _generate_hyde_embedding(self, query: str, language: str = "en") -> list[float]:
        """
        Stage 2b: Hypothetical Document Embedding (HyDE).

        Generates a hypothetical broadcast transcript excerpt that would answer the
        query, then embeds it.  The caller blends this with the raw query embedding
        using hyde_alpha as the interpolation weight, analogous to Zhai & Lafferty's
        Stage 2 query-model interpolation.

        Args:
            query: Original user query
            language: Language code ("en" or "ru")

        Returns:
            Embedding vector for the hypothetical document.
        """
        if language == "ru":
            prompt = (
                "Напиши короткий отрывок (3–5 предложений) из репортажа или "
                "новостного видеотранскрипта, который напрямую отвечает на следующий вопрос. "
                "Используй разговорный стиль, характерный для телевизионных новостей. "
                "Возвращай только текст отрывка, без заголовков и пояснений.\n\n"
                f"Вопрос: {query}"
            )
        else:
            prompt = (
                "Write a short passage (3–5 sentences) from a broadcast news video transcript "
                "that directly answers the following question. "
                "Use informal spoken language typical of on-air reporting. "
                "Return only the passage text, no headings or explanation.\n\n"
                f"Question: {query}"
            )

        logger.info("[Two-Stage] Generating HyDE hypothetical document...")
        try:
            messages = [{"role": "user", "content": prompt}]
            hypothetical_doc = self.generate_answer(
                messages,
                temperature=self.config.two_stage.hyde_temperature,
            )
            logger.debug(f"[Two-Stage] HyDE passage: {hypothetical_doc[:120]}...")
            return self.embed_query(hypothetical_doc)
        except Exception as e:
            logger.warning(
                f"[Two-Stage] HyDE generation failed, falling back to query embedding: {e}"
            )
            return self.embed_query(query)

    def embed_query(self, query: str) -> list[float]:
        """
        Embed the query text using configured provider.

        Args:
            query: The query text

        Returns:
            List of floats representing the query embedding
        """
        if self.config.embedding.provider == "mistral":
            # Use Mistral API embeddings
            logger.debug(f"Embedding query using Mistral API: {query[:100]}...")
            try:
                assert self.mistral_client is not None, "Mistral client not initialized"
                response = self.mistral_client.embeddings.create(
                    model="mistral-embed", inputs=[query]
                )
                embedding = response.data[0].embedding
                if embedding is None:
                    raise RuntimeError("Mistral embeddings API returned None embedding")
                return embedding
            except Exception as e:
                logger.error(f"Failed to generate embeddings with Mistral API: {e}")
                raise RuntimeError(f"Mistral embeddings API error: {e}") from e

        elif self.config.embedding.provider == "local":
            # Use local SentenceTransformer model
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized. Call initialize() first.")

            # Add "query: " prefix for E5 model (improves retrieval performance)
            prefixed_query = f"query: {query}"

            logger.debug(f"Embedding query using local model: {query[:100]}...")
            embedding = self.embedding_model.encode(
                prefixed_query,
                normalize_embeddings=self.config.embedding.normalize_embeddings,
            )

            embedding_list = getattr(embedding, "tolist", lambda: list(embedding))()
            return cast(list[float], embedding_list)

        elif self.config.embedding.provider == "openai":
            # Use OpenAI API embeddings
            logger.debug(f"Embedding query using OpenAI API: {query[:100]}...")
            try:
                assert self.openai_client is not None, "OpenAI client not initialized"
                openai_response = self.openai_client.embeddings.create(
                    model=self.config.openai.embedding_model, input=query
                )
                return openai_response.data[0].embedding
            except Exception as e:
                logger.error(f"Failed to generate embeddings with OpenAI API: {e}")
                raise RuntimeError(f"OpenAI embeddings API error: {e}") from e

        elif self.config.embedding.provider == "gemini":
            # Use Gemini API embeddings
            logger.debug(f"Embedding query using Gemini API: {query[:100]}...")
            try:
                result = self.genai_client.models.embed_content(
                    model=self.config.gemini.embedding_model,
                    contents=[query],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                if result and result.embeddings and len(result.embeddings) > 0:
                    embedding = result.embeddings[0].values
                    if embedding is not None:
                        return embedding
                    else:
                        raise RuntimeError("Gemini embeddings API returned None values")
                else:
                    raise RuntimeError("Gemini embeddings API returned invalid response")
            except Exception as e:
                logger.error(f"Failed to generate embeddings with Gemini API: {e}")
                raise RuntimeError(f"Gemini embeddings API error: {e}") from e

        else:
            raise ValueError(f"Unknown embedding provider: {self.config.embedding.provider}")

    def retrieve_documents(
        self,
        query_vector: list[float],
        top_k: int,
        date_from: str | None = None,
        date_to: str | None = None,
        query_text: str | None = None,
        exclude_speech_free: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant documents from Qdrant.

        Supports hybrid search (vector + BM25) if enabled in config.

        Args:
            query_vector: The query embedding
            top_k: Number of documents to retrieve
            date_from: Filter results from this date (YYYY-MM-DD)
            date_to: Filter results up to this date (YYYY-MM-DD)
            query_text: Original query text (needed for BM25 in hybrid search)
            exclude_speech_free: When True, speech-free (no-transcript) documents
                are removed from results before ranking and truncation.

        Returns:
            List of retrieved documents with metadata
        """
        if self.qdrant_client is None:
            raise RuntimeError("Qdrant client not initialized. Call initialize() first.")

        # Check if hybrid search is enabled and query_text is provided
        use_hybrid = self.config.hybrid_search.enabled and query_text and self.bm25 is not None

        if use_hybrid:
            logger.info(f"Using hybrid search (vector + BM25) for top {top_k} documents...")
        else:
            logger.info(f"Using vector search for top {top_k} documents...")

        # Build date filter if specified - we'll do client-side filtering
        # to avoid Qdrant issues with None date values
        query_filter = None
        if date_from or date_to:
            logger.info(
                f"No server-side filter applied; applying client-side date filter: {date_from} to {date_to}"
            )

        try:
            # If hybrid search is enabled, retrieve more candidates
            if use_hybrid:
                effective_limit = top_k * self.config.hybrid_search.top_k_multiplier
            elif date_from or date_to:
                effective_limit = top_k * 3
            else:
                effective_limit = top_k

            # 1. Vector search
            if date_from or date_to:
                logger.info(
                    f"Querying Qdrant with no server-side filter (client-side date filtering will be applied), limit: {effective_limit}"
                )
            else:
                logger.info(
                    f"Querying Qdrant with filter: {query_filter}, limit: {effective_limit}"
                )
            results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant.collection_name,
                query=query_vector,
                limit=effective_limit,
                query_filter=query_filter,
            ).points

            vector_documents = []
            for idx, hit in enumerate(results):
                hit_payload = hit.payload or {}  # Ensure payload is not None
                doc = {
                    "rank": idx + 1,
                    "score": hit.score,
                    "text": hit_payload.get("text", ""),
                    "path": hit_payload.get("path", ""),
                    "language": hit_payload.get("language", ""),
                    "doc_id": hit_payload.get("doc_id", ""),
                    "date": hit_payload.get("date"),
                    "duration_seconds": hit_payload.get("duration_seconds"),
                    "start_time": hit_payload.get("start_time"),
                    "end_time": hit_payload.get("end_time"),
                    "start_time_seconds": hit_payload.get("start_time_seconds"),
                    "end_time_seconds": hit_payload.get("end_time_seconds"),
                    "is_chunk": hit_payload.get("is_chunk", False),
                    "chunk_index": hit_payload.get("chunk_index"),
                    "total_chunks": hit_payload.get("total_chunks"),
                    "video_id": hit_payload.get("video_id"),
                    "web_title": hit_payload.get("web_title"),
                    "web_date": hit_payload.get("web_date"),
                    "web_date_ts": hit_payload.get("web_date_ts"),
                    "web_description": hit_payload.get("web_description"),
                    "web_url": hit_payload.get("web_url"),
                    "is_speech_free": hit_payload.get("is_speech_free", False),
                }
                vector_documents.append(doc)
                logger.debug(f"[Vector] Rank {idx + 1}: Score={hit.score:.4f}, Path={doc['path']}")

            # 2. Hybrid search: combine with BM25 if enabled
            if use_hybrid and query_text:
                logger.info("Performing BM25 search...")
                bm25_documents = self._search_bm25(
                    query_text, effective_limit, exclude_speech_free=exclude_speech_free
                )
                logger.info(f"Retrieved {len(bm25_documents)} BM25 results")

                # Fuse scores
                if self.config.hybrid_search.fusion_method == "rrf":
                    logger.info("Fusing scores with RRF...")
                    documents = self._fuse_scores_rrf(
                        vector_documents, bm25_documents, k=self.config.hybrid_search.rrf_k
                    )
                else:  # weighted
                    logger.info("Fusing scores with weighted sum...")
                    documents = self._fuse_scores_weighted(
                        vector_documents,
                        bm25_documents,
                        bm25_weight=self.config.hybrid_search.bm25_weight,
                    )
                logger.info(f"Hybrid search produced {len(documents)} fused results")
            else:
                documents = vector_documents

            # Remove speech-free docs if caller does not want them
            if exclude_speech_free:
                before = len(documents)
                documents = [d for d in documents if not d.get("is_speech_free", False)]
                logger.info(
                    f"Excluded {before - len(documents)} speech-free documents (exclude_speech_free=True)"
                )

            # Apply date filtering client-side if requested
            if date_from or date_to:
                from datetime import date as _date
                from datetime import datetime as _dt

                def _parse_date(value: Any) -> _date | None:
                    """Best-effort parse for ISO date strings or datetime/date objects."""
                    if value is None:
                        return None
                    if isinstance(value, _dt):
                        return value.date()
                    if isinstance(value, _date):
                        return value
                    try:
                        return _dt.fromisoformat(str(value)).date()
                    except Exception:
                        return None

                parsed_from = _parse_date(date_from)
                parsed_to = _parse_date(date_to)

                filtered_documents = []
                for doc in documents:
                    # Prioritize web_date over regular date for filtering
                    doc_date_raw = doc.get("web_date") or doc.get("date")
                    doc_date = _parse_date(doc_date_raw)
                    if doc_date is None:
                        continue  # Skip documents without dates

                    include_doc = True
                    if parsed_from and doc_date < parsed_from:
                        include_doc = False
                    if parsed_to and doc_date > parsed_to:
                        include_doc = False

                    if include_doc:
                        filtered_documents.append(doc)

                # Re-rank and limit to top_k
                documents = filtered_documents[:top_k]

                # Update ranks after filtering
                for idx, doc in enumerate(documents):
                    doc["rank"] = idx + 1

            logger.info(f"Retrieved {len(documents)} documents after filtering")
            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise

    def _apply_score_threshold(
        self, documents: list[dict[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        """Drop documents whose score falls below *threshold*.

        Motivated by Cuconasu et al. (SIGIR 2024): near-miss documents that
        score highly from retrieval but are not truly relevant hurt LLM answer
        quality more than simply having fewer documents.  A threshold of 0.0
        (the default) is a no-op.
        """
        if threshold <= 0.0:
            return documents
        return [d for d in documents if d.get("score", 0.0) >= threshold]

    def _order_documents_for_prompt(
        self, documents: list[dict[str, Any]], order: str
    ) -> list[dict[str, Any]]:
        """Re-order *documents* for prompt assembly according to *order*.

        Args:
            documents: Documents in current rank order (best first).
            order: One of ``"rank"`` (no change), ``"reversed"`` (worst first,
                best last — exploits LLM recency bias), or ``"book_end"``
                (best document first, second-best last, remainder in the
                middle — combats 'lost in the middle' attention drop).

        Returns:
            Re-ordered list.  The ``rank`` field is updated so that
            ``[Document N]`` labels in the prompt reflect the new positions.
        """
        if order == "rank" or len(documents) <= 1:
            return documents

        if order == "reversed":
            ordered = list(reversed(documents))
        elif order == "book_end":
            if len(documents) == 2:
                ordered = list(documents)  # already book-ended, but copy to avoid input mutation
            else:
                first = documents[0]
                second_best = documents[1]
                # Keep all remaining documents (including the original last) in order,
                # but remove the second-best from its original position.
                remaining = documents[2:]
                ordered = [first] + remaining + [second_best]
        else:
            logger.warning("Unknown prompt_doc_order %r; falling back to 'rank'", order)
            return documents

        # Re-number so [Document N] labels match the new positions
        for i, doc in enumerate(ordered, start=1):
            doc = dict(doc)
            doc["rank"] = i
            ordered[i - 1] = doc
        return ordered

    def rerank_documents(
        self, query: str, documents: list[dict[str, Any]], top_n: int
    ) -> list[dict[str, Any]]:
        """
        Rerank documents using Cohere Rerank API.

        Args:
            query: The user's query
            documents: List of retrieved documents
            top_n: Number of documents to return after reranking

        Returns:
            Reranked list of documents
        """
        if not self.cohere_client:
            logger.warning("Cohere client not initialized, skipping reranking")
            return documents[:top_n]

        if not documents:
            return documents

        logger.info(f"Reranking {len(documents)} documents with Cohere...")

        try:
            # Prepare documents for reranking
            texts = [doc["text"] for doc in documents]

            # Call Cohere Rerank API
            response = self.cohere_client.rerank(
                model=self.config.cohere.model_name,
                query=query,
                documents=texts,
                top_n=min(top_n, len(documents)),
            )

            # Reorder documents based on rerank results
            reranked = []
            for idx, result in enumerate(response.results):
                doc = documents[result.index].copy()
                doc["rank"] = idx + 1
                doc["rerank_score"] = result.relevance_score
                doc["original_score"] = doc["score"]
                doc["score"] = result.relevance_score  # Use rerank score as primary
                reranked.append(doc)

            logger.info(f"Reranking complete. Returning top {len(reranked)} documents")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original documents.")
            return documents[:top_n]

    def build_prompt(
        self, query: str, documents: list[dict[str, Any]], language: str = "en"
    ) -> list[dict[str, str]]:
        """
        Build the messages for the chat LLM with retrieved context.

        Args:
            query: The user's question
            documents: List of retrieved documents
            language: Language code (e.g., "en", "ru") for response

        Returns:
            List of message dictionaries for the chat API
        """
        # Build context from retrieved documents
        context_parts = []
        # Allow more context per document
        max_chars_per_doc = 2000
        for doc in documents:
            text = doc["text"]
            if len(text) > max_chars_per_doc:
                text = text[:max_chars_per_doc].rstrip() + "..."

            # Document header
            is_speech_free = doc.get("is_speech_free", False)
            doc_header = f"[Document {doc['rank']}]"
            if is_speech_free:
                doc_header += " [No transcript — description only]"
            elif doc.get("is_chunk"):
                doc_header += (
                    f" [Chunk {doc.get('chunk_index', 0) + 1}/{doc.get('total_chunks', 1)}]"
                )
            context_parts.append(doc_header)

            # Include date if available (prioritize web_date over date)
            doc_date = doc.get("web_date") or doc.get("date")
            if doc_date:
                context_parts.append(f"Date: {doc_date}")

            # Include duration if available (format as mm:ss)
            if doc.get("duration_seconds"):
                mins = int(doc["duration_seconds"] // 60)
                secs = int(doc["duration_seconds"] % 60)
                context_parts.append(f"Duration: {mins}:{secs:02d}")

            # Include timecodes only for transcript documents
            if not is_speech_free and doc.get("start_time") and doc.get("end_time"):
                context_parts.append(f"Timecodes: {doc['start_time']} - {doc['end_time']}")

            # Label the text block so the LLM knows what it is reading
            text_label = "Description" if is_speech_free else "Text"
            context_parts.append(f"{text_label}: {text}")
            context_parts.append("")  # Empty line between documents

        context = "\n".join(context_parts)

        # Language-specific system messages
        system_messages = {
            "ru": """Вы — ассистент для журналистов и редакторов, помогающий находить видеоматериалы в новостном архиве. КРИТИЧЕСКИ ВАЖНО: Вы ДОЛЖНЫ отвечать ТОЛЬКО на русском языке.

ВСЕ ВИДЕО — АРХИВНЫЕ ЗАПИСИ, не текущие новости. Правила ответа:
- Всегда указывайте дату записи из метаданных (поле "Date")
- Упоминайте длительность видео (поле "Duration") — важно для редакторов
- Используйте прошедшее время: "В архивном видео от 2021-05-11 показано..."
- Если несколько видео релевантны, перечислите каждое с датой и описанием
- Делайте хорошее, развернутое "Описание" для каждого релевантного видео
- Объясните, почему материал может быть полезен для текущего сюжета
- Некоторые видео могут быть помечены как "[No transcript — description only]" — это означает, что в видео нет речи (музыка, тишина). Для таких видео используйте только поле "Description", не ссылайтесь на тайм-коды

Если дата отсутствует, укажите это. Если материал не найден — скажите прямо, не выдумывайте.""",
            "en": """You are an assistant for journalists and editors, helping them find video footage from a news archive. CRITICAL: You MUST answer ONLY in English.

ALL VIDEOS ARE ARCHIVAL RECORDINGS, not current news. Response rules:
- Always cite the recording date from metadata (the "Date" field)
- Mention video duration (the "Duration" field) — important for editors
- Use past tense: "Archive footage from 2021-05-11 shows..."
- If multiple videos are relevant, list each with date and description
- Provide rich, detailed descriptions explaining the content of each video
- Explain why the footage might be useful for the current story
- Some videos may be marked "[No transcript — description only]" — these contain no speech (music, silence, etc.). For these, rely only on the "Description" field and do not reference timecodes

If date is missing, note this. If no relevant footage is found, say so clearly — do not fabricate.""",
        }

        # Get system message (default to English if not found)
        system_message = system_messages.get(language, system_messages["en"])

        # Build user message with context and question
        user_message = f"""Context from video transcripts:
{context}

Question: {query}"""

        # Return messages in provider-agnostic chat format
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def generate_answer(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> str:
        """
        Generate an answer using configured LLM provider.

        Args:
            messages: List of message dictionaries for the chat
            temperature: Override the provider's configured temperature. Pass a value
                here when you need a different temperature for an intermediate step
                (e.g. query rewriting) while keeping the main answer at the configured
                temperature (typically 0 for deterministic journalist output).

        Returns:
            The generated answer text

        Raises:
            RuntimeError: If API call fails
        """
        if self.config.llm.provider == "mistral":
            logger.info("Generating answer using Mistral API...")
            try:
                assert self.mistral_client is not None, "Mistral client not initialized"
                mistral_response: Any = self.mistral_client.chat.complete(
                    model=self.config.mistral.model_name,
                    messages=cast(Any, messages),
                    max_tokens=self.config.mistral.max_tokens,
                    temperature=temperature
                    if temperature is not None
                    else self.config.mistral.temperature,
                )
                if (
                    mistral_response
                    and mistral_response.choices
                    and len(mistral_response.choices) > 0
                    and mistral_response.choices[0].message
                ):
                    content = mistral_response.choices[0].message.content
                    if isinstance(content, str):
                        answer = content.strip()
                    elif isinstance(content, list) and content:
                        # Handle list of content chunks (e.g., from Mistral API)
                        answer = "".join(str(chunk) for chunk in content).strip()
                    else:
                        raise RuntimeError("Mistral API returned invalid content format")
                else:
                    raise RuntimeError("Mistral API returned invalid response structure")
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Mistral API: {e}")
                raise RuntimeError(f"Mistral API error: {e}") from e

        elif self.config.llm.provider == "openai":
            logger.info("Generating answer using OpenAI API...")
            try:
                assert self.openai_client is not None, "OpenAI client not initialized"
                openai_response: Any = self.openai_client.chat.completions.create(
                    model=self.config.openai.model_name,
                    messages=cast(Any, messages),
                    max_tokens=self.config.openai.max_tokens,
                    temperature=temperature
                    if temperature is not None
                    else self.config.openai.temperature,
                )
                if (
                    openai_response
                    and openai_response.choices
                    and len(openai_response.choices) > 0
                    and openai_response.choices[0].message
                ):
                    content = openai_response.choices[0].message.content
                    if isinstance(content, str):
                        answer = content.strip()
                    elif isinstance(content, list) and content:
                        # Handle list of content chunks (if OpenAI API returns list)
                        answer = "".join(str(chunk) for chunk in content).strip()
                    else:
                        raise RuntimeError("OpenAI API returned invalid content format")
                else:
                    raise RuntimeError("OpenAI API returned invalid response structure")
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with OpenAI API: {e}")
                raise RuntimeError(f"OpenAI API error: {e}") from e

        elif self.config.llm.provider == "claude":
            logger.info("Generating answer using Claude API...")
            try:
                assert self.claude_client is not None, "Claude client not initialized"
                # Extract system message and user messages for Claude API
                system_message = ""
                claude_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        claude_messages.append(msg)

                claude_response: Any = self.claude_client.messages.create(
                    model=self.config.claude.model_name,
                    max_tokens=self.config.claude.max_tokens,
                    temperature=temperature
                    if temperature is not None
                    else self.config.claude.temperature,
                    system=system_message,
                    messages=cast(Any, claude_messages),
                )
                if claude_response and claude_response.content and len(claude_response.content) > 0:
                    content_block = claude_response.content[0]
                    # Check if it's a text block with text attribute
                    try:
                        text_content = getattr(content_block, "text", None)
                        if text_content:
                            if isinstance(text_content, str):
                                answer = text_content.strip()
                            elif isinstance(text_content, list) and text_content:
                                # Handle list of text chunks (if Claude API returns list)
                                answer = "".join(str(chunk) for chunk in text_content).strip()
                            else:
                                raise RuntimeError("Claude API returned invalid text format")
                        else:
                            raise RuntimeError("Claude API returned non-text content block")
                    except AttributeError:
                        raise RuntimeError(
                            "Claude API returned content block without text attribute"
                        )
                else:
                    raise RuntimeError("Claude API returned invalid response structure")
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Claude API: {e}")
                raise RuntimeError(f"Claude API error: {e}") from e

        elif self.config.llm.provider == "gemini":
            logger.info("Generating answer using Gemini API...")
            try:
                # Convert messages to Gemini format
                contents = []
                system_instruction = None

                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] == "user":
                        contents.append(
                            types.Content(role="user", parts=[types.Part(text=msg["content"])])
                        )
                    elif msg["role"] == "assistant":
                        contents.append(
                            types.Content(role="model", parts=[types.Part(text=msg["content"])])
                        )

                # If there's a system instruction, add it to the first user message
                if system_instruction and contents:
                    first_content = contents[0]
                    if first_content.role == "user" and first_content.parts:
                        first_content.parts[
                            0
                        ].text = f"{system_instruction}\n\n{first_content.parts[0].text}"

                response = self.genai_client.models.generate_content(
                    model=self.config.gemini.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self.config.gemini.max_tokens,
                        temperature=temperature
                        if temperature is not None
                        else self.config.gemini.temperature,
                    ),
                )
                if response and response.text:
                    answer = response.text.strip()
                else:
                    raise RuntimeError("Gemini API returned invalid response")
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Gemini API: {e}")
                raise RuntimeError(f"Gemini API error: {e}") from e

        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm.provider}")

    def query(
        self,
        question: str,
        top_k: int | None = None,
        language: str = "en",
        date_from: str | None = None,
        date_to: str | None = None,
        exclude_speech_free: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the complete query pipeline.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve (defaults to config value)
            language: Language code for response (e.g., "en", "ru")
            date_from: Filter results from this date (YYYY-MM-DD)
            date_to: Filter results up to this date (YYYY-MM-DD)
            exclude_speech_free: When True, videos with no transcript are excluded
                from retrieval results entirely.

        Returns:
            Dictionary containing the answer and metadata
        """
        if top_k is None:
            # Get top_k from the appropriate LLM config
            if self.config.llm.provider == "mistral":
                top_k = self.config.mistral.top_k
            elif self.config.llm.provider == "openai":
                top_k = self.config.openai.top_k
            elif self.config.llm.provider == "claude":
                top_k = self.config.claude.top_k
            elif self.config.llm.provider == "gemini":
                top_k = self.config.gemini.top_k
            else:
                top_k = 5  # Default fallback

        # At this point, top_k is guaranteed to be an int
        assert isinstance(top_k, int)

        logger.info(f"Processing query: {question[:100]}... (language: {language})")

        # Step 0: Detect temporal context (if no explicit date filter provided)
        temporal_context = None
        if not date_from and not date_to:
            temporal_context = self._detect_temporal_keywords(question)
            if (
                temporal_context["has_temporal"]
                and temporal_context["time_sensitivity"] == "recent"
            ):
                # Use detected date range for "recent" queries
                date_from = temporal_context.get("date_from")
                date_to = temporal_context.get("date_to")
                logger.info(
                    f"Detected temporal query ('{temporal_context['time_sensitivity']}'), applying date filter: {date_from} to {date_to}"
                )

        # Step 1: Build query variants (Stage 2a: LLM query rewriting)
        two_stage_enabled = self.config.two_stage.enabled
        if two_stage_enabled and self.config.two_stage.query_rewrite_enabled:
            query_variants = self._rewrite_query_for_retrieval(question, language)
        else:
            query_variants = [question]

        # Step 2: Embed primary query, then optionally blend with HyDE (Stage 2b)
        primary_vector = self.embed_query(question)
        embed_calls = 1

        if two_stage_enabled and self.config.two_stage.hyde_enabled:
            import numpy as np

            hyde_vector = self._generate_hyde_embedding(question, language)
            alpha = self.config.two_stage.hyde_alpha
            blended = (1.0 - alpha) * np.array(primary_vector) + alpha * np.array(hyde_vector)
            norm = float(np.linalg.norm(blended))
            if norm > 0:
                blended = blended / norm
            primary_vector = blended.tolist()
            logger.info(f"[Two-Stage] HyDE blend applied (alpha={alpha})")

        # Step 3: Retrieve relevant documents with optional date filter.
        # If reranking is enabled, retrieve more candidates initially.
        retrieval_k = self.config.reranker.initial_k if self.config.reranker.enabled else top_k
        assert isinstance(retrieval_k, int)

        # Per-variant retrieval result ID lists for downstream intent-coverage eval.
        variant_retrieved_ids: list[list[str]] = []

        if two_stage_enabled and len(query_variants) > 1:
            # Retrieve for each variant.  The primary_vector (possibly HyDE-blended)
            # is reused for the original query via index-based check.
            all_variant_docs: list[list[dict[str, Any]]] = []
            for i, variant in enumerate(query_variants):
                if i == 0:
                    variant_vector = primary_vector
                else:
                    variant_vector = self.embed_query(variant)
                    embed_calls += 1
                variant_docs = self.retrieve_documents(
                    variant_vector,
                    retrieval_k,
                    date_from=date_from,
                    date_to=date_to,
                    query_text=variant,
                    exclude_speech_free=exclude_speech_free,
                )
                all_variant_docs.append(variant_docs)
                variant_retrieved_ids.append([d["doc_id"] for d in variant_docs])

            # Merge using the configured strategy.
            merge_strategy = self.config.two_stage.merge_strategy
            if merge_strategy == "diverse_rrf":
                documents = self._merge_variants_diverse_rrf(
                    all_variant_docs,
                    retrieval_k,
                    rrf_k=self.config.two_stage.merge_rrf_k,
                )
            else:
                # Default: greedy coverage-maximising merge (VRisker-style)
                documents = self._merge_variants_coverage(all_variant_docs, retrieval_k)

            # Re-number ranks after merge
            for idx, doc in enumerate(documents):
                doc["rank"] = idx + 1
            logger.info(
                f"[Two-Stage] Merged {len(documents)} unique docs from {len(query_variants)} query variants (strategy={merge_strategy})"
            )
        else:
            documents = self.retrieve_documents(
                primary_vector,
                retrieval_k,
                date_from=date_from,
                date_to=date_to,
                query_text=question,
                exclude_speech_free=exclude_speech_free,
            )
            variant_retrieved_ids = [[d["doc_id"] for d in documents]]

        # Step 4: Apply time-decay boosting if temporal context detected
        if temporal_context and temporal_context["has_temporal"]:
            documents = self._apply_time_decay_boost(
                documents, time_sensitivity=temporal_context["time_sensitivity"]
            )

        # Step 5: Rerank if enabled
        if self.config.reranker.enabled and documents:
            documents = self.rerank_documents(question, documents, top_n=top_k)

        # Step 5b: Drop documents below the minimum score threshold
        min_score = self.config.reranker.min_retrieval_score
        if min_score > 0.0:
            before = len(documents)
            documents = self._apply_score_threshold(documents, min_score)
            if len(documents) < before:
                logger.info(
                    "Score threshold %.3f dropped %d/%d documents",
                    min_score,
                    before - len(documents),
                    before,
                )

        # Step 5c: Re-order documents for prompt assembly
        doc_order = self.config.two_stage.prompt_doc_order
        documents = self._order_documents_for_prompt(documents, doc_order)

        # Step 6: Build the messages with language specification
        messages = self.build_prompt(question, documents, language=language)

        # Step 7: Generate the answer
        answer = self.generate_answer(messages)

        # Track API call counts for transparency (cost estimates currently
        # only include the main answer generation + embedding). These counts
        # enable downstream cost calculations in the MLflow metrics.
        llm_query_rewrite_calls = (
            1 if two_stage_enabled and self.config.two_stage.query_rewrite_enabled else 0
        )
        llm_hyde_calls = 1 if two_stage_enabled and self.config.two_stage.hyde_enabled else 0
        llm_calls = 1 + llm_query_rewrite_calls + llm_hyde_calls

        reranker_calls = 1 if self.config.reranker.enabled and bool(documents) else 0

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": documents,
            "num_documents": len(documents),
            "query_variants": query_variants,
            "variant_retrieved_ids": variant_retrieved_ids,
            "cost.llm_calls_count": llm_calls,
            "cost.llm_query_rewrite_calls": llm_query_rewrite_calls,
            "cost.llm_hyde_calls": llm_hyde_calls,
            "cost.embed_calls_count": embed_calls,
            "cost.reranker_calls_count": reranker_calls,
        }


def run_query(config_path: str, question: str, top_k: int | None = None) -> dict[str, Any]:
    """
    Run a query against the RAG system.

    This is a convenience function for the CLI.

    Args:
        config_path: Path to configuration file
        question: The user's question
        top_k: Number of documents to retrieve

    Returns:
        Dictionary containing the answer and metadata
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    engine = RAGQueryEngine(config)
    engine.initialize()

    return engine.query(question, top_k)
