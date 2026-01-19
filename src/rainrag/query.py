"""Query interface for RainRAG using Mistral/OpenAI/Claude/Gemini API and Qdrant."""

import re
from datetime import datetime, timedelta
from typing import Any, cast

import cohere
import google.generativeai as genai
from anthropic import Anthropic
from loguru import logger
from mistralai import Mistral
from openai import OpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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

    def __init__(self, config: Config):
        """
        Initialize the query engine.

        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.embedding_model: SentenceTransformer | None = None
        self.qdrant_client: QdrantClient | None = None

        # BM25 for hybrid search
        self.bm25: BM25Okapi | None = None
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
            genai.configure(api_key=config.gemini.api_key)
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

        # Load embedding model only if using local provider
        if self.config.embedding.provider == "local":
            logger.info(f"Loading local embedding model: {self.config.embedding.model_name}")
            try:
                try:
                    self.embedding_model = SentenceTransformer(
                        self.config.embedding.model_name,
                        device=self.config.embedding.device,
                        model_kwargs={"dtype": "auto"},  # Prefer new dtype kwarg when supported
                    )
                except TypeError:
                    # Older sentence-transformers versions don't accept model_kwargs
                    self.embedding_model = SentenceTransformer(
                        self.config.embedding.model_name,
                        device=self.config.embedding.device,
                    )
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
            timeout=60,
        )

        # Test connection
        try:
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
                    doc = {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "path": point.payload.get("path", ""),
                        "language": point.payload.get("language", ""),
                        "doc_id": point.payload.get("doc_id", ""),
                        "date": point.payload.get("date"),
                        "duration_seconds": point.payload.get("duration_seconds"),
                        "start_time": point.payload.get("start_time"),
                        "end_time": point.payload.get("end_time"),
                        "start_time_seconds": point.payload.get("start_time_seconds"),
                        "end_time_seconds": point.payload.get("end_time_seconds"),
                        "is_chunk": point.payload.get("is_chunk", False),
                        "chunk_index": point.payload.get("chunk_index"),
                        "total_chunks": point.payload.get("total_chunks"),
                        "video_id": point.payload.get("video_id"),
                    }
                    self.bm25_corpus.append(doc)

                    # Tokenize text for BM25 (simple whitespace + lowercase)
                    tokenized = doc["text"].lower().split()
                    self.bm25_tokenized_corpus.append(tokenized)

                offset = next_offset
                if offset is None:
                    break

            except Exception as e:
                logger.error(f"Error building BM25 index: {e}")
                raise

        # Build BM25 index
        if self.bm25_tokenized_corpus:
            self.bm25 = BM25Okapi(self.bm25_tokenized_corpus)
            logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents")
        else:
            logger.warning("No documents found for BM25 indexing")

    def _search_bm25(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """
        Search using BM25 keyword matching.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of documents with BM25 scores
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Enable hybrid_search and reinitialize.")

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Build result list
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc = self.bm25_corpus[idx].copy()
            doc["rank"] = rank
            doc["score"] = float(scores[idx])
            results.append(doc)

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
                return {doc_id: 1.0 for doc_id in scores}
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
        combined_scores: dict[str, float] = {}
        for doc_id in doc_map:
            vec_score = vector_scores.get(doc_id, 0.0)
            bm_score = bm25_scores.get(doc_id, 0.0)
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

        # Temporal keywords by category
        recent_keywords = {
            # English
            "recent",
            "recently",
            "latest",
            "last",
            "new",
            "current",
            "today",
            "yesterday",
            "this week",
            "this month",
            "this year",
            # Russian
            "недавн",
            "последн",
            "новый",
            "свежий",
            "актуальн",
            "сегодня",
            "вчера",
            "на этой неделе",
            "в этом месяце",
            "в этом году",
        }

        specific_time_patterns = [
            # Dates: 2024, 2024-01-15, January 2024, etc.
            r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",  # YYYY-MM-DD
            r"\b(20\d{2})\b",  # Just year
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(20\d{2})\b",
            r"\b(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+(20\d{2})\b",
        ]

        # Check for recent/latest keywords
        has_recent = any(keyword in query_lower for keyword in recent_keywords)

        # Check for specific time patterns
        has_specific_time = any(
            re.search(pattern, query_lower) for pattern in specific_time_patterns
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
            doc_date_str = doc.get("date")

            if doc_date_str:
                try:
                    # Parse document date
                    doc_date = datetime.strptime(doc_date_str, "%Y-%m-%d")

                    # Calculate age in days
                    age_days = (current_date - doc_date).days

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
                        f"Document from {doc_date_str} (age: {age_days} days) - "
                        f"Original score: {original_score:.4f}, Boosted score: {doc_copy['score']:.4f}, "
                        f"Time boost: {time_boost:.4f}"
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
            # Get the source chunk
            source_points = self.qdrant_client.retrieve(
                collection_name=self.config.qdrant.collection_name,
                ids=[chunk_id],
                with_vectors=True,
                with_payload=True,
            )

            if not source_points:
                logger.warning(f"Chunk not found: {chunk_id}")
                return []

            source_point = source_points[0]
            source_vector = source_point.vector
            # Ensure source_vector is a list of floats for Qdrant
            if hasattr(source_vector, 'tolist'):
                source_vector = source_vector.tolist()
            elif not isinstance(source_vector, list):
                source_vector = list(source_vector)  # type: ignore
            # Type cast for mypy - source_vector should be list[float]
            source_vector = cast(list[float], source_vector)
            source_video_id = source_point.payload.get("video_id")

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
                if hit.id == chunk_id:
                    continue

                # Filter by video_id if requested
                if same_video_only and hit.payload.get("video_id") != source_video_id:
                    continue

                chunk_data = {
                    "doc_id": hit.payload.get("doc_id", ""),
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "path": hit.payload.get("path", ""),
                    "video_id": hit.payload.get("video_id"),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "total_chunks": hit.payload.get("total_chunks"),
                    "start_time": hit.payload.get("start_time"),
                    "end_time": hit.payload.get("end_time"),
                    "start_time_seconds": hit.payload.get("start_time_seconds"),
                    "language": hit.payload.get("language", ""),
                }
                related_chunks.append(chunk_data)

                if len(related_chunks) >= top_k:
                    break

            logger.info(f"Found {len(related_chunks)} related chunks")
            return related_chunks

        except Exception as e:
            logger.error(f"Error finding related chunks: {e}")
            return []

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
                response = self.mistral_client.embeddings.create(
                    model="mistral-embed", inputs=[query]
                )
                return response.data[0].embedding
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

            return embedding.tolist()

        elif self.config.embedding.provider == "openai":
            # Use OpenAI API embeddings
            logger.debug(f"Embedding query using OpenAI API: {query[:100]}...")
            try:
                response = self.openai_client.embeddings.create(  # type: ignore[assignment]
                    model=self.config.openai.embedding_model, input=query
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Failed to generate embeddings with OpenAI API: {e}")
                raise RuntimeError(f"OpenAI embeddings API error: {e}") from e

        elif self.config.embedding.provider == "gemini":
            # Use Gemini API embeddings
            logger.debug(f"Embedding query using Gemini API: {query[:100]}...")
            try:
                result = genai.embed_content(
                    model=self.config.gemini.embedding_model,
                    content=query,
                    task_type="retrieval_query",
                )
                return result["embedding"]
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
            logger.info(f"Will apply client-side date filter: {date_from} to {date_to}")

        try:
            # If hybrid search is enabled, retrieve more candidates
            if use_hybrid:
                effective_limit = top_k * self.config.hybrid_search.top_k_multiplier
            elif date_from or date_to:
                effective_limit = top_k * 3
            else:
                effective_limit = top_k

            # 1. Vector search
            logger.info(f"Querying Qdrant with filter: {query_filter}, limit: {effective_limit}")
            results = self.qdrant_client.query_points(
                collection_name=self.config.qdrant.collection_name,
                query=query_vector,
                limit=effective_limit,
                query_filter=query_filter,
            ).points

            vector_documents = []
            for idx, hit in enumerate(results):
                doc = {
                    "rank": idx + 1,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "path": hit.payload.get("path", ""),
                    "language": hit.payload.get("language", ""),
                    "doc_id": hit.payload.get("doc_id", ""),
                    "date": hit.payload.get("date"),
                    "duration_seconds": hit.payload.get("duration_seconds"),
                    "start_time": hit.payload.get("start_time"),
                    "end_time": hit.payload.get("end_time"),
                    "start_time_seconds": hit.payload.get("start_time_seconds"),
                    "end_time_seconds": hit.payload.get("end_time_seconds"),
                    "is_chunk": hit.payload.get("is_chunk", False),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "total_chunks": hit.payload.get("total_chunks"),
                    "video_id": hit.payload.get("video_id"),
                }
                vector_documents.append(doc)
                logger.debug(f"[Vector] Rank {idx + 1}: Score={hit.score:.4f}, Path={doc['path']}")

            # 2. Hybrid search: combine with BM25 if enabled
            if use_hybrid and query_text:
                logger.info("Performing BM25 search...")
                bm25_documents = self._search_bm25(query_text, effective_limit)
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
                    doc_date_raw = doc.get("date")
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
            doc_header = f"[Document {doc['rank']}]"
            if doc.get("is_chunk"):
                doc_header += (
                    f" [Chunk {doc.get('chunk_index', 0) + 1}/{doc.get('total_chunks', 1)}]"
                )
            context_parts.append(doc_header)

            # Include date if available
            if doc.get("date"):
                context_parts.append(f"Date: {doc['date']}")

            # Include duration if available (format as mm:ss)
            if doc.get("duration_seconds"):
                mins = int(doc["duration_seconds"] // 60)
                secs = int(doc["duration_seconds"] % 60)
                context_parts.append(f"Duration: {mins}:{secs:02d}")

            # Include timecodes if available
            if doc.get("start_time") and doc.get("end_time"):
                context_parts.append(f"Timecodes: {doc['start_time']} - {doc['end_time']}")

            context_parts.append(f"Text: {text}")
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

Если дата отсутствует, укажите это. Если материал не найден — скажите прямо, не выдумывайте.""",
            "en": """You are an assistant for journalists and editors, helping them find video footage from a news archive. CRITICAL: You MUST answer ONLY in English.

ALL VIDEOS ARE ARCHIVAL RECORDINGS, not current news. Response rules:
- Always cite the recording date from metadata (the "Date" field)
- Mention video duration (the "Duration" field) — important for editors
- Use past tense: "Archive footage from 2021-05-11 shows..."
- If multiple videos are relevant, list each with date and description
- Provide rich, detailed descriptions explaining the content of each video
- Explain why the footage might be useful for the current story

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

    def generate_answer(self, messages: list[dict[str, str]]) -> str:
        """
        Generate an answer using configured LLM provider.

        Args:
            messages: List of message dictionaries for the chat

        Returns:
            The generated answer text

        Raises:
            RuntimeError: If API call fails
        """
        if self.config.llm.provider == "mistral":
            logger.info("Generating answer using Mistral API...")
            try:
                response = self.mistral_client.chat.complete(
                    model=self.config.mistral.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.config.mistral.max_tokens,
                    temperature=self.config.mistral.temperature,
                )
                answer = response.choices[0].message.content.strip()  # type: ignore[union-attr]
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Mistral API: {e}")
                raise RuntimeError(f"Mistral API error: {e}") from e

        elif self.config.llm.provider == "openai":
            logger.info("Generating answer using OpenAI API...")
            try:
                response = self.openai_client.chat.completions.create(  # type: ignore[assignment]
                    model=self.config.openai.model_name,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.config.openai.max_tokens,
                    temperature=self.config.openai.temperature,
                )
                answer = response.choices[0].message.content.strip()  # type: ignore[union-attr]
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with OpenAI API: {e}")
                raise RuntimeError(f"OpenAI API error: {e}") from e

        elif self.config.llm.provider == "claude":
            logger.info("Generating answer using Claude API...")
            try:
                # Extract system message and user messages for Claude API
                system_message = ""
                claude_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        claude_messages.append(msg)

                response = self.claude_client.messages.create(  # type: ignore[assignment]
                    model=self.config.claude.model_name,
                    max_tokens=self.config.claude.max_tokens,
                    temperature=self.config.claude.temperature,
                    system=system_message,
                    messages=claude_messages,  # type: ignore[arg-type]
                )
                answer = response.content[0].text.strip()  # type: ignore[attr-defined]
                logger.info("Answer generated successfully")
                return answer
            except Exception as e:
                logger.error(f"Failed to generate answer with Claude API: {e}")
                raise RuntimeError(f"Claude API error: {e}") from e

        elif self.config.llm.provider == "gemini":
            logger.info("Generating answer using Gemini API...")
            try:
                # Convert messages to Gemini format
                model = genai.GenerativeModel(self.config.gemini.model_name)

                # Extract system message and build conversation
                system_instruction = ""
                conversation_parts = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] == "user":
                        conversation_parts.append(msg["content"])
                    elif msg["role"] == "assistant":
                        # Gemini doesn't use explicit assistant messages in the same way
                        # For now, we'll skip assistant messages or handle them differently
                        pass

                # Combine system instruction with user message
                if system_instruction:
                    prompt = f"{system_instruction}\n\n{conversation_parts[-1]}"
                else:
                    prompt = conversation_parts[-1]

                response = model.generate_content(  # type: ignore[assignment]
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=self.config.gemini.max_tokens,
                        temperature=self.config.gemini.temperature,
                    ),
                )
                answer = response.text.strip()  # type: ignore[attr-defined]
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
    ) -> dict[str, Any]:
        """
        Execute the complete query pipeline.

        Args:
            question: The user's question
            top_k: Number of documents to retrieve (defaults to config value)
            language: Language code for response (e.g., "en", "ru")
            date_from: Filter results from this date (YYYY-MM-DD)
            date_to: Filter results up to this date (YYYY-MM-DD)

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
                    f"Detected temporal query ('{temporal_context['time_sensitivity']}'), "
                    f"applying date filter: {date_from} to {date_to}"
                )

        # Step 1: Embed the query
        query_vector = self.embed_query(question)

        # Step 2: Retrieve relevant documents with optional date filter
        # If reranking is enabled, retrieve more candidates
        retrieval_k = self.config.reranker.initial_k if self.config.reranker.enabled else top_k

        documents = self.retrieve_documents(
            query_vector, retrieval_k, date_from=date_from, date_to=date_to, query_text=question
        )

        # Step 2.5: Apply time-decay boosting if temporal context detected
        if temporal_context and temporal_context["has_temporal"]:
            documents = self._apply_time_decay_boost(
                documents, time_sensitivity=temporal_context["time_sensitivity"]
            )

        # Step 3: Rerank if enabled
        if self.config.reranker.enabled and documents:
            documents = self.rerank_documents(question, documents, top_n=top_k)

        # Step 4: Build the messages with language specification
        messages = self.build_prompt(question, documents, language=language)

        # Step 5: Generate the answer
        answer = self.generate_answer(messages)

        return {
            "question": question,
            "answer": answer,
            "retrieved_documents": documents,
            "num_documents": len(documents),
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
