from __future__ import annotations


"""Embedding generation module using multilingual-e5-large."""

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from loguru import logger

from rainrag.config import Config
from rainrag.ingest import Document


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerType


class EmbeddingCache:
    """Cache for storing and loading embeddings."""

    def __init__(self, cache_dir: str):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Path to the cache directory
        """
        super().__init__()
        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback_dir = Path.home() / ".cache" / "rainrag" / "embeddings"
            logger.warning(
                "Embedding cache path {!r} is not writable; falling back to {!r}",
                str(self.cache_dir),
                str(fallback_dir),
            )
            try:
                fallback_dir.mkdir(parents=True, exist_ok=True)
                self.cache_dir = fallback_dir
            except PermissionError:
                raise PermissionError(
                    f"Cannot create embedding cache directory: neither {self.cache_dir} nor {fallback_dir} is writable"
                )
        self.embeddings_file = self.cache_dir / "embeddings.npy"
        self.metadata_file = self.cache_dir / "metadata.jsonl"

    def save(self, embeddings: np.ndarray, documents: list[Document]) -> None:
        """
        Save embeddings and metadata to cache.

        Args:
            embeddings: NumPy array of embeddings (shape: [N, D])
            documents: List of documents corresponding to embeddings
        """
        logger.info(f"Saving {len(embeddings)} embeddings to cache")

        # Save embeddings as NumPy array
        np.save(self.embeddings_file, embeddings)
        logger.info(f"Embeddings saved to {self.embeddings_file}")

        # Save metadata as JSONL
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            for doc in documents:
                json_line = doc.model_dump_json()
                f.write(json_line + "\n")

        logger.info(f"Metadata saved to {self.metadata_file}")

    def load(self) -> tuple[np.ndarray | None, list[Document] | None]:
        """
        Load embeddings and metadata from cache.

        Returns:
            Tuple of (embeddings array, list of documents) or (None, None) if cache doesn't exist
        """
        if not self.embeddings_file.exists() or not self.metadata_file.exists():
            logger.warning("Cache files not found")
            return None, None

        logger.info("Loading embeddings from cache")

        # Load embeddings
        embeddings = np.load(self.embeddings_file)
        logger.info(f"Loaded {len(embeddings)} embeddings")

        # Load metadata
        documents = []
        with open(self.metadata_file, encoding="utf-8") as f:
            for line in f:
                doc_dict = json.loads(line)
                documents.append(Document(**doc_dict))

        logger.info(f"Loaded metadata for {len(documents)} documents")

        # Validate that embeddings and documents have matching lengths
        if len(embeddings) != len(documents):
            logger.error(
                f"Embeddings/metadata mismatch in load(): embeddings count ({len(embeddings)}) != documents count ({len(documents)}), embeddings_file: {self.embeddings_file}, metadata_file: {self.metadata_file}"
            )
            return None, None

        return embeddings, documents

    def exists(self) -> bool:
        """
        Check if cache exists.

        Returns:
            True if cache files exist
        """
        return self.embeddings_file.exists() and self.metadata_file.exists()


class Embedder:
    """Embedding generator using sentence-transformers."""

    def __init__(self, config: Config):
        """
        Initialize the embedder.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.cache = EmbeddingCache(config.paths.embeddings_cache)
        # SentenceTransformerType exists only within TYPE_CHECKING at runtime,
        # so keep this as a string forward reference to avoid NameError.
        self.model: SentenceTransformerType | None = None
        self.openai_client: Any = None
        self.mistral_client: Any = None
        self.genai_client: Any = None
        self._genai_types: Any = None

    def load_model(self) -> None:
        """Load the embedding model based on provider."""
        provider = self.config.embedding.provider

        if provider == "local":
            self._load_local_model()
        elif provider in ["mistral", "openai", "gemini"]:
            logger.info(f"Using {provider.upper()} API for embeddings")
            # For API providers, we don't need to preload models
            # They'll be called during generate_embeddings
            self.model = None
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _load_local_model(self) -> None:
        """Load the local sentence-transformers model."""
        logger.info(f"Loading local model: {self.config.embedding.model_name}")

        # Determine device based on configuration
        configured_device = self.config.embedding.device

        if configured_device == "auto":
            # Auto-selection: try CUDA, then MPS, then CPU
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-selected device: {device}")
        else:
            # Honor configured device if available
            if configured_device.startswith("cuda"):
                if torch.cuda.is_available():
                    device = configured_device
                else:
                    logger.warning(
                        f"CUDA not available, configured device '{configured_device}' not usable, falling back to CPU"
                    )
                    device = "cpu"
            elif configured_device == "mps":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    logger.warning(
                        f"MPS not available, configured device '{configured_device}' not usable, falling back to CPU"
                    )
                    device = "cpu"
            elif configured_device == "cpu":
                device = "cpu"
            else:
                logger.warning(f"Unknown device '{configured_device}', falling back to CPU")
                device = "cpu"

            logger.info(f"Using device: {device}")

        # Load model (module-level symbol allows tests to patch rainrag.embed.SentenceTransformer)
        model_cls_any = SentenceTransformer
        if model_cls_any is None:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer

            model_cls_any = _SentenceTransformer

        model_cls = cast(Any, model_cls_any)
        try:
            self.model = model_cls(
                self.config.embedding.model_name,
                device=device,
                model_kwargs={"dtype": "auto"},  # Prefer new dtype kwarg when supported
            )
        except TypeError as exc:
            # Use package version to determine whether sentence-transformers supports model_kwargs
            try:
                from importlib import metadata as importlib_metadata

                st_version = importlib_metadata.version("sentence-transformers")
            except Exception:
                st_version = None

            try:
                from packaging.version import Version

                parsed_version = Version(st_version) if st_version is not None else None
                cutoff_version = Version("2.2.0")
            except Exception:
                parsed_version = None
                cutoff_version = None

            if (
                parsed_version is not None
                and cutoff_version is not None
                and parsed_version < cutoff_version
            ):
                logger.warning(
                    "sentence-transformers %s does not support model_kwargs; using legacy constructor for model '%s'",
                    st_version,
                    self.config.embedding.model_name,
                )
                self.model = model_cls(
                    self.config.embedding.model_name,
                    device=device,
                )
            else:
                logger.error(
                    "sentence-transformers model construction failed for %s (version=%s): %s",
                    self.config.embedding.model_name,
                    st_version,
                    exc,
                )
                raise

        assert self.model is not None, "Model should be loaded after load_model() call"
        # Set max sequence length
        self.model.max_seq_length = self.config.embedding.max_seq_length

        logger.info(f"Local model loaded on device: {device}")

    def _is_transient_exception(self, exception: Exception) -> bool:
        """
        Check if an exception is transient and should be retried.

        Args:
            exception: The exception to check

        Returns:
            True if the exception is transient (rate limit, network error), False otherwise
        """
        # Check for rate limit errors
        status_code = getattr(exception, "status_code", None)
        if isinstance(status_code, int) and status_code in [
            429,
            500,
            502,
            503,
            504,
        ]:  # Rate limit or server errors
            return True

        # Check for specific error messages that indicate transient issues
        error_str = str(exception).lower()
        transient_indicators = [
            "rate limit",
            "rate_limit",
            "too many requests",
            "service unavailable",
            "temporary failure",
            "connection error",
            "network error",
            "timeout",
            "internal server error",
        ]

        return any(indicator in error_str for indicator in transient_indicators)

    def load_documents(self, docs_path: str) -> list[Document]:
        """
        Load documents from JSONL file.

        Args:
            docs_path: Path to documents JSONL file

        Returns:
            List of Document objects
        """
        docs_file = Path(docs_path)

        if not docs_file.exists():
            raise FileNotFoundError(f"Documents file not found: {docs_path}")

        logger.info(f"Loading documents from {docs_path}")

        documents = []
        with open(docs_file, encoding="utf-8") as f:
            for line in f:
                doc_dict = json.loads(line)
                documents.append(Document(**doc_dict))

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def generate_embeddings(
        self, documents: list[Document], show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of Document objects
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings (shape: [N, D])
        """
        provider = self.config.embedding.provider
        logger.info(f"Generating embeddings for {len(documents)} documents using {provider}")

        if provider == "local":
            return self._generate_local_embeddings(documents, show_progress)
        elif provider == "mistral":
            return self._generate_api_embeddings(documents, show_progress, "mistral")
        elif provider == "openai":
            return self._generate_api_embeddings(documents, show_progress, "openai")
        elif provider == "gemini":
            return self._generate_api_embeddings(documents, show_progress, "gemini")
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _generate_local_embeddings(
        self, documents: list[Document], show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings using local SentenceTransformer model."""
        if self.model is None:
            self.load_model()

        assert self.model is not None, "Model should be loaded after load_model() call"

        # Process documents in chunks to avoid memory issues and show progress
        chunk_size = 10000
        all_embeddings = []

        total_chunks = (len(documents) + chunk_size - 1) // chunk_size
        logger.info(
            f"Starting chunked processing: {len(documents)} documents, chunk_size={chunk_size}"
        )
        logger.info(f"Total chunks to process: {total_chunks}")

        for i in range(0, len(documents), chunk_size):
            chunk_docs = documents[i : i + chunk_size]
            chunk_texts = [f"passage: {doc.text}" for doc in chunk_docs]
            chunk_number = i // chunk_size + 1
            logger.info(
                f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk_texts)} documents)"
            )

            chunk_embeddings = self.model.encode(
                chunk_texts,
                batch_size=self.config.embedding.batch_size,
                show_progress_bar=show_progress
                and len(chunk_texts) > self.config.embedding.batch_size,
                normalize_embeddings=self.config.embedding.normalize_embeddings,
                convert_to_numpy=True,
            )
            chunk_embeddings = np.array(chunk_embeddings)
            all_embeddings.append(chunk_embeddings)
            logger.info(
                f"Completed chunk {chunk_number}, embeddings shape: {chunk_embeddings.shape}"
            )

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Generated embeddings with final shape: {embeddings.shape}")
        return embeddings

    def _generate_api_embeddings(
        self, documents: list[Document], show_progress: bool = True, provider: str = "openai"
    ) -> np.ndarray:
        """Generate embeddings using API providers."""
        # Initialize clients
        model = None  # Initialize to avoid unbound variable
        client: Any = None  # Can be OpenAI, Mistral, or None
        if provider == "openai":
            if self.openai_client is None:
                try:
                    import openai
                except ImportError as e:
                    raise ImportError(
                        "openai package is required for OpenAI embeddings. Install it with: pip install openai"
                    ) from e
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable is required for OpenAI embeddings"
                    )
                self.openai_client = openai.OpenAI(api_key=api_key)
            client = self.openai_client
            model = self.config.openai.embedding_model
        elif provider == "mistral":
            if self.mistral_client is None:
                try:
                    from mistralai import Mistral
                except ImportError as e:
                    raise ImportError(
                        "mistralai package is required for Mistral embeddings. Install it with: pip install mistralai"
                    ) from e
                api_key = os.getenv("MISTRAL_API_KEY")
                if api_key is None:
                    raise ValueError(
                        "MISTRAL_API_KEY environment variable is required for Mistral embeddings"
                    )
                self.mistral_client = Mistral(api_key=api_key)
            client = self.mistral_client
            model = "mistral-embed"
        elif provider == "gemini":
            if self.genai_client is None:
                try:
                    from google import genai
                    from google.genai import types as genai_types
                except ImportError as e:
                    raise ImportError(
                        "google-genai package is required for Gemini embeddings. Install it with: pip install google-genai"
                    ) from e
                self.genai_client = genai.Client(api_key=self.config.gemini.api_key)
                # Store module-level alias for config type usage later
                self._genai_types = genai_types
            client = None  # Gemini uses direct API calls
            model = self.config.gemini.embedding_model
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        # For API embeddings, prefix passages appropriately
        if provider in ["openai", "mistral"]:
            texts = [f"passage: {doc.text}" for doc in documents]
        else:
            texts = [doc.text for doc in documents]

        all_embeddings = []
        batch_size = self.config.embedding.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            if show_progress:
                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
                )

            # Retry logic for transient failures
            max_retries = self.config.embedding.max_retries
            backoff_factor = self.config.embedding.retry_backoff_factor
            batch_index = i // batch_size + 1

            for retry_attempt in range(max_retries + 1):
                try:
                    if provider == "openai":
                        response = client.embeddings.create(model=model, input=batch_texts)
                        batch_embeddings = [item.embedding for item in response.data]
                    elif provider == "mistral":
                        response = client.embeddings.create(model=model, inputs=batch_texts)
                        batch_embeddings = [item.embedding for item in response.data]
                    elif provider == "gemini":
                        if self._genai_types is None:
                            raise RuntimeError(
                                "Gemini client types not initialized. Ensure google-genai is installed."
                            )
                        result = self.genai_client.models.embed_content(
                            model=model,
                            contents=batch_texts,
                            config=self._genai_types.EmbedContentConfig(
                                task_type="RETRIEVAL_DOCUMENT"
                            ),
                        )
                        if result and result.embeddings:
                            batch_embeddings = []
                            for embedding in result.embeddings:
                                if embedding.values is not None:
                                    batch_embeddings.append(embedding.values)
                                else:
                                    raise RuntimeError("Gemini embeddings API returned None values")
                        else:
                            raise RuntimeError("Gemini embeddings API returned invalid response")

                    all_embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop

                except Exception as e:
                    is_transient = self._is_transient_exception(e)

                    if retry_attempt < max_retries and is_transient:
                        delay = backoff_factor * (2**retry_attempt)
                        logger.warning(
                            f"Transient error in batch {batch_index} (attempt {retry_attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay:.1f} seconds..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        # Either non-transient error or max retries exceeded
                        if is_transient:
                            logger.error(
                                f"Failed to generate embeddings for batch {batch_index} after {max_retries + 1} attempts: {e}"
                            )
                        else:
                            logger.error(f"Non-transient error in batch {batch_index}: {e}")
                        raise

        embeddings_array = np.array(all_embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

    def embed(
        self, force_regenerate: bool = False, incremental: bool = False
    ) -> tuple[np.ndarray, list[Document]]:
        """
        Run the embedding pipeline.

        Args:
            force_regenerate: If True, regenerate embeddings even if cache exists
            incremental: If True, only embed documents with new content_hash

        Returns:
            Tuple of (embeddings array, list of documents)
        """
        # Check if cache exists and we're not forcing regeneration
        if not force_regenerate and not incremental and self.cache.exists():
            logger.info("Loading embeddings from cache")
            embeddings, documents = self.cache.load()

            if embeddings is not None and documents is not None:
                return embeddings, documents

        # Load documents
        documents = self.load_documents(self.config.paths.docs_output)

        if not documents:
            logger.error("No documents found to embed")
            raise ValueError("No documents found")

        # Incremental mode: reuse cached embeddings for unchanged content
        if incremental and not force_regenerate and self.cache.exists():
            return self._embed_incremental(documents)

        # Load model
        self.load_model()

        # Generate embeddings
        embeddings = self.generate_embeddings(documents)

        # Save to cache
        self.cache.save(embeddings, documents)

        logger.info("Embedding pipeline complete!")

        return embeddings, documents

    def _embed_incremental(self, documents: list[Document]) -> tuple[np.ndarray, list[Document]]:
        """Incrementally embed documents — reuse cached embeddings for unchanged content.

        Builds a content_hash -> embedding lookup from the existing cache.
        Only documents with a new or changed content_hash are sent to the
        embedding model. Unchanged documents reuse their cached vectors.
        """
        cached_embeddings, cached_docs = self.cache.load()
        if cached_embeddings is None or cached_docs is None:
            logger.info("Cache load failed, falling back to full embedding")
            self.load_model()
            embeddings = self.generate_embeddings(documents)
            self.cache.save(embeddings, documents)
            return embeddings, documents

        # Build content_hash -> embedding lookup from cache
        hash_to_embedding: dict[str, np.ndarray] = {}
        for i, doc in enumerate(cached_docs):
            if doc.content_hash:
                hash_to_embedding[doc.content_hash] = cached_embeddings[i]

        embedding_dim = cached_embeddings.shape[1]
        logger.info(
            f"Incremental embedding: {len(hash_to_embedding)} cached hashes, "
            f"{len(documents)} current documents"
        )

        # Classify documents into cached vs. needs-embedding
        to_embed: list[Document] = []
        to_embed_indices: list[int] = []
        cached_count = 0

        final_embeddings = np.zeros((len(documents), embedding_dim), dtype=np.float32)

        for i, doc in enumerate(documents):
            if doc.content_hash and doc.content_hash in hash_to_embedding:
                final_embeddings[i] = hash_to_embedding[doc.content_hash]
                cached_count += 1
            else:
                to_embed.append(doc)
                to_embed_indices.append(i)

        logger.info(f"Incremental embedding: {cached_count} cached, " f"{len(to_embed)} to embed")

        if to_embed:
            # Load model and embed only the new/changed documents
            self.load_model()
            new_embeddings = self.generate_embeddings(to_embed)
            for j, idx in enumerate(to_embed_indices):
                final_embeddings[idx] = new_embeddings[j]
        else:
            logger.info("All documents have cached embeddings — nothing to embed!")

        # Save updated cache
        self.cache.save(final_embeddings, documents)

        logger.info("Incremental embedding pipeline complete!")
        return final_embeddings, documents


def run_embedding(
    config_path: str = "config.yaml",
    force_regenerate: bool = False,
    incremental: bool = False,
) -> tuple[np.ndarray, list[Document]]:
    """
    Run the embedding generation pipeline.

    Args:
        config_path: Path to configuration file
        force_regenerate: If True, regenerate embeddings even if cache exists
        incremental: If True, only embed documents with new content_hash

    Returns:
        Tuple of (embeddings array, list of documents)
    """
    logger.info("run_embedding function STARTED")
    from rainrag.config import load_config

    config = load_config(config_path)
    logger.info("Config loaded")
    embedder = Embedder(config)
    return embedder.embed(force_regenerate=force_regenerate, incremental=incremental)
