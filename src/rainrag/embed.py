"""Embedding generation module using multilingual-e5-large."""

import json
import os
from pathlib import Path

import google.generativeai as genai
import numpy as np
import openai
import torch
from loguru import logger
from mistralai import Mistral
from sentence_transformers import SentenceTransformer

from rainrag.config import Config
from rainrag.ingest import Document


class EmbeddingCache:
    """Cache for storing and loading embeddings."""

    def __init__(self, cache_dir: str):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        self.config = config
        self.cache = EmbeddingCache(config.paths.embeddings_cache)
        self.model: SentenceTransformer | None = None

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

        # Determine device
        if self.config.embedding.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device = self.config.embedding.device

        # Load model
        try:
            self.model = SentenceTransformer(
                self.config.embedding.model_name,
                device=device,
                model_kwargs={"dtype": "auto"},  # type: ignore  # Prefer new dtype kwarg when supported
            )
        except TypeError:
            # Older sentence-transformers versions don't accept model_kwargs
            self.model = SentenceTransformer(
                self.config.embedding.model_name,
                device=device,
            )

        # Set max sequence length
        self.model.max_seq_length = self.config.embedding.max_seq_length

        logger.info(f"Local model loaded on device: {device}")

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
        logger.info(f"Starting chunked processing: {len(documents)} documents, chunk_size={chunk_size}")
        logger.info(f"Total chunks to process: {total_chunks}")

        for i in range(0, len(documents), chunk_size):
            chunk_docs = documents[i:i + chunk_size]
            chunk_texts = [f"passage: {doc.text}" for doc in chunk_docs]
            chunk_number = i // chunk_size + 1
            logger.info(f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk_texts)} documents)")

            chunk_embeddings = self.model.encode(
                chunk_texts,
                batch_size=self.config.embedding.batch_size,
                show_progress_bar=show_progress and len(chunk_texts) > self.config.embedding.batch_size,
                normalize_embeddings=self.config.embedding.normalize_embeddings,
                convert_to_numpy=True,
            )
            chunk_embeddings = np.array(chunk_embeddings)
            all_embeddings.append(chunk_embeddings)
            logger.info(f"Completed chunk {chunk_number}, embeddings shape: {chunk_embeddings.shape}")

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Generated embeddings with final shape: {embeddings.shape}")
        return embeddings

    def _generate_api_embeddings(
        self, documents: list[Document], show_progress: bool = True, provider: str = "openai"
    ) -> np.ndarray:
        """Generate embeddings using API providers."""
        # Initialize clients
        model = None  # Initialize to avoid unbound variable
        if provider == "openai":
            if not hasattr(self, 'openai_client'):
                self.openai_client = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            client = self.openai_client
            model = self.config.openai.embedding_model
        elif provider == "mistral":
            if not hasattr(self, 'mistral_client'):
                self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
            client = self.mistral_client
            model = "mistral-embed"
        elif provider == "gemini":
            client = None  # Gemini uses direct API calls
            model = self.config.gemini.embedding_model
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

        if model is None:
            raise ValueError(f"Model not initialized for provider: {provider}")

        # For API embeddings, prefix passages appropriately
        if provider in ["openai", "mistral"]:
            texts = [f"passage: {doc.text}" for doc in documents]
        else:
            texts = [doc.text for doc in documents]

        all_embeddings = []
        batch_size = self.config.embedding.batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if show_progress:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            try:
                if provider == "openai":
                    response = client.embeddings.create(model=model, input=batch_texts)  # type: ignore
                    batch_embeddings = [item.embedding for item in response.data]
                elif provider == "mistral":
                    response = client.embeddings.create(model=model, inputs=batch_texts)  # type: ignore
                    batch_embeddings = [item.embedding for item in response.data]
                elif provider == "gemini":
                    batch_embeddings = []
                    for text in batch_texts:
                        result = genai.embed_content(  # type: ignore
                            model=model,
                            content=text,
                            task_type="retrieval_document",
                        )
                        batch_embeddings.append(result["embedding"])

                all_embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                raise

        embeddings_array = np.array(all_embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings_array.shape}")
        return embeddings_array

    def embed(self, force_regenerate: bool = False) -> tuple[np.ndarray, list[Document]]:
        """
        Run the embedding pipeline.

        Args:
            force_regenerate: If True, regenerate embeddings even if cache exists

        Returns:
            Tuple of (embeddings array, list of documents)
        """
        # Check if cache exists and we're not forcing regeneration
        if not force_regenerate and self.cache.exists():
            logger.info("Loading embeddings from cache")
            embeddings, documents = self.cache.load()

            if embeddings is not None and documents is not None:
                return embeddings, documents

        # Load documents
        documents = self.load_documents(self.config.paths.docs_output)

        if not documents:
            logger.error("No documents found to embed")
            raise ValueError("No documents found")

        # Load model
        self.load_model()

        # Generate embeddings
        embeddings = self.generate_embeddings(documents)

        # Save to cache
        self.cache.save(embeddings, documents)

        logger.info("Embedding pipeline complete!")

        return embeddings, documents


def run_embedding(
    config_path: str = "config.yaml", force_regenerate: bool = False
) -> tuple[np.ndarray, list[Document]]:
    """
    Run the embedding generation pipeline.

    Args:
        config_path: Path to configuration file
        force_regenerate: If True, regenerate embeddings even if cache exists

    Returns:
        Tuple of (embeddings array, list of documents)
    """
    logger.info("run_embedding function STARTED")
    from rainrag.config import load_config

    config = load_config(config_path)
    logger.info("Config loaded")
    embedder = Embedder(config)
    return embedder.embed(force_regenerate=force_regenerate)
