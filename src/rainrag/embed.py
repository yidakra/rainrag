"""Embedding generation module using multilingual-e5-large."""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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

    def save(
        self, embeddings: np.ndarray, documents: List[Document]
    ) -> None:
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

    def load(self) -> tuple[Optional[np.ndarray], Optional[List[Document]]]:
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
        with open(self.metadata_file, "r", encoding="utf-8") as f:
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
        self.model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """Load the sentence-transformers model."""
        logger.info(f"Loading model: {self.config.embedding.model_name}")

        # Determine device
        if self.config.embedding.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
        else:
            device = self.config.embedding.device

        # Load model
        self.model = SentenceTransformer(
            self.config.embedding.model_name, device=device
        )

        # Set max sequence length
        self.model.max_seq_length = self.config.embedding.max_seq_length

        logger.info(f"Model loaded on device: {device}")

    def load_documents(self, docs_path: str) -> List[Document]:
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
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                doc_dict = json.loads(line)
                documents.append(Document(**doc_dict))

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def generate_embeddings(
        self, documents: List[Document], show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of Document objects
            show_progress: Whether to show progress bar

        Returns:
            NumPy array of embeddings (shape: [N, D])
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Generating embeddings for {len(documents)} documents")

        # For E5 models, it's recommended to prefix queries with "query: "
        # and passages with "passage: ". Since we're building a search index,
        # we treat all documents as passages.
        texts = [f"passage: {doc.text}" for doc in documents]

        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.embedding.normalize_embeddings,
            convert_to_numpy=True,
        )

        logger.info(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings

    def embed(self, force_regenerate: bool = False) -> tuple[np.ndarray, List[Document]]:
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
) -> tuple[np.ndarray, List[Document]]:
    """
    Run the embedding generation pipeline.

    Args:
        config_path: Path to configuration file
        force_regenerate: If True, regenerate embeddings even if cache exists

    Returns:
        Tuple of (embeddings array, list of documents)
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    embedder = Embedder(config)
    return embedder.embed(force_regenerate=force_regenerate)
