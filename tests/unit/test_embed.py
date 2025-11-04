"""Unit tests for embedding module."""

import json
from pathlib import Path

import numpy as np
import pytest

from rainrag.config import Config
from rainrag.embed import Embedder, EmbeddingCache
from rainrag.ingest import Document


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_cache_creation(self, temp_dir: Path) -> None:
        """Test creating cache directory."""
        cache_dir = temp_dir / "cache"
        cache = EmbeddingCache(str(cache_dir))

        assert cache.cache_dir.exists()
        assert cache.embeddings_file == cache_dir / "embeddings.npy"
        assert cache.metadata_file == cache_dir / "metadata.jsonl"

    def test_cache_save_and_load(self, temp_dir: Path) -> None:
        """Test saving and loading embeddings."""
        cache = EmbeddingCache(str(temp_dir / "cache"))

        # Create test data
        embeddings = np.random.rand(10, 384).astype(np.float32)
        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/to/file{i}.vtt",
                language="en",
                text=f"Test document {i}",
                length=len(f"Test document {i}"),
            )
            for i in range(10)
        ]

        # Save
        cache.save(embeddings, documents)

        # Load
        loaded_embeddings, loaded_docs = cache.load()

        assert loaded_embeddings is not None
        assert loaded_docs is not None
        assert np.allclose(embeddings, loaded_embeddings)
        assert len(loaded_docs) == len(documents)
        assert all(
            loaded_docs[i].id == documents[i].id for i in range(len(documents))
        )

    def test_cache_exists(self, temp_dir: Path) -> None:
        """Test checking if cache exists."""
        cache = EmbeddingCache(str(temp_dir / "cache"))

        assert not cache.exists()

        # Create cache
        embeddings = np.random.rand(5, 384).astype(np.float32)
        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en",
                text=f"Text {i}",
                length=6,
            )
            for i in range(5)
        ]

        cache.save(embeddings, documents)

        assert cache.exists()

    def test_cache_load_nonexistent(self, temp_dir: Path) -> None:
        """Test loading from non-existent cache."""
        cache = EmbeddingCache(str(temp_dir / "nonexistent"))

        embeddings, docs = cache.load()

        assert embeddings is None
        assert docs is None


class TestEmbedder:
    """Tests for Embedder class."""

    def test_embedder_creation(self, test_config: Config) -> None:
        """Test creating embedder."""
        embedder = Embedder(test_config)

        assert embedder.config == test_config
        assert embedder.model is None  # Not loaded yet

    def test_load_model(self, test_config: Config) -> None:
        """Test loading the embedding model."""
        embedder = Embedder(test_config)
        embedder.load_model()

        assert embedder.model is not None
        assert embedder.model.max_seq_length == test_config.embedding.max_seq_length

    def test_load_documents(self, test_config: Config, temp_dir: Path) -> None:
        """Test loading documents from JSONL."""
        # Create test JSONL file
        docs_file = temp_dir / "docs.jsonl"
        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en",
                text=f"Test document {i}",
                length=len(f"Test document {i}"),
            )
            for i in range(5)
        ]

        with open(docs_file, "w") as f:
            for doc in documents:
                f.write(doc.model_dump_json() + "\n")

        # Load documents
        embedder = Embedder(test_config)
        loaded_docs = embedder.load_documents(str(docs_file))

        assert len(loaded_docs) == 5
        assert all(loaded_docs[i].id == documents[i].id for i in range(5))

    def test_load_documents_file_not_found(self, test_config: Config) -> None:
        """Test loading from non-existent file."""
        embedder = Embedder(test_config)

        with pytest.raises(FileNotFoundError):
            embedder.load_documents("/nonexistent/docs.jsonl")

    def test_generate_embeddings(self, test_config: Config) -> None:
        """Test generating embeddings."""
        embedder = Embedder(test_config)
        embedder.load_model()

        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en",
                text=f"This is test document number {i}",
                length=len(f"This is test document number {i}"),
            )
            for i in range(5)
        ]

        embeddings = embedder.generate_embeddings(documents, show_progress=False)

        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] == test_config.qdrant.vector_size
        assert embeddings.dtype == np.float32

        # Check that embeddings are different for different texts
        assert not np.allclose(embeddings[0], embeddings[1])

    def test_generate_embeddings_normalized(self, test_config: Config) -> None:
        """Test that embeddings are normalized when configured."""
        test_config.embedding.normalize_embeddings = True

        embedder = Embedder(test_config)
        embedder.load_model()

        documents = [
            Document(
                id="doc1",
                path="/path/file.vtt",
                language="en",
                text="Test document",
                length=13,
            )
        ]

        embeddings = embedder.generate_embeddings(documents, show_progress=False)

        # Check that vectors are normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_embed_with_cache(self, test_config: Config, temp_dir: Path) -> None:
        """Test embedding with caching."""
        # Create test documents
        docs_file = temp_dir / "data" / "docs.jsonl"
        docs_file.parent.mkdir(parents=True)

        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en",
                text=f"Test document {i}",
                length=len(f"Test document {i}"),
            )
            for i in range(3)
        ]

        with open(docs_file, "w") as f:
            for doc in documents:
                f.write(doc.model_dump_json() + "\n")

        test_config.paths.docs_output = str(docs_file)
        test_config.paths.embeddings_cache = str(temp_dir / "embeddings")

        # First run - should generate embeddings
        embedder = Embedder(test_config)
        embeddings1, docs1 = embedder.embed(force_regenerate=False)

        assert embeddings1.shape[0] == 3
        assert len(docs1) == 3
        assert embedder.cache.exists()

        # Second run - should load from cache
        embedder2 = Embedder(test_config)
        embeddings2, docs2 = embedder2.embed(force_regenerate=False)

        assert np.allclose(embeddings1, embeddings2)
        assert len(docs2) == 3

    def test_embed_force_regenerate(self, test_config: Config, temp_dir: Path) -> None:
        """Test force regeneration of embeddings."""
        # Create test documents
        docs_file = temp_dir / "data" / "docs.jsonl"
        docs_file.parent.mkdir(parents=True)

        documents = [
            Document(
                id="doc1",
                path="/path/file.vtt",
                language="en",
                text="Test document",
                length=13,
            )
        ]

        with open(docs_file, "w") as f:
            for doc in documents:
                f.write(doc.model_dump_json() + "\n")

        test_config.paths.docs_output = str(docs_file)
        test_config.paths.embeddings_cache = str(temp_dir / "embeddings")

        # First run
        embedder = Embedder(test_config)
        embeddings1, _ = embedder.embed(force_regenerate=False)

        # Second run with force_regenerate=True
        embedder2 = Embedder(test_config)
        embeddings2, _ = embedder2.embed(force_regenerate=True)

        # Embeddings should be similar (same text), but regenerated
        assert np.allclose(embeddings1, embeddings2, atol=1e-4)

    def test_embed_no_documents(self, test_config: Config, temp_dir: Path) -> None:
        """Test embedding with no documents."""
        # Create empty JSONL file
        docs_file = temp_dir / "data" / "docs.jsonl"
        docs_file.parent.mkdir(parents=True)
        docs_file.write_text("")

        test_config.paths.docs_output = str(docs_file)

        embedder = Embedder(test_config)

        with pytest.raises(ValueError, match="No documents found"):
            embedder.embed()
