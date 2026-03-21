"""Integration tests for the full pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rainrag.config import Config
from rainrag.embed import Embedder
from rainrag.ingest import Ingester


class TestPipeline:
    """Integration tests for the complete pipeline."""

    def test_ingest_to_embed_pipeline(
        self,
        test_config: Config,
        archive_with_vtt_files: Path,
    ) -> None:
        """Test ingestion followed by embedding generation."""
        with patch("rainrag.embed.SentenceTransformer") as mock_st:
            # Create mock model
            mock_model = MagicMock()
            fake_embeddings = np.random.rand(4, test_config.qdrant.vector_size).astype(np.float32)
            mock_model.encode.return_value = fake_embeddings
            mock_st.return_value = mock_model

            # Setup
            test_config.paths.archive_root = str(archive_with_vtt_files)

            # Step 1: Ingest
            ingester = Ingester(test_config)
            doc_count = ingester.ingest()

            assert doc_count == 4

            # Step 2: Embed
            embedder = Embedder(test_config)
            embeddings, documents = embedder.embed(force_regenerate=True)

            assert embeddings.shape[0] == 4
            assert len(documents) == 4
            assert embeddings.shape[1] == test_config.qdrant.vector_size

            # Verify documents match ingestion
            assert all(doc.id for doc in documents)
            assert all(doc.text for doc in documents)
            assert any(doc.language == "en" for doc in documents)
            assert any(doc.language == "ru" for doc in documents)

    def test_embed_caching_pipeline(
        self,
        test_config: Config,
        archive_with_vtt_files: Path,
    ) -> None:
        """Test that embedding caching works correctly."""
        with patch("rainrag.embed.SentenceTransformer") as mock_st:
            # Create mock model
            mock_model = MagicMock()
            fake_embeddings = np.random.rand(4, test_config.qdrant.vector_size).astype(np.float32)
            mock_model.encode.return_value = fake_embeddings
            mock_st.return_value = mock_model

            # Setup
            test_config.paths.archive_root = str(archive_with_vtt_files)

            # Ingest
            ingester = Ingester(test_config)
            ingester.ingest()

            # First embedding run
            embedder1 = Embedder(test_config)
            embeddings1, docs1 = embedder1.embed(force_regenerate=True)

            # Second embedding run (should use cache)
            embedder2 = Embedder(test_config)
            embedder2.model = None  # Ensure model not loaded
            embeddings2, docs2 = embedder2.embed(force_regenerate=False)

            # Should be identical (from cache)
            assert np.allclose(embeddings1, embeddings2)
            assert len(docs1) == len(docs2)

            # Model should not have been loaded in second run
            assert embedder2.model is None

    def test_multilingual_processing(
        self,
        test_config: Config,
        archive_with_vtt_files: Path,
    ) -> None:
        """Test processing both English and Russian documents."""
        with patch("rainrag.embed.SentenceTransformer") as mock_st:
            # Create mock model that returns different embeddings for different texts
            mock_model = MagicMock()

            def mock_encode_multilingual(texts, **kwargs):
                # Generate different embeddings for each text
                num_texts = len(texts)
                embeddings = []
                for i in range(num_texts):
                    # Create embeddings that vary based on text content
                    base = np.random.rand(test_config.qdrant.vector_size).astype(np.float32)
                    # Add variation based on text index
                    base = base + i * 0.1
                    embeddings.append(base)
                return np.array(embeddings)

            mock_model.encode = mock_encode_multilingual
            mock_st.return_value = mock_model

            # Setup
            test_config.paths.archive_root = str(archive_with_vtt_files)

            # Full pipeline
            ingester = Ingester(test_config)
            ingester.ingest()

            embedder = Embedder(test_config)
            embeddings, documents = embedder.embed(force_regenerate=True)

            # Verify multilingual processing
            languages = {doc.language for doc in documents}
            assert "en" in languages
            assert "ru" in languages

            # Check that embeddings are different for different languages
            en_docs = [i for i, doc in enumerate(documents) if doc.language == "en"]
            ru_docs = [i for i, doc in enumerate(documents) if doc.language == "ru"]

            assert len(en_docs) > 0
            assert len(ru_docs) > 0

            # Embeddings should be different
            en_embeddings = embeddings[en_docs]
            ru_embeddings = embeddings[ru_docs]

            # Not all embeddings should be identical
            assert not np.allclose(en_embeddings[0], ru_embeddings[0])

    def test_empty_archive_handling(
        self,
        test_config: Config,
        temp_dir: Path,
    ) -> None:
        """Test handling of empty archive."""
        # Create empty archive
        empty_archive = temp_dir / "empty"
        empty_archive.mkdir()

        test_config.paths.archive_root = str(empty_archive)

        # Ingest from empty archive
        ingester = Ingester(test_config)
        doc_count = ingester.ingest()

        assert doc_count == 0

        # Embed should fail with no documents
        embedder = Embedder(test_config)
        with pytest.raises(ValueError, match="No documents found"):
            embedder.embed()

    def test_incremental_processing(
        self,
        test_config: Config,
        temp_dir: Path,
        sample_vtt_en: str,
    ) -> None:
        """Test adding new files and reprocessing."""
        with patch("rainrag.embed.SentenceTransformer") as mock_st:
            # Create mock model
            mock_model = MagicMock()

            def mock_encode_variable(texts, **kwargs):
                # Return embeddings matching the number of texts
                num_texts = len(texts)
                return np.random.rand(num_texts, test_config.qdrant.vector_size).astype(np.float32)

            mock_model.encode = mock_encode_variable
            mock_st.return_value = mock_model

            # Setup initial archive
            archive = temp_dir / "archive"
            archive.mkdir(exist_ok=True)

            # Add first file
            file1 = archive / "file1.vtt"
            file1.write_text(sample_vtt_en)

            test_config.paths.archive_root = str(archive)

            # First ingestion
            ingester = Ingester(test_config)
            count1 = ingester.ingest()
            assert count1 == 1

            # Add second file
            file2 = archive / "file2.vtt"
            file2.write_text(sample_vtt_en)

            # Second ingestion (appends to JSONL)
            ingester2 = Ingester(test_config)
            count2 = ingester2.ingest()
            assert count2 == 2

            # Embedding should process all documents
            embedder = Embedder(test_config)
            _, documents = embedder.embed(force_regenerate=True)

            # Note: The JSONL file is overwritten on each ingestion,
            # so it contains 2 documents (from the second run)
            assert len(documents) == 2
