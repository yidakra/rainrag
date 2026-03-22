"""Unit tests for indexing module."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from rainrag.config import Config
from rainrag.index import QdrantIndexer, doc_id_to_uuid
from rainrag.ingest import Document


class TestQdrantIndexer:
    """Tests for QdrantIndexer class."""

    def test_indexer_creation(self, test_config: Config) -> None:
        """Test creating indexer."""
        indexer = QdrantIndexer(test_config)

        assert indexer.config == test_config
        assert indexer.client is None

    @patch("rainrag.index.QdrantClient")
    def test_connect_success(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test successful connection to Qdrant."""
        # Setup mock
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.connect()

        # Verify connection
        assert indexer.client is not None
        mock_client_class.assert_called_once_with(
            host=test_config.qdrant.host,
            port=test_config.qdrant.port,
            prefer_grpc=False,
        )
        mock_client.get_collections.assert_called_once()

    @patch("rainrag.index.QdrantClient")
    def test_connect_failure(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test connection failure."""
        # Setup mock to raise exception
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Connection failed")
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)

        with pytest.raises(Exception, match="Connection failed"):
            indexer.connect()

    @patch("rainrag.index.QdrantClient")
    def test_create_collection_new(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test creating a new collection."""
        # Setup mock
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []  # No existing collections
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client
        indexer.create_collection(recreate=False)

        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == test_config.qdrant.collection_name

    @patch("rainrag.index.QdrantClient")
    def test_create_collection_exists(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test when collection already exists."""
        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = test_config.qdrant.collection_name
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client
        indexer.create_collection(recreate=False)

        # Should not create collection
        mock_client.create_collection.assert_not_called()
        mock_client.delete_collection.assert_not_called()

    @patch("rainrag.index.QdrantClient")
    def test_create_collection_recreate(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test recreating an existing collection."""
        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = test_config.qdrant.collection_name
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client
        indexer.create_collection(recreate=True)

        # Should delete and create collection
        mock_client.delete_collection.assert_called_once_with(test_config.qdrant.collection_name)
        mock_client.create_collection.assert_called_once()

    @patch("rainrag.index.QdrantClient")
    def test_index_documents(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test indexing documents."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create test data
        embeddings = np.random.rand(10, 384).astype(np.float32)
        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en" if i % 2 == 0 else "ru",
                text=f"Test document {i}",
                length=len(f"Test document {i}"),
            )
            for i in range(10)
        ]

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client

        count = indexer.index_documents(embeddings, documents, batch_size=5)

        assert count == 10
        # Should be called twice (10 docs / batch_size=5)
        assert mock_client.upsert.call_count == 2

    @patch("rainrag.index.QdrantClient")
    def test_index_documents_single_batch(
        self, mock_client_class: Mock, test_config: Config
    ) -> None:
        """Test indexing with single batch."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Create test data
        embeddings = np.random.rand(3, 384).astype(np.float32)
        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en",
                text=f"Text {i}",
                length=6,
            )
            for i in range(3)
        ]

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client

        count = indexer.index_documents(embeddings, documents, batch_size=100)

        assert count == 3
        # Should be called once (all in single batch)
        mock_client.upsert.assert_called_once()

        # Verify points structure
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 3

        # Check first point structure
        point = points[0]
        assert hasattr(point, "id")
        assert hasattr(point, "vector")
        assert hasattr(point, "payload")
        assert "doc_id" in point.payload
        assert "text" in point.payload
        assert "language" in point.payload

    @patch("rainrag.index.QdrantClient")
    def test_get_collection_info(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test getting collection info."""
        # Setup mock
        mock_client = MagicMock()
        mock_info = MagicMock()
        mock_info.indexed_vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = "green"
        mock_client.get_collection.return_value = mock_info
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client

        stats = indexer.get_collection_info()

        assert stats["indexed_vectors_count"] == 100
        assert stats["points_count"] == 100
        assert stats["status"] == "green"

    @patch("rainrag.index.QdrantClient")
    def test_get_collection_info_error(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test getting collection info when error occurs."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client

        stats = indexer.get_collection_info()

        assert stats == {}

    @patch("rainrag.index.QdrantClient")
    def test_search(self, mock_client_class: Mock, test_config: Config) -> None:
        """Test searching for similar documents."""
        # Setup mock
        mock_client = MagicMock()
        mock_result1 = MagicMock()
        mock_result1.id = 1
        mock_result1.score = 0.95
        mock_result1.payload = {
            "doc_id": "doc1",
            "text": "Test document",
            "language": "en",
        }

        mock_result2 = MagicMock()
        mock_result2.id = 2
        mock_result2.score = 0.85
        mock_result2.payload = {
            "doc_id": "doc2",
            "text": "Another document",
            "language": "ru",
        }

        # Mock query_points response
        mock_response = MagicMock()
        mock_response.points = [mock_result1, mock_result2]
        mock_client.query_points.return_value = mock_response
        mock_client_class.return_value = mock_client

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client

        query_vector = np.random.rand(384).astype(np.float32)
        results = indexer.search(query_vector, top_k=2, score_threshold=0.8)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["doc_id"] == "doc1"
        assert results[1]["id"] == 2
        assert results[1]["score"] == 0.85

    @patch("rainrag.index.run_embedding")
    @patch("rainrag.index.QdrantClient")
    def test_index_pipeline(
        self,
        mock_client_class: Mock,
        mock_run_embedding: Mock,
        test_config: Config,
    ) -> None:
        """Test full indexing pipeline."""
        # Setup mocks
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_info = MagicMock()
        mock_info.vectors_count = 5
        mock_info.points_count = 5
        mock_info.status = "green"
        mock_client.get_collection.return_value = mock_info
        mock_client_class.return_value = mock_client

        # Mock embedding results
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
        mock_run_embedding.return_value = (embeddings, documents)

        indexer = QdrantIndexer(test_config)
        count = indexer.index(recreate=False)

        assert count == 5
        mock_client.create_collection.assert_called_once()
        mock_client.upsert.assert_called()

    @patch("rainrag.index.QdrantClient")
    def test_index_documents_incremental_diffing(
        self, mock_client_class: Mock, test_config: Config
    ) -> None:
        """Incremental indexing should skip unchanged, upsert changed/new, and delete removed."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Existing points in Qdrant:
        # - doc1: unchanged
        # - old_doc: deleted in new dataset
        doc1_id = "doc1"
        old_doc_id = "old_doc"
        point_doc1 = MagicMock()
        point_doc1.id = doc_id_to_uuid(doc1_id)
        point_doc1.payload = {"content_hash": "h1"}
        point_old = MagicMock()
        point_old.id = doc_id_to_uuid(old_doc_id)
        point_old.payload = {"content_hash": "hold"}
        mock_client.scroll.return_value = ([point_doc1, point_old], None)

        documents = [
            Document(
                id=doc1_id,
                path="/path/doc1.vtt",
                language="en",
                text="same text",
                length=9,
                content_hash="h1",
            ),
            Document(
                id="doc2",
                path="/path/doc2.vtt",
                language="en",
                text="new text",
                length=8,
                content_hash="h2",
            ),
        ]
        embeddings = np.random.rand(2, 384).astype(np.float32)

        indexer = QdrantIndexer(test_config)
        indexer.client = mock_client

        stats = indexer.index_documents_incremental(embeddings, documents, batch_size=50)

        assert stats == {"total": 2, "upserted": 1, "deleted": 1, "unchanged": 1}
        mock_client.delete.assert_called_once()
        mock_client.upsert.assert_called_once()

    @patch("rainrag.index.run_embedding")
    @patch("rainrag.index.QdrantClient")
    def test_index_pipeline_alias_swap_skips_create_collection(
        self, mock_client_class: Mock, mock_run_embedding: Mock, test_config: Config
    ) -> None:
        """Alias-swap mode should bypass create_collection and use staging flow."""
        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        embeddings = np.random.rand(2, 384).astype(np.float32)
        documents = [
            Document(
                id=f"doc{i}",
                path=f"/path/file{i}.vtt",
                language="en",
                text=f"Text {i}",
                length=6,
                content_hash=f"h{i}",
            )
            for i in range(2)
        ]
        mock_run_embedding.return_value = (embeddings, documents)

        test_config.incremental.alias_swap = True
        indexer = QdrantIndexer(test_config)

        indexer.connect = MagicMock()
        indexer.create_collection = MagicMock()
        indexer.index_with_alias_swap = MagicMock(return_value=2)
        indexer.get_collection_info = MagicMock(return_value={})

        count = indexer.index(recreate=False, incremental=True)

        assert count == 2
        indexer.create_collection.assert_not_called()
        indexer.index_with_alias_swap.assert_called_once_with(embeddings, documents)
