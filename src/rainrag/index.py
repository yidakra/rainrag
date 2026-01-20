"""Qdrant indexing module for vector storage."""

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from rainrag.config import Config
from rainrag.embed import run_embedding
from rainrag.ingest import Document


class QdrantIndexer:
    """Indexer for Qdrant vector database."""

    def __init__(self, config: Config):
        """
        Initialize the Qdrant indexer.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.client: QdrantClient | None = None

    def connect(self) -> None:
        """Connect to Qdrant server."""
        logger.info(f"Connecting to Qdrant at {self.config.qdrant.host}:{self.config.qdrant.port}")

        # Disable version check to avoid warnings when client/server versions differ slightly
        # The HTTP API is stable and compatible across minor versions
        self.client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            prefer_grpc=False,
        )

        # Test connection
        try:
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant. Found {len(collections.collections)} collections")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _get_client(self) -> QdrantClient:
        """Return initialized Qdrant client."""
        if self.client is None:
            raise RuntimeError("Qdrant client is not initialized. Call connect() first.")
        return self.client

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create or recreate the collection.

        Args:
            recreate: If True, delete existing collection before creating
        """
        client = self._get_client()
        collection_name = self.config.qdrant.collection_name

        # Check if collection exists
        collections = client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if collection_exists:
            if recreate or self.config.qdrant.recreate_collection:
                logger.warning(f"Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
                collection_exists = False
            else:
                logger.info(f"Collection {collection_name} already exists")
                return

        logger.info(f"Creating collection: {collection_name}")

        # Map distance metric
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclidean": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }

        distance = distance_map.get(self.config.qdrant.distance, models.Distance.COSINE)

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.config.qdrant.vector_size,
                distance=distance,
            ),
        )

        logger.info(f"Collection {collection_name} created successfully")

    def index_documents(
        self, embeddings: np.ndarray, documents: list[Document], batch_size: int = 50
    ) -> int:
        """
        Index documents with their embeddings into Qdrant.

        Args:
            embeddings: NumPy array of embeddings (shape: [N, D])
            documents: List of Document objects
            batch_size: Batch size for uploading

        Returns:
            Number of documents indexed
        """
        client = self._get_client()
        collection_name = self.config.qdrant.collection_name

        logger.info(f"Indexing {len(documents)} documents into {collection_name}")

        # Upload in batches without holding all points in memory
        total = len(documents)
        for i in tqdm(range(0, total, batch_size), desc="Uploading to Qdrant"):
            batch_points: list[models.PointStruct] = []
            end = min(i + batch_size, total)

            for idx in range(i, end):
                embedding = embeddings[idx]
                doc = documents[idx]
                batch_points.append(
                    models.PointStruct(
                        id=idx,  # Use sequential ID for simplicity
                        vector=embedding.tolist(),
                        payload={
                            "doc_id": doc.id,
                            "path": doc.path,
                            "language": doc.language,
                            "text": doc.text,
                            "length": doc.length,
                            "date": doc.date,
                            "date_ts": doc.date_ts,
                            "duration_seconds": doc.duration_seconds,
                            "start_time": doc.start_time,
                            "end_time": doc.end_time,
                        },
                    )
                )

            client.upsert(collection_name=collection_name, points=batch_points)

        logger.info(f"Successfully indexed {len(documents)} documents")

        return len(documents)

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection stats
        """
        client = self._get_client()
        collection_name = self.config.qdrant.collection_name

        try:
            info = client.get_collection(collection_name)

            stats = {
                "name": collection_name,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }

            logger.info(f"Collection info: {stats}")

            return stats

        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.0
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of search results with scores and payloads
        """
        client = self._get_client()
        collection_name = self.config.qdrant.collection_name

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        ).points

        search_results = []
        for result in results:
            search_results.append(
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                }
            )

        return search_results

    def index(self, recreate: bool = False) -> int:
        """
        Run the full indexing pipeline.

        Args:
            recreate: If True, recreate the collection

        Returns:
            Number of documents indexed
        """
        # Connect to Qdrant
        self.connect()

        # Create collection
        self.create_collection(recreate=recreate)

        # Load embeddings (will use cache if available)
        logger.info("Loading embeddings")
        embeddings, documents = run_embedding(config_path="config.yaml")

        # Index documents
        num_indexed = self.index_documents(embeddings, documents)

        # Get collection info
        self.get_collection_info()

        logger.info("Indexing pipeline complete!")

        return num_indexed


def run_indexing(config_path: str = "config.yaml", recreate: bool = False) -> int:
    """
    Run the Qdrant indexing pipeline.

    Args:
        config_path: Path to configuration file
        recreate: If True, recreate the collection

    Returns:
        Number of documents indexed
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    indexer = QdrantIndexer(config)
    return indexer.index(recreate=recreate)
