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
        self.config = config
        self.client: QdrantClient = None

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

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create or recreate the collection.

        Args:
            recreate: If True, delete existing collection before creating
        """
        collection_name = self.config.qdrant.collection_name

        # Check if collection exists
        collections = self.client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)

        if collection_exists:
            if recreate or self.config.qdrant.recreate_collection:
                logger.warning(f"Deleting existing collection: {collection_name}")
                self.client.delete_collection(collection_name)
                collection_exists = False
            else:
                logger.info(f"Collection {collection_name} already exists")
                return

        if not collection_exists:
            logger.info(f"Creating collection: {collection_name}")

            # Map distance metric
            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Euclidean": models.Distance.EUCLID,
                "Dot": models.Distance.DOT,
            }

            distance = distance_map.get(self.config.qdrant.distance, models.Distance.COSINE)

            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.qdrant.vector_size,
                    distance=distance,
                ),
            )

            logger.info(f"Collection {collection_name} created successfully")

    def index_documents(
        self, embeddings: np.ndarray, documents: list[Document], batch_size: int = 100
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
        collection_name = self.config.qdrant.collection_name

        logger.info(f"Indexing {len(documents)} documents into {collection_name}")

        # Prepare points for upload
        points = []

        for idx, (embedding, doc) in enumerate(zip(embeddings, documents, strict=False)):
            point = models.PointStruct(
                id=idx,  # Use sequential ID for simplicity
                vector=embedding.tolist(),
                payload={
                    "doc_id": doc.id,
                    "path": doc.path,
                    "language": doc.language,
                    "text": doc.text,
                    "length": doc.length,
                    "date": doc.date,
                    "duration_seconds": doc.duration_seconds,
                    "start_time": doc.start_time,
                    "end_time": doc.end_time,
                },
            )
            points.append(point)

        # Upload in batches
        (len(points) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
            batch = points[i : i + batch_size]

            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(f"Successfully indexed {len(documents)} documents")

        return len(documents)

    def get_collection_info(self) -> dict:
        """
        Get information about the collection.

        Returns:
            Dictionary with collection stats
        """
        collection_name = self.config.qdrant.collection_name

        try:
            info = self.client.get_collection(collection_name)

            stats = {
                "name": collection_name,
                "vectors_count": info.vectors_count,
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
        collection_name = self.config.qdrant.collection_name

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
        )

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
