"""Qdrant indexing module for vector storage."""

import time
import uuid

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from rainrag.config import Config
from rainrag.embed import run_embedding
from rainrag.ingest import Document


# Stable namespace UUID for generating deterministic point IDs from doc IDs
_NAMESPACE_RAINRAG = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def doc_id_to_uuid(doc_id: str) -> str:
    """Convert a document ID (hex string) to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(_NAMESPACE_RAINRAG, doc_id))


def _build_payload(doc: Document) -> dict:
    """Build the Qdrant payload dict for a document."""
    return {
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
        "web_title": doc.web_title,
        "web_date": doc.web_date,
        "web_date_ts": doc.web_date_ts,
        "web_description": doc.web_description,
        "web_url": doc.web_url,
        "is_speech_free": doc.is_speech_free,
        "content_hash": doc.content_hash,
    }


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
        assert self.client is not None  # Ensures type checker knows client is not None
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

    def _get_distance(self) -> models.Distance:
        """Map configured distance metric to Qdrant enum."""
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "Euclidean": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
        }
        return distance_map.get(self.config.qdrant.distance, models.Distance.COSINE)

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

        distance = self._get_distance()

        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.config.qdrant.vector_size,
                distance=distance,
            ),
        )

        logger.info(f"Collection {collection_name} created successfully")

    def _create_collection_with_name(self, collection_name: str) -> None:
        """Create a new collection with the given name using current config."""
        client = self._get_client()
        distance = self._get_distance()
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.config.qdrant.vector_size,
                distance=distance,
            ),
        )
        logger.info(f"Created staging collection: {collection_name}")

    def index_documents(
        self, embeddings: np.ndarray, documents: list[Document], batch_size: int = 50
    ) -> int:
        """
        Index documents with their embeddings into Qdrant.

        Uses content-based UUIDs derived from doc.id for stable point IDs,
        and stores content_hash in the payload for incremental change detection.

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
                point_id = doc_id_to_uuid(doc.id)
                batch_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=_build_payload(doc),
                    )
                )

            client.upsert(collection_name=collection_name, points=batch_points)

        logger.info(f"Successfully indexed {len(documents)} documents")

        return len(documents)

    def index_documents_incremental(
        self, embeddings: np.ndarray, documents: list[Document], batch_size: int = 50
    ) -> dict:
        """Index documents incrementally — skip unchanged, upsert new/modified, delete removed.

        Scrolls existing collection to build a content_hash lookup, then:
        - Skips documents whose content_hash matches the existing point
        - Upserts documents that are new or have a changed content_hash
        - Deletes points whose doc_id no longer appears in the new document set

        Args:
            embeddings: NumPy array of embeddings (shape: [N, D])
            documents: List of Document objects
            batch_size: Batch size for uploading

        Returns:
            Dict with stats: total, upserted, deleted, unchanged
        """
        client = self._get_client()
        collection_name = self.config.qdrant.collection_name

        # Build new document map: point_uuid -> (embedding, doc)
        new_doc_map: dict[str, tuple[np.ndarray, Document]] = {}
        for i, doc in enumerate(documents):
            point_id = doc_id_to_uuid(doc.id)
            new_doc_map[point_id] = (embeddings[i], doc)

        # Scroll existing points to get their content_hashes
        existing_hashes: dict[str, str | None] = {}  # point_id -> content_hash
        logger.info("Scrolling existing collection for change detection...")
        offset = None
        while True:
            result = client.scroll(
                collection_name=collection_name,
                limit=500,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )
            points, next_offset = result
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                existing_hashes[str(point.id)] = payload.get("content_hash")
            offset = next_offset
            if offset is None:
                break

        logger.info(f"Found {len(existing_hashes)} existing points in collection")

        # Classify changes
        new_point_ids = set(new_doc_map.keys())
        existing_point_ids = set(existing_hashes.keys())

        # Points to delete (exist in Qdrant but not in new set)
        to_delete: list[int | str] = list(existing_point_ids - new_point_ids)

        # Points to upsert (new or content_hash changed)
        to_upsert: list[tuple[str, np.ndarray, Document]] = []
        unchanged_count = 0
        for point_id, (embedding, doc) in new_doc_map.items():
            if (
                point_id in existing_hashes
                and doc.content_hash is not None
                and existing_hashes[point_id] == doc.content_hash
            ):
                unchanged_count += 1
                continue  # Unchanged
            to_upsert.append((point_id, embedding, doc))

        logger.info(
            f"Incremental index: {unchanged_count} unchanged, "
            f"{len(to_upsert)} to upsert, {len(to_delete)} to delete"
        )

        # Delete removed points
        if to_delete:
            logger.info(f"Deleting {len(to_delete)} removed points...")
            # Delete in batches to avoid payload size issues
            for i in range(0, len(to_delete), batch_size):
                batch_ids = to_delete[i : i + batch_size]
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=batch_ids),
                )

        # Upsert new/modified points
        if to_upsert:
            logger.info(f"Upserting {len(to_upsert)} new/modified points...")
            for i in tqdm(
                range(0, len(to_upsert), batch_size), desc="Incremental upsert"
            ):
                batch = to_upsert[i : i + batch_size]
                batch_points = [
                    models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=_build_payload(doc),
                    )
                    for point_id, embedding, doc in batch
                ]
                client.upsert(collection_name=collection_name, points=batch_points)
        else:
            logger.info("No changes to upsert — collection is up to date!")

        stats = {
            "total": len(documents),
            "upserted": len(to_upsert),
            "deleted": len(to_delete),
            "unchanged": unchanged_count,
        }
        logger.info(f"Incremental indexing stats: {stats}")
        return stats

    def index_with_alias_swap(
        self,
        embeddings: np.ndarray,
        documents: list[Document],
        batch_size: int = 50,
    ) -> int:
        """Index into a staging collection and atomically swap a collection alias.

        This provides zero-downtime updates:
        1. Creates a new staging collection with a timestamped name
        2. Indexes all documents into the staging collection
        3. Atomically swaps the alias to point to the staging collection
        4. Deletes the old collection

        Args:
            embeddings: NumPy array of embeddings (shape: [N, D])
            documents: List of Document objects
            batch_size: Batch size for uploading

        Returns:
            Number of documents indexed
        """
        client = self._get_client()
        alias_name = self.config.qdrant.collection_name
        staging_name = f"{alias_name}_staging_{int(time.time())}"

        # Create staging collection
        self._create_collection_with_name(staging_name)

        # Index all documents into staging
        logger.info(f"Indexing {len(documents)} documents into staging collection: {staging_name}")
        total = len(documents)
        for i in tqdm(range(0, total, batch_size), desc="Uploading to staging"):
            batch_points: list[models.PointStruct] = []
            end = min(i + batch_size, total)
            for idx in range(i, end):
                point_id = doc_id_to_uuid(documents[idx].id)
                batch_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings[idx].tolist(),
                        payload=_build_payload(documents[idx]),
                    )
                )
            client.upsert(collection_name=staging_name, points=batch_points)

        # Determine old collection name (what the alias currently points to)
        old_collection_name: str | None = None
        try:
            aliases_response = client.get_aliases()
            aliases = aliases_response.aliases if aliases_response else []
            for alias in aliases:
                if alias.alias_name == alias_name:
                    old_collection_name = alias.collection_name
                    break
        except Exception as e:
            logger.warning(f"Failed to resolve existing alias mapping for {alias_name}: {e}")

        # Swap alias atomically
        logger.info(f"Swapping alias '{alias_name}' to staging collection '{staging_name}'")
        alias_ops: list[
            models.DeleteAliasOperation | models.CreateAliasOperation
        ] = []
        if old_collection_name:
            alias_ops.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=alias_name)
                )
            )
        alias_ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    alias_name=alias_name,
                    collection_name=staging_name,
                )
            )
        )
        client.update_collection_aliases(change_aliases_operations=alias_ops)

        # Delete old collection if it exists and is different from staging
        if old_collection_name and old_collection_name != staging_name:
            try:
                logger.info(f"Deleting old collection: {old_collection_name}")
                client.delete_collection(old_collection_name)
            except Exception as e:
                logger.warning(f"Failed to delete old collection {old_collection_name}: {e}")

        logger.info(f"Alias swap complete. '{alias_name}' -> '{staging_name}'")
        return total

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

    def index(self, recreate: bool = False, incremental: bool = False) -> int:
        """
        Run the full indexing pipeline.

        Args:
            recreate: If True, recreate the collection
            incremental: If True, use incremental indexing (skip unchanged documents)

        Returns:
            Number of documents indexed
        """
        # Connect to Qdrant
        self.connect()

        # Load embeddings (will use cache if available)
        logger.info("Loading embeddings")
        embeddings, documents = run_embedding(
            config_path="config.yaml", incremental=incremental
        )

        # Choose indexing strategy
        if self.config.incremental.alias_swap:
            num_indexed = self.index_with_alias_swap(embeddings, documents)
        else:
            # Create/update the main collection for non-alias flows.
            self.create_collection(recreate=recreate)
            if incremental and self.config.incremental.enabled:
                stats = self.index_documents_incremental(embeddings, documents)
                num_indexed = stats["total"]
            else:
                num_indexed = self.index_documents(embeddings, documents)

        # Get collection info
        self.get_collection_info()

        logger.info("Indexing pipeline complete!")

        return num_indexed


def run_indexing(
    config_path: str = "config.yaml",
    recreate: bool = False,
    incremental: bool = False,
) -> int:
    """
    Run the Qdrant indexing pipeline.

    Args:
        config_path: Path to configuration file
        recreate: If True, recreate the collection
        incremental: If True, use incremental indexing

    Returns:
        Number of documents indexed
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    indexer = QdrantIndexer(config)
    return indexer.index(recreate=recreate, incremental=incremental)
