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


def _is_uuid_string(value: str) -> bool:
    """Return True if value is a valid UUID string."""
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


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
        "web_program": doc.web_program,
        "web_presenters": doc.web_presenters,
        "web_tags": doc.web_tags,
        "web_tags_theme": doc.web_tags_theme,
        "web_tags_person": doc.web_tags_person,
        "web_tags_location": doc.web_tags_location,
        "web_tag_ids": doc.web_tag_ids,
        "web_stories": doc.web_stories,
        "is_speech_free": doc.is_speech_free,
        "content_hash": doc.content_hash,
    }


class QdrantIndexer:
    """Indexer for Qdrant vector database."""

    def __init__(self, config: Config, config_path: str = "config.yaml"):
        """
        Initialize the Qdrant indexer.

        Args:
            config: Configuration object
            config_path: Path to config file used to create the indexer
        """
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.client: QdrantClient | None = None

    def connect(self) -> None:
        """Connect to Qdrant server."""
        logger.info(f"Connecting to Qdrant at {self.config.qdrant.host}:{self.config.qdrant.port}")

        # Disable version check to avoid warnings when client/server versions differ slightly
        # The HTTP API is stable and compatible across minor versions
        self.client = QdrantClient(
            host=self.config.qdrant.host,
            port=self.config.qdrant.port,
            timeout=self.config.qdrant.timeout,
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

    def ensure_collection(self, collection_name: str) -> None:
        """Create a named collection if it does not already exist.

        Used for ephemeral per-upload collections. Uses the same vector params
        as the main collection so the shared embedder/query path work unchanged.
        """
        if self._collection_exists(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return
        self._create_collection_with_name(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        """Delete a named collection if it exists (ephemeral-collection teardown)."""
        client = self._get_client()
        if self._collection_exists(collection_name):
            logger.info(f"Dropping collection: {collection_name}")
            client.delete_collection(collection_name)

    def list_collections(self) -> list[str]:
        """Return the names of all collections on the server."""
        client = self._get_client()
        return [col.name for col in client.get_collections().collections]

    def _collection_exists(self, collection_name: str | None = None) -> bool:
        """Check whether a collection exists."""
        client = self._get_client()
        target_name = collection_name or self.config.qdrant.collection_name
        collections = client.get_collections()
        return any(col.name == target_name for col in collections.collections)

    def _has_legacy_point_ids(self, sample_limit: int = 256) -> bool:
        """Detect legacy point IDs (integer IDs or non-UUID strings).

        Historically this project used sequential integer point IDs.
        Current indexing uses deterministic UUIDv5 string IDs.
        """
        client = self._get_client()
        collection_name = self.config.qdrant.collection_name

        if not self._collection_exists(collection_name):
            return False

        try:
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=sample_limit,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning(
                f"Could not inspect point IDs for legacy detection in {collection_name}: {e}"
            )
            return False

        for point in points:
            point_id = point.id
            if isinstance(point_id, int):
                return True
            if isinstance(point_id, str) and not _is_uuid_string(point_id):
                return True

        return False

    def index_documents(
        self,
        embeddings: np.ndarray,
        documents: list[Document],
        batch_size: int = 50,
        collection_name: str | None = None,
    ) -> int:
        """
        Index documents with their embeddings into Qdrant.

        Uses content-based UUIDs derived from doc.id for stable point IDs,
        and stores content_hash in the payload for incremental change detection.

        Args:
            embeddings: NumPy array of embeddings (shape: [N, D])
            documents: List of Document objects
            batch_size: Batch size for uploading
            collection_name: Override target collection. Defaults to the
                configured collection. Used to index into a single video's
                ephemeral collection.

        Returns:
            Number of documents indexed
        """
        client = self._get_client()
        collection_name = collection_name or self.config.qdrant.collection_name

        effective_batch_size = (
            batch_size if batch_size and batch_size > 0 else self.config.qdrant.upsert_batch_size
        )
        logger.info(
            f"Indexing {len(documents)} documents into {collection_name} (batch_size={effective_batch_size})"
        )

        # Upload in batches without holding all points in memory
        total = len(documents)
        for i in tqdm(range(0, total, effective_batch_size), desc="Uploading to Qdrant"):
            batch_points: list[models.PointStruct] = []
            end = min(i + effective_batch_size, total)

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
        effective_batch_size = (
            batch_size if batch_size and batch_size > 0 else self.config.qdrant.upsert_batch_size
        )

        # Build new document map: point_uuid -> (embedding, doc)
        new_doc_map: dict[str, tuple[np.ndarray, Document]] = {}
        for i, doc in enumerate(documents):
            point_id = doc_id_to_uuid(doc.id)
            new_doc_map[point_id] = (embeddings[i], doc)

        # Scroll existing points to get their content_hashes
        existing_hashes: dict[models.ExtendedPointId, str | None] = {}  # point_id -> content_hash

        # Use collection info to make the loop observable and bounded
        collection_info = self.get_collection_info()
        existing_points_raw = collection_info.get("points_count") if collection_info else None
        existing_points = existing_points_raw if isinstance(existing_points_raw, int) else None
        if existing_points is not None:
            logger.info(f"Collection has {existing_points} existing points; starting scroll.")

        max_iterations = self.config.qdrant.max_scroll_iterations
        max_duration = self.config.qdrant.max_scroll_duration
        scroll_batch_size = self.config.qdrant.scroll_batch_size
        effective_max_iterations = max_iterations

        if existing_points is not None and existing_points > 0:
            required_iterations = (existing_points + scroll_batch_size - 1) // scroll_batch_size + 2
            if required_iterations > effective_max_iterations:
                logger.warning(
                    "Configured max_scroll_iterations=%d is too low for points_count=%d at batch_size=%d; "
                    + "temporarily raising to %d for this run.",
                    max_iterations,
                    existing_points,
                    scroll_batch_size,
                    required_iterations,
                )
            effective_max_iterations = max(effective_max_iterations, required_iterations)

        logger.info(
            f"Scrolling existing collection for change detection... (batch_size={scroll_batch_size}, "
            + f"max_iterations={effective_max_iterations}, max_duration={max_duration}s)"
        )

        offset = None
        iterations = 0
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > max_duration:
                logger.warning(
                    f"Aborting scroll loop: reached max scroll duration {max_duration}s; processed {len(existing_hashes)} points so far."
                )
                break

            if iterations >= effective_max_iterations:
                logger.warning(
                    f"Aborting scroll loop: reached max scroll iterations {effective_max_iterations}; processed {len(existing_hashes)} points so far."
                )
                break

            result = client.scroll(
                collection_name=collection_name,
                limit=scroll_batch_size,
                offset=offset,
                with_payload=["content_hash"],
                with_vectors=False,
            )

            points, next_offset = result
            if not points:
                logger.info("Scroll loop ended: no more points returned.")
                break

            for point in points:
                payload = point.payload or {}
                existing_hashes[point.id] = payload.get("content_hash")

            offset = next_offset
            iterations += 1

            logger.debug(
                "Scroll progress: iteration %d, retrieved %d points, offset=%s",
                iterations,
                len(points),
                repr(offset),
            )

            if offset is None:
                logger.info("Scroll loop ended: no further offset to continue.")
                break

        logger.info(f"Found {len(existing_hashes)} existing points in collection")

        # Classify changes
        new_point_ids = set(new_doc_map.keys())
        existing_point_ids = set(existing_hashes.keys())

        # Points to delete (exist in Qdrant but not in new set)
        to_delete: list[models.ExtendedPointId] = list(existing_point_ids - new_point_ids)

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
            f"Incremental index: {unchanged_count} unchanged, {len(to_upsert)} to upsert, {len(to_delete)} to delete"
        )

        # Delete removed points
        if to_delete:
            logger.info(f"Deleting {len(to_delete)} removed points...")
            # Delete in batches to avoid payload size issues
            for i in range(0, len(to_delete), effective_batch_size):
                batch_ids = to_delete[i : i + effective_batch_size]
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=batch_ids),
                )

        # Upsert new/modified points
        if to_upsert:
            logger.info(
                f"Upserting {len(to_upsert)} new/modified points... (batch_size={effective_batch_size})"
            )
            start = time.monotonic()
            total_batches = (len(to_upsert) + effective_batch_size - 1) // effective_batch_size
            for batch_idx, i in enumerate(
                tqdm(range(0, len(to_upsert), effective_batch_size), desc="Incremental upsert"),
                start=1,
            ):
                batch = to_upsert[i : i + effective_batch_size]
                batch_points = [
                    models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=_build_payload(doc),
                    )
                    for point_id, embedding, doc in batch
                ]
                client.upsert(collection_name=collection_name, points=batch_points)

                if batch_idx == 1 or batch_idx % 50 == 0 or batch_idx == total_batches:
                    processed = min(batch_idx * effective_batch_size, len(to_upsert))
                    elapsed = time.monotonic() - start
                    logger.info(
                        f"Incremental upsert progress: {processed}/{len(to_upsert)} points "
                        + f"({batch_idx}/{total_batches} batches, {elapsed:.1f}s elapsed)"
                    )
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
        alias_ops: list[models.DeleteAliasOperation | models.CreateAliasOperation] = []
        if old_collection_name:
            alias_ops.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias_name))
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
            config_path=self.config_path,
            incremental=incremental,
        )

        # Choose indexing strategy
        if self.config.incremental.alias_swap:
            num_indexed = self.index_with_alias_swap(embeddings, documents)
        else:
            # Create/update the main collection for non-alias flows.
            self.create_collection(recreate=recreate)

            # Migration guard: prior versions used integer / non-UUID point IDs.
            # Without a one-time cleanup, UUID-based upserts can leave stale legacy points.
            if not recreate and self._has_legacy_point_ids():
                logger.warning(
                    "Detected legacy non-UUID point IDs in existing collection. "
                    + "Recreating collection to migrate safely to UUID-based point IDs."
                )
                self.create_collection(recreate=True)

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
    indexer = QdrantIndexer(config, config_path=config_path)
    return indexer.index(recreate=recreate, incremental=incremental)
