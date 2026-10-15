"""The indexer treats a collection alias as an existing collection.

The blue-green rebuild (index_with_alias_swap) leaves the configured name as an
alias to a ``<name>_staging_<ts>`` collection. The server refuses to create a
collection under an alias name, so every existence check must see the alias.

These tests use the in-memory ``QdrantClient(":memory:")``, which supports
aliases and resolves reads and writes through them like the server does. It
does not refuse a create under an alias name, so the client is wrapped in a
spy to assert that no create is attempted.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from rainrag.config import Config
from rainrag.index import CollectionIsAliasError, QdrantIndexer, doc_id_to_uuid
from rainrag.ingest import Document


STAGING = "test_collection_staging_1"


def _documents(n: int) -> list[Document]:
    return [
        Document(
            id=f"doc{i}",
            path=f"/path/file{i}.vtt",
            language="en",
            text=f"Text {i}",
            length=6,
            content_hash=f"h{i}",
        )
        for i in range(n)
    ]


@pytest.fixture
def aliased_indexer(test_config: Config) -> tuple[QdrantIndexer, MagicMock]:
    """An indexer whose configured name is an alias to a staging collection."""
    alias = test_config.qdrant.collection_name
    real = QdrantClient(":memory:")
    real.create_collection(
        collection_name=STAGING,
        vectors_config=models.VectorParams(
            size=test_config.qdrant.vector_size, distance=models.Distance.COSINE
        ),
    )
    real.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(alias_name=alias, collection_name=STAGING)
            )
        ]
    )
    spy = MagicMock(wraps=real)
    indexer = QdrantIndexer(test_config)
    indexer.client = spy
    return indexer, spy


def test_collection_exists_is_true_for_an_alias(
    aliased_indexer: tuple[QdrantIndexer, MagicMock], test_config: Config
) -> None:
    indexer, _ = aliased_indexer
    assert indexer._collection_exists() is True
    assert indexer._collection_exists(test_config.qdrant.collection_name) is True
    assert indexer._collection_exists(STAGING) is True
    assert indexer._collection_exists("missing") is False


def test_create_collection_on_an_alias_does_not_create(
    aliased_indexer: tuple[QdrantIndexer, MagicMock],
) -> None:
    indexer, spy = aliased_indexer
    indexer.create_collection()
    spy.create_collection.assert_not_called()
    spy.delete_collection.assert_not_called()


def test_create_collection_recreate_on_an_alias_refuses(
    aliased_indexer: tuple[QdrantIndexer, MagicMock], test_config: Config
) -> None:
    indexer, spy = aliased_indexer
    with pytest.raises(CollectionIsAliasError) as exc_info:
        indexer.create_collection(recreate=True)
    message = str(exc_info.value)
    assert test_config.qdrant.collection_name in message
    assert STAGING in message
    spy.create_collection.assert_not_called()
    spy.delete_collection.assert_not_called()


def test_drop_collection_leaves_an_alias_alone(
    aliased_indexer: tuple[QdrantIndexer, MagicMock], test_config: Config
) -> None:
    indexer, spy = aliased_indexer
    indexer.drop_collection(test_config.qdrant.collection_name)
    spy.delete_collection.assert_not_called()
    assert STAGING in indexer.list_collections()


@patch("rainrag.index.run_embedding")
def test_incremental_index_writes_through_the_alias(
    mock_run_embedding: MagicMock,
    aliased_indexer: tuple[QdrantIndexer, MagicMock],
    test_config: Config,
) -> None:
    indexer, spy = aliased_indexer
    documents = _documents(3)
    embeddings = np.random.rand(3, test_config.qdrant.vector_size).astype(np.float32)
    mock_run_embedding.return_value = (embeddings, documents)
    indexer.connect = MagicMock()

    assert test_config.incremental.enabled
    assert not test_config.incremental.alias_swap
    count = indexer.index(incremental=True)

    assert count == 3
    spy.create_collection.assert_not_called()
    spy.delete_collection.assert_not_called()
    assert spy.count(STAGING).count == 3
    stored, _ = spy.scroll(STAGING, limit=10)
    assert {str(p.id) for p in stored} == {doc_id_to_uuid(d.id) for d in documents}
    aliases = spy.get_aliases().aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [
        (test_config.qdrant.collection_name, STAGING)
    ]


@patch("rainrag.index.run_embedding")
def test_recreate_on_an_alias_rebuilds_blue_green(
    mock_run_embedding: MagicMock,
    aliased_indexer: tuple[QdrantIndexer, MagicMock],
    test_config: Config,
) -> None:
    indexer, spy = aliased_indexer
    alias = test_config.qdrant.collection_name
    documents = _documents(2)
    embeddings = np.random.rand(2, test_config.qdrant.vector_size).astype(np.float32)
    mock_run_embedding.return_value = (embeddings, documents)
    indexer.connect = MagicMock()

    count = indexer.index(recreate=True)

    assert count == 2
    # The live collection is replaced only after the new one is built and the
    # alias points at it; no create or delete ever targets the alias name.
    created = [c.kwargs.get("collection_name") for c in spy.create_collection.call_args_list]
    assert alias not in created
    assert len(created) == 1 and created[0].startswith(f"{alias}_staging_")
    new_target = created[0]
    aliases = spy.get_aliases().aliases
    assert [(a.alias_name, a.collection_name) for a in aliases] == [(alias, new_target)]
    assert spy.count(new_target).count == 2
    assert indexer.list_collections() == [new_target]
