"""The indexer creates the keyword payload indexes that query.py filters on.

Without them Qdrant answers a payload filter by reading every point; on the
production collection (2.7 million points) the seed lookup of /similar-videos
timed out after 60 s. The in-memory QdrantClient accepts create_payload_index
but keeps no index, so these tests assert the calls through a spy or a mock.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qdrant_client import QdrantClient, models
from typer.testing import CliRunner

from rainrag.config import Config
from rainrag.index import PAYLOAD_INDEX_FIELDS, QdrantIndexer


_KEYWORD = SimpleNamespace(data_type=models.PayloadSchemaType.KEYWORD)


def _mock_indexer(test_config: Config, schema: dict, aliases: list | None = None):
    """Return an indexer whose Qdrant client is a mock with the given payload schema and aliases."""
    client = MagicMock()
    client.get_collection.return_value = SimpleNamespace(payload_schema=schema)
    client.get_aliases.return_value = SimpleNamespace(aliases=aliases or [])
    indexer = QdrantIndexer(test_config)
    indexer.client = client
    return indexer, client


def _indexed(client: MagicMock) -> list[tuple[str, str]]:
    """Return (collection, field) for every create_payload_index call on the client."""
    return [
        (c.kwargs["collection_name"], c.kwargs["field_name"])
        for c in client.create_payload_index.call_args_list
    ]


def test_the_fields_are_the_ones_query_filters_on() -> None:
    """The indexed fields are exactly the payload keys that query.py filters on."""
    assert set(PAYLOAD_INDEX_FIELDS) == {"video_id", "path", "language", "doc_id"}


def test_every_missing_field_gets_a_keyword_index(test_config: Config) -> None:
    """With no index yet, each field gets a keyword index, and the call waits for it."""
    indexer, client = _mock_indexer(test_config, schema={})
    created = indexer.ensure_payload_indexes()
    name = test_config.qdrant.collection_name
    assert created == list(PAYLOAD_INDEX_FIELDS)
    assert _indexed(client) == [(name, f) for f in PAYLOAD_INDEX_FIELDS]
    for c in client.create_payload_index.call_args_list:
        assert c.kwargs["field_schema"] == models.PayloadSchemaType.KEYWORD
        assert c.kwargs["wait"] is True


def test_an_existing_index_is_not_created_again(test_config: Config) -> None:
    """A field that already has a keyword index is skipped."""
    indexer, client = _mock_indexer(test_config, schema={"video_id": _KEYWORD, "path": _KEYWORD})
    assert indexer.ensure_payload_indexes() == ["language", "doc_id"]
    assert [f for _, f in _indexed(client)] == ["language", "doc_id"]


def test_a_second_call_creates_nothing(test_config: Config) -> None:
    """When every field is indexed, the call creates nothing."""
    indexer, client = _mock_indexer(
        test_config, schema=dict.fromkeys(PAYLOAD_INDEX_FIELDS, _KEYWORD)
    )
    assert indexer.ensure_payload_indexes() == []
    client.create_payload_index.assert_not_called()


def test_a_keyword_index_given_as_a_string_counts(test_config: Config) -> None:
    """A keyword type reported as the string 'keyword' counts as done."""
    indexer, client = _mock_indexer(
        test_config, schema={f: SimpleNamespace(data_type="keyword") for f in PAYLOAD_INDEX_FIELDS}
    )
    assert indexer.ensure_payload_indexes() == []
    client.create_payload_index.assert_not_called()


def test_an_index_of_another_type_is_replaced(test_config: Config) -> None:
    """A text index is deleted and replaced with a keyword index."""
    text = SimpleNamespace(data_type=models.PayloadSchemaType.TEXT)
    schema = {"video_id": _KEYWORD, "path": _KEYWORD, "language": _KEYWORD, "doc_id": text}
    indexer, client = _mock_indexer(test_config, schema=schema)
    assert indexer.ensure_payload_indexes() == ["doc_id"]
    name = test_config.qdrant.collection_name
    client.delete_payload_index.assert_called_once_with(
        collection_name=name, field_name="doc_id", wait=True
    )
    assert _indexed(client) == [(name, "doc_id")]


def test_an_index_of_unknown_type_is_left_alone(test_config: Config) -> None:
    """An index whose type cannot be read is neither deleted nor rebuilt."""
    unknown = SimpleNamespace(data_type=None)
    indexer, client = _mock_indexer(
        test_config, schema=dict.fromkeys(PAYLOAD_INDEX_FIELDS, unknown)
    )
    assert indexer.ensure_payload_indexes() == []
    client.delete_payload_index.assert_not_called()
    client.create_payload_index.assert_not_called()


def test_an_alias_is_indexed_on_its_collection(test_config: Config) -> None:
    """An alias is resolved, and the indexes go on the collection behind it."""
    alias = test_config.qdrant.collection_name
    indexer, client = _mock_indexer(
        test_config,
        schema={},
        aliases=[SimpleNamespace(alias_name=alias, collection_name="live_staging_1")],
    )
    indexer.ensure_payload_indexes()
    client.get_collection.assert_called_once_with("live_staging_1")
    assert {n for n, _ in _indexed(client)} == {"live_staging_1"}


def _spy_indexer(test_config: Config) -> tuple[QdrantIndexer, MagicMock]:
    """Return an indexer over an in-memory Qdrant client wrapped in a spy."""
    spy = MagicMock(wraps=QdrantClient(":memory:"))
    indexer = QdrantIndexer(test_config)
    indexer.client = spy
    return indexer, spy


def test_a_new_collection_is_indexed(test_config: Config) -> None:
    """create_collection indexes a collection it creates."""
    indexer, spy = _spy_indexer(test_config)
    indexer.create_collection()
    name = test_config.qdrant.collection_name
    assert _indexed(spy) == [(name, f) for f in PAYLOAD_INDEX_FIELDS]


def test_an_existing_collection_is_indexed_without_a_recreate(test_config: Config) -> None:
    """create_collection indexes an existing collection without deleting it."""
    indexer, spy = _spy_indexer(test_config)
    indexer.create_collection()
    spy.create_payload_index.reset_mock()
    spy.delete_collection.reset_mock()

    indexer.create_collection()

    spy.delete_collection.assert_not_called()
    # The in-memory client keeps no index, so every field reads as missing.
    assert [f for _, f in _indexed(spy)] == list(PAYLOAD_INDEX_FIELDS)


def test_a_staging_collection_is_indexed(test_config: Config) -> None:
    """The staging collection of the blue-green rebuild is indexed."""
    indexer, spy = _spy_indexer(test_config)
    indexer._create_collection_with_name("x_staging_1")
    assert _indexed(spy) == [("x_staging_1", f) for f in PAYLOAD_INDEX_FIELDS]


def test_the_cli_command_reports_what_it_created() -> None:
    """ensure-indexes prints the fields it indexed, or that all exist."""
    from rainrag.cli import app

    runner = CliRunner()
    indexer = MagicMock()
    indexer.ensure_payload_indexes.return_value = ["video_id", "path"]
    with (
        patch("rainrag.cli.setup_logging"),
        patch("rainrag.cli.load_config"),
        patch("rainrag.index.QdrantIndexer", return_value=indexer),
    ):
        result = runner.invoke(app, ["ensure-indexes"])
    assert result.exit_code == 0, result.output
    indexer.connect.assert_called_once()
    assert "Created payload indexes: video_id, path" in result.output

    indexer.ensure_payload_indexes.return_value = []
    with (
        patch("rainrag.cli.setup_logging"),
        patch("rainrag.cli.load_config"),
        patch("rainrag.index.QdrantIndexer", return_value=indexer),
    ):
        result = runner.invoke(app, ["ensure-indexes"])
    assert "All payload indexes already exist" in result.output


def test_the_cli_command_fails_loudly() -> None:
    """ensure-indexes exits 1 and prints the error when indexing fails."""
    from rainrag.cli import app

    indexer = MagicMock()
    indexer.ensure_payload_indexes.side_effect = RuntimeError("qdrant down")
    with (
        patch("rainrag.cli.setup_logging"),
        patch("rainrag.cli.load_config"),
        patch("rainrag.index.QdrantIndexer", return_value=indexer),
    ):
        result = CliRunner().invoke(app, ["ensure-indexes"])
    assert result.exit_code == 1
    assert "Creating payload indexes failed: qdrant down" in result.output


def test_a_new_collection_gets_int8_quantization_in_ram(test_config: Config) -> None:
    """create_collection keeps an int8 copy of the vectors in RAM."""
    indexer, spy = _spy_indexer(test_config)
    indexer.create_collection()
    quant = spy.create_collection.call_args.kwargs["quantization_config"]
    assert quant.scalar.type == models.ScalarType.INT8
    assert quant.scalar.always_ram is True


def test_quantization_can_be_turned_off(test_config: Config) -> None:
    """With scalar_quantization false, collections are created without quantization."""
    test_config.qdrant.scalar_quantization = False
    indexer, spy = _spy_indexer(test_config)
    indexer._create_collection_with_name("x_staging_2")
    assert spy.create_collection.call_args.kwargs["quantization_config"] is None
