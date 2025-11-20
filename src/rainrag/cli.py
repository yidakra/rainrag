"""CLI interface for RainRAG using Typer."""

from pathlib import Path

import typer
from loguru import logger

from rainrag.config import load_config
from rainrag.embed import run_embedding
from rainrag.index import run_indexing
from rainrag.ingest import run_ingestion
from rainrag.query import run_query


app = typer.Typer(
    name="rainrag",
    help="Local-first RAG pipeline for VTT subtitle processing",
    add_completion=False,
)


def setup_logging(config_path: str = "config.yaml") -> None:
    """
    Setup logging configuration.

    Args:
        config_path: Path to configuration file
    """
    try:
        config = load_config(config_path)

        # Remove default logger
        logger.remove()

        # Add console logger
        logger.add(
            lambda msg: typer.echo(msg, err=True),
            format=config.logging.format,
            level=config.logging.level,
            colorize=True,
        )

        # Add file logger if specified
        if config.logging.log_file:
            log_file = Path(config.logging.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                config.logging.log_file,
                format=config.logging.format,
                level=config.logging.level,
                rotation="10 MB",
                retention="1 week",
            )

    except Exception as e:
        typer.echo(f"Warning: Failed to setup logging: {e}", err=True)


@app.command()
def ingest(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """
    Ingest and parse VTT files from archive directory.

    This command will:
    1. Recursively find all .vtt files in the archive directory
    2. Parse and clean the transcript text
    3. Detect language (ru/en) from file path
    4. Save parsed documents to JSONL format
    """
    setup_logging(config)

    try:
        typer.echo("🚀 Starting ingestion pipeline...")

        doc_count = run_ingestion(config)

        typer.echo(f"✅ Ingestion complete! Processed {doc_count} documents")

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        typer.echo(f"❌ Ingestion failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def embed(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force regeneration of embeddings even if cache exists",
    ),
) -> None:
    """
    Generate embeddings for parsed documents.

    This command will:
    1. Load parsed documents from JSONL
    2. Load the multilingual-e5-large model
    3. Generate embeddings for all documents
    4. Cache embeddings locally
    """
    setup_logging(config)

    try:
        typer.echo("🚀 Starting embedding generation...")

        embeddings, documents = run_embedding(config, force_regenerate=force)

        typer.echo(f"✅ Embedding complete! Generated embeddings for {len(documents)} documents")
        typer.echo(f"   Embedding shape: {embeddings.shape}")

    except Exception as e:
        logger.exception(f"Embedding failed: {e}")
        typer.echo(f"❌ Embedding failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def index(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    recreate: bool = typer.Option(
        False,
        "--recreate",
        "-r",
        help="Recreate the collection (deletes existing data)",
    ),
) -> None:
    """
    Index embeddings into Qdrant vector store.

    This command will:
    1. Connect to local Qdrant instance
    2. Create or update the collection
    3. Index all embeddings with metadata
    4. Verify the indexing
    """
    setup_logging(config)

    try:
        typer.echo("🚀 Starting indexing pipeline...")

        if recreate:
            typer.echo("⚠️  Warning: Recreating collection (existing data will be deleted)")

        num_indexed = run_indexing(config, recreate=recreate)

        typer.echo(f"✅ Indexing complete! Indexed {num_indexed} documents")

    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        typer.echo(f"❌ Indexing failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def pipeline(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    skip_ingest: bool = typer.Option(
        False,
        "--skip-ingest",
        help="Skip ingestion step",
    ),
    skip_embed: bool = typer.Option(
        False,
        "--skip-embed",
        help="Skip embedding step",
    ),
    recreate_index: bool = typer.Option(
        False,
        "--recreate-index",
        help="Recreate the Qdrant collection",
    ),
) -> None:
    """
    Run the full pipeline: ingest -> embed -> index.

    This is a convenience command that runs all three steps in sequence.
    """
    setup_logging(config)

    try:
        typer.echo("🚀 Starting full pipeline...")

        # Step 1: Ingest
        if not skip_ingest:
            typer.echo("\n📂 Step 1/3: Ingestion")
            doc_count = run_ingestion(config)
            typer.echo(f"   ✓ Processed {doc_count} documents")
        else:
            typer.echo("\n📂 Step 1/3: Ingestion (skipped)")

        # Step 2: Embed
        if not skip_embed:
            typer.echo("\n🧮 Step 2/3: Embedding")
            embeddings, documents = run_embedding(config)
            typer.echo(f"   ✓ Generated embeddings for {len(documents)} documents")
        else:
            typer.echo("\n🧮 Step 2/3: Embedding (skipped)")

        # Step 3: Index
        typer.echo("\n📊 Step 3/3: Indexing")
        num_indexed = run_indexing(config, recreate=recreate_index)
        typer.echo(f"   ✓ Indexed {num_indexed} documents")

        typer.echo("\n✅ Full pipeline complete!")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        typer.echo(f"\n❌ Pipeline failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask"),
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    top_k: int = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Number of documents to retrieve (default from config)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show retrieved documents and sources",
    ),
) -> None:
    """
    Ask a question and get an answer based on the video transcripts.

    This command will:
    1. Embed your question using the same model as documents
    2. Search Qdrant for the most relevant transcript chunks
    3. Generate an answer using the configured LLM provider (Mistral/OpenAI/Claude/Gemini)

    Example:
        rainrag ask "О чём говорили в выпуске про энергетику?"
    """
    setup_logging(config)

    try:
        typer.echo("🔍 Processing your question...\n")

        result = run_query(config, question, top_k)

        # Display the answer
        typer.echo("💡 Answer:")
        typer.echo("=" * 70)
        typer.echo(result["answer"])
        typer.echo("=" * 70)

        # Display metadata
        typer.echo(f"\n📊 Retrieved {result['num_documents']} relevant documents")

        # Display sources if verbose
        if verbose:
            typer.echo("\n📚 Sources:")
            for doc in result["retrieved_documents"]:
                typer.echo(f"\n  [{doc['rank']}] Score: {doc['score']:.4f}")
                typer.echo(f"      Path: {doc['path']}")
                typer.echo(f"      Language: {doc['language']}")
                typer.echo(f"      Preview: {doc['text'][:200]}...")

    except Exception as e:
        logger.exception(f"Query failed: {e}")
        typer.echo(f"❌ Query failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def info(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """
    Display information about the current configuration and state.
    """
    setup_logging(config)

    try:
        from rainrag.config import load_config
        from rainrag.index import QdrantIndexer

        cfg = load_config(config)

        typer.echo("📋 RainRAG Configuration")
        typer.echo("=" * 50)
        typer.echo("\nPaths:")
        typer.echo(f"  Archive root:      {cfg.paths.archive_root}")
        typer.echo(f"  Docs output:       {cfg.paths.docs_output}")
        typer.echo(f"  Embeddings cache:  {cfg.paths.embeddings_cache}")

        typer.echo("\nEmbedding:")
        typer.echo(f"  Provider:          {cfg.embedding.provider}")
        typer.echo(f"  Model:             {cfg.embedding.model_name}")
        if cfg.embedding.provider == "local":
            typer.echo(f"  Device:            {cfg.embedding.device}")
        typer.echo(f"  Batch size:        {cfg.embedding.batch_size}")

        typer.echo("\nQdrant:")
        typer.echo(f"  Host:              {cfg.qdrant.host}:{cfg.qdrant.port}")
        typer.echo(f"  Collection:        {cfg.qdrant.collection_name}")
        typer.echo(f"  Vector size:       {cfg.qdrant.vector_size}")
        typer.echo(f"  Distance metric:   {cfg.qdrant.distance}")

        typer.echo("\nLLM Provider:")
        typer.echo(f"  Provider:          {cfg.llm.provider}")

        if cfg.llm.provider == "mistral":
            typer.echo(f"  Model:             {cfg.mistral.model_name}")
            typer.echo(f"  Max tokens:        {cfg.mistral.max_tokens}")
            typer.echo(f"  Temperature:       {cfg.mistral.temperature}")
            typer.echo(f"  Top-k docs:        {cfg.mistral.top_k}")
        elif cfg.llm.provider == "openai":
            typer.echo(f"  Model:             {cfg.openai.model_name}")
            typer.echo(f"  Max tokens:        {cfg.openai.max_tokens}")
            typer.echo(f"  Temperature:       {cfg.openai.temperature}")
            typer.echo(f"  Top-k docs:        {cfg.openai.top_k}")
        elif cfg.llm.provider == "claude":
            typer.echo(f"  Model:             {cfg.claude.model_name}")
            typer.echo(f"  Max tokens:        {cfg.claude.max_tokens}")
            typer.echo(f"  Temperature:       {cfg.claude.temperature}")
            typer.echo(f"  Top-k docs:        {cfg.claude.top_k}")
        elif cfg.llm.provider == "gemini":
            typer.echo(f"  Model:             {cfg.gemini.model_name}")
            typer.echo(f"  Max tokens:        {cfg.gemini.max_tokens}")
            typer.echo(f"  Temperature:       {cfg.gemini.temperature}")
            typer.echo(f"  Top-k docs:        {cfg.gemini.top_k}")

        # Try to get collection info
        try:
            indexer = QdrantIndexer(cfg)
            indexer.connect()
            stats = indexer.get_collection_info()

            if stats:
                typer.echo("\nCollection Status:")
                typer.echo(f"  Points count:      {stats.get('points_count', 'N/A')}")
                typer.echo(f"  Vectors count:     {stats.get('vectors_count', 'N/A')}")
                typer.echo(f"  Status:            {stats.get('status', 'N/A')}")

        except Exception as e:
            typer.echo(f"\n⚠️  Could not connect to Qdrant: {e}")

    except Exception as e:
        logger.exception(f"Failed to get info: {e}")
        typer.echo(f"❌ Failed to get info: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def mcp(
    config: str = typer.Option(
        "config.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    transport: str = typer.Option(
        None,
        "--transport",
        "-t",
        help="Transport protocol (stdio, sse, streamable-http). Defaults to config value.",
    ),
    host: str = typer.Option(
        None,
        "--host",
        "-h",
        help="Host for HTTP-based transports. Defaults to config value.",
    ),
    port: int = typer.Option(
        None,
        "--port",
        "-p",
        help="Port for HTTP-based transports. Defaults to config value.",
    ),
) -> None:
    """
    Run the MCP (Model Context Protocol) server.

    This exposes the RainRAG system as an MCP server that can be used by
    AI assistants like Claude Desktop, ChatGPT, and Cursor.

    The server provides two main tools:
    - query_rag: Full RAG pipeline (retrieve + generate answer)
    - retrieve_documents: Retrieval only (no LLM generation)

    Transport options:
    - stdio: Standard input/output (for Claude Desktop, Cursor)
    - streamable-http: HTTP server (for remote connections)
    - sse: Server-Sent Events

    Examples:
        # Run with stdio transport (default for Claude Desktop)
        rainrag mcp

        # Run with HTTP transport on custom port
        rainrag mcp --transport streamable-http --port 8080

        # Run with specific config file
        rainrag mcp --config custom-config.yaml
    """
    setup_logging(config)

    try:
        from rainrag.config import load_config
        from rainrag.mcp_server import run_server

        # Load config to get defaults
        cfg = load_config(config)

        # Use CLI args if provided, otherwise use config values
        transport_to_use = transport or cfg.mcp.transport
        host_to_use = host or cfg.mcp.host
        port_to_use = port or cfg.mcp.port

        typer.echo("🚀 Starting MCP server...")
        typer.echo(f"   Transport: {transport_to_use}")
        if transport_to_use != "stdio":
            typer.echo(f"   Address: {host_to_use}:{port_to_use}")
        typer.echo("")

        # Run the MCP server (this will block)
        run_server(
            config_path=config,
            transport=transport_to_use,
            host=host_to_use,
            port=port_to_use,
        )

    except KeyboardInterrupt:
        typer.echo("\n\n⏹️  MCP server stopped")
        raise typer.Exit(code=0)
    except Exception as e:
        logger.exception(f"MCP server failed: {e}")
        typer.echo(f"❌ MCP server failed: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
