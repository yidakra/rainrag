"""MCP server for RainRAG - exposes RAG functionality to AI assistants."""

from typing import Any

import uvicorn
from loguru import logger
from mcp.server.fastmcp import FastMCP

from rainrag.config import Config, load_config
from rainrag.query import RAGQueryEngine


# Global query engine instance (initialized on startup)
_query_engine: RAGQueryEngine | None = None
_config: Config | None = None


def get_query_engine() -> RAGQueryEngine:
    """Get the initialized query engine instance."""
    if _query_engine is None:
        raise RuntimeError("Query engine not initialized. This should not happen.")
    return _query_engine


def initialize_server(config_path: str = "config.yaml") -> None:
    """
    Initialize the RAG query engine with configuration.

    Args:
        config_path: Path to configuration file
    """
    global _query_engine, _config

    logger.info(f"Loading configuration from {config_path}")
    _config = load_config(config_path)

    logger.info("Initializing RAG query engine...")
    _query_engine = RAGQueryEngine(_config)
    _query_engine.initialize()
    logger.info("MCP server initialized successfully")


# Create FastMCP server instance
mcp = FastMCP("RainRAG", json_response=True)


@mcp.tool()
def query_rag(
    question: str,
    language: str = "en",
    top_k: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """
    Query the RainRAG system with a question and get an answer with context.

    This tool performs full RAG (Retrieval-Augmented Generation):
    1. Embeds your question
    2. Retrieves relevant video transcript chunks from the vector database
    3. Uses an LLM to generate a comprehensive answer based on the retrieved context

    Args:
        question: The question to ask about the video transcripts
        language: Language for the response - "en" for English or "ru" for Russian (default: "en")
        top_k: Number of document chunks to retrieve (default: uses config value, typically 5)
        date_from: Filter results from this date (YYYY-MM-DD format)
        date_to: Filter results up to this date (YYYY-MM-DD format)

    Returns:
        Dictionary containing:
        - answer: The generated answer from the LLM
        - question: The original question
        - retrieved_documents: List of relevant document chunks with metadata
        - num_documents: Number of documents retrieved

    Example:
        >>> query_rag("What topics are discussed in the videos?", language="en", top_k=3)
        >>> query_rag("protests", date_from="2020-01-01", date_to="2020-12-31")
    """
    engine = get_query_engine()
    logger.info(f"MCP query received: {question[:100]}... (language: {language})")

    try:
        result = engine.query(
            question=question,
            top_k=top_k,
            language=language,
            date_from=date_from,
            date_to=date_to,
        )
        logger.info("MCP query completed successfully")
        return result
    except Exception as e:
        logger.error(f"MCP query failed: {e}")
        raise


@mcp.tool()
def retrieve_documents(
    question: str,
    top_k: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve relevant document chunks without generating an answer.

    This tool only performs the retrieval step:
    1. Embeds your question
    2. Retrieves relevant video transcript chunks from the vector database
    3. Returns the chunks with their metadata (no LLM generation)

    Use this when you just want to see what context is available without
    generating an answer.

    Args:
        question: The question or query to search for relevant documents
        top_k: Number of document chunks to retrieve (default: 5)
        date_from: Filter results from this date (YYYY-MM-DD format)
        date_to: Filter results up to this date (YYYY-MM-DD format)

    Returns:
        Dictionary containing:
        - question: The original question
        - documents: List of relevant document chunks with metadata
          Each document includes: rank, score, text, path, language, doc_id, date, duration_seconds
        - num_documents: Number of documents retrieved

    Example:
        >>> retrieve_documents("machine learning algorithms", top_k=3)
        >>> retrieve_documents("protests", date_from="2020-01-01", date_to="2020-12-31")
    """
    engine = get_query_engine()
    logger.info(f"MCP retrieval request: {question[:100]}... (top_k: {top_k})")

    try:
        # Embed the query
        query_vector = engine.embed_query(question)

        # Retrieve documents with optional date filter
        documents = engine.retrieve_documents(
            query_vector, top_k, date_from=date_from, date_to=date_to
        )

        logger.info("MCP retrieval completed successfully")
        return {
            "question": question,
            "documents": documents,
            "num_documents": len(documents),
        }
    except Exception as e:
        logger.error(f"MCP retrieval failed: {e}")
        raise


@mcp.resource("config://current")
def get_current_config() -> str:
    """
    Get the current RainRAG configuration summary.

    Returns a formatted string with key configuration details including
    embedding provider, LLM provider, and vector database settings.

    Returns:
        Formatted configuration summary
    """
    if _config is None:
        return "Configuration not loaded"

    config_info = f"""RainRAG Configuration:
- Embedding Provider: {_config.embedding.provider}
- Embedding Model: {_config.embedding.model_name}
- LLM Provider: {_config.llm.provider}
- Vector Database: Qdrant at {_config.qdrant.host}:{_config.qdrant.port}
- Collection: {_config.qdrant.collection_name}
- Vector Size: {_config.qdrant.vector_size}

LLM Details:"""

    if _config.llm.provider == "mistral":
        config_info += f"\n- Mistral Model: {_config.mistral.model_name}"
    elif _config.llm.provider == "openai":
        config_info += f"\n- OpenAI Model: {_config.openai.model_name}"
    elif _config.llm.provider == "claude":
        config_info += f"\n- Claude Model: {_config.claude.model_name}"
    elif _config.llm.provider == "gemini":
        config_info += f"\n- Gemini Model: {_config.gemini.model_name}"

    return config_info


def run_server(
    config_path: str = "config.yaml",
    transport: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """
    Run the MCP server.

    Args:
        config_path: Path to configuration file
        transport: Transport protocol - "stdio", "sse", or "streamable-http" (defaults to config value)
        host: Host address for HTTP transports (defaults to config value)
        port: Port for HTTP transports (defaults to config value)
    """
    # Initialize the server with config
    initialize_server(config_path)

    # Use provided values or fall back to config
    transport_to_use = transport or _config.mcp.transport
    host_to_use = host or _config.mcp.host
    port_to_use = port or _config.mcp.port

    logger.info(f"Starting MCP server with transport: {transport_to_use}")

    # Run the server with specified transport
    if transport_to_use == "streamable-http":
        logger.info(f"MCP server running at http://{host_to_use}:{port_to_use}/mcp")
        # For HTTP transport, we need to use uvicorn directly
        # Get the ASGI app from FastMCP and run it with uvicorn
        asgi_app = mcp.streamable_http_app()
        uvicorn.run(asgi_app, host=host_to_use, port=port_to_use)
    else:
        logger.info(f"MCP server running with {transport_to_use} transport")
        mcp.run(transport=transport_to_use)


if __name__ == "__main__":
    # Default: run with stdio transport (for use with Claude Desktop, etc.)
    run_server()
