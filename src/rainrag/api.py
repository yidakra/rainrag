"""FastAPI backend for RainRAG query interface."""

from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import os

from rainrag.config import load_config
from rainrag.query import RAGQueryEngine


# Global query engine instance
query_engine: Optional[RAGQueryEngine] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., description="The user's question", min_length=1)
    language: str = Field(
        default="ru", description="Response language (ru or en)", pattern="^(ru|en)$"
    )
    top_k: Optional[int] = Field(
        default=None, description="Number of context chunks to retrieve", ge=1, le=20
    )


class ContextChunk(BaseModel):
    """Model for a single context chunk."""

    text: str
    filename: str
    language: str
    score: float
    rank: int
    doc_id: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    context: List[ContextChunk]
    question: str
    num_documents: int


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    qdrant_connected: bool
    model_loaded: bool
    vllm_model: str
    qdrant_collection: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the query engine on startup."""
    global query_engine

    logger.info("Initializing RainRAG API...")

    # Load configuration
    config_path = os.getenv("RAINRAG_CONFIG", "config.yaml")
    config = load_config(config_path)

    # Initialize query engine
    query_engine = RAGQueryEngine(config)
    query_engine.initialize()

    logger.info("RainRAG API initialized successfully")

    yield

    # Cleanup (if needed)
    logger.info("Shutting down RainRAG API...")


# Create FastAPI app
app = FastAPI(
    title="RainRAG API",
    description="Query interface for RainRAG - Multilingual RAG system for video transcripts",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware to allow cross-origin requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_auth_token(authorization: Optional[str] = Header(None)) -> bool:
    """Verify authentication token if configured."""
    required_token = os.getenv("RAINRAG_AUTH_TOKEN")

    # If no token is configured, allow all requests
    if not required_token:
        return True

    # Check if authorization header is provided
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Extract token from "Bearer <token>" format
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

    # Verify token
    if token != required_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    return True


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        System health status
    """
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    qdrant_connected = query_engine.qdrant_client is not None
    model_loaded = query_engine.embedding_model is not None

    return HealthResponse(
        status="healthy" if (qdrant_connected and model_loaded) else "degraded",
        qdrant_connected=qdrant_connected,
        model_loaded=model_loaded,
        vllm_model=query_engine.config.vllm.model_name,
        qdrant_collection=query_engine.config.qdrant.collection_name,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, authorized: bool = Header(default=True)):
    """
    Query the RAG system.

    Args:
        request: Query request containing question and parameters
        authorized: Authorization check result (injected by dependency)

    Returns:
        Answer and retrieved context chunks
    """
    # Verify authentication
    verify_auth_token()

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    try:
        logger.info(f"Received query: {request.question[:100]}... (language: {request.language})")

        # Execute query
        result = query_engine.query(question=request.question, top_k=request.top_k)

        # Format response
        context_chunks = [
            ContextChunk(
                text=doc["text"],
                filename=doc["path"],
                language=doc.get("language", "unknown"),
                score=doc["score"],
                rank=doc["rank"],
                doc_id=doc.get("doc_id", ""),
            )
            for doc in result["retrieved_documents"]
        ]

        response = QueryResponse(
            answer=result["answer"],
            context=context_chunks,
            question=result["question"],
            num_documents=result["num_documents"],
        )

        logger.info(f"Query completed successfully. Retrieved {len(context_chunks)} documents")

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RainRAG API",
        "version": "0.1.0",
        "description": "Multilingual RAG system for video transcripts",
        "endpoints": {
            "POST /query": "Submit a question to the RAG system",
            "GET /health": "Check system health",
            "GET /docs": "OpenAPI documentation",
        },
    }


def start_server(
    host: str = "0.0.0.0",
    port: int = 8001,
    config_path: str = "config.yaml",
    reload: bool = False,
):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        config_path: Path to configuration file
        reload: Enable auto-reload for development
    """
    import uvicorn

    # Set config path environment variable
    os.environ["RAINRAG_CONFIG"] = config_path

    logger.info(f"Starting RainRAG API server on {host}:{port}")

    uvicorn.run(
        "rainrag.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    start_server()
