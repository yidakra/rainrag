"""FastAPI backend for RainRAG query interface."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from rainrag.config import Config, load_config
from rainrag.query import RAGQueryEngine


# Global query engine instance and config
query_engine: RAGQueryEngine | None = None
config: Config | None = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    question: str = Field(..., description="The user's question", min_length=1)
    language: str = Field(
        default="ru", description="Response language (ru or en)", pattern="^(ru|en)$"
    )
    top_k: int | None = Field(
        default=None, description="Number of context chunks to retrieve", ge=1, le=20
    )
    date_from: str | None = Field(
        default=None, description="Filter results from this date (YYYY-MM-DD)"
    )
    date_to: str | None = Field(
        default=None, description="Filter results up to this date (YYYY-MM-DD)"
    )


class ContextChunk(BaseModel):
    """Model for a single context chunk."""

    text: str
    filename: str
    language: str
    score: float
    rank: int
    doc_id: str
    rerank_score: float | None = None  # Cohere reranking score (if available)
    video_url: str | None = None
    vtt_url: str | None = None
    group_id: str | None = None  # Base name for grouping multilingual versions
    date: str | None = None  # Recording date (from video mtime)
    duration_seconds: float | None = None  # Video duration in seconds
    start_time: str | None = None  # First timecode (HH:MM:SS)
    end_time: str | None = None  # Last timecode (HH:MM:SS)


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    answer: str
    context: list[ContextChunk]
    question: str
    num_documents: int


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str
    qdrant_connected: bool
    model_loaded: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    qdrant_collection: str
    # Deprecated fields (kept for backwards compatibility)
    mistral_model: str


def timecode_to_seconds(timecode: str) -> int | None:
    """
    Convert HH:MM:SS timecode to seconds.

    Args:
        timecode: Timecode in HH:MM:SS format

    Returns:
        Total seconds or None if parsing fails
    """
    try:
        parts = timecode.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        return None
    except (ValueError, AttributeError):
        return None


def find_video_file(vtt_path: str) -> str | None:
    """
    Find video file corresponding to a VTT file.

    Looks for video files with the same base name as the VTT file.
    Supports multiple resolutions (prefers 1080p > 720p > 480p > 360p > 180p).

    Args:
        vtt_path: Path to the VTT file

    Returns:
        Path to video file if found, None otherwise
    """
    if config is None or not config.video.enabled:
        return None

    vtt_file = Path(vtt_path)

    # Determine the directory to search

    # Get base name without VTT extension
    base_name = vtt_file.stem
    # Remove language suffixes like .en or .ru
    for lang_suffix in [".en", ".ru"]:
        if base_name.endswith(lang_suffix):
            base_name = base_name[: -len(lang_suffix)]
            break

    # Search for video file in the same directory as VTT
    vtt_dir = vtt_file.parent

    # Quality preference order (highest to lowest)
    quality_order = ["1080p", "720p", "480p", "360p", "180p"]

    # First, try to find video files with quality suffixes
    for quality in quality_order:
        for ext in config.video.extensions:
            # Try with underscore separator (e.g., hash_1080p.mp4)
            video_file = vtt_dir / f"{base_name}_{quality}{ext}"
            if video_file.exists():
                return str(video_file)

    # If no quality-suffixed video found, look for exact match
    for ext in config.video.extensions:
        video_file = vtt_dir / f"{base_name}{ext}"
        if video_file.exists():
            return str(video_file)

    # Finally, try to find any video file that starts with the base name
    try:
        for video_file in vtt_dir.iterdir():
            if not video_file.is_file():
                continue

            # Check if file starts with base name and has a video extension
            if video_file.name.startswith(base_name) and any(
                video_file.suffix.lower() == ext for ext in config.video.extensions
            ):
                return str(video_file)
    except Exception as e:
        logger.warning(f"Error searching for video files: {e}")

    return None


def get_video_base_name(vtt_path: str) -> str:
    """
    Extract the base name from a VTT file path for grouping.

    Removes language suffixes (.en, .ru) to group multilingual versions together.

    Args:
        vtt_path: Path to the VTT file

    Returns:
        Base name without language suffix
    """
    vtt_file = Path(vtt_path)
    base_name = vtt_file.stem

    # Remove language suffixes like .en or .ru
    for lang_suffix in [".en", ".ru"]:
        if base_name.endswith(lang_suffix):
            base_name = base_name[: -len(lang_suffix)]
            break

    # Include the parent directory to make it unique across different videos
    # This creates a group_id like "archive/subdir/hash"
    return f"{vtt_file.parent.name}/{base_name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the query engine on startup."""
    global query_engine, config

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


def verify_auth_token(authorization: str | None = Header(None)) -> bool:
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
    model_loaded = (
        query_engine.embedding_model is not None
        or query_engine.config.embedding.provider in ["mistral", "openai", "gemini"]
    )

    # Get LLM model based on provider
    llm_provider = query_engine.config.llm.provider
    if llm_provider == "mistral":
        llm_model = query_engine.config.mistral.model_name
    elif llm_provider == "openai":
        llm_model = query_engine.config.openai.model_name
    elif llm_provider == "claude":
        llm_model = query_engine.config.claude.model_name
    elif llm_provider == "gemini":
        llm_model = query_engine.config.gemini.model_name
    else:
        llm_model = "unknown"

    # Get embedding model based on provider
    embedding_provider = query_engine.config.embedding.provider
    if embedding_provider == "local":
        embedding_model = query_engine.config.embedding.model_name
    elif embedding_provider == "mistral":
        embedding_model = "mistral-embed"
    elif embedding_provider == "openai":
        embedding_model = query_engine.config.openai.embedding_model
    elif embedding_provider == "gemini":
        embedding_model = query_engine.config.gemini.embedding_model
    else:
        embedding_model = "unknown"

    return HealthResponse(
        status="healthy" if (qdrant_connected and model_loaded) else "degraded",
        qdrant_connected=qdrant_connected,
        model_loaded=model_loaded,
        llm_provider=llm_provider,
        llm_model=llm_model,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        qdrant_collection=query_engine.config.qdrant.collection_name,
        # Deprecated field (kept for backwards compatibility)
        mistral_model=llm_model,
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

        # Execute query with language and date filter parameters
        result = query_engine.query(
            question=request.question,
            top_k=request.top_k,
            language=request.language,
            date_from=request.date_from,
            date_to=request.date_to,
        )

        # Format response with video and VTT URLs
        context_chunks = []
        for doc in result["retrieved_documents"]:
            vtt_path = doc["path"]

            # Generate URLs for video and VTT files
            video_url = None
            vtt_url = None

            if config and config.video.enabled:
                # Find corresponding video file
                video_file = find_video_file(vtt_path)
                if video_file:
                    # Create relative path from video_root
                    video_root = (
                        config.paths.video_root
                        if config.paths.video_root
                        else config.paths.archive_root
                    )
                    try:
                        video_rel = Path(video_file).relative_to(video_root)
                        video_url = f"/video/{video_rel}"
                        # Add timecode fragment for video player to start at
                        start_time = doc.get("start_time")
                        if start_time:
                            start_seconds = timecode_to_seconds(start_time)
                            if start_seconds is not None:
                                video_url = f"{video_url}#t={start_seconds}"
                    except ValueError:
                        logger.warning(f"Video file {video_file} is not under video_root")

                # Create VTT URL
                archive_root = config.paths.archive_root
                try:
                    vtt_rel = Path(vtt_path).relative_to(archive_root)
                    vtt_url = f"/vtt/{vtt_rel}"
                except ValueError:
                    logger.warning(f"VTT file {vtt_path} is not under archive_root")

            # Get group ID for grouping multilingual versions
            group_id = get_video_base_name(vtt_path)

            context_chunks.append(
                ContextChunk(
                    text=doc["text"],
                    filename=doc["path"],
                    language=doc.get("language", "unknown"),
                    score=doc["score"],
                    rank=doc["rank"],
                    doc_id=doc.get("doc_id", ""),
                    rerank_score=doc.get("rerank_score"),
                    video_url=video_url,
                    vtt_url=vtt_url,
                    group_id=group_id,
                    date=doc.get("date"),
                    duration_seconds=doc.get("duration_seconds"),
                    start_time=doc.get("start_time"),
                    end_time=doc.get("end_time"),
                )
            )

        response = QueryResponse(
            answer=result["answer"],
            context=context_chunks,
            question=result["question"],
            num_documents=result["num_documents"],
        )

        logger.info(f"Query completed successfully. Retrieved {len(context_chunks)} documents")

        return response

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"Query failed: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


class RelatedChunksRequest(BaseModel):
    """Request model for related chunks endpoint."""

    chunk_id: str = Field(..., description="The ID of the source chunk", min_length=1)
    top_k: int = Field(default=5, description="Number of related chunks to return", ge=1, le=10)
    same_video_only: bool = Field(
        default=False, description="If true, only return chunks from the same video"
    )


class RelatedChunksResponse(BaseModel):
    """Response model for related chunks endpoint."""

    chunk_id: str
    related_chunks: list[ContextChunk]
    num_related: int


@app.post("/related-chunks", response_model=RelatedChunksResponse)
async def get_related_chunks(
    request: RelatedChunksRequest, authorized: bool = Header(default=True)
):
    """
    Find chunks related to a given chunk based on vector similarity.

    Args:
        request: Related chunks request containing chunk_id and parameters
        authorized: Authorization check result (injected by dependency)

    Returns:
        List of related chunks with similarity scores
    """
    verify_auth_token()

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    try:
        logger.info(
            f"Finding related chunks for: {request.chunk_id} "
            f"(top_k={request.top_k}, same_video_only={request.same_video_only})"
        )

        # Find related chunks
        related_docs = query_engine.find_related_chunks(
            chunk_id=request.chunk_id, top_k=request.top_k, same_video_only=request.same_video_only
        )

        # Format response with video and VTT URLs
        related_chunks = []
        for doc in related_docs:
            vtt_path = doc["path"]

            # Generate URLs for video and VTT files
            video_url = None
            vtt_url = None

            if config and config.video.enabled:
                # Find corresponding video file
                video_file = find_video_file(vtt_path)
                if video_file:
                    # Create relative path from video_root
                    video_root = (
                        config.paths.video_root
                        if config.paths.video_root
                        else config.paths.archive_root
                    )
                    try:
                        video_rel = Path(video_file).relative_to(video_root)
                        video_url = f"/video/{video_rel}"
                        # Add timecode fragment for video player to start at
                        start_time = doc.get("start_time")
                        if start_time:
                            start_seconds = timecode_to_seconds(start_time)
                            if start_seconds is not None:
                                video_url = f"{video_url}#t={start_seconds}"
                    except ValueError:
                        pass

                # VTT file URL
                try:
                    vtt_rel = Path(vtt_path).relative_to(config.paths.archive_root)
                    vtt_url = f"/vtt/{vtt_rel}"
                except ValueError:
                    pass

            # Get base name for grouping
            group_id = get_video_base_name(vtt_path)

            chunk = ContextChunk(
                text=doc["text"],
                filename=Path(vtt_path).name,
                language=doc.get("language", "unknown"),
                score=doc["score"],
                rank=related_docs.index(doc) + 1,
                doc_id=doc["doc_id"],
                video_url=video_url,
                vtt_url=vtt_url,
                group_id=group_id,
            )
            related_chunks.append(chunk)

        logger.info(f"Returning {len(related_chunks)} related chunks")

        return RelatedChunksResponse(
            chunk_id=request.chunk_id,
            related_chunks=related_chunks,
            num_related=len(related_chunks),
        )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"Finding related chunks failed: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Finding related chunks failed: {str(e)}")


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
            "GET /video/{file_path:path}": "Serve video files",
            "GET /vtt/{file_path:path}": "Serve VTT subtitle files",
            "GET /docs": "OpenAPI documentation",
        },
    }


@app.api_route("/video/{file_path:path}", methods=["GET", "HEAD"])
async def serve_video(file_path: str):
    """
    Serve video files.

    Args:
        file_path: Path to the video file (relative to video_root)

    Returns:
        Video file as streaming response
    """
    verify_auth_token()

    if config is None or not config.video.enabled:
        raise HTTPException(status_code=404, detail="Video serving is disabled")

    # Convert to absolute path
    video_root = config.paths.video_root if config.paths.video_root else config.paths.archive_root
    full_path = Path(video_root) / file_path

    # Security check: ensure the path is within video_root
    try:
        full_path = full_path.resolve()
        video_root_resolved = Path(video_root).resolve()
        if not str(full_path).startswith(str(video_root_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Path resolution error: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Check if file exists
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    # Check if file has valid video extension
    if full_path.suffix.lower() not in config.video.extensions:
        raise HTTPException(status_code=400, detail="Invalid video file type")

    logger.info(f"Serving video file: {full_path}")

    # Determine media type based on extension
    media_types = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
    }
    media_type = media_types.get(full_path.suffix.lower(), "video/mp4")

    # NOTE: Do not force Content-Disposition: attachment for videos.
    # Some browsers refuse to play media in <video> when it is served as an attachment.
    return FileResponse(path=str(full_path), media_type=media_type)


@app.api_route("/vtt/{file_path:path}", methods=["GET", "HEAD"])
async def serve_vtt(file_path: str):
    """
    Serve VTT subtitle files.

    Args:
        file_path: Path to the VTT file (relative to archive_root)

    Returns:
        VTT file as text response
    """
    verify_auth_token()

    if config is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Convert to absolute path
    archive_root = config.paths.archive_root
    full_path = Path(archive_root) / file_path

    # Security check: ensure the path is within archive_root
    try:
        full_path = full_path.resolve()
        archive_root_resolved = Path(archive_root).resolve()
        if not str(full_path).startswith(str(archive_root_resolved)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Path resolution error: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Check if file exists
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="VTT file not found")

    # Check if file has valid VTT extension
    if not any(str(full_path).endswith(ext) for ext in config.video.vtt_extensions):
        raise HTTPException(status_code=400, detail="Invalid VTT file type")

    logger.info(f"Serving VTT file: {full_path}")

    return FileResponse(
        path=str(full_path),
        media_type="text/vtt",
        filename=full_path.name,
    )


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
