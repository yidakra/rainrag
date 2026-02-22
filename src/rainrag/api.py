"""FastAPI backend for RainRAG query interface."""

import hmac
import os
import string
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query
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
    original_score: float | None = None  # Original score before reranking (if reranked)
    video_url: str | None = None
    vtt_url: str | None = None
    group_id: str | None = None  # Base name for grouping multilingual versions
    date: str | None = None  # Recording date (from video mtime)
    duration_seconds: float | None = None  # Video duration in seconds
    start_time: str | None = None  # First timecode (HH:MM:SS)
    end_time: str | None = None  # Last timecode (HH:MM:SS)

    # Web metadata fields
    web_title: str | None = None
    web_date: str | None = None
    web_date_ts: float | None = None
    web_description: str | None = None
    web_url: str | None = None


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
    # Search features configuration
    hybrid_search_enabled: bool = False
    reranker_enabled: bool = False
    chunking_enabled: bool = (
        False  # Reports whether chunking is enabled (used for temporal features)
    )
    fusion_method: str = "rrf"
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

    # If indexed path is stale/foreign, rely on suffix-based relative reconstruction later.

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


def _resolve_media_relative(path_str: str, root: Path) -> Path | None:
    """Resolve a media file path to a root-relative path with migration-safe fallbacks."""
    candidate = Path(path_str)

    def _infer_hashed_relative(file_name_path: Path) -> Path | None:
        """Infer hash-sharded relative path (<hh>/<hh>/.../<filename>) from filename."""
        name = file_name_path.name
        stem = file_name_path.stem

        # Normalize VTT stem: remove language suffix from <hash>.ru.vtt / <hash>.en.vtt
        for lang_suffix in (".ru", ".en"):
            if stem.endswith(lang_suffix):
                stem = stem[: -len(lang_suffix)]
                break

        # For videos, basename is typically <hash>_1080p.mp4, <hash>.mp4, etc.
        hash_part = stem.split("_", 1)[0]
        if len(hash_part) == 40 and all(ch in string.hexdigits for ch in hash_part):
            shard_dirs = [hash_part[i : i + 2].lower() for i in range(0, 40, 2)]
            return Path(*shard_dirs) / name

        return None

    # 1) Standard case: path already under root
    try:
        return candidate.resolve().relative_to(root)
    except Exception:
        pass

    # 2) Heuristic: recover relative suffix after root basename (e.g. /transcoded/...)
    root_marker = f"/{root.name}/"
    normalized = path_str.replace("\\", "/")
    marker_pos = normalized.rfind(root_marker)
    if marker_pos != -1:
        suffix = normalized[marker_pos + len(root_marker) :].lstrip("/")
        guessed = Path(suffix)
        if (root / guessed).exists():
            return guessed

    # 3) Heuristic: derive hash-sharded path from filename and verify existence
    inferred = _infer_hashed_relative(candidate)
    if inferred is not None and (root / inferred).exists():
        return inferred

    # 4) Generic fallback: try matching trailing suffix of the original path
    parts = list(candidate.parts)
    for n in range(min(len(parts), 30), 3, -1):
        tail = Path(*parts[-n:])
        if (root / tail).exists():
            return tail

    return None


def generate_media_urls(
    vtt_path: str, start_time: str | None = None
) -> tuple[str | None, str | None]:
    """
    Generate video and VTT URLs for a given VTT file path.

    Args:
        vtt_path: Path to the VTT file
        start_time: Optional start time for video URL timecode

    Returns:
        Tuple of (video_url, vtt_url) - both may be None if video is disabled or files not found
    """
    video_url = None
    vtt_url = None

    if config and config.video.enabled:
        archive_root = Path(config.paths.archive_root).resolve()
        vtt_rel = _resolve_media_relative(vtt_path, archive_root)

        # Canonical VTT path under current archive root (if resolvable)
        canonical_vtt_path = (
            str((archive_root / vtt_rel).resolve()) if vtt_rel is not None else vtt_path
        )

        # Find corresponding video file
        video_file = find_video_file(canonical_vtt_path)
        if video_file:
            # Create relative path from video_root
            video_root = Path(
                config.paths.video_root if config.paths.video_root else config.paths.archive_root
            ).resolve()
            video_rel = _resolve_media_relative(video_file, video_root)
            if video_rel is not None:
                video_url = f"/video/{video_rel}"
                # Add timecode fragment for video player to start at
                if start_time:
                    start_seconds = timecode_to_seconds(start_time)
                    if start_seconds is not None:
                        video_url = f"{video_url}#t={start_seconds}"
            else:
                logger.warning(f"Could not resolve video path under video_root: {video_file}")

        # VTT file URL
        if vtt_rel is not None:
            vtt_url = f"/vtt/{vtt_rel}"
        else:
            logger.warning(f"Could not resolve VTT path under archive_root: {vtt_path}")

    return video_url, vtt_url


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

    # Test-mode escape hatch: avoid expensive model/client startup when endpoint
    # tests patch query_engine/config directly.
    if os.getenv("RAINRAG_SKIP_API_STARTUP_INIT", "").lower() in {"1", "true", "yes"}:
        logger.info("Skipping RainRAG API startup initialization (RAINRAG_SKIP_API_STARTUP_INIT)")
        yield
        logger.info("Shutting down RainRAG API...")
        return

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


def verify_auth_token(authorization: str | None = None, access_token: str | None = None) -> bool:
    """Verify authentication token if configured."""
    required_token = os.getenv("RAINRAG_AUTH_TOKEN")

    # If no token is configured, allow all requests
    if not required_token:
        return True

    # Allow explicit query token for browser media requests where custom headers
    # are not available on <video>/<a> elements.
    if access_token and hmac.compare_digest(access_token, required_token):
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
        # Search features configuration
        hybrid_search_enabled=query_engine.config.hybrid_search.enabled,
        reranker_enabled=query_engine.config.reranker.enabled,
        chunking_enabled=query_engine.config.chunking.enabled,  # Reports chunking status (enables temporal features)
        fusion_method=query_engine.config.hybrid_search.fusion_method,
        # Deprecated field (kept for backwards compatibility)
        mistral_model=llm_model,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, authorized: bool = Header(True)):
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
            video_url, vtt_url = generate_media_urls(vtt_path, doc.get("start_time"))

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
                    original_score=doc.get("original_score"),
                    video_url=video_url,
                    vtt_url=vtt_url,
                    group_id=group_id,
                    date=doc.get("date"),
                    duration_seconds=doc.get("duration_seconds"),
                    start_time=doc.get("start_time"),
                    end_time=doc.get("end_time"),
                    web_title=doc.get("web_title"),
                    web_date=doc.get("web_date"),
                    web_date_ts=doc.get("web_date_ts"),
                    web_description=doc.get("web_description"),
                    web_url=doc.get("web_url"),
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
    request: RelatedChunksRequest,
    authorized: bool = Header(True),
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
            f"Finding related chunks for: {request.chunk_id} (top_k={request.top_k}, same_video_only={request.same_video_only})"
        )

        # Find related chunks
        related_docs = query_engine.find_related_chunks(
            chunk_id=request.chunk_id, top_k=request.top_k, same_video_only=request.same_video_only
        )

        # Format response with video and VTT URLs
        related_chunks = []
        for idx, doc in enumerate(related_docs):
            vtt_path = doc["path"]

            # Generate URLs for video and VTT files
            video_url, vtt_url = generate_media_urls(vtt_path, doc.get("start_time"))

            # Get base name for grouping
            group_id = get_video_base_name(vtt_path)

            chunk = ContextChunk(
                text=doc["text"],
                filename=Path(vtt_path).name,
                language=doc.get("language", "unknown"),
                score=doc["score"],
                rank=idx + 1,
                doc_id=doc.get("doc_id", ""),
                video_url=video_url,
                vtt_url=vtt_url,
                group_id=group_id,
                date=doc.get("date"),
                duration_seconds=doc.get("duration_seconds"),
                start_time=doc.get("start_time"),
                end_time=doc.get("end_time"),
                web_title=doc.get("web_title"),
                web_date=doc.get("web_date"),
                web_date_ts=doc.get("web_date_ts"),
                web_description=doc.get("web_description"),
                web_url=doc.get("web_url"),
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
async def serve_video(
    file_path: str,
    authorization: Annotated[str | None, Header()] = None,
    auth: Annotated[str | None, Query()] = None,
):
    """
    Serve video files.

    Args:
        file_path: Path to the video file (relative to video_root)

    Returns:
        Video file as streaming response
    """
    verify_auth_token(authorization=authorization, access_token=auth)

    if config is None or not config.video.enabled:
        raise HTTPException(status_code=404, detail="Video serving is disabled")

    # Convert to absolute path
    video_root = Path(
        config.paths.video_root if config.paths.video_root else config.paths.archive_root
    ).resolve()
    full_path = (video_root / file_path).resolve()

    # Security check: ensure the path is within video_root
    try:
        if not str(full_path).startswith(str(video_root)):
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
async def serve_vtt(
    file_path: str,
    authorization: Annotated[str | None, Header()] = None,
    auth: Annotated[str | None, Query()] = None,
):
    """
    Serve VTT subtitle files.

    Args:
        file_path: Path to the VTT file (relative to archive_root)

    Returns:
        VTT file as text response
    """
    verify_auth_token(authorization=authorization, access_token=auth)

    if config is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    # Convert to absolute path
    archive_root = Path(config.paths.archive_root).resolve()
    full_path = (archive_root / file_path).resolve()

    # Security check: ensure the path is within archive_root
    try:
        if not str(full_path).startswith(str(archive_root)):
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
