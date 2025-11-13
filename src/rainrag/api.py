"""FastAPI backend for RainRAG query interface."""

from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import os

from rainrag.config import load_config, Config
from rainrag.query import RAGQueryEngine


# Global query engine instance and config
query_engine: Optional[RAGQueryEngine] = None
config: Optional[Config] = None


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
    video_url: Optional[str] = None
    vtt_url: Optional[str] = None
    group_id: Optional[str] = None  # Base name for grouping multilingual versions


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


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    display_name: str
    chat_template: str
    is_active: bool


class ModelsResponse(BaseModel):
    """Response model for models endpoint."""

    models: List[ModelInfo]
    current_model: str


class SwitchModelRequest(BaseModel):
    """Request model for switching models."""

    model_name: str = Field(..., description="Full model name to switch to")
    chat_template: str = Field(
        default="auto",
        description="Chat template to use (auto, mistral, gemma, chatml, generic)",
    )


def find_video_file(vtt_path: str) -> Optional[str]:
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
    video_root = config.paths.video_root if config.paths.video_root else config.paths.archive_root

    # Get base name without VTT extension
    base_name = vtt_file.stem
    # Remove language suffixes like .en or .ru
    for lang_suffix in [".en", ".ru"]:
        if base_name.endswith(lang_suffix):
            base_name = base_name[:-len(lang_suffix)]
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
            if video_file.name.startswith(base_name):
                if any(video_file.suffix.lower() == ext for ext in config.video.extensions):
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
            base_name = base_name[:-len(lang_suffix)]
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


# Supported models configuration
SUPPORTED_MODELS = {
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": {
        "display_name": "Mistral Small 3.2 24B",
        "chat_template": "mistral",
    },
    "google/gemma-2-27b-it": {
        "display_name": "Gemma 2 27B",
        "chat_template": "gemma",
    },
    "gpt-oss:20b": {
        "display_name": "GPT-OSS 20B",
        "chat_template": "chatml",
    },
}


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List available LLM models.

    Returns:
        List of supported models with metadata
    """
    if query_engine is None or config is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    current_model = config.vllm.model_name
    models = []

    for model_name, model_info in SUPPORTED_MODELS.items():
        models.append(
            ModelInfo(
                name=model_name,
                display_name=model_info["display_name"],
                chat_template=model_info["chat_template"],
                is_active=(model_name == current_model),
            )
        )

    return ModelsResponse(models=models, current_model=current_model)


@app.post("/models/switch")
async def switch_model(request: SwitchModelRequest):
    """
    Switch to a different LLM model dynamically.

    Args:
        request: Model switch request with model name and optional chat template

    Returns:
        Success status with new model information
    """
    global query_engine, config

    if query_engine is None or config is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    # Validate model name
    if request.model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {request.model_name}. "
            f"Supported models: {list(SUPPORTED_MODELS.keys())}",
        )

    try:
        logger.info(f"Switching model from {config.vllm.model_name} to {request.model_name}")

        # Update config with new model
        config.vllm.model_name = request.model_name
        config.vllm.chat_template = request.chat_template

        # Reinitialize query engine with new model
        # Note: We keep the same embedding model and Qdrant client
        old_embedding_model = query_engine.embedding_model
        old_qdrant_client = query_engine.qdrant_client

        # Create new query engine with updated config
        query_engine = RAGQueryEngine(config)

        # Reuse existing connections
        query_engine.embedding_model = old_embedding_model
        query_engine.qdrant_client = old_qdrant_client

        logger.info(f"Successfully switched to model: {request.model_name}")

        return {
            "status": "success",
            "message": f"Switched to {SUPPORTED_MODELS[request.model_name]['display_name']}",
            "model_name": request.model_name,
            "chat_template": query_engine.chat_template,
        }

    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to switch model: {str(e)}"
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

        # Execute query with language parameter
        result = query_engine.query(question=request.question, top_k=request.top_k, language=request.language)

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
                    video_root = config.paths.video_root if config.paths.video_root else config.paths.archive_root
                    try:
                        video_rel = Path(video_file).relative_to(video_root)
                        video_url = f"/video/{video_rel}"
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
                    video_url=video_url,
                    vtt_url=vtt_url,
                    group_id=group_id,
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


@app.get("/video/{file_path:path}")
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

    return FileResponse(
        path=str(full_path),
        media_type=media_type,
        filename=full_path.name,
    )


@app.get("/vtt/{file_path:path}")
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
