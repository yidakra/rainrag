"""FastAPI backend for RainRAG query interface."""

import asyncio
import hashlib
import hmac
import os
import re
import string
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from loguru import logger
from prometheus_client import Counter, Gauge


# Constants loaded once at import time (avoids repeated getenv calls per request)
# Guard against invalid env values (float() would otherwise raise and prevent
# the module from importing).
_query_timeout_raw = os.getenv("RAINRAG_QUERY_TIMEOUT_SECONDS", "240")
try:
    _query_timeout_seconds = float(_query_timeout_raw)
except (TypeError, ValueError) as exc:
    logger.warning(
        "Invalid RAINRAG_QUERY_TIMEOUT_SECONDS={!r}, using default 240.0: {}",
        _query_timeout_raw,
        exc,
    )
    _query_timeout_seconds = 240.0
QUERY_TIMEOUT_SECONDS: float = _query_timeout_seconds

_max_concurrent_queries_raw = os.getenv("RAINRAG_MAX_CONCURRENT_QUERIES", "8")
try:
    _max_concurrent_queries = int(_max_concurrent_queries_raw)
except (TypeError, ValueError) as exc:
    logger.warning(
        "Invalid RAINRAG_MAX_CONCURRENT_QUERIES={!r}, using default 8: {}",
        _max_concurrent_queries_raw,
        exc,
    )
    _max_concurrent_queries = 8
MAX_CONCURRENT_QUERIES: int = _max_concurrent_queries

# Lazy semaphore to avoid creating asyncio.Semaphore at import time in Python 3.10+
_global_query_semaphore: asyncio.Semaphore | None = None


def _get_query_semaphore() -> asyncio.Semaphore:
    global _global_query_semaphore
    if _global_query_semaphore is None:
        _global_query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
    return _global_query_semaphore


def _create_query_timeout_counter() -> Counter:
    """Create a timeout counter while avoiding duplicate-registration errors.

    The Prometheus client raises ``ValueError`` if you try to register the
    same metric name twice in the default registry.  We intentionally do NOT
    access private registry internals; instead, if the metric name is already
    registered, we fall back to creating the counter in a fresh registry.

    Note: the fallback counter will not be scraped by the default Prometheus
    registry, but it allows the application to safely increment the counter
    without crashing on import.
    """
    name = "rainrag_query_timeouts_total"
    try:
        return Counter(name, "Number of query requests that timed out")
    except ValueError:
        logger.warning(
            "Metric {!r} already registered in default registry; creating unregistered counter (won't be scraped).",
            name,
        )
        from prometheus_client import CollectorRegistry

        return Counter(
            name, "Number of query requests that timed out", registry=CollectorRegistry()
        )


# Prometheus metrics
# Ensure the counter isn't registered twice when the module is imported under
# different package names (e.g. `rainrag.api` vs `src.rainrag.api`).
QUERY_TIMEOUT_COUNTER: Counter = _create_query_timeout_counter()


def _create_active_queries_gauge() -> Gauge:
    """Create an active query gauge in a safe way for multiple module imports."""
    name = "rainrag_active_queries"
    try:
        return Gauge(name, "Number of currently active query handlers")
    except ValueError:
        logger.warning(
            "Metric {!r} already registered in default registry; creating unregistered gauge.",
            name,
        )
        from prometheus_client import CollectorRegistry

        return Gauge(
            name, "Number of currently active query handlers", registry=CollectorRegistry()
        )


ACTIVE_QUERIES: Gauge = _create_active_queries_gauge()

from fastapi import FastAPI, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from rainrag.config import Config, load_config
from rainrag.query import RAGQueryEngine
from rainrag.video_session import VideoSessionManager


# Global query engine instance and config
query_engine: RAGQueryEngine | None = None
# Manager for single-video upload sessions (initialized on startup when enabled)
video_session_manager: VideoSessionManager | None = None
config: Config | None = None
HLS_CACHE_ROOT = Path(os.getenv("RAINRAG_HLS_CACHE_DIR", "/tmp/rainrag_hls_cache"))
_hls_prewarm_inflight: set[str] = set()
_hls_prewarm_lock = threading.Lock()
_hls_generation_locks: dict[str, threading.Lock] = {}
_hls_generation_locks_guard = threading.Lock()
HLS_FFMPEG_TIMEOUT_SECONDS = int(os.getenv("RAINRAG_HLS_FFMPEG_TIMEOUT_SECONDS", "300"))
HLS_CACHE_TTL_SECONDS = int(os.getenv("RAINRAG_HLS_CACHE_TTL_SECONDS", "86400"))
HLS_CACHE_MAX_DIRS = int(os.getenv("RAINRAG_HLS_CACHE_MAX_DIRS", "512"))


def _parse_csv_env(name: str, default: list[str]) -> list[str]:
    """Parse comma-separated env values into a de-duplicated list."""
    raw = os.getenv(name, "")
    if not raw.strip():
        return default

    values: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        value = item.strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values or default


def _is_mock_like(obj: Any) -> bool:
    """Detect objects that behave like unittest.mock.Mock without importing it.

    This is used primarily in unit tests where the query callable is patched with
    a Mock. Running Mock instances via ``asyncio.to_thread`` can deadlock the
    test runner teardown, so we invoke them directly when we detect this behavior.
    """

    # The official unittest.mock.Mock has attributes like ``called`` and
    # ``assert_called``. We use duck-typing to avoid importing test-only utilities.
    return callable(obj) and hasattr(obj, "called") and hasattr(obj, "assert_called")


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
    metadata_fallback_hits: int = 0


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

    # Quality preference order (highest to lowest).
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


def _select_media_file_from_directory(
    directory: Path,
    *,
    kind: str,
) -> Path | None:
    """Pick a likely media file from a hash-sharded directory.

    This is a defensive fallback for callers that pass a directory path
    instead of a specific file path (for example /video/<hash-shards>/).
    """
    if not directory.is_dir():
        return None

    try:
        files = [p for p in directory.iterdir() if p.is_file()]
    except OSError:
        return None

    if kind == "video":
        quality_order = ["1080p", "720p", "480p", "360p", "180p"]
        for quality in quality_order:
            for file_path in files:
                suffix = file_path.suffix.lower()
                if suffix in config.video.extensions and f"_{quality}" in file_path.stem:
                    return file_path
        for file_path in files:
            if file_path.suffix.lower() in config.video.extensions:
                return file_path
        return None

    if kind == "vtt":
        preferred_suffixes = [".ru.vtt", ".en.vtt"]
        for preferred in preferred_suffixes:
            for file_path in files:
                if file_path.name.endswith(preferred):
                    return file_path
        for file_path in files:
            if any(file_path.name.endswith(ext) for ext in config.video.vtt_extensions):
                return file_path
        return None

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
    except (ValueError, OSError):
        # ValueError: path not under root; OSError: resolution failed (permissions, etc.)
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


def _build_quality_video_map(directory: Path, target_stem: str | None = None) -> dict[str, Path]:
    """Return available quality MP4 files in a directory keyed by quality label."""
    if config is None or not directory.is_dir():
        return {}

    valid_exts = {ext.lower() for ext in config.video.extensions}
    quality_re = re.compile(r"^(?P<base>.+)_(?P<quality>1080p|720p|480p|360p|180p)$")
    out: dict[str, Path] = {}
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in valid_exts:
            continue
        match = quality_re.match(file_path.stem)
        if not match:
            continue
        if target_stem is not None and match.group("base") != target_stem:
            continue
        quality = match.group("quality")
        out[quality] = file_path
    return out


def _resolve_hls_file_context(file_path: str) -> tuple[Path, str | None]:
    """Resolve HLS input path to (directory, target video base stem)."""
    if config is None or not config.video.enabled:
        raise HTTPException(status_code=404, detail="Video serving is disabled")

    video_root = Path(
        config.paths.video_root if config.paths.video_root else config.paths.archive_root
    ).resolve()
    full_path = (video_root / file_path).resolve()

    try:
        if not str(full_path).startswith(str(video_root)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception as e:
        logger.error(f"Path resolution error: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")

    if full_path.exists() and full_path.is_dir():
        return full_path, None

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    if full_path.suffix.lower() not in config.video.extensions:
        raise HTTPException(status_code=400, detail="Invalid video file type")

    match = re.match(r"^(?P<base>.+)_(1080p|720p|480p|360p|180p)$", full_path.stem)
    target_base = match.group("base") if match else None
    return full_path.parent, target_base


def _hls_cache_key(video_file: Path, quality: str) -> str:
    """Build deterministic cache key for generated HLS assets."""
    try:
        stat = video_file.stat()
        signature = f"{video_file.resolve()}|{quality}|{int(stat.st_mtime)}|{stat.st_size}"
    except OSError:
        signature = f"{video_file.resolve()}|{quality}"
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()


def _get_hls_generation_lock(cache_key: str) -> threading.Lock:
    with _hls_generation_locks_guard:
        lock = _hls_generation_locks.get(cache_key)
        if lock is None:
            lock = threading.Lock()
            _hls_generation_locks[cache_key] = lock
        return lock


def _cleanup_hls_cache_best_effort() -> None:
    """Best-effort cleanup of stale/excess HLS cache directories."""
    try:
        HLS_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        dirs = [d for d in HLS_CACHE_ROOT.iterdir() if d.is_dir()]
    except OSError:
        return

    import time as _time

    now = _time.time()
    stale_before = now - max(HLS_CACHE_TTL_SECONDS, 0)

    for d in dirs:
        try:
            if d.stat().st_mtime < stale_before:
                for item in d.iterdir():
                    item.unlink(missing_ok=True)
                d.rmdir()
        except OSError:
            continue

    try:
        dirs = sorted(
            [d for d in HLS_CACHE_ROOT.iterdir() if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
    except OSError:
        return

    overflow = len(dirs) - max(HLS_CACHE_MAX_DIRS, 0)
    if overflow <= 0:
        return
    for d in dirs[:overflow]:
        try:
            for item in d.iterdir():
                item.unlink(missing_ok=True)
            d.rmdir()
        except OSError:
            continue


def _ensure_hls_variant_cache(video_file: Path, quality: str) -> tuple[str, Path]:
    """Generate segmented HLS files for a single quality, cached on disk."""
    cache_key = _hls_cache_key(video_file, quality)
    out_dir = HLS_CACHE_ROOT / cache_key
    playlist_path = out_dir / "index.m3u8"

    if playlist_path.exists():
        return cache_key, playlist_path
    lock = _get_hls_generation_lock(cache_key)
    with lock:
        if playlist_path.exists():
            return cache_key, playlist_path
        _cleanup_hls_cache_best_effort()
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"hls_{cache_key}_", dir=str(HLS_CACHE_ROOT)
        ) as tmp_dir:
            tmp_path = Path(tmp_dir)
            tmp_playlist = tmp_path / "index.m3u8"
            segment_pattern = str(tmp_path / "seg_%05d.ts")
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_file),
                "-c",
                "copy",
                "-f",
                "hls",
                "-hls_time",
                "6",
                "-hls_list_size",
                "0",
                "-hls_playlist_type",
                "vod",
                "-hls_flags",
                "independent_segments",
                "-hls_segment_filename",
                segment_pattern,
                str(tmp_playlist),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=HLS_FFMPEG_TIMEOUT_SECONDS,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail="ffmpeg is not installed") from exc
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail="HLS generation timed out") from exc

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate HLS playlist for {quality}: {result.stderr.strip()}",
                )

            for item in tmp_path.iterdir():
                target = out_dir / item.name
                item.replace(target)

    if not playlist_path.exists():
        raise HTTPException(status_code=500, detail="HLS playlist generation incomplete")
    return cache_key, playlist_path


def _rewrite_hls_variant_playlist(
    playlist_path: Path, cache_key: str, auth: str | None = None
) -> str:
    """Rewrite local segment names to API-served segment URLs."""
    auth_q = f"?auth={quote(auth, safe='')}" if auth else ""
    lines: list[str] = []
    for raw_line in playlist_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            lines.append(line)
            continue
        seg_name = Path(line).name
        lines.append(f"/hls/asset/{cache_key}/{quote(seg_name, safe='')}{auth_q}")
    return "\n".join(lines) + "\n"


def _schedule_hls_prewarm(available: dict[str, Path]) -> None:
    """Schedule non-blocking HLS prewarm for available quality files."""
    if not available:
        return

    # Prefer lower startup bitrates first.
    ordered = [q for q in ("180p", "360p", "480p", "720p", "1080p") if q in available]
    if not ordered:
        return

    key_file = available[ordered[0]]
    prewarm_key = str(key_file.resolve())
    with _hls_prewarm_lock:
        if prewarm_key in _hls_prewarm_inflight:
            return
        _hls_prewarm_inflight.add(prewarm_key)

    def _runner() -> None:
        try:
            for q in ordered:
                try:
                    _ensure_hls_variant_cache(available[q], q)
                except Exception as exc:
                    logger.debug(f"HLS prewarm skipped for {available[q].name} {q}: {exc}")
        finally:
            with _hls_prewarm_lock:
                _hls_prewarm_inflight.discard(prewarm_key)

    threading.Thread(target=_runner, daemon=True).start()


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
    global query_engine, config, video_session_manager

    logger.info("Initializing RainRAG API...")

    # Escape hatch: avoid expensive model/client startup in tests.
    skip_flag = os.getenv("RAINRAG_SKIP_API_STARTUP_INIT", "").lower() in {"1", "true", "yes"}

    if skip_flag:
        logger.info("Skipping RainRAG API startup initialization")
        yield
        logger.info("Shutting down RainRAG API...")
        return

    # Load configuration
    config_path = os.getenv("RAINRAG_CONFIG", "config.yaml")
    config = load_config(config_path)

    # Initialize query engine
    query_engine = RAGQueryEngine(config)
    query_engine.initialize()

    # Initialize the single-video upload manager (transcribe → scoped Q&A)
    if config.video_upload.enabled:
        try:
            video_session_manager = VideoSessionManager(config)
            logger.info("Video upload session manager initialized")
        except Exception:
            logger.exception("Failed to initialize video session manager; upload mode disabled")
            video_session_manager = None

    logger.info("RainRAG API initialized successfully")

    yield

    # Cleanup (if needed)
    if video_session_manager is not None:
        video_session_manager.shutdown()
    logger.info("Shutting down RainRAG API...")


# Create FastAPI app
app = FastAPI(
    title="RainRAG API",
    description="Query interface for RainRAG - Multilingual RAG system for video transcripts",
    version="0.1.0",
    lifespan=lifespan,
)

# External deployment defaults (can be overridden in environment)
allowed_hosts = _parse_csv_env(
    "RAINRAG_ALLOWED_HOSTS", ["rag.tvrain.tv", "localhost", "127.0.0.1", "testserver"]
)
cors_origins = _parse_csv_env(
    "RAINRAG_CORS_ORIGINS",
    ["https://rag.tvrain.tv", "http://localhost:7860", "http://127.0.0.1:7860"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

# CORS for browser-based clients (Streamlit UI / API consumers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
async def query(request: QueryRequest):
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

        # Concurrency control: bounded active query slots to avoid thread/task explosion.
        acquired = False
        try:
            await asyncio.wait_for(_get_query_semaphore().acquire(), timeout=5.0)
            acquired = True
        except asyncio.TimeoutError as exc:
            logger.warning(
                "Too many concurrent queries (%d). Rejecting request (question=%r)",
                MAX_CONCURRENT_QUERIES,
                request.question[:100],
            )
            raise HTTPException(
                status_code=429,
                detail="Server busy: too many concurrent queries. Please retry shortly.",
            ) from exc

        active_query_incremented = False
        try:
            try:
                ACTIVE_QUERIES.inc()
                active_query_incremented = True
            except Exception:
                logger.exception("Failed to increment active query gauge")

            background_task = None
            try:
                # Execute query in a worker thread so the event loop remains responsive
                # (health checks/UI should not freeze while a long LLM call is running).
                # Use module-level constant to avoid repeated environment lookups.
                query_callable = query_engine.query

                # ASGI unit tests often patch query_engine.query with Mock objects.
                # Running a Mock via asyncio.to_thread can deadlock test-loop shutdown.
                if _is_mock_like(query_callable):
                    result = query_callable(
                        question=request.question,
                        top_k=request.top_k,
                        language=request.language,
                        date_from=request.date_from,
                        date_to=request.date_to,
                    )
                else:
                    background_task = asyncio.create_task(
                        asyncio.to_thread(
                            query_callable,
                            question=request.question,
                            top_k=request.top_k,
                            language=request.language,
                            date_from=request.date_from,
                            date_to=request.date_to,
                        )
                    )
                    try:
                        result = await asyncio.wait_for(
                            background_task, timeout=QUERY_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError as exc:
                        if not background_task.done():
                            background_task.cancel()
                            try:
                                await asyncio.wait_for(background_task, timeout=2.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError) as cancel_exc:
                                logger.warning(
                                    "Background query task did not stop after cancellation (%s: %s)",
                                    type(cancel_exc).__name__,
                                    cancel_exc,
                                )

                        try:
                            QUERY_TIMEOUT_COUNTER.inc()
                        except Exception:
                            logger.exception("Failed to increment timeout metric")

                        logger.warning(
                            "Query timed out (question={!r}, top_k={}, language={}, timeout={})",
                            request.question[:100],
                            request.top_k,
                            request.language,
                            QUERY_TIMEOUT_SECONDS,
                        )
                        raise HTTPException(
                            status_code=504,
                            detail=(
                                "Query timed out while generating answer. "
                                "Please try a shorter or more specific question."
                            ),
                        ) from exc
            finally:
                if active_query_incremented:
                    try:
                        ACTIVE_QUERIES.dec()
                    except Exception:
                        logger.exception("Failed to decrement active query gauge")

        finally:
            if acquired:
                _get_query_semaphore().release()

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
            metadata_fallback_hits=int(result.get("metadata_fallback_hits") or 0),
        )

        logger.info(f"Query completed successfully. Retrieved {len(context_chunks)} documents")

        return response

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f"Query failed: {e}")
        logger.error(f"Full traceback:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ---------------------------------------------------------------------------
# Single-video upload mode: upload → transcribe → scoped Q&A
# ---------------------------------------------------------------------------


class VideoQueryRequest(BaseModel):
    """Request model for querying a single uploaded video's transcript."""

    question: str = Field(..., description="The user's question", min_length=1)
    language: str = Field(
        default="ru", description="Response language (ru or en)", pattern="^(ru|en)$"
    )
    top_k: int | None = Field(
        default=None, description="Number of context chunks to retrieve", ge=1, le=20
    )


def _require_video_manager() -> VideoSessionManager:
    if video_session_manager is None:
        raise HTTPException(status_code=503, detail="Video upload mode is not available")
    return video_session_manager


def _build_query_response(result: dict[str, Any]) -> QueryResponse:
    """Format a query-engine result dict into the API QueryResponse."""
    context_chunks = []
    for doc in result["retrieved_documents"]:
        vtt_path = doc["path"]
        video_url, vtt_url = generate_media_urls(vtt_path, doc.get("start_time"))
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
                group_id=get_video_base_name(vtt_path),
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
    return QueryResponse(
        answer=result["answer"],
        context=context_chunks,
        question=result["question"],
        num_documents=result["num_documents"],
        metadata_fallback_hits=int(result.get("metadata_fallback_hits") or 0),
    )


@app.post("/video-sessions")
async def create_video_session(file: UploadFile):
    """Upload a video, kick off transcription + indexing, return the session."""
    verify_auth_token()
    manager = _require_video_manager()

    max_bytes = manager.cfg.max_upload_mb * 1024 * 1024
    tmp_root = Path(manager.cfg.tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    # Stream the upload to a temp file, enforcing the size cap as we go.
    written = 0
    fd, tmp_name = tempfile.mkstemp(dir=str(tmp_root), suffix=".upload")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds maximum size of {manager.cfg.max_upload_mb} MB",
                    )
                out.write(chunk)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        await file.close()

    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload")

    filename = file.filename or "upload.mp4"
    session = manager.create_session(tmp_path, filename)
    return session.public_dict()


@app.get("/video-sessions/{session_id}")
async def get_video_session(session_id: str):
    """Return the current status/progress of an upload session."""
    verify_auth_token()
    manager = _require_video_manager()
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.public_dict()


@app.api_route("/video-sessions/{session_id}/media", methods=["GET", "HEAD"])
async def serve_video_session_media(
    session_id: str,
    authorization: Annotated[str | None, Header()] = None,
    auth: Annotated[str | None, Query()] = None,
):
    """Stream an uploaded session's source video back for in-page playback.

    Accepts the token as a query parameter as well as a header, because a
    <video> element cannot send custom headers. Serves the recorded session
    path directly — it is never built from client input, so there is no
    traversal surface here.
    """
    verify_auth_token(authorization=authorization, access_token=auth)
    manager = _require_video_manager()

    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    video_path = Path(session.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Session video is no longer available")

    media_types = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".m4v": "video/mp4",
    }
    media_type = media_types.get(video_path.suffix.lower(), "video/mp4")
    # FileResponse honours Range requests, which is what lets the player seek to
    # a cited timecode without downloading the whole file first.
    return FileResponse(path=str(video_path), media_type=media_type)


@app.delete("/video-sessions/{session_id}")
async def delete_video_session(session_id: str):
    """Delete an upload session: drop its collection and working files."""
    verify_auth_token()
    manager = _require_video_manager()
    if not manager.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.post("/video-sessions/{session_id}/query", response_model=QueryResponse)
async def query_video_session(session_id: str, request: VideoQueryRequest):
    """Answer a question scoped to a single uploaded video's transcript."""
    verify_auth_token()
    manager = _require_video_manager()

    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Session not ready (status={session.status})",
        )

    # Concurrency control shared with the main /query path.
    try:
        await asyncio.wait_for(_get_query_semaphore().acquire(), timeout=5.0)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="Server busy: too many concurrent queries. Please retry shortly.",
        ) from exc

    try:
        task = asyncio.create_task(
            asyncio.to_thread(
                query_engine.query,
                question=request.question,
                top_k=request.top_k,
                language=request.language,
                collection_name=session.collection_name,
                single_video=True,
            )
        )
        try:
            result = await asyncio.wait_for(task, timeout=QUERY_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            if not task.done():
                task.cancel()
            raise HTTPException(
                status_code=504, detail="Query timed out while generating answer."
            ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video-session query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    finally:
        _get_query_semaphore().release()

    return _build_query_response(result)


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


class VideoNameSearchLanguageResult(BaseModel):
    """Media URLs for a single language version of a video found by name search."""

    video_url: str | None = None
    vtt_url: str | None = None


class VideoNameSearchResult(BaseModel):
    """A single result from a video name search."""

    video_hash: str
    name: str
    date: str | None = None
    web_url: str | None = None
    teleshow_name: str | None = None
    languages: dict[str, VideoNameSearchLanguageResult]


class VideoNameSearchResponse(BaseModel):
    """Response model for the video name search endpoint."""

    results: list[VideoNameSearchResult]
    query: str


@app.get("/search-by-name", response_model=VideoNameSearchResponse)
async def search_by_name(
    q: str = Query(..., min_length=1, description="Video title search query"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of results to return"),
    authorization: str | None = Header(None),
):
    """Search for videos by title in the local web metadata cache.

    Performs a case-insensitive substring match against the ``name`` field of
    every cached ``{hash}.json`` file in the configured metadata directory.
    Returns matching videos with media URLs for all locally available language
    versions (ru / en).
    """
    verify_auth_token(authorization=authorization)

    if config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")

    from pathlib import Path as _Path

    from rainrag.ingest import WebMetadataLoader
    from rainrag.web_metadata_api import WebMetadataAPIClient

    metadata_path = _Path(config.web_metadata.path)
    api_client = None
    if config.web_metadata.source in {"api", "hybrid"}:
        try:
            api_client = WebMetadataAPIClient.from_env(
                base_url=config.web_metadata.api_url,
                token_env=config.web_metadata.api_token_env,
            )
        except ValueError as exc:
            if config.web_metadata.source == "api":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Web metadata API is required but token is missing. "
                        + f"Set {config.web_metadata.api_token_env}."
                    ),
                ) from exc
            logger.warning(
                "Web metadata API token missing for hybrid mode; falling back to local-only search."
            )

    loader = WebMetadataLoader(
        metadata_path,
        source=config.web_metadata.source,
        api_client=api_client,
    )

    # Run the directory scan in a thread so the async event loop stays responsive
    matches = await asyncio.to_thread(loader.search_by_name, q)
    matches = matches[:limit]

    archive_root = _Path(config.paths.archive_root).resolve()

    results: list[VideoNameSearchResult] = []
    for article in matches:
        video_hash = article.get("video_hash", "").strip()
        if not video_hash:
            continue

        try:
            archive_rel = WebMetadataLoader.hash_to_archive_dir(video_hash)
        except ValueError:
            logger.warning(f"Invalid video_hash in metadata: {video_hash!r}")
            continue

        archive_dir = archive_root / archive_rel

        languages: dict[str, VideoNameSearchLanguageResult] = {}
        for lang in ("ru", "en"):
            vtt_file = archive_dir / f"{video_hash}.{lang}.vtt"
            if vtt_file.exists():
                video_url, vtt_url = generate_media_urls(str(vtt_file))
                languages[lang] = VideoNameSearchLanguageResult(
                    video_url=video_url,
                    vtt_url=vtt_url,
                )

        teleshow = article.get("teleshow")
        teleshow_name = teleshow.get("name") if isinstance(teleshow, dict) else None

        results.append(
            VideoNameSearchResult(
                video_hash=video_hash,
                name=article.get("name", ""),
                date=article.get("date_active_start"),
                web_url=article.get("url"),
                teleshow_name=teleshow_name,
                languages=languages,
            )
        )

    logger.info(f"Name search for {q!r}: {len(results)} results (limit={limit})")
    return VideoNameSearchResponse(results=results, query=q)


class VideoUrlLookupResponse(BaseModel):
    """Response model for the video URL lookup endpoint."""

    result: VideoNameSearchResult | None
    url: str


@app.get("/search-by-url", response_model=VideoUrlLookupResponse)
async def search_by_url(
    url: str = Query(..., min_length=1, description="Exact tvrain.tv web URL for the video"),
    authorization: str | None = Header(None),
):
    """Look up a video by its exact tvrain.tv web URL.

    Performs an exact match against the ``url`` field of cached ``{hash}.json``
    files and returns the video hash, title, date, and media URLs if found.
    """
    verify_auth_token(authorization=authorization)

    if config is None:
        raise HTTPException(status_code=503, detail="Config not initialized")

    from pathlib import Path as _Path

    from rainrag.ingest import WebMetadataLoader
    from rainrag.web_metadata_api import WebMetadataAPIClient

    metadata_path = _Path(config.web_metadata.path)
    api_client = None
    if config.web_metadata.source in {"api", "hybrid"}:
        try:
            api_client = WebMetadataAPIClient.from_env(
                base_url=config.web_metadata.api_url,
                token_env=config.web_metadata.api_token_env,
            )
        except ValueError as exc:
            if config.web_metadata.source == "api":
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Web metadata API is required but token is missing. "
                        + f"Set {config.web_metadata.api_token_env}."
                    ),
                ) from exc
            logger.warning(
                "Web metadata API token missing for hybrid mode; falling back to local-only search."
            )

    loader = WebMetadataLoader(
        metadata_path,
        source=config.web_metadata.source,
        api_client=api_client,
    )

    article = await asyncio.to_thread(loader.search_by_url, url)

    if article is None:
        logger.info(f"URL lookup for {url!r}: not found")
        return VideoUrlLookupResponse(result=None, url=url)

    video_hash = article.get("video_hash", "").strip()
    if not video_hash:
        return VideoUrlLookupResponse(result=None, url=url)

    try:
        archive_rel = WebMetadataLoader.hash_to_archive_dir(video_hash)
    except ValueError:
        logger.warning(f"Invalid video_hash in metadata: {video_hash!r}")
        return VideoUrlLookupResponse(result=None, url=url)

    archive_root = _Path(config.paths.archive_root).resolve()
    archive_dir = archive_root / archive_rel

    languages: dict[str, VideoNameSearchLanguageResult] = {}
    for lang in ("ru", "en"):
        vtt_file = archive_dir / f"{video_hash}.{lang}.vtt"
        if vtt_file.exists():
            video_url, vtt_url = generate_media_urls(str(vtt_file))
            languages[lang] = VideoNameSearchLanguageResult(
                video_url=video_url,
                vtt_url=vtt_url,
            )

    teleshow = article.get("teleshow")
    teleshow_name = teleshow.get("name") if isinstance(teleshow, dict) else None

    result = VideoNameSearchResult(
        video_hash=video_hash,
        name=article.get("name", ""),
        date=article.get("date_active_start"),
        web_url=article.get("url"),
        teleshow_name=teleshow_name,
        languages=languages,
    )

    logger.info(f"URL lookup for {url!r}: found {video_hash}")
    return VideoUrlLookupResponse(result=result, url=url)


@app.post("/related-chunks", response_model=RelatedChunksResponse)
async def get_related_chunks(request: RelatedChunksRequest):
    """
    Find chunks related to a given chunk based on vector similarity.

    Args:
        request: Related chunks request containing chunk_id and parameters

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

    # If a directory is passed, choose best available video file in that directory.
    if full_path.exists() and full_path.is_dir():
        selected = _select_media_file_from_directory(full_path, kind="video")
        if selected is None:
            raise HTTPException(status_code=404, detail="Video directory is empty")
        full_path = selected

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


@app.api_route("/hls/master/{file_path:path}", methods=["GET", "HEAD"])
async def serve_hls_master(
    file_path: str,
    authorization: Annotated[str | None, Header()] = None,
    auth: Annotated[str | None, Query()] = None,
    cb: Annotated[str | None, Query()] = None,
):
    """Serve a master HLS playlist referencing available MP4 quality variants."""
    verify_auth_token(authorization=authorization, access_token=auth)

    directory, target_base = _resolve_hls_file_context(file_path)
    available = _build_quality_video_map(directory, target_stem=target_base)
    if not available:
        raise HTTPException(status_code=404, detail="No quality variants found for HLS")
    _schedule_hls_prewarm(available)

    # List lower bitrates first to reduce startup latency in ABR auto mode.
    quality_order = ["180p", "360p", "480p", "720p", "1080p"]
    bandwidth_map = {
        "1080p": 8_000_000,
        "720p": 4_500_000,
        "480p": 2_500_000,
        "360p": 1_200_000,
        "180p": 450_000,
    }
    resolution_map = {
        "1080p": "1920x1080",
        "720p": "1280x720",
        "480p": "854x480",
        "360p": "640x360",
        "180p": "320x180",
    }

    params: list[str] = []
    if auth:
        params.append(f"auth={quote(auth, safe='')}")
    if cb:
        params.append(f"cb={quote(cb, safe='')}")
    auth_q = f"?{'&'.join(params)}" if params else ""
    encoded_path = quote(file_path, safe="/")

    lines = ["#EXTM3U", "#EXT-X-VERSION:3"]
    for quality in quality_order:
        if quality not in available:
            continue
        lines.append(
            f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth_map[quality]},RESOLUTION={resolution_map[quality]},NAME="{quality}"'
        )
        lines.append(f"/hls/variant/{quality}/{encoded_path}{auth_q}")

    return Response(
        content="\n".join(lines) + "\n",
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.api_route("/hls/variant/{quality}/{file_path:path}", methods=["GET", "HEAD"])
async def serve_hls_variant(
    quality: str,
    file_path: str,
    authorization: Annotated[str | None, Header()] = None,
    auth: Annotated[str | None, Query()] = None,
    cb: Annotated[str | None, Query()] = None,
):
    """Serve single-variant HLS playlist pointing to the MP4 file of requested quality."""
    verify_auth_token(authorization=authorization, access_token=auth)

    if quality not in {"1080p", "720p", "480p", "360p", "180p"}:
        raise HTTPException(status_code=400, detail="Invalid quality")

    directory, target_base = _resolve_hls_file_context(file_path)
    available = _build_quality_video_map(directory, target_stem=target_base)
    selected = available.get(quality)
    if selected is None:
        raise HTTPException(status_code=404, detail="Requested quality not available")

    if config is None:
        raise HTTPException(status_code=503, detail="Server not initialized")

    video_root = Path(
        config.paths.video_root if config.paths.video_root else config.paths.archive_root
    ).resolve()
    rel = _resolve_media_relative(str(selected), video_root)
    if rel is None:
        raise HTTPException(status_code=500, detail="Could not resolve variant path")

    # Build real segmented HLS for robust player compatibility.
    cache_key, playlist_path = await asyncio.to_thread(_ensure_hls_variant_cache, selected, quality)
    content = _rewrite_hls_variant_playlist(playlist_path, cache_key, auth=auth)
    if cb:
        cb_q = quote(cb, safe="")

        def _inject_cb(match: re.Match[str]) -> str:
            base = match.group(1)
            q = match.group(2)
            if q:
                sep = "&" if len(q) > 1 else ""
                return f"{base}{q}{sep}cb={cb_q}"
            return f"{base}?cb={cb_q}"

        content = re.sub(r"(\n/hls/asset/[^\n?#]+)(\?[^\n#]*)?", _inject_cb, content)
    return Response(
        content=content,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-store"},
    )


@app.api_route("/hls/asset/{cache_key}/{file_name}", methods=["GET", "HEAD"])
async def serve_hls_asset(
    cache_key: str,
    file_name: str,
    authorization: Annotated[str | None, Header()] = None,
    auth: Annotated[str | None, Query()] = None,
    cb: Annotated[str | None, Query()] = None,
):
    """Serve generated HLS segments and variant-local assets."""
    verify_auth_token(authorization=authorization, access_token=auth)
    safe_key = cache_key.lower()
    if not re.fullmatch(r"[a-f0-9]{40}", safe_key):
        raise HTTPException(status_code=404, detail="HLS asset not found")
    safe_name = Path(file_name).name
    full_path = (HLS_CACHE_ROOT / safe_key / safe_name).resolve()
    root = (HLS_CACHE_ROOT / safe_key).resolve()
    if not str(full_path).startswith(str(root)):
        raise HTTPException(status_code=403, detail="Access denied")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="HLS asset not found")
    media_type = "video/MP2T" if full_path.suffix.lower() == ".ts" else "application/octet-stream"
    return FileResponse(
        path=str(full_path), media_type=media_type, headers={"Cache-Control": "no-store"}
    )


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

    # If a directory is passed, choose preferred VTT file from that directory.
    if full_path.exists() and full_path.is_dir():
        selected = _select_media_file_from_directory(full_path, kind="vtt")
        if selected is None:
            raise HTTPException(status_code=404, detail="VTT directory is empty")
        full_path = selected

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
