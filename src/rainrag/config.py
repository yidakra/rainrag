"""Configuration management for RainRAG."""

import logging
import os
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


_logger = logging.getLogger(__name__)


class PathsConfig(BaseModel):
    """Configuration for file paths."""

    archive_root: str = Field(description="Root directory containing .vtt files")
    docs_output: str = Field(description="Output path for parsed documents")
    embeddings_cache: str = Field(description="Directory for cached embeddings")
    video_root: str = Field(
        default="",
        description="Root directory containing video files (same as archive_root if not specified)",
    )


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    provider: str = Field(
        default="local",
        description="Embedding provider: 'local' for local model, 'mistral' for Mistral API, 'openai' for OpenAI API, 'gemini' for Google Gemini API",
    )
    model_name: str = Field(default="intfloat/multilingual-e5-large")
    batch_size: int = Field(default=32)
    max_seq_length: int = Field(
        default=512,
        description="Maximum sequence length in tokens for embedding model (multilingual-e5-large supports up to 512)",
    )
    device: str = Field(
        default="auto",
        description="Device selection: 'auto' (default), 'cuda', 'cuda:0', 'mps', or 'cpu'",
    )
    normalize_embeddings: bool = Field(default=True)
    prefix: str = Field(
        default="",
        description="Optional string prepended to texts before embedding (e.g. 'passage: ' for E5 models)",
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum number of retries for transient embedding API failures",
    )
    retry_backoff_factor: float = Field(
        default=2.0, gt=0, description="Exponential backoff factor for retries"
    )


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector store."""

    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    timeout: int = Field(
        default=180,
        ge=1,
        description="HTTP timeout in seconds for Qdrant requests",
    )
    collection_name: str = Field(default="broadcast_transcripts")
    vector_size: int = Field(default=1024)
    distance: str = Field(default="Cosine")
    recreate_collection: bool = Field(default=False)
    scroll_batch_size: int = Field(
        default=500,
        ge=1,
        description="Number of points to fetch per scroll request for incremental indexing",
    )
    max_scroll_iterations: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of client.scroll iterations during incremental indexing",
    )
    max_scroll_duration: int = Field(
        default=300,
        ge=1,
        description="Maximum duration in seconds for incremental scroll loop before aborting",
    )
    upsert_batch_size: int = Field(
        default=250,
        ge=1,
        description="Batch size for Qdrant upsert operations during indexing",
    )


class MistralConfig(BaseModel):
    """Configuration for Mistral API."""

    api_key: str = Field(description="Mistral API key")
    model_name: str = Field(
        default="mistral-small-latest",
        description="Mistral model to use: mistral-small-latest, mistral-medium-latest, mistral-large-latest, etc.",
    )
    embedding_model: str = Field(
        default="mistral-embed",
        description="Mistral embedding model",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class OpenAIConfig(BaseModel):
    """Configuration for OpenAI API."""

    api_key: str = Field(description="OpenAI API key")
    model_name: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use: gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model: text-embedding-3-small, text-embedding-3-large, etc.",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class ClaudeConfig(BaseModel):
    """Configuration for Anthropic Claude API."""

    api_key: str = Field(default="", description="Anthropic API key")
    model_name: str = Field(
        default="claude-3-5-sonnet-20240620",
        description="Claude model to use: claude-3-5-sonnet-20240620, claude-3-opus-20240229, claude-3-haiku-20240307, etc.",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini API."""

    api_key: str = Field(default="", description="Google API key")
    model_name: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use: gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash-exp, etc.",
    )
    embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model: models/text-embedding-004, models/embedding-001, etc.",
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    temperature: float = Field(default=0.3)
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class LLMConfig(BaseModel):
    """Configuration for LLM provider selection."""

    provider: str = Field(
        default="mistral",
        description="LLM provider: 'mistral' for Mistral API, 'openai' for OpenAI API, 'claude' for Anthropic Claude API, 'gemini' for Google Gemini API",
    )


class ChunkingConfig(BaseModel):
    """Configuration for VTT chunking."""

    enabled: bool = Field(default=True, description="Enable chunking of VTT files")
    strategy: Literal["time", "token", "hybrid"] = Field(
        default="hybrid",
        description="Chunking strategy: 'time' (time-based only), 'token' (token-based only), 'hybrid' (time-based with token validation)",
    )
    chunk_duration_seconds: int = Field(
        default=300,
        description="Duration of each chunk in seconds (default: 5 minutes) - used for 'time' and 'hybrid' strategies",
    )
    overlap_seconds: int = Field(
        default=30,
        description="Overlap between adjacent chunks in seconds (default: 30). Prevents information loss at chunk boundaries.",
    )
    min_chunk_tokens: int = Field(
        default=50,
        description="Minimum tokens per chunk (chunks smaller than this may be merged with neighbors)",
    )
    max_chunk_tokens: int | None = Field(
        default=None,
        description="Maximum tokens per chunk (auto-detected from embedding model if not set). Chunks exceeding this will be split.",
    )
    token_buffer: int = Field(
        default=50,
        description="Safety buffer to reserve for special tokens (e.g., 'passage:' prefix)",
    )


class HybridSearchConfig(BaseModel):
    """Configuration for hybrid search (vector + BM25)."""

    enabled: bool = Field(
        default=False,
        description="Enable hybrid search combining vector similarity and BM25 keyword matching",
    )
    bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 scores in hybrid search (0.0-1.0). Vector weight = 1 - bm25_weight",
    )
    top_k_multiplier: int = Field(
        default=3,
        description="Retrieve top_k * multiplier documents before reranking (to get more BM25 candidates)",
    )
    fusion_method: str = Field(
        default="rrf",
        description="Score fusion method: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (weighted sum)",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF constant (default: 60, standard value from literature)",
    )


class ProcessingConfig(BaseModel):
    """Configuration for data processing."""

    num_workers: int = Field(default=4)
    max_file_size: int = Field(default=10485760)  # 10 MB
    min_text_length: int = Field(default=50)


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    log_file: str = Field(default="./logs/rainrag.log")


class VideoConfig(BaseModel):
    """Configuration for video file serving."""

    enabled: bool = Field(default=True, description="Enable video file serving")
    extensions: list[str] = Field(
        default=[".mp4", ".mkv", ".webm", ".avi", ".mov"],
        description="Video file extensions to look for",
    )
    vtt_extensions: list[str] = Field(
        default=[".vtt", ".en.vtt", ".ru.vtt"],
        description="VTT file extensions to serve",
    )


class VideoUploadConfig(BaseModel):
    """Configuration for the single-video upload / scoped-Q&A mode.

    OpenAI transcription is available without local accelerator dependencies.
    The local provider runs faster-whisper in a separate interpreter, keeping
    its heavy CUDA/ctranslate2 stack out of the RainRAG environment.
    """

    # Off unless a deployment asks for it: OpenAI needs an API key and the local
    # provider needs a separate faster-whisper environment. A config that omits
    # this section should not expose an upload flow that cannot transcribe.
    enabled: bool = Field(default=False, description="Enable the video-upload mode")
    provider: Literal["openai", "local"] = Field(
        default="local", description="Transcription provider: openai or local"
    )
    openai_model: Literal["whisper-1"] = Field(
        default="whisper-1",
        description="OpenAI model used for timestamped upload transcription",
    )
    openai_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable containing the OpenAI API key",
    )
    openai_workers: int = Field(
        default=2,
        ge=1,
        description="Maximum concurrent OpenAI upload transcriptions",
    )
    openai_chunk_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="Target duration for silence-aware OpenAI audio chunks",
    )
    openai_silence_window_seconds: float = Field(
        default=30.0,
        ge=0,
        description="Window around chunk targets used to find nearby silence",
    )
    livevtt_python: str = Field(
        default="/home/ubuntu/livevtt/.venv/bin/python",
        description="Python interpreter that has faster-whisper installed",
    )
    transcribe_script: str = Field(
        default="scripts/transcribe_one.py",
        description="Path to the single-file transcription script",
    )
    model: str = Field(default="large-v3-turbo", description="faster-whisper model")
    compute_type: str = Field(default="int8_float16", description="ctranslate2 compute type")
    device: str = Field(default="cuda", description="cuda or cpu")
    device_index: int = Field(
        default=0, description="Fallback CUDA device when detection is unavailable"
    )
    device_indices: list[int] = Field(
        default_factory=list,
        description=(
            "CUDA devices transcription may use, one concurrent upload per device. "
            "Empty (the default) detects every visible GPU at startup, so a two-GPU "
            "box transcribes two uploads at once without being configured to."
        ),
    )
    language: str = Field(
        default="auto",
        description="Source language code, or 'auto' to detect it from the audio",
    )
    language_detection_segments: int = Field(
        default=4,
        description=(
            "Audio windows sampled when auto-detecting the language. More than one "
            "guards against a musical or foreign-language intro deciding the whole file."
        ),
    )
    multilingual: bool = Field(
        default=True,
        description=(
            "With language='auto', re-detect the language per window so a video that "
            "switches languages is decoded correctly. Costs some transcription speed."
        ),
    )
    beam_size: int = Field(default=5, description="Decoding beam size")
    tmp_root: str = Field(
        default="./data/video_sessions",
        description="Root directory for per-session working files",
    )
    collection_prefix: str = Field(
        default="session_", description="Prefix for ephemeral per-session collections"
    )
    session_ttl_seconds: int = Field(
        default=21600, description="Session lifetime before automatic cleanup (seconds)"
    )
    max_upload_mb: int = Field(default=512, description="Maximum accepted upload size in megabytes")
    yt_dlp_cookies_path: str = Field(
        default="",
        description=(
            "Optional cookies.txt (Netscape format) handed to yt-dlp. Needed for sites that "
            "no longer serve video anonymously -- X restricted guest access, and YouTube "
            "sometimes requires a signed-in session. Export it from a logged-in browser. "
            "Treat the file as a credential: it authenticates as that account. Missing or "
            "unreadable files are ignored, so the endpoint keeps working for other sites."
        ),
    )
    telegram_enabled: bool = Field(
        default=False,
        description=(
            "Fetch t.me links over MTProto instead of yt-dlp. Needs api_id/api_hash and a "
            "one-time user login (scripts/telegram_login.py). yt-dlp only manages roughly a "
            "third of public Telegram video posts, because Telegram often omits the video "
            "element from the web embed it scrapes."
        ),
    )
    telegram_api_id_env: str = Field(
        default="TELEGRAM_API_ID", description="Env var holding the my.telegram.org api_id"
    )
    telegram_api_hash_env: str = Field(
        default="TELEGRAM_API_HASH", description="Env var holding the my.telegram.org api_hash"
    )
    telegram_session_path: str = Field(
        default="./data/telegram.session",
        description=(
            "Telethon session file. This grants full access to the logged-in account: "
            "treat it as a credential and keep it out of version control."
        ),
    )
    telegram_flood_sleep_threshold: int = Field(
        default=60,
        description=(
            "Absorb Telegram FloodWait pauses up to this many seconds; longer waits surface "
            "as an error instead of blocking the request."
        ),
    )
    sweep_orphans_on_start: bool = Field(
        default=True,
        description=(
            "On startup, drop leftover session collections and working directories from "
            "previous runs. Assumes a single API process owns them; disable if running "
            "multiple API workers against one Qdrant instance."
        ),
    )


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = Field(
        default="stdio",
        description='Transport protocol: "stdio", "sse", or "streamable-http"',
    )
    host: str = Field(default="localhost")
    port: int = Field(default=8000)


class CohereConfig(BaseModel):
    """Configuration for Cohere API (reranking and embeddings)."""

    api_key: str = Field(default="", description="Cohere API key")
    model_name: str = Field(
        default="rerank-v3.5",
        description="Cohere rerank model: rerank-v3.5, rerank-english-v3.0, rerank-multilingual-v3.0",
    )
    embedding_model: str = Field(
        default="embed-multilingual-v3.0",
        description="Cohere embedding model: embed-multilingual-v3.0, embed-english-v3.0, etc.",
    )


class RerankerConfig(BaseModel):
    """Configuration for reranking."""

    enabled: bool = Field(default=False, description="Enable reranking of search results")
    provider: str = Field(default="cohere", description="Reranker provider: 'cohere'")
    top_n: int = Field(default=5, description="Number of documents to return after reranking")
    initial_k: int = Field(
        default=20, description="Number of candidates to retrieve before reranking"
    )
    min_retrieval_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum score threshold for documents passed to the LLM. "
            "Documents scoring below this value are dropped before prompt assembly, "
            "even if they rank within top_k. "
            "Motivated by Cuconasu et al. (SIGIR 2024): high-scoring-but-irrelevant "
            "documents harm LLM accuracy more than no documents at all. "
            "Set to 0.0 (default) to disable filtering."
        ),
    )


class TwoStageConfig(BaseModel):
    """Configuration for modern two-stage retrieval (query rewrite + HyDE fusion)."""

    enabled: bool = Field(
        default=False,
        description=(
            "Enable two-stage retrieval pipeline. "
            "Sub-features like query_rewrite_enabled and hyde_enabled are only respected when enabled=True. "
            "Set enabled=True to activate the two-stage behavior; otherwise sub-feature keys are ignored."
        ),
    )

    # Stage 2a: LLM query rewriting
    query_rewrite_enabled: bool = Field(
        default=True,
        description=(
            "Rewrite query into transcript-register variants before retrieval. "
            "This setting is ignored unless TwoStageConfig.enabled=True."
        ),
    )
    query_rewrite_variants: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of rewritten variants to generate (original is always included)",
    )
    query_rewrite_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "Temperature for the query-rewriting LLM call. "
            "Higher values produce more diverse paraphrases. "
            "Kept separate from answer-generation temperature, which should stay at 0 "
            "for deterministic, source-grounded journalist output."
        ),
    )

    # Stage 2b: HyDE (Hypothetical Document Embedding)
    hyde_enabled: bool = Field(
        default=False,
        description="Generate a hypothetical transcript passage and blend its embedding with the query embedding",
    )
    hyde_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Interpolation weight for HyDE embedding (0=raw query only, 1=HyDE only)",
    )
    hyde_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "Temperature for the HyDE hypothetical-document generation call. "
            "Higher values produce more varied hypothetical passages, improving embedding coverage. "
            "Kept separate from answer-generation temperature, which should stay at 0 "
            "for deterministic, source-grounded journalist output."
        ),
    )

    # Multi-variant merge strategy (applied when query_rewrite_enabled and >1 variant)
    merge_strategy: Literal["coverage", "diverse_rrf"] = Field(
        default="coverage",
        description=(
            "Strategy for merging results from multiple query variants. "
            "'coverage' uses a greedy set-cover selection (VRisker-style) that "
            "maximises the number of variants with at least one selected document, "
            "preventing majority-intent documents from drowning out minority readings. "
            "'diverse_rrf' uses multi-source RRF with a concave diversity weight "
            "(1/sqrt(variant_count)) that gives diminishing returns to consensus "
            "documents and upweights candidates unique to under-served variants."
        ),
    )
    merge_rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "RRF constant k used by the 'diverse_rrf' merge strategy. "
            "Standard literature default is 60 (tuned for two-source fusion). "
            "Smaller values (e.g. 20–40) amplify rank differences more aggressively "
            "and may help when variant result lists are short (top_k ≤ 5)."
        ),
    )

    # Prompt document ordering (Axis F)
    prompt_doc_order: Literal["rank", "reversed", "book_end"] = Field(
        default="rank",
        description=(
            "Order in which retrieved documents are presented in the LLM prompt. "
            "'rank' (default) places the highest-scoring document first and lowest last. "
            "'reversed' places the lowest-scoring document first and highest last, "
            "exploiting recency bias in LLMs (primary position effect). "
            "'book_end' places the two highest-scoring documents at the start and end "
            "of the context window with lower-scoring documents in the middle, "
            "combating the 'lost in the middle' attention deficit identified by "
            "Liu et al. (2023) and further motivated by Cuconasu et al. (SIGIR 2024)."
        ),
    )


class IncrementalConfig(BaseModel):
    """Configuration for incremental re-indexing pipeline."""

    enabled: bool = Field(
        default=True,
        description="Enable incremental processing (only re-process changed documents)",
    )
    manifest_path: str = Field(
        default="./data/manifest.json",
        description="Path to file manifest for tracking source file changes",
    )
    alias_swap: bool = Field(
        default=False,
        description="Use two-phase indexing with collection alias swap for zero-downtime updates",
    )


class WebMetadataConfig(BaseModel):
    """Configuration for web metadata integration."""

    enabled: bool = Field(
        default=False, description="Enable loading of web metadata from JSON files"
    )
    source: Literal["local", "api", "hybrid"] = Field(
        default="local",
        description=(
            "Metadata source: 'local' reads from path/ directory only; "
            "'api' fetches from the library API only (with local cache); "
            "'hybrid' tries local first, falls back to API for missing hashes"
        ),
    )
    path: str = Field(
        default="./web_metadata", description="Path to directory containing web metadata JSON files"
    )
    api_url: str = Field(
        default="https://library.tvrain.tv",
        description="Base URL for the library metadata API",
    )
    api_token_env: str = Field(
        default="LIBRARY_API_TOKEN",
        description="Environment variable name holding the Bearer token for the library API",
    )
    api_batch_days: int = Field(
        default=180,
        ge=1,
        le=180,
        description="Number of days to look back when running batch export (max 180)",
    )
    min_content_length: int = Field(
        default=10, ge=1, description="Minimum content length for web description text"
    )
    include_in_text: bool = Field(
        default=False,
        description="If True, append web metadata to document text for retrieval",
    )
    append_label: str = Field(
        default="[Web]",
        description="Label used when appending web metadata to document text",
    )
    append_to_each_chunk: bool = Field(
        default=True,
        description="If True, append web metadata to each chunk text",
    )
    fields: list[str] = Field(
        default=["title", "date", "description", "url"],
        description=(
            "Web metadata fields to append to document text. Accepted: title, date, "
            "description, url, program, presenters, tags, stories. Changing this list "
            "changes the embedded text and so forces a full re-embed, not just a reindex"
        ),
    )
    require_web_metadata: bool = Field(
        default=False,
        description="If True, only ingest videos that have corresponding web metadata; if False, ingest all videos with empty web fields when metadata is missing",
    )
    ingest_speech_free: bool = Field(
        default=True,
        description="If True, create a metadata-only document for speech-free videos (empty VTT, no subtitle cues) when web metadata is available; if False, skip them entirely",
    )


class Config(BaseModel):
    """Main configuration model."""

    paths: PathsConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    llm: LLMConfig
    mistral: MistralConfig
    openai: OpenAIConfig
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    cohere: CohereConfig = Field(default_factory=CohereConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    two_stage: TwoStageConfig = Field(default_factory=TwoStageConfig)
    processing: ProcessingConfig
    logging: LoggingConfig
    video: VideoConfig = Field(default_factory=VideoConfig)
    video_upload: VideoUploadConfig = Field(default_factory=VideoUploadConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    web_metadata: WebMetadataConfig = Field(default_factory=WebMetadataConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)

    def get_max_chunk_tokens(self) -> int:
        """
        Get the maximum chunk size in tokens based on embedding model.

        Returns the configured max_chunk_tokens if set, otherwise auto-detects
        based on the embedding model's known limits.

        Returns:
            Maximum tokens per chunk (accounting for token_buffer)
        """
        # Use explicit config if set
        if self.chunking.max_chunk_tokens is not None:
            return self.chunking.max_chunk_tokens

        # Auto-detect based on embedding model
        model_limits = {
            # Local models
            "intfloat/multilingual-e5-large": 512,
            "intfloat/multilingual-e5-base": 512,
            "intfloat/multilingual-e5-small": 512,
            "sentence-transformers/all-MiniLM-L6-v2": 256,
            "sentence-transformers/all-mpnet-base-v2": 384,
            # OpenAI models
            "text-embedding-3-large": 8191,
            "text-embedding-3-small": 8191,
            "text-embedding-ada-002": 8191,
            # Cohere models
            "embed-english-v3.0": 512,
            "embed-multilingual-v3.0": 512,
            "embed-english-light-v3.0": 512,
            "embed-multilingual-light-v3.0": 512,
            # Mistral models
            "mistral-embed": 512,
            # Gemini models
            "models/text-embedding-004": 3072,
            "models/embedding-001": 2048,
        }

        # Get model name based on provider
        if self.embedding.provider == "local":
            model_name = self.embedding.model_name
        elif self.embedding.provider == "openai":
            model_name = self.openai.embedding_model
        elif self.embedding.provider == "gemini":
            model_name = self.gemini.embedding_model
        elif self.embedding.provider == "mistral":
            model_name = self.mistral.embedding_model
        elif self.embedding.provider == "cohere":
            model_name = self.cohere.embedding_model
        else:
            # Default to configured max_seq_length
            return max(1, self.embedding.max_seq_length - self.chunking.token_buffer)

        # Look up model limit
        limit = model_limits.get(model_name, self.embedding.max_seq_length)

        # Subtract safety buffer
        return max(1, limit - self.chunking.token_buffer)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object
    """
    # Load environment variables from .env file if it exists
    load_dotenv()

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config_data = cast(dict[str, Any], yaml.safe_load(f))

    def _get_secret_or_env(env_name: str) -> str | None:
        """Read secret from <ENV>_FILE path first, then fallback to <ENV>."""
        file_var = f"{env_name}_FILE"
        secret_file = os.getenv(file_var)
        if secret_file:
            try:
                value = Path(secret_file).read_text(encoding="utf-8").strip()
                if value:
                    return value
            except OSError as exc:
                # If the secret file cannot be read, log the issue and fall back to the
                # environment variable. We don't want this to crash the program.
                _logger.debug(
                    "Failed to read %s from %s: %s",
                    file_var,
                    secret_file,
                    exc,
                )

        value = os.getenv(env_name)
        if value:
            return value
        return None

    providers = [
        ("MISTRAL_API_KEY", "mistral", "api_key"),
        ("OPENAI_API_KEY", "openai", "api_key"),
        ("ANTHROPIC_API_KEY", "claude", "api_key"),
        ("GOOGLE_API_KEY", "gemini", "api_key"),
        ("COHERE_API_KEY", "cohere", "api_key"),
    ]

    for env_name, section, key in providers:
        value = _get_secret_or_env(env_name)
        if value:
            if section not in config_data:
                config_data[section] = {}
            config_data[section][key] = value

    return Config(**config_data)
