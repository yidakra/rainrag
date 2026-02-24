"""Ingestion module for parsing VTT subtitle files."""

import hashlib
import html
import json
import re
import subprocess
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel
from tqdm import tqdm

from rainrag.config import Config


class Document(BaseModel):
    """Parsed document model."""

    id: str
    path: str
    language: str
    text: str
    length: int
    date: str | None = None  # ISO date from source video mtime
    date_ts: float | None = None  # Timestamp (seconds) for range filtering
    duration_seconds: float | None = None  # Video duration in seconds
    start_time: str | None = None  # First timecode in VTT (HH:MM:SS)
    end_time: str | None = None  # Last timecode in VTT (HH:MM:SS)

    # Chunking and video identification fields
    video_id: str | None = None  # Common ID for all chunks/languages of same video
    is_chunk: bool = False  # True if this is a chunk, False if full document
    chunk_index: int | None = None  # Index of chunk (0-based)
    total_chunks: int | None = None  # Total number of chunks for this video
    start_time_seconds: float | None = None  # Start time in seconds for precise filtering
    end_time_seconds: float | None = None  # End time in seconds for precise filtering

    # Web metadata fields
    web_title: str | None = None  # Title from web metadata
    web_date: str | None = None  # ISO date from web metadata (preferred over date)
    web_date_ts: float | None = None  # Timestamp from web metadata
    web_description: str | None = None  # Description from web metadata
    web_url: str | None = None  # URL from web metadata

    # Content hash for incremental processing
    content_hash: str | None = None  # SHA-256 hash of embedded text for change detection


class VTTCue(BaseModel):
    """Represents a single VTT subtitle cue."""

    start_time: str  # HH:MM:SS format
    end_time: str  # HH:MM:SS format
    start_seconds: float | None  # Timestamp in seconds
    end_seconds: float | None  # Timestamp in seconds
    text: str  # Cleaned text


class VTTChunk(BaseModel):
    """Represents a time-based chunk of VTT content."""

    chunk_index: int
    start_time: str  # HH:MM:SS format
    end_time: str  # HH:MM:SS format
    start_seconds: float | None
    end_seconds: float | None
    text: str
    cue_count: int  # Number of cues in this chunk


class VTTParser:
    """Parser for WebVTT subtitle files."""

    # Pattern to match VTT timestamp lines (e.g., "00:00:00.000 --> 00:00:05.000")
    TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*$")

    # Pattern to match VTT cue identifiers (numeric or alphanumeric IDs)
    CUE_ID_PATTERN = re.compile(r"^\d+$|^[a-zA-Z0-9_-]+$")

    # Pattern to remove VTT markup tags (e.g., <v Speaker>, <c>, positioning tags)
    MARKUP_PATTERN = re.compile(r"<[^>]+>")

    # Large value to effectively disable time-based chunking in hybrid mode
    INFINITE_CHUNK_DURATION = 9999999

    @staticmethod
    def extract_video_hash(path: Path | str) -> str:
        """
        Extract the video hash from a file path or filename.

        Handles filenames like "abc123.ru.vtt" -> "abc123" or "abc123.vtt" -> "abc123"

        Args:
            path: File path or filename string

        Returns:
            Video hash without language suffix or extension
        """
        stem = Path(path).stem  # Remove extension: "abc123.ru"
        return stem.rsplit(".", 1)[0] if "." in stem else stem  # Remove language suffix: "abc123"

    @staticmethod
    def detect_language(file_path: Path) -> str:
        """
        Detect language from file path or name.

        Looks for 'ru', 'rus', 'russian' or 'en', 'eng', 'english' in path.

        Args:
            file_path: Path to the VTT file

        Returns:
            Language code ('ru' or 'en')
        """
        path_str = str(file_path).lower()
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", path_str)))

        # Check for Russian indicators
        if any(indicator in tokens for indicator in ["ru", "rus", "russian"]):
            return "ru"

        # Check for English indicators
        if any(indicator in tokens for indicator in ["en", "eng", "english"]):
            return "en"

        # Default to English if no indicators found
        logger.warning(f"Could not detect language for {file_path}, defaulting to 'en'")
        return "en"

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean VTT text by removing markup and normalizing whitespace.

        Args:
            text: Raw text from VTT file

        Returns:
            Cleaned text
        """
        # Remove VTT markup tags
        text = VTTParser.MARKUP_PATTERN.sub("", text)

        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def timestamp_to_seconds(timestamp: str) -> float | None:
        """
        Convert HH:MM:SS timestamp to seconds.

        Args:
            timestamp: Timestamp in HH:MM:SS format

        Returns:
            Time in seconds, or None if parsing fails
        """
        try:
            parts = timestamp.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse timestamp '{timestamp}': {e}")
            return None

    @staticmethod
    def seconds_to_timestamp(seconds: float) -> str:
        """
        Convert seconds to HH:MM:SS timestamp.

        Args:
            seconds: Time in seconds

        Returns:
            Timestamp in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def estimate_tokens(text: str, language: str = "en") -> int:
        """
        Estimate token count for text using character-to-token ratio.

        This is a fast approximation that doesn't require loading tokenizers.
        For precise counting, use the actual model tokenizer.

        Args:
            text: Text to estimate tokens for
            language: Language code for better estimation

        Returns:
            Estimated token count
        """
        # Character-to-token ratios (conservative estimates)
        # Based on empirical measurements with various models
        ratios = {
            "en": 4.0,  # English: ~4 chars per token
            "ru": 2.5,  # Russian/Cyrillic: ~2.5 chars per token (denser)
            "zh": 1.5,  # Chinese: ~1.5 chars per token
            "ja": 1.5,  # Japanese: ~1.5 chars per token
            "ar": 3.0,  # Arabic: ~3 chars per token
        }

        ratio = ratios.get(language, 3.5)  # Default: 3.5 chars/token
        estimated_tokens = len(text) / ratio

        return int(estimated_tokens)

    @classmethod
    def parse_vtt(cls, file_path: Path) -> str | None:
        """
        Parse a VTT file and extract clean transcript text.

        Args:
            file_path: Path to the VTT file

        Returns:
            Cleaned transcript text or None if parsing fails
        """
        result = cls.parse_vtt_with_timecodes(file_path)
        return result[0] if result else None

    @classmethod
    def parse_vtt_with_timecodes(cls, file_path: Path) -> tuple[str, str | None, str | None] | None:
        """
        Parse a VTT file and extract text with first/last timecodes.

        Args:
            file_path: Path to the VTT file

        Returns:
            Tuple of (text, start_time, end_time) or None if parsing fails.
            Timecodes are in HH:MM:SS format.
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            cues = cls._parse_vtt_lines_to_cues(lines)
            if cues is None:
                logger.debug(f"File {file_path} does not appear to be a valid VTT file")
                return None

            # Extract text and timecodes from cues
            text_parts = [cue.text for cue in cues if cue.text]
            full_text = " ".join(text_parts)

            # Get first and last timecodes
            start_time = cues[0].start_time if cues else None
            end_time = cues[-1].end_time if cues else None

            return (full_text, start_time, end_time) if full_text else None

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    @classmethod
    def _finalize_cue(cls, timestamp_line: str, text_lines: list[str]) -> VTTCue | None:
        """
        Finalize a cue from timestamp line and text lines.

        Args:
            timestamp_line: VTT timestamp line (e.g., "00:00:00.000 --> 00:00:10.160")
            text_lines: List of text lines for the cue

        Returns:
            VTTCue object if valid, None if invalid timestamp or empty text
        """
        # Parse timestamp: "00:00:00.000 --> 00:00:10.160"
        parts = timestamp_line.split("-->")
        if len(parts) != 2:
            return None

        start = parts[0].strip().split(".")[0]  # Remove milliseconds
        end = parts[1].strip().split()[0].split(".")[0]  # Remove ms and positioning

        # Handle invalid timestamp parts
        if not start or not end:
            return None

        # Combine and clean text
        combined_text = " ".join(text_lines)
        cleaned_text = cls.clean_text(combined_text)

        if not cleaned_text:
            return None

        cue = VTTCue(
            start_time=start,
            end_time=end,
            start_seconds=cls.timestamp_to_seconds(start),
            end_seconds=cls.timestamp_to_seconds(end),
            text=cleaned_text,
        )
        return cue

    @classmethod
    def _parse_vtt_lines_to_cues(cls, lines: list[str]) -> list[VTTCue] | None:
        """
        Parse VTT file lines into individual cues with timestamps.

        Args:
            lines: List of lines from a VTT file

        Returns:
            List of VTTCue objects or None if parsing fails
        """
        # VTT files should start with "WEBVTT"
        if not lines or not lines[0].strip().startswith("WEBVTT"):
            return None

        cues: list[VTTCue] = []
        current_timestamp_line = None
        current_text_lines: list[str] = []

        for i, line in enumerate(lines[1:], 1):  # Skip the WEBVTT header
            line = line.strip()

            # Skip empty lines - finalize current cue if any
            if not line:
                if current_timestamp_line and current_text_lines:
                    cue = cls._finalize_cue(current_timestamp_line, current_text_lines)
                    if cue:
                        cues.append(cue)

                current_timestamp_line = None
                current_text_lines = []
                continue

            # Capture timestamp lines
            if cls.TIMESTAMP_PATTERN.match(line):
                current_timestamp_line = line
                continue

            # Skip cue identifiers only if immediately followed by timestamp
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if cls.CUE_ID_PATTERN.match(line) and cls.TIMESTAMP_PATTERN.match(next_line):
                continue

            # Skip NOTE comments
            if line.lstrip().upper().startswith("NOTE"):
                continue

            # This is actual subtitle text
            current_text_lines.append(line)

        # Don't forget the last cue if file doesn't end with empty line
        if current_timestamp_line and current_text_lines:
            cue = cls._finalize_cue(current_timestamp_line, current_text_lines)
            if cue:
                cues.append(cue)

        return cues if cues else None

    @classmethod
    def parse_vtt_to_cues(cls, file_path: Path) -> list[VTTCue] | None:
        """
        Parse a VTT file into individual cues with timestamps.

        Args:
            file_path: Path to the VTT file

        Returns:
            List of VTTCue objects or None if parsing fails
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()

            cues = cls._parse_vtt_lines_to_cues(lines)
            if cues is None:
                logger.debug(f"File {file_path} does not appear to be a valid VTT file")
            return cues

        except Exception as e:
            logger.error(f"Error parsing {file_path} to cues: {e}")
            return None

    @classmethod
    def create_chunks_from_cues(
        cls, cues: list[VTTCue], chunk_duration_seconds: int = 300, overlap_seconds: int = 30
    ) -> list[VTTChunk]:
        """
        Create time-based chunks from a list of VTT cues with optional overlap.

        Args:
            cues: List of VTTCue objects
            chunk_duration_seconds: Duration of each chunk in seconds
            overlap_seconds: Overlap between adjacent chunks in seconds (default: 30)

        Returns:
            List of VTTChunk objects with overlapping context
        """
        if not cues:
            return []

        chunks: list[VTTChunk] = []
        chunk_index = 0
        chunk_start_seconds = 0.0
        chunk_cues: list[VTTCue] = []

        for cue in cues:
            # Skip cues with invalid timestamps
            if cue.start_seconds is None or cue.end_seconds is None:
                logger.warning(
                    f"Skipping cue with invalid timestamp: {cue.start_time} -> {cue.end_time}"
                )
                continue

            # Check if this cue starts a new chunk
            if cue.start_seconds >= chunk_start_seconds + chunk_duration_seconds:
                # Finalize current chunk if it has content
                if chunk_cues:
                    chunk_text = " ".join(c.text for c in chunk_cues)

                    chunk = VTTChunk(
                        chunk_index=chunk_index,
                        start_time=chunk_cues[0].start_time,
                        end_time=chunk_cues[-1].end_time,
                        start_seconds=chunk_cues[0].start_seconds,
                        end_seconds=chunk_cues[-1].end_seconds,
                        text=chunk_text,
                        cue_count=len(chunk_cues),
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap
                # Move forward by (chunk_duration - overlap) to create overlap
                chunk_start_seconds += chunk_duration_seconds - overlap_seconds

                # Keep overlapping cues from previous chunk
                chunk_cues = [
                    c
                    for c in chunk_cues
                    if c.start_seconds is not None and c.start_seconds >= chunk_start_seconds
                ]

            chunk_cues.append(cue)

        # Add final chunk
        if chunk_cues:
            chunk_text = " ".join(c.text for c in chunk_cues)

            chunk = VTTChunk(
                chunk_index=chunk_index,
                start_time=chunk_cues[0].start_time,
                end_time=chunk_cues[-1].end_time,
                start_seconds=chunk_cues[0].start_seconds,
                end_seconds=chunk_cues[-1].end_seconds,
                text=chunk_text,
                cue_count=len(chunk_cues),
            )
            chunks.append(chunk)

        return chunks

    @classmethod
    def _merge_undersized_chunks(
        cls, chunks: list[VTTChunk], min_tokens: int, max_tokens: int, language: str
    ) -> list[VTTChunk]:
        """
        Merge chunks that are smaller than min_tokens with their neighbors.

        Args:
            chunks: List of VTTChunk objects to process
            min_tokens: Minimum token count threshold for merging
            max_tokens: Maximum token count threshold for merged chunks
            language: Language code for token estimation

        Returns:
            List of merged VTTChunk objects
        """
        if not chunks:
            return chunks

        merged_chunks: list[VTTChunk] = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            current_tokens = cls.estimate_tokens(current_chunk.text, language)

            # If current chunk is large enough, keep it
            if current_tokens >= min_tokens:
                merged_chunks.append(current_chunk)
                i += 1
                continue

            # Current chunk is too small, try to merge with previous chunk
            if merged_chunks:
                prev_chunk = merged_chunks[-1]
                combined_text = prev_chunk.text + " " + current_chunk.text
                combined_tokens = cls.estimate_tokens(combined_text, language)

                # Only merge if the combined chunk doesn't exceed a reasonable limit
                # Use a conservative limit to prevent creating oversized chunks
                if combined_tokens <= max_tokens:
                    # Merge with previous
                    merged_chunk = VTTChunk(
                        chunk_index=prev_chunk.chunk_index,
                        start_time=prev_chunk.start_time,
                        end_time=current_chunk.end_time,
                        start_seconds=prev_chunk.start_seconds,
                        end_seconds=current_chunk.end_seconds,
                        text=combined_text,
                        cue_count=prev_chunk.cue_count + current_chunk.cue_count,
                    )
                    merged_chunks[-1] = merged_chunk
                    i += 1
                    continue

            # If we couldn't merge with previous, try merging with next chunk
            if i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                combined_text = current_chunk.text + " " + next_chunk.text
                combined_tokens = cls.estimate_tokens(combined_text, language)

                if combined_tokens <= max_tokens:
                    # Merge with next
                    merged_chunk = VTTChunk(
                        chunk_index=current_chunk.chunk_index,
                        start_time=current_chunk.start_time,
                        end_time=next_chunk.end_time,
                        start_seconds=current_chunk.start_seconds,
                        end_seconds=next_chunk.end_seconds,
                        text=combined_text,
                        cue_count=current_chunk.cue_count + next_chunk.cue_count,
                    )
                    merged_chunks.append(merged_chunk)
                    i += 2  # Skip the next chunk since we merged it
                    continue

            # If we couldn't merge with either neighbor, keep the small chunk as-is
            # This preserves temporal boundaries when merging would create problems
            merged_chunks.append(current_chunk)
            i += 1

        return merged_chunks

    @classmethod
    def _create_chunk_from_cues(cls, cues: list[VTTCue], chunk_index: int) -> VTTChunk:
        """Create a VTTChunk from a list of cues."""
        if not cues:
            raise ValueError("cues must not be empty")
        text = " ".join(c.text for c in cues)
        return VTTChunk(
            chunk_index=chunk_index,
            start_time=cues[0].start_time,
            end_time=cues[-1].end_time,
            start_seconds=cues[0].start_seconds,
            end_seconds=cues[-1].end_seconds,
            text=text,
            cue_count=len(cues),
        )

    @classmethod
    def _split_cues_by_tokens(
        cls, cues: list[VTTCue], max_tokens: int, language: str, start_chunk_index: int
    ) -> tuple[list[VTTChunk], int]:
        """Split cues into chunks based on token limits.

        Args:
            cues: List of VTTCue objects to split
            max_tokens: Maximum tokens per chunk
            language: Language code for token estimation
            start_chunk_index: Starting chunk index for new chunks

        Returns:
            Tuple of (produced chunks, next available chunk index)
        """
        chunks: list[VTTChunk] = []
        current_cues: list[VTTCue] = []
        current_tokens = 0
        chunk_index = start_chunk_index

        for cue in cues:
            cue_tokens = cls.estimate_tokens(cue.text, language)

            # If cue itself exceeds limit, add it as a separate chunk
            if cue_tokens > max_tokens:
                # First finalize any pending chunk
                if current_cues:
                    chunks.append(cls._create_chunk_from_cues(current_cues, chunk_index))
                    chunk_index += 1
                    current_cues = []
                    current_tokens = 0

                # Add the oversized cue as its own chunk
                chunks.append(cls._create_chunk_from_cues([cue], chunk_index))
                chunk_index += 1
                continue

            # Check if adding this cue would exceed limit
            if (
                current_tokens + cue_tokens + (1 if current_cues else 0) > max_tokens
                and current_cues
            ):
                # Finalize current chunk
                chunks.append(cls._create_chunk_from_cues(current_cues, chunk_index))
                chunk_index += 1

                # Start new chunk
                current_cues = [cue]
                current_tokens = cue_tokens
            else:
                current_cues.append(cue)
                current_tokens += cue_tokens + (1 if current_cues else 0)

        # Add final chunk
        if current_cues:
            chunks.append(cls._create_chunk_from_cues(current_cues, chunk_index))
            chunk_index += 1

        return chunks, chunk_index

    @classmethod
    def create_chunks_hybrid(
        cls,
        cues: list[VTTCue],
        chunk_duration_seconds: int = 300,
        overlap_seconds: int = 30,
        max_tokens: int = 462,  # 512 - 50 buffer
        min_tokens: int = 50,
        language: str = "en",
    ) -> list[VTTChunk]:
        """
        Create chunks using hybrid strategy: time-based with token validation and overlap.

        This method:
        1. Creates initial time-based chunks with overlap (semantic coherence)
        2. Validates chunks don't exceed max_tokens
        3. Splits oversized chunks by token count
        4. Merges undersized chunks with neighbors

        Args:
            cues: List of VTTCue objects
            chunk_duration_seconds: Target duration for time-based chunks
            overlap_seconds: Overlap between adjacent chunks in seconds
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk (for merging)
            language: Language code for token estimation

        Returns:
            List of VTTChunk objects optimized for token limits with overlap
        """
        if not cues:
            return []

        # Step 1: Create initial time-based chunks with overlap
        time_chunks = cls.create_chunks_from_cues(cues, chunk_duration_seconds, overlap_seconds)

        # Step 2: Validate and split oversized chunks
        validated_chunks: list[VTTChunk] = []
        chunk_index = 0

        for time_chunk in time_chunks:
            estimated_tokens = cls.estimate_tokens(time_chunk.text, language)

            # If chunk fits within limit, keep it
            if estimated_tokens <= max_tokens:
                # Update chunk index
                validated_chunk = VTTChunk(
                    chunk_index=chunk_index,
                    start_time=time_chunk.start_time,
                    end_time=time_chunk.end_time,
                    start_seconds=time_chunk.start_seconds,
                    end_seconds=time_chunk.end_seconds,
                    text=time_chunk.text,
                    cue_count=time_chunk.cue_count,
                )
                validated_chunks.append(validated_chunk)
                chunk_index += 1
            else:
                # Split oversized chunk by token count
                # Re-parse this time chunk's cues and split by tokens
                chunk_cues = [
                    cue
                    for cue in cues
                    if (
                        cue.start_seconds is not None
                        and cue.end_seconds is not None
                        and time_chunk.start_seconds is not None
                        and time_chunk.end_seconds is not None
                        and time_chunk.start_seconds <= cue.start_seconds < time_chunk.end_seconds
                    )
                ]

                # Build token-based sub-chunks
                sub_chunks, chunk_index = cls._split_cues_by_tokens(
                    chunk_cues, max_tokens, language, chunk_index
                )
                validated_chunks.extend(sub_chunks)

        # Step 3: Merge undersized chunks with neighbors
        merged_chunks = cls._merge_undersized_chunks(
            validated_chunks, min_tokens, max_tokens, language
        )

        return merged_chunks

    @staticmethod
    def generate_id(file_path: Path, chunk_index: int | None = None) -> str:
        """
        Generate a unique ID for a document based on its path and optional chunk index.

        Args:
            file_path: Path to the file
            chunk_index: Optional chunk index for chunked documents

        Returns:
            Unique hash ID
        """
        path_str = str(file_path.absolute())
        if chunk_index is not None:
            path_str = f"{path_str}_chunk_{chunk_index}"
        return hashlib.sha256(path_str.encode()).hexdigest()[:16]

    @staticmethod
    def generate_video_id(file_path: Path) -> str:
        """
        Generate a video ID that's shared across all languages/chunks of the same video.

        For files like "abc123.en.vtt" and "abc123.ru.vtt", this returns the same ID.

        Args:
            file_path: Path to the VTT file

        Returns:
            Video ID (hash of base filename without language suffix)
        """
        # Extract the hash from filename (e.g., "abc123.ru.vtt" -> "abc123")
        video_hash = VTTParser.extract_video_hash(file_path)

        # Combine with parent directory for uniqueness
        parent_str = str(file_path.parent.absolute())
        video_key = f"{parent_str}/{video_hash}"

        return hashlib.sha256(video_key.encode()).hexdigest()[:16]

    @staticmethod
    def find_source_video(vtt_path: Path) -> Path | None:
        """
        Find the source video file for a VTT subtitle file.

        VTT files are named like: {hash}.{lang}.vtt
        Source videos are named like: {hash} (no extension) or {hash}.mp4

        Args:
            vtt_path: Path to the VTT file

        Returns:
            Path to source video or None if not found
        """
        # Extract the hash from filename (e.g., "abc123.ru.vtt" -> "abc123")
        video_hash = VTTParser.extract_video_hash(vtt_path)

        parent = vtt_path.parent

        # Try to find source video (usually extensionless or .mp4)
        candidates = [
            parent / video_hash,  # No extension (original)
            parent / f"{video_hash}.mp4",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    @staticmethod
    def get_video_metadata(video_path: Path) -> tuple[str | None, float | None, float | None]:
        """
        Extract metadata from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (date_iso, duration_seconds, date_ts)
        """
        date_iso = None
        date_ts = None
        duration = None

        # Get mtime as date
        try:
            mtime = video_path.stat().st_mtime
            date_iso = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            date_ts = mtime
        except Exception as e:
            logger.debug(f"Could not get mtime for {video_path}: {e}")

        # Get duration via ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                duration_str = data.get("format", {}).get("duration")
                if duration_str:
                    duration = float(duration_str)
        except Exception as e:
            logger.debug(f"Could not get duration for {video_path}: {e}")

        return date_iso, duration, date_ts


class WebMetadataLoader:
    """Loader for web metadata JSON files."""

    def __init__(self, metadata_path: Path) -> None:
        """
        Initialize the web metadata loader.

        Args:
            metadata_path: Path to directory containing web metadata JSON files
        """
        super().__init__()
        self.metadata_path = metadata_path

    def load_metadata(self, video_hash: str) -> dict[str, Any] | None:
        """
        Load web metadata for a video by its hash.

        Args:
            video_hash: Video hash (filename without extension)

        Returns:
            Dictionary containing web metadata, or None if not found
        """
        metadata_file = self.metadata_path / f"{video_hash}.json"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.debug(f"Could not load web metadata for {video_hash}: {e}")
            return None

    def extract_clean_metadata(self, raw_metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Extract and clean web metadata for use in documents.

        Args:
            raw_metadata: Raw metadata from JSON file

        Returns:
            Cleaned metadata dictionary
        """

        # Extract basic fields
        title = raw_metadata.get("name", "").strip()
        date_active_start = raw_metadata.get("date_active_start")
        url = raw_metadata.get("url", "").strip()

        # Clean and combine text content
        preview_text = self._clean_html_text(raw_metadata.get("preview_text", ""))
        detail_text = self._clean_html_text(raw_metadata.get("detail_text", ""))

        # Skip if detail_text is empty after cleaning
        if not detail_text:
            logger.debug(f"Skipping video with empty detail_text: {title}")
            return {}

        # Combine preview and detail text for description
        description_parts = []
        if preview_text:
            description_parts.append(preview_text)
        if detail_text:
            description_parts.append(detail_text)

        description = " ".join(description_parts) if description_parts else None

        # Parse date
        web_date = None
        web_date_ts = None
        if date_active_start:
            try:
                # Parse ISO format with Z suffix
                dt = datetime.fromisoformat(date_active_start.replace("Z", "+00:00"))
                web_date = dt.strftime("%Y-%m-%d")
                web_date_ts = dt.timestamp()
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not parse web date {date_active_start}: {e}")

        return {
            "web_title": title if title else None,
            "web_date": web_date,
            "web_date_ts": web_date_ts,
            "web_description": description,
            "web_url": url if url else None,
        }

    @staticmethod
    def _clean_html_text(text: str | None) -> str:
        """Strip HTML tags and normalize whitespace in metadata text."""
        if not text:
            return ""
        cleaned = html.unescape(text)
        cleaned = cleaned.replace("\u00a0", " ")
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def list_video_hashes(self) -> list[str]:
        """
        List video hashes that have web metadata available.

        Returns:
            List of video hash strings (derived from metadata filenames).
        """
        if not self.metadata_path.exists():
            return []

        hashes: list[str] = []
        for metadata_file in self.metadata_path.glob("*.json"):
            if metadata_file.is_file():
                hashes.append(metadata_file.stem)
        return hashes

    @staticmethod
    def hash_to_archive_dir(video_hash: str) -> Path:
        """
        Convert a video hash to its archive directory path (relative).

        Example:
            0c14229efcba436a4c22f8d96f67e0cb93bc9076
            -> 0c/14/22/9e/fc/ba/43/6a/4c/22/f8/d9/6f/67/e0/cb/93/bc/90/76

        Args:
            video_hash: Video hash string

        Returns:
            Relative Path object with split hash segments
        """
        # Validate input
        video_hash = video_hash.strip()
        if not video_hash:
            raise ValueError("video_hash cannot be empty")

        if len(video_hash) % 2 != 0:
            raise ValueError(f"video_hash must have even length, got {len(video_hash)}")

        # Check for valid hex characters
        if not all(c in "0123456789abcdefABCDEF" for c in video_hash):
            raise ValueError(
                "video_hash contains invalid characters, must be hexadecimal (0-9, a-f, A-F)"
            )

        parts = [video_hash[i : i + 2] for i in range(0, len(video_hash), 2)]
        return Path(*parts)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of document text for change detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ManifestEntry(BaseModel):
    """Entry in the file manifest tracking source file state."""

    mtime: float
    size: int
    file_hash: str  # SHA-256 of file content
    doc_ids: list[str] = []  # Document IDs produced from this file


class FileManifest:
    """Tracks source file metadata for incremental ingestion.

    The manifest records each source VTT file's mtime, size, and content hash.
    On subsequent runs, files are classified as new, modified, or unchanged
    by comparing their current state against the manifest.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.entries: dict[str, ManifestEntry] = {}

    def load(self) -> bool:
        """Load manifest from disk. Returns True if loaded successfully."""
        if not self.manifest_path.exists():
            logger.info("No existing manifest found, will do full ingestion")
            return False
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                raw = json.load(f)
            self.entries = {k: ManifestEntry(**v) for k, v in raw.items()}
            logger.info(f"Loaded manifest with {len(self.entries)} file entries")
            return True
        except Exception as e:
            logger.warning(f"Failed to load manifest, will do full ingestion: {e}")
            self.entries = {}
            return False

    def save(self) -> None:
        """Save manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump() for k, v in self.entries.items()}
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved manifest with {len(self.entries)} file entries")

    def classify_file(self, file_path: Path) -> str:
        """Classify a file as 'new', 'modified', or 'unchanged'.

        Uses mtime + size as a fast check. Falls back to content hash for
        borderline cases (same size, different mtime).
        """
        key = str(file_path.absolute())
        if key not in self.entries:
            return "new"

        entry = self.entries[key]
        try:
            stat = file_path.stat()
        except OSError:
            return "new"

        # Fast check: if mtime and size match, file is unchanged
        if stat.st_mtime == entry.mtime and stat.st_size == entry.size:
            return "unchanged"

        # Size changed -> definitely modified
        if stat.st_size != entry.size:
            return "modified"

        # Same size, different mtime -> check content hash
        current_hash = self._compute_file_hash(file_path)
        if current_hash == entry.file_hash:
            return "unchanged"

        return "modified"

    def update_entry(self, file_path: Path, doc_ids: list[str]) -> None:
        """Update manifest entry for a file after processing."""
        key = str(file_path.absolute())
        stat = file_path.stat()
        self.entries[key] = ManifestEntry(
            mtime=stat.st_mtime,
            size=stat.st_size,
            file_hash=self._compute_file_hash(file_path),
            doc_ids=doc_ids,
        )

    def get_deleted_files(self, current_files: set[str]) -> list[str]:
        """Return manifest keys for files that no longer exist."""
        return [k for k in self.entries if k not in current_files]

    def get_deleted_doc_ids(self, deleted_file_keys: list[str]) -> list[str]:
        """Return document IDs associated with deleted files."""
        doc_ids = []
        for key in deleted_file_keys:
            if key in self.entries:
                doc_ids.extend(self.entries[key].doc_ids)
        return doc_ids

    def remove_entries(self, keys: list[str]) -> None:
        """Remove entries for deleted files."""
        for key in keys:
            self.entries.pop(key, None)

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file's content."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class Ingester:
    """Main ingestion class for crawling and parsing VTT files."""

    def __init__(self, config: Config):
        """
        Initialize the ingester.

        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        self.parser = VTTParser()
        self.invalid_vtt_count = 0
        self.no_cues_count = 0

        # Initialize web metadata loader with validation
        if config.web_metadata.enabled:
            web_metadata_path = Path(config.web_metadata.path)
            if not web_metadata_path.exists():
                logger.warning(f"Web metadata path does not exist: {web_metadata_path}")
                self.web_metadata_loader = None
            elif not web_metadata_path.is_dir():
                logger.warning(f"Web metadata path is not a directory: {web_metadata_path}")
                self.web_metadata_loader = None
            else:
                self.web_metadata_loader = WebMetadataLoader(web_metadata_path)
        else:
            self.web_metadata_loader = None

    def find_vtt_files(self, root_path: Path) -> Generator[Path, None, None]:
        """
        Recursively find all .vtt files in a directory.

        Args:
            root_path: Root directory to search

        Yields:
            Path objects for VTT files
        """
        logger.info(f"Scanning for .vtt files in {root_path}")

        vtt_files: list[Path] = []

        if self.config.web_metadata.require_web_metadata:
            if not self.web_metadata_loader:
                logger.warning("require_web_metadata is True but web_metadata not available")
                vtt_files = []
            else:
                video_hashes = self.web_metadata_loader.list_video_hashes()
                logger.info(f"Restricting scan to {len(video_hashes)} web_metadata hashes")

                for video_hash in video_hashes:
                    relative_dir = self.web_metadata_loader.hash_to_archive_dir(video_hash)
                    target_dir = root_path / relative_dir
                    if not target_dir.exists():
                        continue
                    vtt_files.extend(target_dir.rglob("*.vtt"))
        else:
            # Parallel scanning for better performance
            import os

            def scan_subdirectory(subdir: Path) -> list[Path]:
                vtt_files = []
                for root, _dirs, files in os.walk(subdir):
                    for file in files:
                        if file.endswith(".vtt"):
                            vtt_files.append(Path(root) / file)
                return vtt_files

            subdirs = [d for d in root_path.iterdir() if d.is_dir()]
            vtt_files = []

            if subdirs:
                logger.info(f"Scanning {len(subdirs)} subdirectories in parallel...")
                with ThreadPoolExecutor(max_workers=min(16, len(subdirs))) as executor:
                    results = executor.map(scan_subdirectory, subdirs)
                    for i, result in enumerate(results):
                        vtt_files.extend(result)
                        if (i + 1) % 10 == 0:
                            logger.info(
                                f"Processed {i + 1}/{len(subdirs)} subdirectories, total files: {len(vtt_files)}"
                            )
                        if len(vtt_files) % 10000 == 0 and len(vtt_files) > 0:
                            logger.info(f"Scanned {len(vtt_files)} .vtt files so far...")
            else:
                # Fallback for flat structure
                vtt_files = list(root_path.rglob("*.vtt"))

        logger.info(f"Found {len(vtt_files)} .vtt files")

        for vtt_file in vtt_files:
            # Check file size limit
            if vtt_file.stat().st_size > self.config.processing.max_file_size:
                logger.warning(
                    f"Skipping {vtt_file} (size {vtt_file.stat().st_size} exceeds limit)"
                )
                continue

            yield vtt_file

    def process_file(self, file_path: Path) -> list[Document]:
        """
        Process a single VTT file, optionally chunking it.

        Args:
            file_path: Path to the VTT file

        Returns:
            List of Document objects (may contain multiple chunks or single full document)
        """
        # Detect language
        language = self.parser.detect_language(file_path)

        # Generate video ID (shared across languages/chunks)
        video_id = self.parser.generate_video_id(file_path)

        # Extract metadata from source video
        date_iso = None
        date_ts = None
        duration = None
        source_video = self.parser.find_source_video(file_path)
        if source_video:
            date_iso, duration, date_ts = self.parser.get_video_metadata(source_video)

        # Load web metadata if enabled
        web_metadata = {}
        if self.web_metadata_loader:
            # Extract video hash from filename
            video_hash = VTTParser.extract_video_hash(file_path)
            raw_web_data = self.web_metadata_loader.load_metadata(video_hash)
            if raw_web_data:
                web_metadata = self.web_metadata_loader.extract_clean_metadata(raw_web_data)
                # Skip this video if web metadata indicates empty content
                if not web_metadata:
                    logger.debug(f"Skipping {file_path} due to empty web metadata content")
                    return []
            elif self.config.web_metadata.require_web_metadata:
                # Skip this video if web metadata is required but not found
                logger.debug(f"Skipping {file_path} because web metadata is required but not found")
                return []

        # Prioritize web metadata dates over video mtime
        final_date = (
            web_metadata.get("web_date") if web_metadata.get("web_date") is not None else date_iso
        )
        final_date_ts = (
            web_metadata.get("web_date_ts")
            if web_metadata.get("web_date_ts") is not None
            else date_ts
        )

        # Check if chunking is enabled
        if self.config.chunking.enabled:
            # Parse VTT into cues
            cues = self.parser.parse_vtt_to_cues(file_path)

            if cues is None:
                self.invalid_vtt_count += 1
                if self.invalid_vtt_count % 1000 == 0:
                    logger.warning(
                        "Invalid VTT files so far: "
                        + f"{self.invalid_vtt_count} (latest: {file_path})"
                    )
                else:
                    logger.debug(f"Invalid VTT file: {file_path}")
                return []

            if not cues:
                self.no_cues_count += 1
                if self.no_cues_count % 1000 == 0:
                    logger.warning(
                        "VTT files with no cues so far: "
                        + f"{self.no_cues_count} (latest: {file_path})"
                    )
                else:
                    logger.debug(f"No cues extracted from {file_path}")
                return []

            # Get max tokens based on embedding model
            max_tokens = self.config.get_max_chunk_tokens()
            adjusted_max_tokens = max_tokens
            metadata_block = self._build_web_metadata_block(web_metadata)
            if (
                self.config.web_metadata.include_in_text
                and self.config.web_metadata.append_to_each_chunk
                and metadata_block
            ):
                metadata_tokens = VTTParser.estimate_tokens(metadata_block, language)
                adjusted_max_tokens = max(1, max_tokens - metadata_tokens)
                if adjusted_max_tokens < max_tokens:
                    logger.debug(
                        "Reserving token budget for metadata: "
                        + f"max_tokens={max_tokens}, metadata_tokens={metadata_tokens}, "
                        + f"adjusted_max_tokens={adjusted_max_tokens}"
                    )

            # Create chunks based on strategy
            if self.config.chunking.strategy == "hybrid":
                # Hybrid: time-based with token validation (RECOMMENDED)
                chunks = self.parser.create_chunks_hybrid(
                    cues,
                    chunk_duration_seconds=self.config.chunking.chunk_duration_seconds,
                    overlap_seconds=self.config.chunking.overlap_seconds,
                    max_tokens=adjusted_max_tokens,
                    min_tokens=self.config.chunking.min_chunk_tokens,
                    language=language,
                )
            elif self.config.chunking.strategy == "time":
                # Time-based only (may exceed token limits!)
                chunks = self.parser.create_chunks_from_cues(
                    cues,
                    chunk_duration_seconds=self.config.chunking.chunk_duration_seconds,
                    overlap_seconds=self.config.chunking.overlap_seconds,
                )
            elif self.config.chunking.strategy == "token":
                # Pure token-based chunking (disable overlap for token-only)
                chunks = self.parser.create_chunks_hybrid(
                    cues,
                    chunk_duration_seconds=self.parser.INFINITE_CHUNK_DURATION,  # Effectively disable time chunking
                    overlap_seconds=0,  # No overlap for pure token strategy
                    max_tokens=adjusted_max_tokens,
                    min_tokens=self.config.chunking.min_chunk_tokens,
                    language=language,
                )
            else:
                raise ValueError(
                    f"Unknown chunking strategy: '{self.config.chunking.strategy}'. Valid strategies are: 'hybrid', 'time', 'token'"
                )

            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return []

            logger.debug(
                f"Created {len(chunks)} chunks from {file_path} using '{self.config.chunking.strategy}' strategy (max_tokens={adjusted_max_tokens})"
            )

            # Create Document objects for each chunk
            documents: list[Document] = []
            total_chunks = len(chunks)

            for chunk in chunks:
                # Chunks are already validated by the chunking strategy
                # No need for additional length checks
                text = self._append_web_metadata(
                    chunk.text,
                    web_metadata,
                    max_tokens=max_tokens,
                    language=language,
                    metadata_block=metadata_block,
                )
                doc_id = self.parser.generate_id(file_path, chunk.chunk_index)

                doc = Document(
                    id=doc_id,
                    path=str(file_path.absolute()),
                    language=language,
                    text=text,
                    length=len(text),
                    date=final_date,
                    date_ts=final_date_ts,
                    duration_seconds=duration,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    video_id=video_id,
                    is_chunk=True,
                    chunk_index=chunk.chunk_index,
                    total_chunks=total_chunks,
                    start_time_seconds=chunk.start_seconds,
                    end_time_seconds=chunk.end_seconds,
                    web_title=web_metadata.get("web_title"),
                    web_date=web_metadata.get("web_date"),
                    web_date_ts=web_metadata.get("web_date_ts"),
                    web_description=web_metadata.get("web_description"),
                    web_url=web_metadata.get("web_url"),
                    content_hash=compute_content_hash(text),
                )
                documents.append(doc)

            logger.debug(f"Created {len(documents)} chunks from {file_path}")
            return documents

        else:
            # Chunking disabled - process as single document (legacy behavior)
            result = self.parser.parse_vtt_with_timecodes(file_path)

            if not result:
                logger.warning(f"No text extracted from {file_path}")
                return []

            text, start_time, end_time = result
            text = self._append_web_metadata(text, web_metadata)

            # Check minimum text length
            if len(text) < self.config.processing.min_text_length:
                logger.debug(f"Skipping {file_path} (text too short: {len(text)} chars)")
                return []

            # Generate document ID
            doc_id = self.parser.generate_id(file_path)

            # Calculate start/end in seconds
            start_seconds = self.parser.timestamp_to_seconds(start_time) if start_time else None
            end_seconds = self.parser.timestamp_to_seconds(end_time) if end_time else None

            # Create document
            doc = Document(
                id=doc_id,
                path=str(file_path.absolute()),
                language=language,
                text=text,
                length=len(text),
                date=final_date,
                date_ts=final_date_ts,
                duration_seconds=duration,
                start_time=start_time,
                end_time=end_time,
                video_id=video_id,
                is_chunk=False,
                start_time_seconds=start_seconds,
                end_time_seconds=end_seconds,
                web_title=web_metadata.get("web_title"),
                web_date=web_metadata.get("web_date"),
                web_date_ts=web_metadata.get("web_date_ts"),
                web_description=web_metadata.get("web_description"),
                web_url=web_metadata.get("web_url"),
                content_hash=compute_content_hash(text),
            )

            return [doc]

    def _append_web_metadata(
        self,
        text: str,
        web_metadata: dict[str, Any],
        max_tokens: int | None = None,
        language: str = "en",
        metadata_block: str | None = None,
    ) -> str:
        """Append web metadata to the document text when configured."""
        if not self.config.web_metadata.include_in_text:
            return text

        if not web_metadata:
            return text

        if not self.config.web_metadata.append_to_each_chunk:
            return text

        block = metadata_block or self._build_web_metadata_block(web_metadata)
        if not block:
            return text

        combined = f"{text}\n\n{block}"
        if max_tokens is not None:
            combined_tokens = VTTParser.estimate_tokens(combined, language)
            if combined_tokens > max_tokens:
                logger.debug(
                    "Metadata append would exceed max_tokens; "
                    + f"dropping metadata (tokens {combined_tokens} > {max_tokens})."
                )
                return text
        return combined

    def _build_web_metadata_block(self, web_metadata: dict[str, Any]) -> str | None:
        if not web_metadata:
            return None

        field_map = {
            "title": ("Title", web_metadata.get("web_title")),
            "date": ("Date", web_metadata.get("web_date")),
            "description": ("Description", web_metadata.get("web_description")),
            "url": ("URL", web_metadata.get("web_url")),
        }

        unknowns = [field for field in self.config.web_metadata.fields if field not in field_map]
        if unknowns:
            logger.warning(
                f"Unrecognized web metadata fields: {unknowns}. Accepted fields: {list(field_map.keys())}"
            )

        lines = []
        for field in self.config.web_metadata.fields:
            label, value = field_map.get(field, (None, None))
            if label and value:
                lines.append(f"{label}: {value}")

        if not lines:
            return None

        label = self.config.web_metadata.append_label
        return "\n".join([label, *lines])

    def ingest(self, incremental: bool = False) -> int:
        """
        Run the ingestion pipeline.

        Args:
            incremental: If True, only re-process changed files using the manifest.

        Returns:
            Number of documents processed
        """
        if incremental and self.config.incremental.enabled:
            return self._ingest_incremental()
        return self._ingest_full()

    def _ingest_full(self) -> int:
        """Run the full ingestion pipeline (processes all files)."""
        root_path = Path(self.config.paths.archive_root)

        if not root_path.exists():
            logger.error(f"Archive root does not exist: {root_path}")
            raise FileNotFoundError(f"Archive root not found: {root_path}")

        output_path = Path(self.config.paths.docs_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting full ingestion from {root_path}")
        logger.info(f"Output will be written to {output_path}")

        doc_count = 0
        file_count = 0

        # Build manifest for future incremental runs
        manifest = FileManifest(self.config.incremental.manifest_path)

        # Open output file for streaming writes
        with open(output_path, "w", encoding="utf-8") as out_file:
            # Process all VTT files
            vtt_files = list(self.find_vtt_files(root_path))

            for file_path in tqdm(vtt_files, desc="Processing VTT files"):
                docs = self.process_file(file_path)

                if docs:
                    file_count += 1
                    doc_ids = []
                    # Write each document as JSONL (one JSON object per line)
                    for doc in docs:
                        json_line = doc.model_dump_json()
                        out_file.write(json_line + "\n")
                        doc_count += 1
                        doc_ids.append(doc.id)

                    # Update manifest entry for this file
                    manifest.update_entry(file_path, doc_ids)

                    if doc_count % 100 == 0:
                        logger.info(f"Processed {doc_count} documents from {file_count} files")

        chunking_status = "enabled" if self.config.chunking.enabled else "disabled"
        logger.info(
            f"Ingestion complete! Processed {file_count} files into {doc_count} documents (chunking: {chunking_status})"
        )
        if self.invalid_vtt_count or self.no_cues_count:
            logger.info(
                f"Ingestion summary: invalid_vtt={self.invalid_vtt_count}, no_cues={self.no_cues_count}"
            )
        logger.info(f"Output saved to {output_path}")

        # Save manifest for future incremental runs
        manifest.save()

        return doc_count

    def _ingest_incremental(self) -> int:
        """Run incremental ingestion — only re-process changed files."""
        root_path = Path(self.config.paths.archive_root)

        if not root_path.exists():
            logger.error(f"Archive root does not exist: {root_path}")
            raise FileNotFoundError(f"Archive root not found: {root_path}")

        output_path = Path(self.config.paths.docs_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load manifest
        manifest = FileManifest(self.config.incremental.manifest_path)
        if not manifest.load():
            logger.info("No manifest found, falling back to full ingestion")
            return self._ingest_full()

        # Load previous docs into a lookup by doc_id
        previous_docs: dict[str, Document] = {}
        if output_path.exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            doc = Document(**json.loads(line))
                            previous_docs[doc.id] = doc
                logger.info(f"Loaded {len(previous_docs)} previous documents")
            except Exception as e:
                logger.warning(f"Failed to load previous docs, doing full ingestion: {e}")
                return self._ingest_full()

        logger.info(f"Starting incremental ingestion from {root_path}")

        # Scan current files
        vtt_files = list(self.find_vtt_files(root_path))
        current_file_keys = {str(f.absolute()) for f in vtt_files}

        # Classify files
        new_files: list[Path] = []
        modified_files: list[Path] = []
        unchanged_files: list[Path] = []

        for file_path in vtt_files:
            state = manifest.classify_file(file_path)
            if state == "new":
                new_files.append(file_path)
            elif state == "modified":
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)

        # Detect deleted files
        deleted_keys = manifest.get_deleted_files(current_file_keys)
        deleted_doc_ids = set(manifest.get_deleted_doc_ids(deleted_keys))

        logger.info(
            f"Incremental scan: {len(new_files)} new, {len(modified_files)} modified, "
            f"{len(unchanged_files)} unchanged, {len(deleted_keys)} deleted files"
        )

        # Collect doc_ids that need to be replaced (from modified files)
        modified_old_doc_ids: set[str] = set()
        for file_path in modified_files:
            key = str(file_path.absolute())
            if key in manifest.entries:
                modified_old_doc_ids.update(manifest.entries[key].doc_ids)

        # Write output: unchanged docs + newly processed docs
        doc_count = 0
        file_count = 0
        files_to_process = new_files + modified_files

        with open(output_path, "w", encoding="utf-8") as out_file:
            # Write unchanged documents (skip deleted and modified-old)
            skip_ids = deleted_doc_ids | modified_old_doc_ids
            for doc_id, doc in previous_docs.items():
                if doc_id not in skip_ids:
                    out_file.write(doc.model_dump_json() + "\n")
                    doc_count += 1

            unchanged_count = doc_count
            logger.info(f"Kept {unchanged_count} unchanged documents")

            # Process new and modified files
            for file_path in tqdm(files_to_process, desc="Processing changed files"):
                docs = self.process_file(file_path)

                if docs:
                    file_count += 1
                    doc_ids = []
                    for doc in docs:
                        out_file.write(doc.model_dump_json() + "\n")
                        doc_count += 1
                        doc_ids.append(doc.id)
                    manifest.update_entry(file_path, doc_ids)

        # Remove deleted file entries from manifest
        manifest.remove_entries(deleted_keys)

        # Save updated manifest
        manifest.save()

        logger.info(
            f"Incremental ingestion complete! "
            f"{unchanged_count} unchanged + {doc_count - unchanged_count} new/modified documents = {doc_count} total"
        )
        if deleted_doc_ids:
            logger.info(f"Removed {len(deleted_doc_ids)} documents from {len(deleted_keys)} deleted files")

        return doc_count


def run_ingestion(config_path: str = "config.yaml", incremental: bool = False) -> int:
    """
    Run the ingestion pipeline.

    Args:
        config_path: Path to configuration file
        incremental: If True, only re-process changed files

    Returns:
        Number of documents processed
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    ingester = Ingester(config)
    return ingester.ingest(incremental=incremental)
