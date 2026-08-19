"""Ingestion module for parsing VTT subtitle files."""

import hashlib
import html
import json
import os
import re
import subprocess
import time
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from rainrag.config import Config


# The library CMS returns tag categories as English slugs. `theme`/`person`/
# `location` are the taxonomy proper; `lite` is the Дождь Lite section rubric
# (only 9 distinct values, yet ~73% of all tag volume), so it is dropped —
# keeping it would swamp the real tags in every list it appears in.
TAXONOMY_TAG_CATEGORIES = ("theme", "person", "location")

# Every web-metadata key carried on a Qdrant payload. Shared so the indexer,
# the query engine and the API cannot drift apart as fields are added.
WEB_METADATA_PAYLOAD_FIELDS = (
    "web_title",
    "web_date",
    "web_date_ts",
    "web_description",
    "web_url",
    "web_program",
    "web_presenters",
    "web_tags",
    "web_tags_theme",
    "web_tags_person",
    "web_tags_location",
    "web_tag_ids",
    "web_stories",
)


def document_web_fields(web_metadata: dict[str, Any]) -> dict[str, Any]:
    """Map cleaned web metadata onto ``Document`` keyword arguments.

    Kept in one place because three call sites build Documents (chunked,
    unchunked and speech-free) and they must stay in step.
    """
    fields: dict[str, Any] = {
        key: web_metadata.get(key)
        for key in ("web_title", "web_date", "web_date_ts", "web_description", "web_url")
    }
    fields["web_program"] = web_metadata.get("web_program")
    for key in (
        "web_presenters",
        "web_tags",
        "web_tags_theme",
        "web_tags_person",
        "web_tags_location",
        "web_tag_ids",
        "web_stories",
    ):
        fields[key] = web_metadata.get(key) or []
    return fields


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

    # Library CMS taxonomy. Coverage is uneven by era: program/presenters are
    # populated throughout, tags only through 2025-09, stories only through 2022.
    # Treat every one of these as optional at query time.
    web_program: str | None = None  # Programme / cycle name (CMS `teleshow`)
    web_presenters: list[str] = Field(default_factory=list)  # Hosts (CMS `presentors`)
    web_tags: list[str] = Field(default_factory=list)  # All taxonomy tag names
    web_tags_theme: list[str] = Field(default_factory=list)  # category=theme
    web_tags_person: list[str] = Field(default_factory=list)  # category=person (incl. orgs)
    web_tags_location: list[str] = Field(default_factory=list)  # category=location
    # Stable CMS tag ids, for joining to the master tag list. Not positionally
    # aligned with web_tags: names are deduped, ids are not, and the CMS files
    # some tags (e.g. Украина) under both theme and location with distinct ids.
    web_tag_ids: list[int] = Field(default_factory=list)
    web_stories: list[str] = Field(default_factory=list)  # Event arcs (CMS `stories`)

    # Content hash for incremental processing
    content_hash: str | None = None  # SHA-256 hash of embedded text for change detection

    # Speech-free flag
    is_speech_free: bool = False  # True when VTT had no cues (silent video); text is metadata-only


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

        Looks for 'ru', 'rus', 'russian' or 'en', 'eng', 'english' in the filename
        first, then in the immediate parent directory. Ancestor directories are
        deliberately ignored: an unrelated segment (a temp dir named 'tmpab_ru_9x',
        a mount point, a user's home) would otherwise decide the language of every
        file beneath it.

        Args:
            file_path: Path to the VTT file

        Returns:
            Language code ('ru' or 'en')
        """
        path = Path(file_path)
        for candidate in (path.name, path.parent.name):
            tokens = set(filter(None, re.split(r"[^a-z0-9]+", candidate.lower())))

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

            # Valid VTT but no speech cues (silent/music-only video)
            if not cues:
                return ("", None, None)

            # Extract text and timecodes from cues
            text_parts = [cue.text for cue in cues if cue.text]
            full_text = " ".join(text_parts)

            # Get first and last timecodes
            start_time = cues[0].start_time
            end_time = cues[-1].end_time

            return (full_text, start_time, end_time)

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
    def _parse_vtt_lines_to_cues(cls, lines: Sequence[str]) -> list[VTTCue] | None:
        """Parse VTT file lines into cues.

        _parse_vtt_lines_to_cues takes `lines: Sequence[str]` from a VTT file and
        returns a list of `VTTCue` objects when parsing succeeds.
        The returned list may be empty for valid speech-free VTT (header-only
        content with no cues). Returns `None` only when parsing fails, e.g.
        missing `WEBVTT` header or malformed input.

        Args:
            lines: Sequence of lines from a VTT file

        Returns:
            list[VTTCue]: parsed cues (possibly empty for speech-free VTT)
            None: parsing failure
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

        return cues

    @classmethod
    def parse_vtt_to_cues(cls, file_path: Path) -> list[VTTCue] | None:
        """
        Parse a VTT file into individual cues with timestamps.

        This method wraps `_parse_vtt_lines_to_cues` and returns a list of
        `VTTCue` objects when the file is valid. For a speech-free but
        otherwise valid VTT file, the returned list may be empty. Returns
        `None` only when parsing fails (e.g. missing `WEBVTT` header or other
        invalid format issue).

        Args:
            file_path: Path to the VTT file

        Returns:
            list[VTTCue]: parsed cues (possibly empty for valid no-cue files)
            None: parsing failure

        Note:
            Callers should handle the empty-list case as valid no-speech content,
            and `None` as parse failure.
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
    """Loader for web metadata JSON files, with optional API backend.

    Supports three *source* modes controlled by
    :pyattr:`rainrag.config.WebMetadataConfig.source`:

    ``local``
        Read only from the on-disk ``metadata_path`` directory (default,
        backwards-compatible).
    ``api``
        Fetch every hash from the remote API.  Results are cached locally
        under ``metadata_path`` so subsequent lookups are free.
    ``hybrid``
        Try the local file first; if missing, fall back to the API and
        cache the result.
    """

    def __init__(
        self,
        metadata_path: Path,
        *,
        source: str = "local",
        api_client: Any | None = None,
    ) -> None:
        """
        Initialize the web metadata loader.

        Args:
            metadata_path: Path to directory containing web metadata JSON files
            source: One of ``"local"``, ``"api"``, ``"hybrid"``.
            api_client: Optional :class:`WebMetadataAPIClient` instance
                        (required when *source* is ``"api"`` or ``"hybrid"``).
        """
        super().__init__()
        allowed_sources = {"local", "api", "hybrid"}
        source_normalized = source.lower().strip()
        if source_normalized not in allowed_sources:
            raise ValueError(
                f"Invalid web metadata loader source {source!r}; "
                + f"expected one of {sorted(allowed_sources)}."
            )
        self.metadata_path = metadata_path
        self.source = source_normalized
        self.api_client = api_client
        self._api_batch_cache_ttl_seconds = int(
            os.getenv("RAINRAG_WEB_METADATA_BATCH_TTL_SECONDS", "900")
        )
        self._api_batch_cache_mtime: float | None = None
        self._api_batch_cache: list[dict[str, Any]] = []

    @staticmethod
    def _safe_metadata_hash(video_hash: str) -> str | None:
        """Return safe hash for metadata filename, or None when invalid."""
        candidate = video_hash.strip()
        if not candidate:
            return None
        if Path(candidate).name != candidate:
            return None
        if not re.fullmatch(r"[a-fA-F0-9]{40}", candidate):
            return None
        return candidate.lower()

    def _load_api_batch_candidates(self) -> list[dict[str, Any]]:
        """Load API batch with in-process TTL cache and persist safe files."""
        if self.api_client is None:
            return []

        now = time.time()
        if (
            self._api_batch_cache
            and self._api_batch_cache_mtime is not None
            and (now - self._api_batch_cache_mtime) < self._api_batch_cache_ttl_seconds
        ):
            return self._api_batch_cache

        api_candidates = self.api_client.export_batch()
        self._api_batch_cache = api_candidates
        self._api_batch_cache_mtime = now

        self.metadata_path.mkdir(parents=True, exist_ok=True)
        for article in api_candidates:
            safe_hash = self._safe_metadata_hash(str(article.get("video_hash", "")))
            if not safe_hash:
                continue
            try:
                (self.metadata_path / f"{safe_hash}.json").write_text(
                    json.dumps(article, ensure_ascii=False, indent=None), encoding="utf-8"
                )
            except OSError:
                continue
        return api_candidates

    # ------------------------------------------------------------------
    # Local helpers
    # ------------------------------------------------------------------

    def _load_local(self, video_hash: str) -> dict[str, Any] | None:
        """Read a single ``{hash}.json`` from the local directory."""
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

    def _fetch_api(self, video_hash: str) -> dict[str, Any] | None:
        """Fetch metadata from the remote API and cache locally."""
        if self.api_client is None:
            return None
        try:
            data = self.api_client.fetch_by_hash(video_hash)
        except Exception as exc:
            logger.warning(f"API fetch failed for {video_hash}: {exc}")
            return None
        if data is None:
            return None
        # Cache to disk so future runs pick it up locally
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        cache_file = self.metadata_path / f"{video_hash}.json"
        try:
            cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=None), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning(f"Failed to cache metadata for {video_hash}: {exc}")
        return data

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_metadata(self, video_hash: str) -> dict[str, Any] | None:
        """
        Load web metadata for a video by its hash.

        The lookup strategy depends on ``self.source``:

        * ``local`` -- disk only
        * ``api`` -- API only (result cached)
        * ``hybrid`` -- disk first, then API fallback

        Args:
            video_hash: Video hash (filename without extension)

        Returns:
            Dictionary containing web metadata, or None if not found
        """
        if self.source == "local":
            # No API to refetch from; Ingester warns up front if the cache is stale.
            return self._load_local(video_hash)

        # Both API modes prefer the local cache to avoid redundant calls, but a
        # file cached before the API exposed the taxonomy would silently index
        # without tags, so refetch those.
        local = self._load_local(video_hash)
        if local is not None and not self._predates_taxonomy(local):
            return local

        fresh = self._fetch_api(video_hash)
        if fresh is not None:
            return fresh
        # The refetch failed; stale metadata still beats none at all.
        return local

    @staticmethod
    def _predates_taxonomy(data: dict[str, Any]) -> bool:
        """True for cache files written before the API exposed the taxonomy.

        The library API always emits a ``tags`` key now — empty for the many
        untagged articles, but present — so a missing key means the file was
        cached by an older build and cannot be trusted to be complete.
        """
        return "tags" not in data

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

        # Combine preview and detail text for description.
        # detail_text may be empty for speech-free videos that only carry a title;
        # that is still useful metadata so we do not skip here.
        description_parts = []
        if preview_text:
            description_parts.append(preview_text)
        if detail_text:
            description_parts.append(detail_text)

        description = " ".join(description_parts) if description_parts else None

        # Skip only when there is genuinely nothing to index
        if not title and not description:
            logger.debug(
                f"Skipping video with no usable metadata content: title={title!r} description={description!r}"
            )
            return {}

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

        tags_by_category = self._extract_tags(raw_metadata.get("tags"))

        return {
            "web_title": title if title else None,
            "web_date": web_date,
            "web_date_ts": web_date_ts,
            "web_description": description,
            "web_url": url if url else None,
            "web_program": self._extract_program(raw_metadata.get("teleshow")),
            "web_presenters": self._extract_people(raw_metadata.get("presentors")),
            "web_tags": self._dedupe(
                name for category in TAXONOMY_TAG_CATEGORIES for name in tags_by_category[category]
            ),
            "web_tags_theme": tags_by_category["theme"],
            "web_tags_person": tags_by_category["person"],
            "web_tags_location": tags_by_category["location"],
            "web_tag_ids": self._extract_tag_ids(raw_metadata.get("tags")),
            "web_stories": self._extract_names(raw_metadata.get("stories")),
        }

    @staticmethod
    def _dedupe(names: Any) -> list[str]:
        """Collapse duplicates while preserving first-seen order."""
        return list(dict.fromkeys(names))

    @staticmethod
    def _extract_program(teleshow: Any) -> str | None:
        """Pull the programme name out of the CMS `teleshow` object."""
        if not isinstance(teleshow, dict):
            return None
        name = str(teleshow.get("name") or "").strip()
        return name or None

    @classmethod
    def _extract_people(cls, people: Any) -> list[str]:
        """Render CMS person objects as "Firstname Lastname".

        The article endpoint splits names into ``firstname``/``lastname`` while
        the site embeds a single ``name``; accept either.
        """
        if not isinstance(people, list):
            return []
        rendered: list[str] = []
        for person in people:
            if isinstance(person, str):
                name = person.strip()
            elif isinstance(person, dict):
                name = str(person.get("name") or "").strip() or " ".join(
                    part
                    for key in ("firstname", "lastname")
                    if (part := str(person.get(key) or "").strip())
                )
            else:
                continue
            if name:
                rendered.append(name)
        return cls._dedupe(rendered)

    @classmethod
    def _extract_names(cls, items: Any) -> list[str]:
        """Pull `name` out of a list of CMS objects (tolerating bare strings)."""
        if not isinstance(items, list):
            return []
        names: list[str] = []
        for item in items:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                continue
            if name:
                names.append(name)
        return cls._dedupe(names)

    @classmethod
    def _extract_tags(cls, tags: Any) -> dict[str, list[str]]:
        """Group tag names by taxonomy category, dropping the `lite` rubric."""
        grouped: dict[str, list[str]] = {category: [] for category in TAXONOMY_TAG_CATEGORIES}
        if not isinstance(tags, list):
            return grouped
        for tag in tags:
            if not isinstance(tag, dict):
                continue
            category = str(tag.get("category") or "").strip().lower()
            if category not in grouped:
                continue
            name = str(tag.get("name") or "").strip()
            if name:
                grouped[category].append(name)
        return {category: cls._dedupe(names) for category, names in grouped.items()}

    @staticmethod
    def _extract_tag_ids(tags: Any) -> list[int]:
        """Collect stable CMS tag ids for taxonomy tags, for joins to the master list."""
        if not isinstance(tags, list):
            return []
        ids: list[int] = []
        for tag in tags:
            if not isinstance(tag, dict):
                continue
            if str(tag.get("category") or "").strip().lower() not in TAXONOMY_TAG_CATEGORIES:
                continue
            try:
                tag_id = int(tag["id"])
            except (KeyError, TypeError, ValueError):
                continue
            ids.append(tag_id)
        return list(dict.fromkeys(ids))

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

    def search_by_name(self, query: str) -> list[dict[str, Any]]:
        """Search metadata by title (case-insensitive substring match).

        Source behavior:
        - ``local``: search local ``{hash}.json`` files
        - ``api``: search recent API batch export (and cache to disk)
        - ``hybrid``: merge local + recent API results
        """
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        # Collect candidates from local cache
        local_candidates: list[dict[str, Any]] = []
        if self.metadata_path.exists():
            for metadata_file in self.metadata_path.glob("*.json"):
                if not metadata_file.is_file():
                    continue
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                local_candidates.append(data)

        # Optionally collect candidates from API export
        api_candidates: list[dict[str, Any]] = []
        if self.source in {"api", "hybrid"} and self.api_client is not None:
            try:
                api_candidates = self._load_api_batch_candidates()
            except Exception as exc:
                logger.warning(f"API batch export failed during name search: {exc}")

        if self.source == "local":
            candidates = local_candidates
        elif self.source == "api":
            candidates = api_candidates
        else:
            # hybrid: local first, then API; de-duplicate by hash/url
            seen: set[str] = set()
            candidates = []
            for item in local_candidates + api_candidates:
                key = str(item.get("video_hash") or item.get("url") or "")
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(item)

        results = [item for item in candidates if query_lower in item.get("name", "").lower()]

        # Stable two-pass sort: date descending first, then relevance tier
        results.sort(key=lambda a: a.get("date_active_start", ""), reverse=True)
        results.sort(
            key=lambda a: (
                0
                if a.get("name", "").lower() == query_lower
                else 1
                if a.get("name", "").lower().startswith(query_lower)
                else 2
            )
        )
        return results

    def search_by_url(self, url: str) -> dict[str, Any] | None:
        """Find a single metadata entry by exact web URL match.

        Searches local cache and, if configured, the API batch export.
        Returns the first matching entry or ``None``.
        """
        url = url.strip()
        if not url:
            return None

        if self.metadata_path.exists():
            for metadata_file in self.metadata_path.glob("*.json"):
                if not metadata_file.is_file():
                    continue
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("url") == url:
                    return data

        if self.source in {"api", "hybrid"} and self.api_client is not None:
            try:
                for item in self._load_api_batch_candidates():
                    if item.get("url") == url:
                        return item
            except Exception as exc:
                logger.warning(f"API batch export failed during URL search: {exc}")

        return None

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
    doc_ids: list[str] = Field(default_factory=list)  # Document IDs produced from this file


class FileManifest:
    """Tracks source file metadata for incremental ingestion.

    The manifest records each source VTT file's mtime, size, and content hash.
    On subsequent runs, files are classified as new, modified, or unchanged
    by comparing their current state against the manifest.
    """

    def __init__(self, manifest_path: str | Path) -> None:
        super().__init__()
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
        # Allow small tolerance for filesystem mtime precision differences
        if abs(stat.st_mtime - entry.mtime) < 0.001 and stat.st_size == entry.size:
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
        try:
            stat = file_path.stat()
            self.entries[key] = ManifestEntry(
                mtime=stat.st_mtime,
                size=stat.st_size,
                file_hash=self._compute_file_hash(file_path),
                doc_ids=doc_ids,
            )
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                f"Failed to update manifest entry for missing file {file_path}: {exc}. Removing stale manifest entry if present."
            )
            self.entries.pop(key, None)

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


def warn_if_metadata_cache_predates_taxonomy(metadata_path: Path, sample_size: int = 50) -> bool:
    """Warn when every sampled cache file was written before `tags` existed.

    The library API gained ``tags``/``stories`` in August 2026, and cached
    ``{hash}.json`` files written earlier simply have no such key. A local cache
    also wins over the API in ``api`` mode, so a stale cache yields a silently
    tag-free reindex — which reads as "the API has no tags" rather than "refresh
    the cache". Returns True when a warning was emitted.
    """
    if not metadata_path.is_dir():
        return False

    sampled = 0
    for cache_file in metadata_path.glob("*.json"):
        if sampled >= sample_size:
            break
        try:
            article = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(article, dict):
            continue
        sampled += 1
        if "tags" in article:
            return False

    if sampled == 0:
        return False

    logger.warning(
        f"None of {sampled} sampled web metadata files in {metadata_path} carry a 'tags' key, "
        + "so they predate the library taxonomy. Tag, story and programme fields will be empty. "
        + "Refresh the cache with `rainrag sync-metadata` before reindexing."
    )
    return True


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
        self.speech_free_count = 0  # valid VTT with no cues (silent video)
        self.speech_free_with_metadata_count = 0  # subset that had web metadata → indexed

        # Initialize web metadata loader with validation
        if config.web_metadata.enabled:
            web_metadata_path = Path(config.web_metadata.path)
            source = config.web_metadata.source

            # Build optional API client for api / hybrid modes
            api_client = None
            if source in ("api", "hybrid"):
                try:
                    from rainrag.web_metadata_api import WebMetadataAPIClient

                    api_client = WebMetadataAPIClient.from_env(
                        base_url=config.web_metadata.api_url,
                        token_env=config.web_metadata.api_token_env,
                    )
                    logger.info(
                        "Web metadata API client initialised "
                        + f"(base_url={config.web_metadata.api_url})"
                    )
                except Exception as exc:
                    logger.warning(f"Could not initialise web metadata API client: {exc}")
                    if source == "api":
                        logger.warning(
                            "source='api' but API client failed to init; "
                            + "web metadata will be unavailable"
                        )
                        self.web_metadata_loader = None
                        return
                    if source == "hybrid":
                        logger.warning(
                            "source='hybrid' but API client failed to init; "
                            + "web metadata loader will operate in local-only mode"
                        )

            # For local/hybrid modes, validate the directory
            if source in ("local", "hybrid") and not web_metadata_path.exists():
                if source == "local":
                    logger.warning(f"Web metadata path does not exist: {web_metadata_path}")
                    self.web_metadata_loader = None
                    return
                # hybrid: directory will be created on first API cache write
                web_metadata_path.mkdir(parents=True, exist_ok=True)
            elif source in ("local", "hybrid") and not web_metadata_path.is_dir():
                logger.warning(f"Web metadata path is not a directory: {web_metadata_path}")
                self.web_metadata_loader = None
                return

            # For api mode, ensure the cache directory exists
            if source == "api":
                if web_metadata_path.exists() and web_metadata_path.is_file():
                    raise FileExistsError(
                        f"Web metadata path for api mode exists as a file, not a directory: {web_metadata_path}"
                    )
                web_metadata_path.mkdir(parents=True, exist_ok=True)

            self.web_metadata_loader = WebMetadataLoader(
                web_metadata_path,
                source=source,
                api_client=api_client,
            )
            # Only local mode is stuck with a stale cache; the API modes refetch.
            if source == "local":
                warn_if_metadata_cache_predates_taxonomy(web_metadata_path)
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

    def _make_speech_free_doc(
        self,
        file_path: Path,
        language: str,
        video_id: str,
        web_metadata: dict[str, Any],
        final_date: str | None,
        final_date_ts: float | None,
        duration: float | None,
    ) -> Document | None:
        """Create a metadata-only document for a speech-free (silent) video.

        Called when the VTT file is structurally valid but contains no subtitle
        cues (e.g. a music video or a video with no speech).  If
        ``web_metadata.ingest_speech_free`` is False, or if ``web_metadata`` is
        empty, returns None and the video is skipped entirely.

        The document text is built from the available web metadata fields
        (title + description).  ``is_speech_free`` is set to True so callers
        can filter these documents out of transcript-only queries.
        """
        if not self.config.web_metadata.ingest_speech_free:
            return None
        if not web_metadata:
            return None

        # Build text from web metadata
        parts: list[str] = []
        web_title = web_metadata.get("web_title") or ""
        web_desc = web_metadata.get("web_description") or ""
        if web_title:
            parts.append(web_title)
        if web_desc:
            parts.append(web_desc)

        text = "\n\n".join(parts).strip()
        if len(text) < self.config.processing.min_text_length:
            return None

        doc_id = self.parser.generate_id(file_path)
        return Document(
            id=doc_id,
            path=str(file_path.absolute()),
            language=language,
            text=text,
            length=len(text),
            date=final_date,
            date_ts=final_date_ts,
            duration_seconds=duration,
            start_time=None,
            end_time=None,
            video_id=video_id,
            is_chunk=False,
            chunk_index=None,
            total_chunks=None,
            start_time_seconds=None,
            end_time_seconds=None,
            **document_web_fields(web_metadata),
            is_speech_free=True,
        )

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
                # Valid VTT structure but no subtitle cues — silent/music-only video
                self.speech_free_count += 1
                doc = self._make_speech_free_doc(
                    file_path, language, video_id, web_metadata, final_date, final_date_ts, duration
                )
                if doc:
                    self.speech_free_with_metadata_count += 1
                    logger.debug(
                        f"Created metadata-only document for speech-free video: {file_path}"
                    )
                    return [doc]
                logger.debug(f"Skipping speech-free video (no usable web metadata): {file_path}")
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
                    **document_web_fields(web_metadata),
                    content_hash=compute_content_hash(text),
                )
                documents.append(doc)

            logger.debug(f"Created {len(documents)} chunks from {file_path}")
            return documents

        else:
            # Chunking disabled - process as single document (legacy behavior)
            result = self.parser.parse_vtt_with_timecodes(file_path)

            if result is None:
                # Truly invalid VTT (missing WEBVTT header or unreadable)
                self.invalid_vtt_count += 1
                logger.debug(f"Invalid VTT file: {file_path}")
                return []

            text, start_time, end_time = result

            if not text:
                # Valid VTT structure but no speech cues (silent/music-only video)
                self.speech_free_count += 1
                doc = self._make_speech_free_doc(
                    file_path, language, video_id, web_metadata, final_date, final_date_ts, duration
                )
                if doc:
                    self.speech_free_with_metadata_count += 1
                    logger.debug(
                        f"Created metadata-only document for speech-free video: {file_path}"
                    )
                    return [doc]
                logger.debug(f"Skipping speech-free video (no usable web metadata): {file_path}")
                return []

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
                **document_web_fields(web_metadata),
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

        def joined(key: str) -> str | None:
            values = web_metadata.get(key) or []
            return ", ".join(values) if values else None

        field_map = {
            "title": ("Title", web_metadata.get("web_title")),
            "date": ("Date", web_metadata.get("web_date")),
            "description": ("Description", web_metadata.get("web_description")),
            "url": ("URL", web_metadata.get("web_url")),
            # Opt-in via web_metadata.fields. Adding any of these changes the
            # embedded text, so it invalidates content hashes and forces a full
            # re-embed of the corpus — not just a payload refresh.
            "program": ("Program", web_metadata.get("web_program")),
            "presenters": ("Presenters", joined("web_presenters")),
            "tags": ("Tags", joined("web_tags")),
            "stories": ("Stories", joined("web_stories")),
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
        if self.invalid_vtt_count or self.speech_free_count:
            logger.info(
                f"Ingestion summary: invalid_vtt={self.invalid_vtt_count}, speech_free={self.speech_free_count} (indexed with metadata: {self.speech_free_with_metadata_count})"
            )
        logger.info(f"Output saved to {output_path}")

        # Save manifest for future incremental runs
        try:
            manifest.save()
        except Exception as exc:
            logger.warning(
                f"Failed to save manifest after ingestion: {exc}. Ingestion succeeded, but incremental manifest may be stale."
            )

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
            + f"{len(unchanged_files)} unchanged, {len(deleted_keys)} deleted files"
        )

        # Collect doc_ids that need to be replaced (from modified files)
        modified_old_doc_ids: set[str] = set()
        for file_path in modified_files:
            key = str(file_path.absolute())
            if key in manifest.entries:
                modified_old_doc_ids.update(manifest.entries[key].doc_ids)

        # Write output: unchanged docs + newly processed docs in a temp file for atomic replace
        doc_count = 0
        file_count = 0
        files_to_process = new_files + modified_files

        temp_output_path = output_path.with_name(output_path.name + ".tmp")

        with open(temp_output_path, "w", encoding="utf-8") as out_file:
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

            # Ensure disk durability before job completion
            out_file.flush()
            os.fsync(out_file.fileno())

        # Atomically replace the previous docs file with the new content
        temp_output_path.replace(output_path)

        # Remove deleted file entries from manifest
        manifest.remove_entries(deleted_keys)

        # Save updated manifest
        manifest.save()

        logger.info(
            f"Incremental ingestion complete! {unchanged_count} unchanged + "
            + f"{doc_count - unchanged_count} new/modified documents = {doc_count} total"
        )
        if deleted_doc_ids:
            logger.info(
                f"Removed {len(deleted_doc_ids)} documents from {len(deleted_keys)} deleted files"
            )

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
