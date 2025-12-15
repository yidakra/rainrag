"""Ingestion module for parsing VTT subtitle files."""

import hashlib
import json
import re
import subprocess
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

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


class VTTCue(BaseModel):
    """Represents a single VTT subtitle cue."""

    start_time: str  # HH:MM:SS format
    end_time: str  # HH:MM:SS format
    start_seconds: float  # Timestamp in seconds
    end_seconds: float  # Timestamp in seconds
    text: str  # Cleaned text


class VTTChunk(BaseModel):
    """Represents a time-based chunk of VTT content."""

    chunk_index: int
    start_time: str  # HH:MM:SS format
    end_time: str  # HH:MM:SS format
    start_seconds: float
    end_seconds: float
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

        # Check for Russian indicators
        if any(indicator in path_str for indicator in ["ru", "rus", "russian"]):
            return "ru"

        # Check for English indicators
        if any(indicator in path_str for indicator in ["en", "eng", "english"]):
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
    def timestamp_to_seconds(timestamp: str) -> float:
        """
        Convert HH:MM:SS timestamp to seconds.

        Args:
            timestamp: Timestamp in HH:MM:SS format

        Returns:
            Time in seconds
        """
        try:
            parts = timestamp.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError) as e:
            logger.warning(f"Failed to parse timestamp '{timestamp}': {e}")
            return 0.0

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

            # VTT files should start with "WEBVTT"
            if not lines or not lines[0].strip().startswith("WEBVTT"):
                logger.warning(f"File {file_path} does not appear to be a valid VTT file")
                return None

            text_lines: list[str] = []
            timecodes: list[tuple[str, str]] = []  # (start, end) pairs
            skip_next = False

            for line in lines[1:]:  # Skip the WEBVTT header
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Capture timestamp lines
                if cls.TIMESTAMP_PATTERN.match(line):
                    # Extract start and end times: "00:00:00.000 --> 00:00:10.160"
                    parts = line.split("-->")
                    if len(parts) == 2:
                        start = parts[0].strip().split(".")[0]  # Remove milliseconds
                        end = (
                            parts[1].strip().split()[0].split(".")[0]
                        )  # Remove ms and any positioning
                        timecodes.append((start, end))
                    skip_next = False
                    continue

                # Skip cue identifiers (lines that are just numbers or IDs)
                if cls.CUE_ID_PATTERN.match(line) and skip_next is False:
                    skip_next = True
                    continue

                # Skip NOTE comments
                if line.startswith("NOTE"):  # type: ignore[unreachable]
                    continue

                # Skip cue identifiers (lines that are just numbers or IDs)
                if cls.CUE_ID_PATTERN.match(line):
                    continue

                # This is actual subtitle text
                cleaned = cls.clean_text(line)
                if cleaned:
                    text_lines.append(cleaned)

            # Join all text lines with spaces
            full_text = " ".join(text_lines)

            # Get first and last timecodes
            start_time = timecodes[0][0] if timecodes else None
            end_time = timecodes[-1][1] if timecodes else None

            return (full_text, start_time, end_time) if full_text else None

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

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

            # VTT files should start with "WEBVTT"
            if not lines or not lines[0].strip().startswith("WEBVTT"):
                logger.warning(f"File {file_path} does not appear to be a valid VTT file")
                return None

            cues: list[VTTCue] = []
            current_timestamp_line = None
            current_text_lines: list[str] = []
            skip_next = False

            for line in lines[1:]:  # Skip the WEBVTT header
                line = line.strip()

                # Skip empty lines - finalize current cue if any
                if not line:
                    if current_timestamp_line and current_text_lines:
                        # Parse timestamp: "00:00:00.000 --> 00:00:10.160"
                        parts = current_timestamp_line.split("-->")
                        if len(parts) == 2:
                            start = parts[0].strip().split(".")[0]  # Remove milliseconds
                            end = parts[1].strip().split()[0].split(".")[0]  # Remove ms and positioning

                            # Combine and clean text
                            combined_text = " ".join(current_text_lines)
                            cleaned_text = cls.clean_text(combined_text)

                            if cleaned_text:
                                cue = VTTCue(
                                    start_time=start,
                                    end_time=end,
                                    start_seconds=cls.timestamp_to_seconds(start),
                                    end_seconds=cls.timestamp_to_seconds(end),
                                    text=cleaned_text,
                                )
                                cues.append(cue)

                        current_timestamp_line = None
                        current_text_lines = []
                    continue

                # Capture timestamp lines
                if cls.TIMESTAMP_PATTERN.match(line):
                    current_timestamp_line = line
                    skip_next = False
                    continue

                # Skip cue identifiers
                if cls.CUE_ID_PATTERN.match(line) and skip_next is False:
                    skip_next = True
                    continue

                # Skip NOTE comments
                if line.startswith("NOTE"):
                    continue

                # This is actual subtitle text
                current_text_lines.append(line)

            # Don't forget the last cue if file doesn't end with empty line
            if current_timestamp_line and current_text_lines:
                parts = current_timestamp_line.split("-->")
                if len(parts) == 2:
                    start = parts[0].strip().split(".")[0]
                    end = parts[1].strip().split()[0].split(".")[0]
                    combined_text = " ".join(current_text_lines)
                    cleaned_text = cls.clean_text(combined_text)
                    if cleaned_text:
                        cue = VTTCue(
                            start_time=start,
                            end_time=end,
                            start_seconds=cls.timestamp_to_seconds(start),
                            end_seconds=cls.timestamp_to_seconds(end),
                            text=cleaned_text,
                        )
                        cues.append(cue)

            return cues if cues else None

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
            # Check if this cue starts a new chunk
            if cue.start_seconds >= chunk_start_seconds + chunk_duration_seconds:
                # Finalize current chunk if it has content
                if chunk_cues:
                    chunk_end_seconds = chunk_cues[-1].end_seconds
                    chunk_text = " ".join(c.text for c in chunk_cues)

                    chunk = VTTChunk(
                        chunk_index=chunk_index,
                        start_time=chunk_cues[0].start_time,
                        end_time=chunk_cues[-1].end_time,
                        start_seconds=chunk_start_seconds,
                        end_seconds=chunk_end_seconds,
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
                    c for c in chunk_cues
                    if c.start_seconds >= chunk_start_seconds
                ]

            chunk_cues.append(cue)

        # Add final chunk
        if chunk_cues:
            chunk_end_seconds = chunk_cues[-1].end_seconds
            chunk_text = " ".join(c.text for c in chunk_cues)

            chunk = VTTChunk(
                chunk_index=chunk_index,
                start_time=chunk_cues[0].start_time,
                end_time=chunk_cues[-1].end_time,
                start_seconds=chunk_start_seconds,
                end_seconds=chunk_end_seconds,
                text=chunk_text,
                cue_count=len(chunk_cues),
            )
            chunks.append(chunk)

        return chunks

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
                    if time_chunk.start_seconds <= cue.start_seconds < time_chunk.end_seconds
                ]

                # Build token-based sub-chunks
                current_cues: list[VTTCue] = []
                current_tokens = 0

                for cue in chunk_cues:
                    cue_tokens = cls.estimate_tokens(cue.text, language)

                    # Check if adding this cue would exceed limit
                    if current_tokens + cue_tokens > max_tokens and current_cues:
                        # Finalize current sub-chunk
                        sub_chunk_text = " ".join(c.text for c in current_cues)
                        sub_chunk = VTTChunk(
                            chunk_index=chunk_index,
                            start_time=current_cues[0].start_time,
                            end_time=current_cues[-1].end_time,
                            start_seconds=current_cues[0].start_seconds,
                            end_seconds=current_cues[-1].end_seconds,
                            text=sub_chunk_text,
                            cue_count=len(current_cues),
                        )
                        validated_chunks.append(sub_chunk)
                        chunk_index += 1

                        # Start new sub-chunk
                        current_cues = [cue]
                        current_tokens = cue_tokens
                    else:
                        current_cues.append(cue)
                        current_tokens += cue_tokens

                # Add final sub-chunk
                if current_cues:
                    sub_chunk_text = " ".join(c.text for c in current_cues)
                    sub_chunk = VTTChunk(
                        chunk_index=chunk_index,
                        start_time=current_cues[0].start_time,
                        end_time=current_cues[-1].end_time,
                        start_seconds=current_cues[0].start_seconds,
                        end_seconds=current_cues[-1].end_seconds,
                        text=sub_chunk_text,
                        cue_count=len(current_cues),
                    )
                    validated_chunks.append(sub_chunk)
                    chunk_index += 1

        # Step 3: Merge undersized chunks (optional optimization)
        # For now, we'll keep all chunks to preserve temporal boundaries
        # Future enhancement: merge chunks < min_tokens with neighbors

        return validated_chunks

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
        stem = file_path.stem  # "abc123.ru"
        # Remove language suffix if present
        video_hash = stem.rsplit(".", 1)[0] if "." in stem else stem  # "abc123"

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
        stem = vtt_path.stem  # "abc123.ru"
        video_hash = stem.rsplit(".", 1)[0] if "." in stem else stem  # "abc123"

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


class Ingester:
    """Main ingestion class for crawling and parsing VTT files."""

    def __init__(self, config: Config):
        """
        Initialize the ingester.

        Args:
            config: Configuration object
        """
        self.config = config
        self.parser = VTTParser()

    def find_vtt_files(self, root_path: Path) -> Generator[Path, None, None]:
        """
        Recursively find all .vtt files in a directory.

        Args:
            root_path: Root directory to search

        Yields:
            Path objects for VTT files
        """
        logger.info(f"Scanning for .vtt files in {root_path}")

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

        # Check if chunking is enabled
        if self.config.chunking.enabled:
            # Parse VTT into cues
            cues = self.parser.parse_vtt_to_cues(file_path)

            if not cues:
                logger.warning(f"No cues extracted from {file_path}")
                return []

            # Get max tokens based on embedding model
            max_tokens = self.config.get_max_chunk_tokens()

            # Create chunks based on strategy
            if self.config.chunking.strategy == "hybrid":
                # Hybrid: time-based with token validation (RECOMMENDED)
                chunks = self.parser.create_chunks_hybrid(
                    cues,
                    chunk_duration_seconds=self.config.chunking.chunk_duration_seconds,
                    overlap_seconds=self.config.chunking.overlap_seconds,
                    max_tokens=max_tokens,
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
            else:  # token strategy
                # Pure token-based chunking (disable overlap for token-only)
                chunks = self.parser.create_chunks_hybrid(
                    cues,
                    chunk_duration_seconds=9999999,  # Effectively disable time chunking
                    overlap_seconds=0,  # No overlap for pure token strategy
                    max_tokens=max_tokens,
                    min_tokens=self.config.chunking.min_chunk_tokens,
                    language=language,
                )

            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return []

            logger.debug(
                f"Created {len(chunks)} chunks from {file_path} using '{self.config.chunking.strategy}' strategy (max_tokens={max_tokens})"
            )

            # Create Document objects for each chunk
            documents: list[Document] = []
            total_chunks = len(chunks)

            for chunk in chunks:
                # Chunks are already validated by the chunking strategy
                # No need for additional length checks
                text = chunk.text
                doc_id = self.parser.generate_id(file_path, chunk.chunk_index)

                doc = Document(
                    id=doc_id,
                    path=str(file_path.absolute()),
                    language=language,
                    text=text,
                    length=len(text),
                    date=date_iso,
                    date_ts=date_ts,
                    duration_seconds=duration,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    video_id=video_id,
                    is_chunk=True,
                    chunk_index=chunk.chunk_index,
                    total_chunks=total_chunks,
                    start_time_seconds=chunk.start_seconds,
                    end_time_seconds=chunk.end_seconds,
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
                date=date_iso,
                date_ts=date_ts,
                duration_seconds=duration,
                start_time=start_time,
                end_time=end_time,
                video_id=video_id,
                is_chunk=False,
                start_time_seconds=start_seconds,
                end_time_seconds=end_seconds,
            )

            return [doc]

    def ingest(self) -> int:
        """
        Run the full ingestion pipeline.

        Returns:
            Number of documents processed
        """
        root_path = Path(self.config.paths.archive_root)

        if not root_path.exists():
            logger.error(f"Archive root does not exist: {root_path}")
            raise FileNotFoundError(f"Archive root not found: {root_path}")

        output_path = Path(self.config.paths.docs_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting ingestion from {root_path}")
        logger.info(f"Output will be written to {output_path}")

        doc_count = 0
        file_count = 0

        # Open output file for streaming writes
        with open(output_path, "w", encoding="utf-8") as out_file:
            # Process all VTT files
            vtt_files = list(self.find_vtt_files(root_path))

            for file_path in tqdm(vtt_files, desc="Processing VTT files"):
                docs = self.process_file(file_path)

                if docs:
                    file_count += 1
                    # Write each document as JSONL (one JSON object per line)
                    for doc in docs:
                        json_line = doc.model_dump_json()
                        out_file.write(json_line + "\n")
                        doc_count += 1

                    if doc_count % 100 == 0:
                        logger.info(f"Processed {doc_count} documents from {file_count} files")

        chunking_status = "enabled" if self.config.chunking.enabled else "disabled"
        logger.info(
            f"Ingestion complete! Processed {file_count} files into {doc_count} documents (chunking: {chunking_status})"
        )
        logger.info(f"Output saved to {output_path}")

        return doc_count


def run_ingestion(config_path: str = "config.yaml") -> int:
    """
    Run the ingestion pipeline.

    Args:
        config_path: Path to configuration file

    Returns:
        Number of documents processed
    """
    from rainrag.config import load_config

    config = load_config(config_path)
    ingester = Ingester(config)
    return ingester.ingest()
