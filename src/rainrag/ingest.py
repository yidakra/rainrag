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

    @staticmethod
    def generate_id(file_path: Path) -> str:
        """
        Generate a unique ID for a document based on its path.

        Args:
            file_path: Path to the file

        Returns:
            Unique hash ID
        """
        path_str = str(file_path.absolute())
        return hashlib.sha256(path_str.encode()).hexdigest()[:16]

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

    def process_file(self, file_path: Path) -> Document | None:
        """
        Process a single VTT file.

        Args:
            file_path: Path to the VTT file

        Returns:
            Document object or None if processing fails
        """
        # Parse VTT content with timecodes
        result = self.parser.parse_vtt_with_timecodes(file_path)

        if not result:
            logger.warning(f"No text extracted from {file_path}")
            return None

        text, start_time, end_time = result

        # Check minimum text length
        if len(text) < self.config.processing.min_text_length:
            logger.debug(f"Skipping {file_path} (text too short: {len(text)} chars)")
            return None

        # Detect language
        language = self.parser.detect_language(file_path)

        # Generate document ID
        doc_id = self.parser.generate_id(file_path)

        # Extract metadata from source video
        date_iso = None
        date_ts = None
        duration = None
        source_video = self.parser.find_source_video(file_path)
        if source_video:
            date_iso, duration, date_ts = self.parser.get_video_metadata(source_video)

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
        )

        return doc

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

        # Open output file for streaming writes
        with open(output_path, "w", encoding="utf-8") as out_file:
            # Process all VTT files
            vtt_files = list(self.find_vtt_files(root_path))

            for file_path in tqdm(vtt_files, desc="Processing VTT files"):
                doc = self.process_file(file_path)

                if doc:
                    # Write as JSONL (one JSON object per line)
                    json_line = doc.model_dump_json()
                    out_file.write(json_line + "\n")
                    doc_count += 1

                    if doc_count % 100 == 0:
                        logger.info(f"Processed {doc_count} documents")

        logger.info(f"Ingestion complete! Processed {doc_count} documents")
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
