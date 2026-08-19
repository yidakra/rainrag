"""Unit tests for the ingestion module."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rainrag.config import Config
from rainrag.ingest import Document, Ingester, VTTParser, WebMetadataLoader


def _create_fake_loader(
    web_dir: Path, metadata_factory: Callable[[str], dict]
) -> WebMetadataLoader:
    """Return a minimal WebMetadataLoader stub that uses the given factory.

    Args:
        web_dir: directory to pass to real loader when cleaning metadata.
        metadata_factory: callable taking a video_hash and returning a metadata dict.
    """

    class _FakeLoader(WebMetadataLoader):
        def __init__(self, metadata_path: Path):
            super().__init__(metadata_path)

        def load_metadata(self, video_hash):
            return metadata_factory(video_hash)

        def extract_clean_metadata(self, raw_metadata):
            # delegate parsing to a real loader rooted at web_dir
            loader = WebMetadataLoader(web_dir)
            return loader.extract_clean_metadata(raw_metadata)

    return _FakeLoader(web_dir)


_EMPTY_VTT = "WEBVTT\n\n"  # Valid header-only VTT (no cues)


class TestVTTParser:
    """Tests for VTT parser."""

    def test_parse_vtt_basic(self, temp_dir: Path, sample_vtt_en: str) -> None:
        """Test basic VTT parsing."""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(sample_vtt_en)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "Hello, this is a test subtitle" in text
        assert "second line of text" in text
        assert "And this has markup tags that should be removed" in text

    def test_parse_vtt_with_cue_ids(self, temp_dir: Path, sample_vtt_ru: str) -> None:
        """Test VTT parsing with cue IDs."""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(sample_vtt_ru)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "Привет, это тестовые субтитры" in text
        assert "вторая строка текста" in text

    def test_parse_vtt_removes_markup(self, temp_dir: Path) -> None:
        """Test that VTT markup tags are removed."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
<v Speaker>Hello</v> <c>world</c>
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "<v" not in text
        assert "<c>" not in text
        assert "Hello world" in text

    def test_parse_vtt_removes_timestamps(self, temp_dir: Path, sample_vtt_en: str) -> None:
        """Test that timestamps are not included in output."""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(sample_vtt_en)

        text = VTTParser.parse_vtt(vtt_file)

        assert text is not None
        assert "-->" not in text
        assert "00:00" not in text

    def test_parse_invalid_vtt(self, temp_dir: Path, invalid_vtt: str) -> None:
        """Test parsing invalid VTT file."""
        vtt_file = temp_dir / "invalid.vtt"
        vtt_file.write_text(invalid_vtt)

        text = VTTParser.parse_vtt(vtt_file)

        # Should return None or warn for invalid files
        assert text is None

    def test_parse_nonexistent_file(self, temp_dir: Path) -> None:
        """Test parsing non-existent file."""
        vtt_file = temp_dir / "nonexistent.vtt"

        text = VTTParser.parse_vtt(vtt_file)

        assert text is None

    def test_empty_vtt_parse_to_cues_returns_empty_list(self, temp_dir: Path) -> None:
        """parse_vtt_to_cues must return [] (not None) for a header-only VTT."""
        vtt_path = temp_dir / "empty.vtt"
        vtt_path.write_text(_EMPTY_VTT)

        result = VTTParser.parse_vtt_to_cues(vtt_path)
        assert result == []

    def test_empty_vtt_parse_with_timecodes_returns_empty_string(self, temp_dir: Path) -> None:
        """parse_vtt_with_timecodes must return ('', None, None) for a header-only VTT."""
        vtt_file = temp_dir / "silent.vtt"
        vtt_file.write_text(_EMPTY_VTT)

        result = VTTParser.parse_vtt_with_timecodes(vtt_file)
        assert result == ("", None, None)

    def test_clean_text_normalizes_whitespace(self) -> None:
        """Test that whitespace is normalized."""
        text = "Hello    world\n\n\nwith   spaces"
        cleaned = VTTParser.clean_text(text)

        assert "    " not in cleaned
        assert cleaned == "Hello world with spaces"

    def test_clean_text_removes_markup(self) -> None:
        """Test markup removal."""
        text = "<v Speaker>Hello</v> <c>world</c>"
        cleaned = VTTParser.clean_text(text)

        assert "<v" not in cleaned
        assert "<c>" not in cleaned

    def test_detect_language_russian(self, temp_dir: Path) -> None:
        """Test Russian language detection."""
        test_paths = [
            temp_dir / "russian" / "test.vtt",
            temp_dir / "broadcast_ru.vtt",
            temp_dir / "rus" / "subtitle.vtt",
        ]

        for path in test_paths:
            lang = VTTParser.detect_language(path)
            assert lang == "ru", f"Failed for path: {path}"

    def test_detect_language_english(self, temp_dir: Path) -> None:
        """Test English language detection."""
        test_paths = [
            temp_dir / "english" / "test.vtt",
            temp_dir / "broadcast_en.vtt",
            temp_dir / "eng" / "subtitle.vtt",
        ]

        for path in test_paths:
            lang = VTTParser.detect_language(path)
            assert lang == "en", f"Failed for path: {path}"

    def test_detect_language_default(self, temp_dir: Path) -> None:
        """Test default language when no indicators found."""
        path = temp_dir / "unknown" / "test.vtt"
        lang = VTTParser.detect_language(path)

        # Should default to English
        assert lang == "en"

    def test_detect_language_ignores_ancestor_directories(self) -> None:
        """Ancestor directories must not decide the language.

        Regression: detection used to tokenize the whole path, so a temp dir
        named e.g. 'tmpab_ru_9x' made every file under it Russian.
        """
        assert VTTParser.detect_language(Path("/tmp/tmpab_ru_9x/english/test.vtt")) == "en"
        assert VTTParser.detect_language(Path("/tmp/tmpab_ru_9x/broadcast_en.vtt")) == "en"
        assert VTTParser.detect_language(Path("/srv/en_archive/russian/subtitle.vtt")) == "ru"

    def test_detect_language_filename_beats_parent(self) -> None:
        """The filename suffix is more specific than the containing directory."""
        assert VTTParser.detect_language(Path("/archive/russian/clip.en.vtt")) == "en"
        assert VTTParser.detect_language(Path("/archive/english/clip.ru.vtt")) == "ru"

    def test_generate_id_consistency(self, temp_dir: Path) -> None:
        """Test that ID generation is consistent."""
        path = temp_dir / "test.vtt"

        id1 = VTTParser.generate_id(path)
        id2 = VTTParser.generate_id(path)

        assert id1 == id2

    def test_generate_id_uniqueness(self, temp_dir: Path) -> None:
        """Test that different paths generate different IDs."""
        path1 = temp_dir / "test1.vtt"
        path2 = temp_dir / "test2.vtt"

        id1 = VTTParser.generate_id(path1)
        id2 = VTTParser.generate_id(path2)

        assert id1 != id2

    def test_timestamp_to_seconds(self) -> None:
        """Test timestamp to seconds conversion."""
        assert VTTParser.timestamp_to_seconds("00:00:00") == 0.0
        assert VTTParser.timestamp_to_seconds("00:01:00") == 60.0
        assert VTTParser.timestamp_to_seconds("00:00:30") == 30.0
        assert VTTParser.timestamp_to_seconds("01:30:45") == 5445.0

    def test_seconds_to_timestamp(self) -> None:
        """Test seconds to timestamp conversion."""
        assert VTTParser.seconds_to_timestamp(0) == "00:00:00"
        assert VTTParser.seconds_to_timestamp(60) == "00:01:00"
        assert VTTParser.seconds_to_timestamp(30) == "00:00:30"
        assert VTTParser.seconds_to_timestamp(5445) == "01:30:45"

    def test_generate_video_id_consistency(self, temp_dir: Path) -> None:
        """Test that video_id is same for different languages of same video."""
        path_en = temp_dir / "abc123.en.vtt"
        path_ru = temp_dir / "abc123.ru.vtt"

        id_en = VTTParser.generate_video_id(path_en)
        id_ru = VTTParser.generate_video_id(path_ru)

        # Same video in different languages should have same video_id
        assert id_en == id_ru

    def test_generate_video_id_different_videos(self, temp_dir: Path) -> None:
        """Test that different videos have different video_ids."""
        path1 = temp_dir / "abc123.en.vtt"
        path2 = temp_dir / "xyz456.en.vtt"

        id1 = VTTParser.generate_video_id(path1)
        id2 = VTTParser.generate_video_id(path2)

        assert id1 != id2

    def test_parse_vtt_to_cues(self, temp_dir: Path) -> None:
        """Test parsing VTT into individual cues."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
First subtitle

00:00:05.000 --> 00:00:10.000
Second subtitle

00:00:10.000 --> 00:00:15.000
Third subtitle
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)

        assert cues is not None
        assert len(cues) == 3
        assert cues[0].text == "First subtitle"
        assert cues[0].start_time == "00:00:00"
        assert cues[0].end_time == "00:00:05"
        assert cues[0].start_seconds == 0.0
        assert cues[0].end_seconds == 5.0

    def test_create_chunks_from_cues(self, temp_dir: Path) -> None:
        """Test creating time-based chunks from cues."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:01:00.000
Text in first minute

00:02:00.000 --> 00:03:00.000
Text in third minute

00:06:00.000 --> 00:07:00.000
Text in seventh minute
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)
        assert cues is not None

        # Create 5-minute chunks (300 seconds) without overlap
        chunks = VTTParser.create_chunks_from_cues(
            cues, chunk_duration_seconds=300, overlap_seconds=0
        )

        assert len(chunks) == 2  # 0-5min and 5-10min
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_seconds == 0.0
        assert "Text in first minute" in chunks[0].text
        assert "Text in third minute" in chunks[0].text

        assert chunks[1].chunk_index == 1
        assert chunks[1].start_seconds == 360.0  # When cue 3 starts (00:06:00)
        assert "Text in seventh minute" in chunks[1].text

    def test_estimate_tokens_english(self) -> None:
        """Test token estimation for English text."""
        # English: ~4 chars per token
        text = "This is a test sentence with approximately twenty characters per word."
        estimated = VTTParser.estimate_tokens(text, "en")
        # 72 characters / 4 = ~18 tokens
        assert 15 <= estimated <= 21

    def test_estimate_tokens_russian(self) -> None:
        """Test token estimation for Russian text."""
        # Russian: ~2.5 chars per token
        text = "Это тестовое предложение с примерным количеством символов."
        estimated = VTTParser.estimate_tokens(text, "ru")
        # 60 characters / 2.5 = ~24 tokens
        assert 20 <= estimated <= 28

    def test_estimate_tokens_default(self) -> None:
        """Test token estimation with unknown language falls back to default."""
        text = "A" * 350  # 350 characters
        estimated = VTTParser.estimate_tokens(text, "unknown")
        # Default: ~3.5 chars per token -> 350 / 3.5 = 100 tokens
        assert 95 <= estimated <= 105

    def test_create_chunks_hybrid_no_splitting(self, temp_dir: Path) -> None:
        """Test hybrid chunking when time-based chunks fit within token limits."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:02:00.000
Short content that fits easily within token limits

00:05:00.000 --> 00:07:00.000
Another short segment
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)
        assert cues is not None

        # Create chunks with generous token limit (should not split)
        chunks = VTTParser.create_chunks_hybrid(
            cues,
            chunk_duration_seconds=300,
            overlap_seconds=0,
            max_tokens=500,
            min_tokens=1,
            language="en",
        )

        # Should create 2 chunks (0-5min and 5-10min), no splitting needed
        assert len(chunks) == 2
        assert "Short content" in chunks[0].text
        assert "Another short segment" in chunks[1].text

    def test_create_chunks_hybrid_with_splitting(self, temp_dir: Path) -> None:
        """Test hybrid chunking splits oversized time-based chunks."""
        # Create a VTT with moderately long content split across multiple cues in first 5 minutes
        # Split the text into segments that will exceed token limits when combined but not individually
        words = [f"Word{i}" for i in range(150)]
        segment1 = " ".join(words[:40])  # ~40 tokens
        segment2 = " ".join(words[40:80])  # ~40 tokens
        segment3 = " ".join(words[80:110])  # ~30 tokens

        vtt_content = f"""WEBVTT

00:00:00.000 --> 00:01:20.000
{segment1}

00:01:20.000 --> 00:02:40.000
{segment2}

00:02:40.000 --> 00:04:00.000
{segment3}

00:05:00.000 --> 00:06:00.000
Short text in next chunk
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)
        assert cues is not None

        # Create chunks with token limit that will force splitting of combined segments
        chunks = VTTParser.create_chunks_hybrid(
            cues, chunk_duration_seconds=300, max_tokens=70, min_tokens=10, language="en"
        )

        # First time-based chunk should be split due to token limit
        # Combined segments (~110 tokens) exceed 70 token limit, should split into multiple chunks
        # Plus the second time chunk makes at least 3 total
        assert len(chunks) >= 3

        # Verify that splitting occurred (we should have multiple chunks from the first time period)
        # The exact token limits may vary due to merging logic, so just check we got reasonable chunks
        token_counts = [VTTParser.estimate_tokens(chunk.text, "en") for chunk in chunks]
        assert all(count > 0 for count in token_counts), (
            f"All chunks should have content: {token_counts}"
        )

        # Verify per-chunk token limits with tolerance for estimator variance
        assert all(count <= 70 * 1.3 for count in token_counts), (
            f"Chunks should not exceed max_tokens=70 with tolerance: {token_counts}"
        )

    def test_create_chunks_hybrid_multiple_time_chunks(self, temp_dir: Path) -> None:
        """Test hybrid chunking creates multiple time-based chunks when content is sparse."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Content in first 5 minute chunk

00:05:00.000 --> 00:06:00.000
Content in second 5 minute chunk

00:10:00.000 --> 00:11:00.000
Content in third 5 minute chunk
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)
        assert cues is not None

        chunks = VTTParser.create_chunks_hybrid(
            cues, chunk_duration_seconds=300, max_tokens=500, min_tokens=5, language="en"
        )

        # Should create 3 time-based chunks (0-5min, 5-10min, 10-15min)
        assert len(chunks) == 3
        assert "first 5 minute chunk" in chunks[0].text
        assert "second 5 minute chunk" in chunks[1].text
        assert "third 5 minute chunk" in chunks[2].text

    def test_create_chunks_with_overlap(self, temp_dir: Path) -> None:
        """Test chunk overlap prevents information loss at boundaries."""
        vtt_content = """WEBVTT

00:04:30.000 --> 00:04:45.000
Important context before boundary

00:04:50.000 --> 00:05:10.000
Critical information spanning the 5-minute mark

00:05:15.000 --> 00:05:30.000
Follow-up information after boundary
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)
        assert cues is not None

        # Create chunks with 5-minute duration and 30-second overlap
        chunks = VTTParser.create_chunks_from_cues(
            cues, chunk_duration_seconds=300, overlap_seconds=30
        )

        # Should create 2 chunks: 0-5min and 4:30-10min (30s overlap)
        assert len(chunks) == 2

        # Verify first chunk (0-5min window, content starts at 4:30)
        assert chunks[0].start_seconds == 270.0  # First cue at 4:30 (270s)
        assert "Important context before boundary" in chunks[0].text
        assert "Critical information spanning" in chunks[0].text

        # Verify second chunk (4:30-10min window, content starts at 4:30 due to overlap)
        assert chunks[1].start_seconds == 270.0  # Overlap starts at 4:30 (270s)
        # Second chunk should contain overlapping content from first chunk
        assert "Important context before boundary" in chunks[1].text
        assert "Critical information spanning" in chunks[1].text
        assert "Follow-up information after boundary" in chunks[1].text


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self) -> None:
        """Test creating a document."""
        doc = Document(
            id="test123",
            path="/path/to/file.vtt",
            language="en",
            text="Hello world",
            length=11,
        )

        assert doc.id == "test123"
        assert doc.path == "/path/to/file.vtt"
        assert doc.language == "en"
        assert doc.text == "Hello world"
        assert doc.length == 11

    def test_document_serialization(self) -> None:
        """Test document JSON serialization."""
        doc = Document(
            id="test123",
            path="/path/to/file.vtt",
            language="ru",
            text="Привет",
            length=6,
        )

        json_str = doc.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["id"] == "test123"
        assert parsed["language"] == "ru"
        assert parsed["text"] == "Привет"


class TestIngester:
    """Tests for Ingester class."""

    def test_find_vtt_files(self, test_config: Config, archive_with_vtt_files: Path) -> None:
        """Test finding VTT files in archive."""
        test_config.paths.archive_root = str(archive_with_vtt_files)
        ingester = Ingester(test_config)

        vtt_files = list(ingester.find_vtt_files(archive_with_vtt_files))

        assert len(vtt_files) == 4  # 2 in subdirs + 2 in mixed
        assert all(f.suffix == ".vtt" for f in vtt_files)

    def test_process_file_success(
        self, test_config: Config, temp_dir: Path, sample_vtt_en: str
    ) -> None:
        """Test successful file processing."""
        vtt_file = temp_dir / "english" / "test.vtt"
        vtt_file.parent.mkdir(parents=True)
        vtt_file.write_text(sample_vtt_en)

        # Disable chunking for this test to get single document
        test_config.chunking.enabled = False
        ingester = Ingester(test_config)
        docs = ingester.process_file(vtt_file)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.language == "en"
        assert "Hello, this is a test subtitle" in doc.text
        assert doc.length > 0
        assert doc.path == str(vtt_file.absolute())
        assert doc.video_id is not None
        assert doc.is_chunk is False

    def test_process_file_too_short(self, test_config: Config, temp_dir: Path) -> None:
        """Test skipping files with text too short."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hi
"""
        vtt_file = temp_dir / "short.vtt"
        vtt_file.write_text(vtt_content)

        test_config.chunking.enabled = False
        ingester = Ingester(test_config)
        docs = ingester.process_file(vtt_file)

        # Should be empty list because text is too short (min_text_length=10)
        assert len(docs) == 0

    def test_process_file_invalid(
        self, test_config: Config, temp_dir: Path, invalid_vtt: str
    ) -> None:
        """Test processing invalid VTT file."""
        vtt_file = temp_dir / "invalid.vtt"
        vtt_file.write_text(invalid_vtt)

        test_config.chunking.enabled = False
        ingester = Ingester(test_config)
        docs = ingester.process_file(vtt_file)

        assert len(docs) == 0

    # ------------------------------------------------------------------
    # Speech-free (empty VTT) handling
    # ------------------------------------------------------------------

    def test_empty_vtt_without_web_metadata_skipped(
        self, test_config: Config, temp_dir: Path
    ) -> None:
        """Speech-free video with no web metadata must be skipped (no documents)."""
        vtt_file = temp_dir / "silent.vtt"
        vtt_file.write_text(_EMPTY_VTT)

        test_config.chunking.enabled = False
        test_config.web_metadata.enabled = False
        ingester = Ingester(test_config)
        docs = ingester.process_file(vtt_file)

        assert docs == []
        assert ingester.speech_free_count == 1
        assert ingester.speech_free_with_metadata_count == 0

    def test_empty_vtt_chunking_without_web_metadata_skipped(
        self, test_config: Config, temp_dir: Path
    ) -> None:
        """Speech-free video skipped in chunking mode when no metadata available."""
        vtt_file = temp_dir / "silent.vtt"
        vtt_file.write_text(_EMPTY_VTT)

        test_config.chunking.enabled = True
        test_config.web_metadata.enabled = False
        ingester = Ingester(test_config)
        docs = ingester.process_file(vtt_file)

        assert docs == []
        assert ingester.speech_free_count == 1
        assert ingester.speech_free_with_metadata_count == 0

    def test_empty_vtt_with_web_metadata_creates_doc(
        self, test_config: Config, temp_dir: Path
    ) -> None:
        """Speech-free video with web metadata must produce a metadata-only document."""

        vtt_file = temp_dir / "silent.vtt"
        vtt_file.write_text(_EMPTY_VTT)

        # Set up web metadata directory with a file matching the vtt filename hash
        web_dir = temp_dir / "web_metadata"
        web_dir.mkdir()
        # Use a fake hash matching what extract_video_hash would extract from the path.
        # We write directly to the Ingester's web_metadata_loader to bypass hash extraction.
        test_config.chunking.enabled = False
        test_config.web_metadata.enabled = True
        test_config.web_metadata.path = str(web_dir)
        test_config.web_metadata.ingest_speech_free = True

        ingester = Ingester(test_config)

        # Use shared fake loader factory instead of repeating class
        ingester.web_metadata_loader = _create_fake_loader(
            web_dir,
            lambda _: {
                "name": "Silent Video Title",
                "preview_text": "",
                "detail_text": "<p>Some description of the silent video.</p>",
                "date_active_start": "2025-01-15T10:00:00Z",
                "url": "https://example.com/video/1",
                "video_hash": "fakehash",
            },
        )

        docs = ingester.process_file(vtt_file)

        assert len(docs) == 1
        doc = docs[0]
        assert doc.is_speech_free is True
        assert "Silent Video Title" in doc.text
        assert doc.start_time is None
        assert doc.end_time is None
        assert doc.web_title == "Silent Video Title"
        assert doc.is_chunk is False
        assert ingester.speech_free_count == 1
        assert ingester.speech_free_with_metadata_count == 1

    def test_empty_vtt_ingest_speech_free_false_skips_even_with_metadata(
        self, test_config: Config, temp_dir: Path
    ) -> None:
        """When ingest_speech_free=False, speech-free videos must be skipped."""
        vtt_file = temp_dir / "silent.vtt"
        vtt_file.write_text(_EMPTY_VTT)

        test_config.chunking.enabled = False
        test_config.web_metadata.enabled = True
        test_config.web_metadata.ingest_speech_free = False
        test_config.web_metadata.path = str(temp_dir)

        ingester = Ingester(test_config)

        # assign loader using factory and config path for extraction
        web_dir = Path(test_config.web_metadata.path)
        ingester.web_metadata_loader = _create_fake_loader(
            web_dir,
            lambda _: {
                "name": "Title",
                "preview_text": "",
                "detail_text": "<p>Description text here.</p>",
                "date_active_start": None,
                "url": "",
                "video_hash": "fakehash",
            },
        )
        docs = ingester.process_file(vtt_file)

        assert docs == []
        assert ingester.speech_free_with_metadata_count == 0
        # Even though ingester.web_metadata_loader returns metadata for this file,
        # test_config.web_metadata.ingest_speech_free is False, so ingester.process_file
        # skips ingestion and keeps speech_free_with_metadata_count at 0.
        # ingester.speech_free_count is still incremented to 1 because the file
        # is recognized as speech-free.
        assert ingester.speech_free_count == 1

    def test_invalid_vtt_counter_not_incremented_for_empty_vtt(
        self, test_config: Config, temp_dir: Path
    ) -> None:
        """A speech-free VTT must increment speech_free_count, not invalid_vtt_count."""
        vtt_file = temp_dir / "silent.vtt"
        vtt_file.write_text(_EMPTY_VTT)

        test_config.chunking.enabled = False
        test_config.web_metadata.enabled = False
        ingester = Ingester(test_config)
        ingester.process_file(vtt_file)

        assert ingester.invalid_vtt_count == 0
        assert ingester.speech_free_count == 1

    def test_ingest_pipeline(self, test_config: Config, archive_with_vtt_files: Path) -> None:
        """Test full ingestion pipeline without chunking."""
        test_config.paths.archive_root = str(archive_with_vtt_files)
        test_config.chunking.enabled = False  # Disable chunking for predictable count
        ingester = Ingester(test_config)

        doc_count = ingester.ingest()

        # Should process 4 VTT files
        assert doc_count == 4

        # Check output file exists
        output_file = Path(test_config.paths.docs_output)
        assert output_file.exists()

        # Check JSONL content
        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == 4

        # Parse and validate documents
        docs = [json.loads(line) for line in lines]
        assert all("id" in doc for doc in docs)
        assert all("text" in doc for doc in docs)
        assert all("language" in doc for doc in docs)
        assert all("video_id" in doc for doc in docs)

        # Check we have both languages
        languages = {doc["language"] for doc in docs}
        assert "en" in languages
        assert "ru" in languages

    def test_ingest_pipeline_with_empty_vtt_no_metadata(
        self, test_config: Config, archive_with_vtt_files: Path
    ) -> None:
        """Empty VTT files without web metadata must be excluded from the JSONL output."""
        # Add an empty VTT into the archive
        silent_dir = archive_with_vtt_files / "silent"
        silent_dir.mkdir()
        (silent_dir / "silent_en.vtt").write_text(_EMPTY_VTT)

        test_config.paths.archive_root = str(archive_with_vtt_files)
        test_config.chunking.enabled = False
        test_config.web_metadata.enabled = False
        ingester = Ingester(test_config)

        doc_count = ingester.ingest()

        # Still 4 — the empty VTT is not counted
        assert doc_count == 4
        assert ingester.speech_free_count == 1
        assert ingester.speech_free_with_metadata_count == 0
        assert ingester.invalid_vtt_count == 0

        # Verify no speech-free docs in JSONL
        with open(test_config.paths.docs_output) as f:
            docs = [json.loads(line) for line in f]
        assert all(not doc.get("is_speech_free", False) for doc in docs)

    def test_ingest_pipeline_empty_vtt_with_metadata_creates_doc(
        self, test_config: Config, archive_with_vtt_files: Path, temp_dir: Path
    ) -> None:
        """Empty VTT + web metadata must produce one metadata-only doc in the JSONL."""
        # Add an empty VTT whose filename encodes a fake hash
        silent_dir = archive_with_vtt_files / "silent"
        silent_dir.mkdir()
        (silent_dir / "silent_en.vtt").write_text(_EMPTY_VTT)

        test_config.paths.archive_root = str(archive_with_vtt_files)
        test_config.chunking.enabled = False
        test_config.web_metadata.enabled = True
        test_config.web_metadata.ingest_speech_free = True

        ingester = Ingester(test_config)

        # inject loader via shared factory, using temp_dir as web root
        ingester.web_metadata_loader = _create_fake_loader(
            temp_dir,
            lambda video_hash: {
                "name": "Silent Video",
                "preview_text": "",
                "detail_text": "<p>Silent footage of the archive.</p>",
                "date_active_start": "2024-06-01T00:00:00Z",
                "url": "https://example.com/silent",
                "video_hash": video_hash,
            },
        )

        doc_count = ingester.ingest()

        # 4 normal docs + 1 speech-free metadata doc
        assert doc_count == 5
        assert ingester.speech_free_with_metadata_count == 1

        with open(test_config.paths.docs_output) as f:
            docs = [json.loads(line) for line in f]

        speech_free_docs = [d for d in docs if d.get("is_speech_free")]
        assert len(speech_free_docs) == 1
        assert speech_free_docs[0]["start_time"] is None
        assert speech_free_docs[0]["end_time"] is None
        assert "Silent Video" in speech_free_docs[0]["text"]

    def test_ingest_nonexistent_archive(self, test_config: Config, temp_dir: Path) -> None:
        """Test ingestion with non-existent archive directory."""
        test_config.paths.archive_root = str(temp_dir / "nonexistent")
        ingester = Ingester(test_config)

        with pytest.raises(FileNotFoundError):
            ingester.ingest()

    def test_ingest_empty_archive(self, test_config: Config, temp_dir: Path) -> None:
        """Test ingestion with empty archive directory."""
        archive_dir = temp_dir / "empty"
        archive_dir.mkdir()

        test_config.paths.archive_root = str(archive_dir)
        ingester = Ingester(test_config)

        doc_count = ingester.ingest()

        # Should process 0 files
        assert doc_count == 0

    def test_file_size_limit(self, test_config: Config, temp_dir: Path) -> None:
        """Test that files exceeding size limit are skipped."""
        # Create a large VTT file
        large_content = "WEBVTT\n\n" + ("00:00:00.000 --> 00:00:05.000\nText\n\n" * 10000)
        vtt_file = temp_dir / "large.vtt"
        vtt_file.write_text(large_content)

        # Set a small size limit
        test_config.processing.max_file_size = 100  # 100 bytes
        test_config.paths.archive_root = str(temp_dir)

        ingester = Ingester(test_config)
        vtt_files = list(ingester.find_vtt_files(temp_dir))

        # Large file should be skipped
        assert len(vtt_files) == 0

    def test_process_file_with_chunking(self, test_config: Config, temp_dir: Path) -> None:
        """Test file processing with chunking enabled."""
        # Create a VTT file with 10 minutes of content
        vtt_content = """WEBVTT

00:00:00.000 --> 00:01:00.000
Content in first minute

00:02:00.000 --> 00:03:00.000
Content in third minute

00:06:00.000 --> 00:07:00.000
Content in seventh minute

00:09:00.000 --> 00:10:00.000
Content in tenth minute
"""
        vtt_file = temp_dir / "test.en.vtt"
        vtt_file.write_text(vtt_content)

        # Enable chunking with 5-minute chunks and no overlap
        test_config.chunking.enabled = True
        test_config.chunking.chunk_duration_seconds = 300  # 5 minutes
        test_config.chunking.overlap_seconds = 0  # No overlap for this test
        test_config.chunking.min_chunk_tokens = 1  # Very low to prevent merging

        ingester = Ingester(test_config)
        docs = ingester.process_file(vtt_file)

        # Should create 2 chunks (0-5min and 5-10min)
        assert len(docs) == 2

        # Verify first chunk
        assert docs[0].is_chunk is True
        assert docs[0].chunk_index == 0
        assert docs[0].total_chunks == 2
        assert "Content in first minute" in docs[0].text
        assert docs[0].start_time_seconds == 0.0

        # Verify second chunk
        assert docs[1].is_chunk is True
        assert docs[1].chunk_index == 1
        assert docs[1].total_chunks == 2
        assert "Content in seventh minute" in docs[1].text
        assert docs[1].start_time_seconds == 360.0  # First cue at 6:00 (360 seconds)

        # Verify both have same video_id
        assert docs[0].video_id == docs[1].video_id


class TestConfig:
    """Tests for Config model and methods."""

    def test_get_max_chunk_tokens_e5_model(self, test_config: Config) -> None:
        """Test token limit detection for E5 embedding models."""
        test_config.embedding.provider = "local"
        test_config.embedding.model_name = "intfloat/multilingual-e5-large"
        test_config.chunking.token_buffer = 50

        max_tokens = test_config.get_max_chunk_tokens()

        # E5 has 512 token limit, minus 50 buffer = 462
        assert max_tokens == 462

    def test_get_max_chunk_tokens_openai_model(self, test_config: Config) -> None:
        """Test token limit detection for OpenAI embedding models."""
        test_config.embedding.provider = "openai"
        test_config.openai.embedding_model = "text-embedding-3-small"
        test_config.chunking.token_buffer = 50

        max_tokens = test_config.get_max_chunk_tokens()

        # OpenAI has 8191 token limit, minus 50 buffer = 8141
        assert max_tokens == 8141

    def test_get_max_chunk_tokens_gemini_model(self, test_config: Config) -> None:
        """Test token limit detection for Gemini embedding models."""
        test_config.embedding.provider = "gemini"
        test_config.gemini.embedding_model = "models/text-embedding-004"
        test_config.chunking.token_buffer = 50

        max_tokens = test_config.get_max_chunk_tokens()

        # Gemini text-embedding-004 has 3072 token limit, minus 50 buffer = 3022
        assert max_tokens == 3022

    def test_get_max_chunk_tokens_explicit_override(self, test_config: Config) -> None:
        """Test that explicit max_chunk_tokens overrides auto-detection."""
        test_config.embedding.provider = "local"
        test_config.embedding.model_name = "intfloat/multilingual-e5-large"
        test_config.chunking.max_chunk_tokens = 300  # Explicit override

        max_tokens = test_config.get_max_chunk_tokens()

        # Should use explicit value, not auto-detected
        assert max_tokens == 300

    def test_get_max_chunk_tokens_unknown_model(self, test_config: Config) -> None:
        """Test fallback for unknown embedding models."""
        test_config.embedding.provider = "local"
        test_config.embedding.model_name = "unknown/model"
        test_config.embedding.max_seq_length = 512
        test_config.chunking.token_buffer = 50

        max_tokens = test_config.get_max_chunk_tokens()

        # Should fall back to max_seq_length minus buffer
        assert max_tokens == 462


class TestLibraryTaxonomyExtraction:
    """Extraction of the library CMS taxonomy fields into cleaned metadata."""

    @staticmethod
    def _clean(tmp_path: Path, **article) -> dict:
        article.setdefault("name", "Выпуск")
        return WebMetadataLoader(tmp_path).extract_clean_metadata(article)

    def test_program_and_presenters_are_extracted(self, tmp_path):
        cleaned = self._clean(
            tmp_path,
            teleshow={"id": 2320, "name": "Женщины сверху", "active": True},
            presentors=[
                {"id": 363, "firstname": "Анна", "lastname": "Монгайт"},
                {"id": 12, "firstname": "Тихон", "lastname": ""},
            ],
        )
        assert cleaned["web_program"] == "Женщины сверху"
        assert cleaned["web_presenters"] == ["Анна Монгайт", "Тихон"]

    def test_presenters_accept_the_single_name_shape(self, tmp_path):
        """The site embeds one `name`; the article endpoint splits it in two."""
        cleaned = self._clean(tmp_path, presentors=[{"name": "Анна Монгайт"}])
        assert cleaned["web_presenters"] == ["Анна Монгайт"]

    def test_lite_rubric_is_dropped_from_tags(self, tmp_path):
        cleaned = self._clean(
            tmp_path,
            tags=[
                {"id": 1, "name": "Спорт", "category": "lite"},
                {"id": 2, "name": "Война", "category": "theme"},
            ],
        )
        assert cleaned["web_tags"] == ["Война"]
        assert cleaned["web_tag_ids"] == [2]

    def test_tags_are_grouped_by_category(self, tmp_path):
        cleaned = self._clean(
            tmp_path,
            tags=[
                {"id": 258, "name": "Украина", "category": "theme"},
                {"id": 139, "name": "Сергей Собянин", "category": "person"},
                {"id": 476, "name": "Египет", "category": "location"},
            ],
        )
        assert cleaned["web_tags_theme"] == ["Украина"]
        assert cleaned["web_tags_person"] == ["Сергей Собянин"]
        assert cleaned["web_tags_location"] == ["Египет"]
        assert cleaned["web_tags"] == ["Украина", "Сергей Собянин", "Египет"]

    def test_name_duplicated_across_categories_collapses_but_keeps_both_ids(self, tmp_path):
        """The CMS files Украина under both theme and location with distinct ids."""
        cleaned = self._clean(
            tmp_path,
            tags=[
                {"id": 258, "name": "Украина", "category": "theme"},
                {"id": 1248, "name": "Украина", "category": "location"},
            ],
        )
        assert cleaned["web_tags"] == ["Украина"]
        assert cleaned["web_tag_ids"] == [258, 1248]

    def test_unknown_categories_are_ignored(self, tmp_path):
        cleaned = self._clean(tmp_path, tags=[{"id": 9, "name": "Нечто", "category": "brand-new"}])
        assert cleaned["web_tags"] == []
        assert cleaned["web_tag_ids"] == []

    def test_stories_are_extracted(self, tmp_path):
        cleaned = self._clean(
            tmp_path, stories=[{"id": 320563, "name": "Коронавирус"}, {"id": 1, "name": ""}]
        )
        assert cleaned["web_stories"] == ["Коронавирус"]

    def test_missing_taxonomy_yields_empty_values_not_none(self, tmp_path):
        """The vast majority of the archive has no tags; that must not be an error."""
        cleaned = self._clean(tmp_path, preview_text="Текст")
        assert cleaned["web_program"] is None
        assert cleaned["web_presenters"] == []
        assert cleaned["web_tags"] == []
        assert cleaned["web_tag_ids"] == []
        assert cleaned["web_stories"] == []

    @pytest.mark.parametrize("junk", [None, "not-a-list", 42, [None, 7, "плоская строка"]])
    def test_malformed_payloads_do_not_raise(self, tmp_path, junk):
        cleaned = self._clean(tmp_path, tags=junk, stories=junk, presentors=junk, teleshow=junk)
        assert isinstance(cleaned["web_tags"], list)
        assert isinstance(cleaned["web_stories"], list)
        assert isinstance(cleaned["web_presenters"], list)
        assert cleaned["web_program"] is None

    def test_tag_ids_survive_string_encoding(self, tmp_path):
        cleaned = self._clean(
            tmp_path,
            tags=[
                {"id": "258", "name": "Украина", "category": "theme"},
                {"id": None, "name": "Война", "category": "theme"},
            ],
        )
        assert cleaned["web_tag_ids"] == [258]
        assert cleaned["web_tags"] == ["Украина", "Война"]


class TestTaxonomyReachesThePayload:
    """The fields must survive Document construction and payload building."""

    def test_document_web_fields_covers_every_payload_field(self, tmp_path):
        from rainrag.index import _build_payload
        from rainrag.ingest import WEB_METADATA_PAYLOAD_FIELDS, document_web_fields

        cleaned = WebMetadataLoader(tmp_path).extract_clean_metadata(
            {
                "name": "Выпуск",
                "teleshow": {"name": "Здесь и сейчас"},
                "presentors": [{"firstname": "Анна", "lastname": "Монгайт"}],
                "tags": [{"id": 258, "name": "Украина", "category": "theme"}],
                "stories": [{"id": 1, "name": "Болотное дело"}],
            }
        )
        doc = Document(
            id="d", path="/a.vtt", language="ru", text="t", length=1, **document_web_fields(cleaned)
        )
        payload = _build_payload(doc)
        for field in WEB_METADATA_PAYLOAD_FIELDS:
            assert field in payload, f"{field} missing from the Qdrant payload"
        assert payload["web_program"] == "Здесь и сейчас"
        assert payload["web_presenters"] == ["Анна Монгайт"]
        assert payload["web_tags"] == ["Украина"]
        assert payload["web_stories"] == ["Болотное дело"]

    def test_untagged_document_defaults_to_empty_lists(self):
        from rainrag.index import _build_payload

        payload = _build_payload(Document(id="d", path="/a.vtt", language="ru", text="t", length=1))
        assert payload["web_tags"] == []
        assert payload["web_program"] is None


class TestTaxonomyInEmbeddedText:
    """Taxonomy is opt-in for the embedded text block."""

    @staticmethod
    def _ingester(config: Config, fields: list[str]) -> Ingester:
        config.web_metadata.fields = fields
        config.web_metadata.append_label = "[Web]"
        return Ingester(config)

    def test_taxonomy_is_absent_unless_requested(self, test_config: Config):
        ingester = self._ingester(test_config, ["title"])
        block = ingester._build_web_metadata_block(
            {"web_title": "Выпуск", "web_tags": ["Война"], "web_program": "Здесь и сейчас"}
        )
        assert block == "[Web]\nTitle: Выпуск"

    def test_requested_taxonomy_is_rendered(self, test_config: Config):
        ingester = self._ingester(test_config, ["program", "presenters", "tags", "stories"])
        block = ingester._build_web_metadata_block(
            {
                "web_program": "Здесь и сейчас",
                "web_presenters": ["Анна Монгайт", "Тихон Дзядко"],
                "web_tags": ["Война", "Украина"],
                "web_stories": ["Болотное дело"],
            }
        )
        assert block == (
            "[Web]\nProgram: Здесь и сейчас\nPresenters: Анна Монгайт, Тихон Дзядко\n"
            "Tags: Война, Украина\nStories: Болотное дело"
        )

    def test_empty_taxonomy_lists_emit_no_lines(self, test_config: Config):
        ingester = self._ingester(test_config, ["title", "tags", "stories"])
        block = ingester._build_web_metadata_block({"web_title": "Выпуск", "web_tags": []})
        assert block == "[Web]\nTitle: Выпуск"

    def test_taxonomy_fields_are_not_warned_about_as_unknown(
        self, test_config: Config, monkeypatch
    ):
        """Config accepts the new names, so the unknown-field warning must stay silent.

        ingest logs through loguru, which does not reach caplog; capture the call.
        """
        warnings: list[str] = []
        monkeypatch.setattr(
            "rainrag.ingest.logger.warning", lambda message, *a, **k: warnings.append(str(message))
        )
        ingester = self._ingester(test_config, ["program", "presenters", "tags", "stories"])
        ingester._build_web_metadata_block({"web_program": "Здесь и сейчас"})
        assert warnings == []

    def test_a_genuinely_unknown_field_still_warns(self, test_config: Config, monkeypatch):
        warnings: list[str] = []
        monkeypatch.setattr(
            "rainrag.ingest.logger.warning", lambda message, *a, **k: warnings.append(str(message))
        )
        ingester = self._ingester(test_config, ["not_a_field"])
        ingester._build_web_metadata_block({"web_title": "Выпуск"})
        assert any("Unrecognized web metadata fields" in w for w in warnings)


class TestStaleMetadataCacheWarning:
    """A cache written before the API exposed tags must not fail silently."""

    @staticmethod
    def _write(directory: Path, name: str, article: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{name}.json").write_text(json.dumps(article), encoding="utf-8")

    def test_warns_when_no_sampled_file_has_a_tags_key(self, tmp_path):
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        for i in range(3):
            self._write(tmp_path, f"hash{i}", {"name": "Выпуск", "teleshow": {"name": "П"}})
        assert warn_if_metadata_cache_predates_taxonomy(tmp_path) is True

    def test_silent_when_any_file_carries_tags(self, tmp_path):
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        self._write(tmp_path, "old", {"name": "Выпуск"})
        self._write(tmp_path, "new", {"name": "Выпуск", "tags": []})
        assert warn_if_metadata_cache_predates_taxonomy(tmp_path) is False

    def test_an_empty_tags_list_still_counts_as_fresh(self, tmp_path):
        """Untagged-but-current articles legitimately return an empty list."""
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        self._write(tmp_path, "only", {"name": "Выпуск", "tags": []})
        assert warn_if_metadata_cache_predates_taxonomy(tmp_path) is False

    def test_silent_on_empty_or_missing_directory(self, tmp_path):
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        assert warn_if_metadata_cache_predates_taxonomy(tmp_path) is False
        assert warn_if_metadata_cache_predates_taxonomy(tmp_path / "nope") is False

    def test_unreadable_files_are_skipped_not_counted(self, tmp_path):
        from rainrag.ingest import warn_if_metadata_cache_predates_taxonomy

        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        assert warn_if_metadata_cache_predates_taxonomy(tmp_path) is False
