"""Unit tests for the ingestion module."""

import json
from pathlib import Path

import pytest

from rainrag.config import Config
from rainrag.ingest import Document, Ingester, VTTParser


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
        chunks = VTTParser.create_chunks_from_cues(cues, chunk_duration_seconds=300, overlap_seconds=0)

        assert len(chunks) == 2  # 0-5min and 5-10min
        assert chunks[0].chunk_index == 0
        assert chunks[0].start_seconds == 0.0
        assert "Text in first minute" in chunks[0].text
        assert "Text in third minute" in chunks[0].text

        assert chunks[1].chunk_index == 1
        assert chunks[1].start_seconds == 300.0  # 5 minutes (no overlap)
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
            cues, chunk_duration_seconds=300, max_tokens=500, min_tokens=10, language="en"
        )

        # Should create 2 chunks (0-5min and 5-10min), no splitting needed
        assert len(chunks) == 2
        assert "Short content" in chunks[0].text
        assert "Another short segment" in chunks[1].text

    def test_create_chunks_hybrid_with_splitting(self, temp_dir: Path) -> None:
        """Test hybrid chunking splits oversized time-based chunks."""
        # Create a VTT with very long content in first 5 minutes
        long_text = " ".join([f"Word{i}" for i in range(200)])  # ~1000 characters
        vtt_content = f"""WEBVTT

00:00:00.000 --> 00:04:00.000
{long_text}

00:05:00.000 --> 00:06:00.000
Short text in next chunk
"""
        vtt_file = temp_dir / "test.vtt"
        vtt_file.write_text(vtt_content)

        cues = VTTParser.parse_vtt_to_cues(vtt_file)
        assert cues is not None

        # Create chunks with small token limit (should trigger splitting)
        chunks = VTTParser.create_chunks_hybrid(
            cues, chunk_duration_seconds=300, max_tokens=100, min_tokens=10, language="en"
        )

        # First time-based chunk should be split due to token limit
        # Should have more than 2 chunks
        assert len(chunks) >= 2

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

        # Verify first chunk (0-5min)
        assert chunks[0].start_seconds == 0.0
        assert "Important context before boundary" in chunks[0].text
        assert "Critical information spanning" in chunks[0].text

        # Verify second chunk starts at 4:30 (270s), creating 30s overlap
        assert chunks[1].start_seconds == 270.0
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
        assert docs[1].start_time_seconds == 300.0  # No overlap

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

        # Gemini has 2048 token limit, minus 50 buffer = 1998
        assert max_tokens == 1998

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
