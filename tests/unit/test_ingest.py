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

        ingester = Ingester(test_config)
        doc = ingester.process_file(vtt_file)

        assert doc is not None
        assert doc.language == "en"
        assert "Hello, this is a test subtitle" in doc.text
        assert doc.length > 0
        assert doc.path == str(vtt_file.absolute())

    def test_process_file_too_short(
        self, test_config: Config, temp_dir: Path
    ) -> None:
        """Test skipping files with text too short."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hi
"""
        vtt_file = temp_dir / "short.vtt"
        vtt_file.write_text(vtt_content)

        ingester = Ingester(test_config)
        doc = ingester.process_file(vtt_file)

        # Should be None because text is too short (min_text_length=10)
        assert doc is None

    def test_process_file_invalid(
        self, test_config: Config, temp_dir: Path, invalid_vtt: str
    ) -> None:
        """Test processing invalid VTT file."""
        vtt_file = temp_dir / "invalid.vtt"
        vtt_file.write_text(invalid_vtt)

        ingester = Ingester(test_config)
        doc = ingester.process_file(vtt_file)

        assert doc is None

    def test_ingest_pipeline(
        self, test_config: Config, archive_with_vtt_files: Path
    ) -> None:
        """Test full ingestion pipeline."""
        test_config.paths.archive_root = str(archive_with_vtt_files)
        ingester = Ingester(test_config)

        doc_count = ingester.ingest()

        # Should process 4 VTT files
        assert doc_count == 4

        # Check output file exists
        output_file = Path(test_config.paths.docs_output)
        assert output_file.exists()

        # Check JSONL content
        with open(output_file, "r") as f:
            lines = f.readlines()

        assert len(lines) == 4

        # Parse and validate documents
        docs = [json.loads(line) for line in lines]
        assert all("id" in doc for doc in docs)
        assert all("text" in doc for doc in docs)
        assert all("language" in doc for doc in docs)

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
