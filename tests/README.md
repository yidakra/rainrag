# RainRAG Test Suite

This directory contains the comprehensive test suite for RainRAG.

## Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── unit/                    # Unit tests for individual modules
│   ├── test_config.py       # Configuration system tests
│   ├── test_ingest.py       # VTT parsing and ingestion tests
│   ├── test_embed.py        # Embedding generation tests
│   └── test_index.py        # Qdrant indexing tests
├── integration/             # Integration tests
│   └── test_pipeline.py     # Full pipeline tests
└── fixtures/                # Test data fixtures
```

## Running Tests

### All Tests

```bash
# Run all tests
make test

# Or with poetry
poetry run pytest
```

### Unit Tests Only

```bash
# Run only unit tests
make test-unit

# Or with pytest
poetry run pytest tests/unit
```

### Integration Tests Only

```bash
# Run only integration tests
make test-integration

# Or with pytest
poetry run pytest tests/integration
```

### With Coverage

```bash
# Run tests with coverage report
make test-cov

# View HTML coverage report
open htmlcov/index.html
```

### Specific Test File

```bash
# Run tests from a specific file
poetry run pytest tests/unit/test_ingest.py

# Run a specific test
poetry run pytest tests/unit/test_ingest.py::TestVTTParser::test_parse_vtt_basic
```

### Verbose Output

```bash
# Run with verbose output
poetry run pytest -v

# Run with very verbose output (show all test names)
poetry run pytest -vv
```

## Test Coverage

The test suite includes:

### Unit Tests

#### Configuration (`test_config.py`)
- ✅ Configuration model validation
- ✅ Loading from YAML files
- ✅ Default values
- ✅ Custom configurations
- ✅ Error handling for invalid configs

#### Ingestion (`test_ingest.py`)
- ✅ VTT file parsing
- ✅ Timestamp and markup removal
- ✅ Language detection (Russian/English)
- ✅ Text cleaning and normalization
- ✅ Document ID generation
- ✅ File size limits
- ✅ Minimum text length filtering
- ✅ Full ingestion pipeline
- ✅ JSONL output format
- ✅ Empty archive handling

#### Embedding (`test_embed.py`)
- ✅ Embedding cache management
- ✅ Model loading
- ✅ Document loading from JSONL
- ✅ Embedding generation
- ✅ Vector normalization
- ✅ Cache save and load
- ✅ Force regeneration
- ✅ Empty document handling

#### Indexing (`test_index.py`)
- ✅ Qdrant connection
- ✅ Collection creation
- ✅ Collection recreation
- ✅ Document indexing
- ✅ Batch uploading
- ✅ Search functionality
- ✅ Collection info retrieval
- ✅ Error handling

### Integration Tests

#### Pipeline (`test_pipeline.py`)
- ✅ Full ingest → embed pipeline
- ✅ Embedding caching between runs
- ✅ Multilingual document processing
- ✅ Empty archive handling
- ✅ Incremental processing

## Test Fixtures

The test suite uses pytest fixtures for common test scenarios:

- `temp_dir`: Temporary directory for test files
- `sample_vtt_en`: Sample English VTT content
- `sample_vtt_ru`: Sample Russian VTT content
- `invalid_vtt`: Invalid VTT content for error testing
- `test_config`: Pre-configured test configuration
- `archive_with_vtt_files`: Complete archive with sample files

## Writing New Tests

### Unit Test Template

```python
"""Tests for new module."""

import pytest
from rainrag.new_module import NewClass


class TestNewClass:
    """Tests for NewClass."""

    def test_basic_functionality(self) -> None:
        """Test basic functionality."""
        obj = NewClass()
        result = obj.do_something()

        assert result is not None
        assert result == expected_value
```

### Integration Test Template

```python
"""Integration test for new feature."""

from pathlib import Path
import pytest
from rainrag.config import Config


class TestNewFeature:
    """Integration tests for new feature."""

    def test_feature_integration(
        self,
        test_config: Config,
        temp_dir: Path,
    ) -> None:
        """Test feature integrated with pipeline."""
        # Setup
        # Execute
        # Assert
```

## Best Practices

1. **Use descriptive test names**: Test names should clearly describe what is being tested
2. **One assertion per test**: Each test should verify one specific behavior
3. **Use fixtures**: Reuse common setup code via fixtures
4. **Mock external dependencies**: Use mocks for Qdrant, file I/O where appropriate
5. **Test error cases**: Don't just test the happy path
6. **Keep tests fast**: Unit tests should run in milliseconds

## Continuous Integration

Tests are automatically run on:
- Every commit (if CI is configured)
- Pull requests
- Main branch merges

## Troubleshooting

### Tests Failing Locally

1. **Clear caches**:
   ```bash
   make clean
   rm -rf .pytest_cache
   ```

2. **Reinstall dependencies**:
   ```bash
   poetry install
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.10+
   ```

### Import Errors

Make sure you're running tests with poetry:
```bash
poetry run pytest
```

Or activate the virtual environment:
```bash
poetry shell
pytest
```

### Slow Tests

Run only fast unit tests:
```bash
make test-unit
```

Skip integration tests:
```bash
poetry run pytest -m "not integration"
```

## Coverage Goals

We aim for:
- **Overall coverage**: >80%
- **Critical modules**: >90% (config, ingest, embed, index)
- **CLI module**: >70%

Check current coverage:
```bash
make test-cov
```
