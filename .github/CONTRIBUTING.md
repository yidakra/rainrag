# Contributing to RainRAG

Thank you for considering contributing to RainRAG! This document outlines the development workflow and tools we use.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/rainrag.git
cd rainrag

# Install dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Verify setup
uv run pytest
uv run ruff check src/ tests/
uv run mypy src/rainrag/
```

---

## 🛠️ Development Tools

### Code Quality Tools

We use several tools to maintain code quality:

#### **Ruff** - Linter & Formatter
Fast Python linter and formatter (replaces flake8, black, isort).

```bash
# Check for issues
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check src/ tests/ --fix

# Format code
uv run ruff format src/ tests/

# Check formatting without changing files
uv run ruff format --check src/ tests/
```

#### **Mypy** - Type Checker
Static type checking for Python.

```bash
# Run type checking
uv run mypy src/rainrag/

# Install missing type stubs
uv run mypy --install-types
```

#### **Bandit** - Security Linter
Finds common security issues in Python code.

```bash
# Run security checks
uv run bandit -r src/
```

#### **Pre-commit** - Git Hooks
Automatically runs checks before each commit.

```bash
# Install hooks (one time)
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files

# Skip hooks for a commit (not recommended)
git commit --no-verify -m "message"
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/rainrag --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_query.py

# Run tests matching pattern
uv run pytest -k "test_mistral"

# Run with verbose output
uv run pytest -v

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/
```

### Test Coverage

We aim for >80% test coverage. View coverage report:

```bash
uv run pytest --cov=src/rainrag --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

---

## 📝 Code Style Guidelines

### Python Style
- **Line length:** 100 characters
- **Quotes:** Double quotes preferred
- **Import order:** stdlib → third-party → first-party
- **Type hints:** Use modern syntax (`str | None` not `Optional[str]`)
- **Docstrings:** Google style

### Example

```python
"""Module for handling embeddings."""

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rainrag.config import Config


def generate_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings for texts.

    Args:
        texts: List of texts to embed
        model: SentenceTransformer model
        batch_size: Batch size for processing

    Returns:
        Array of embeddings with shape (len(texts), embedding_dim)
    """
    return model.encode(texts, batch_size=batch_size)
```

---

## 🔄 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

Write your code following the style guidelines above.

### 3. Run Quality Checks

```bash
# Format code
uv run ruff format src/ tests/

# Check for issues
uv run ruff check src/ tests/ --fix

# Run tests
uv run pytest

# Type check (optional, but recommended)
uv run mypy src/rainrag/
```

### 4. Commit Changes

Pre-commit hooks will automatically run before each commit:

```bash
git add .
git commit -m "feat: add support for new provider"
```

If hooks fail, fix the issues and commit again.

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## 📋 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or updating tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

**Examples:**
```
feat: add Cohere provider support
fix: handle empty query results gracefully
docs: update README with new installation steps
test: add tests for multilingual processing
refactor: extract prompt templates to config
```

---

## 🏗️ Architecture Overview

### Project Structure

```
rainrag/
├── src/rainrag/          # Main application code
│   ├── api.py           # FastAPI endpoints
│   ├── cli.py           # CLI commands
│   ├── config.py        # Configuration management
│   ├── embed.py         # Embedding generation
│   ├── index.py         # Qdrant indexing
│   ├── ingest.py        # VTT file processing
│   └── query.py         # RAG query engine
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── fixtures/       # Test fixtures
├── docs/               # Documentation
└── .github/            # GitHub workflows
```

### Key Components

- **Ingestion:** Processes VTT subtitle files
- **Embedding:** Generates vector embeddings (local or API-based)
- **Indexing:** Stores vectors in Qdrant
- **Querying:** RAG pipeline with multiple LLM providers

---

## 🐛 Debugging

### Enable Debug Logging

```python
# In config.yaml
logging:
  level: "DEBUG"
```

### Run with Debugger

```bash
# Using Python debugger
uv run python -m pdb -m rainrag.cli ask "your question"

# Using IPython
uv run ipython
>>> from rainrag.query import RAGQueryEngine
>>> engine = RAGQueryEngine.from_config()
>>> %debug
```

### Common Issues

**Issue:** Tests fail with import errors
**Solution:** Run `uv sync` to ensure all dependencies are installed

**Issue:** Type checking shows many errors
**Solution:** This is expected. We're incrementally improving type coverage.

**Issue:** Pre-commit hooks are slow
**Solution:** You can skip mypy hook in `.pre-commit-config.yaml` if needed

---

## 📚 Additional Resources

- [Project Roadmap](../NEXT_STEPS.md)
- [Architecture Decisions](../docs/) (if available)
- [API Documentation](http://localhost:8000/docs) (when server is running)

---

## ❓ Questions?

- Open an issue on GitHub
- Check existing issues and discussions
- Review the [NEXT_STEPS.md](../NEXT_STEPS.md) roadmap

---

**Happy coding! 🎉**
