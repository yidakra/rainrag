# Plan: Migrate from Poetry to uv

## Overview

Migrate rainrag from **Poetry 1.7–1.8** to **uv** for faster dependency resolution, installs, and a simpler toolchain. uv replaces both Poetry and pip with a single Rust-based tool.

## Current State

- **Package manager**: Poetry (1.7.1 in Docker, latest/1.8.0 in CI)
- **Config files**: `pyproject.toml` (Poetry format), `poetry.toml`, `poetry.lock`
- **Dependency groups**: main (44 packages), `eval` (optional), `dev`
- **Entry points**: `rainrag` CLI, `rainrag-eval` CLI
- **Build backend**: `poetry.core.masonry.api`
- **Touchpoints**: Makefile (22 `poetry run` invocations), 2 Dockerfiles, 3 CI workflows, `poetry.toml`

## Migration Steps

### Step 1: Convert `pyproject.toml` from Poetry to PEP 621 + uv

**Changes to `pyproject.toml`:**

1. Replace `[tool.poetry]` metadata with standard `[project]` table:
   - `[tool.poetry] name/version/description/authors/readme` → `[project] name/version/description/authors/readme`
   - `[tool.poetry.dependencies] python = ">=3.10,<3.14"` → `[project] requires-python = ">=3.10,<3.14"`
   - Move all main dependencies to `[project] dependencies = [...]` using PEP 508 strings
   - `[tool.poetry.scripts]` → `[project.scripts]`

2. Convert dependency groups:
   - `[tool.poetry.group.eval.dependencies]` → `[project.optional-dependencies] eval = [...]`
   - `[tool.poetry.group.dev.dependencies]` → `[dependency-groups] dev = [...]` (PEP 735)

3. Update build system:
   - `requires = ["poetry-core"]` → `requires = ["hatchling"]`
   - `build-backend = "poetry.core.masonry.api"` → `build-backend = "hatchling.build"`
   - Add `[tool.hatch.build.targets.wheel] packages = ["src/rainrag", "src/rainrag_eval"]`

4. Translate dependency version specifiers:
   - Poetry `^X.Y.Z` → PEP 440 `>=X.Y.Z,<(X+1).0.0` (for major > 0)
   - Poetry `~X.Y.Z` → PEP 440 `~=X.Y.Z`
   - Extras syntax stays the same: `uvicorn[standard]>=0.31.1,<1.0.0`

**No changes** to `[tool.black]`, `[tool.ruff]`, `[tool.mypy]`, `[tool.bandit]` — these are tool configs, not Poetry-specific.

### Step 2: Generate `uv.lock` and delete Poetry files

1. Run `uv lock` to generate `uv.lock` (replaces `poetry.lock`)
2. Delete `poetry.lock`
3. Delete `poetry.toml` (uv uses `[tool.uv]` in `pyproject.toml` if needed)
4. Add to `pyproject.toml` if needed:
   ```toml
   [tool.uv]
   dev-dependencies = [...]  # if using uv-specific dev deps format
   ```

### Step 3: Update Makefile

Replace all `poetry run` with `uv run` and `poetry install` with `uv sync`:

| Before | After |
|--------|-------|
| `poetry install` | `uv sync` |
| `poetry install --with eval` | `uv sync --extra eval` |
| `poetry run pytest` | `uv run pytest` |
| `poetry run black src/` | `uv run black src/` |
| `poetry run ruff check src/` | `uv run ruff check src/` |
| `poetry run mypy src/` | `uv run mypy src/` |
| `poetry run python -m uvicorn ...` | `uv run python -m uvicorn ...` |
| `poetry run streamlit run ...` | `uv run streamlit run ...` |
| `poetry run rainrag ...` | `uv run rainrag ...` |

Also update `setup-dev` help text: remove `poetry shell` reference (uv doesn't use shell activation — `uv run` handles it).

### Step 4: Update Dockerfile (CPU)

```dockerfile
# Build stage
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y build-essential curl git && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev)
RUN uv sync --frozen --no-dev --no-install-project

# Runtime stage
FROM python:3.10-slim
RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY config.yaml pyproject.toml README.md /app/

RUN mkdir -p /data/archive /data/rainrag /data/embeddings /data/logs

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

ENTRYPOINT ["rainrag"]
CMD ["--help"]
```

Key differences:
- Uses `COPY --from=ghcr.io/astral-sh/uv:latest` (no pip install needed)
- Copies `.venv` directory instead of site-packages (cleaner isolation)
- No need for `pip install -e .` in runtime stage

### Step 5: Update Dockerfile.gpu

Same pattern as CPU Dockerfile but with `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` base and conda-aware paths.

### Step 6: Update CI Workflows

**`.github/workflows/test.yml`:**
- Replace `snok/install-poetry@v1` with `astral-sh/setup-uv@v4`
- Replace cache key from `poetry.lock` hash to `uv.lock` hash
- Cache path changes from `.venv` to uv's managed cache (`~/.cache/uv`)
- `poetry install` → `uv sync`
- `poetry run pytest` → `uv run pytest`

```yaml
- name: Install uv
  uses: astral-sh/setup-uv@v4

- name: Set up Python ${{ matrix.python-version }}
  run: uv python install ${{ matrix.python-version }}

- name: Install dependencies
  run: uv sync --frozen

- name: Run tests
  run: uv run pytest -v --tb=short --cov=src/rainrag --cov-report=xml --cov-report=term
```

**`.github/workflows/eval-tests.yml`:**
- Same uv setup
- `poetry install --with eval` → `uv sync --frozen --extra eval`
- `poetry run pytest` → `uv run pytest`

**`.github/workflows/lint.yml`:**
- This workflow already uses `pip install` directly for ruff and mypy — can optionally switch to `uv pip install` or `uvx` for tool execution, but this is low priority since it works fine as-is.

### Step 7: Update `.pre-commit-config.yaml`

No changes needed — pre-commit hooks use their own isolated environments and don't depend on Poetry.

### Step 8: Update `.gitignore`

- Ensure `uv.lock` is **not** in `.gitignore` (it should be committed)
- Remove any Poetry-specific ignores if present

### Step 9: Update documentation references

- Update any README or docs that mention `poetry install`, `poetry run`, `poetry shell`
- Update setup instructions to reference `uv sync` and `uv run`

## Files Modified (Summary)

| File | Change |
|------|--------|
| `pyproject.toml` | Rewrite metadata to PEP 621, change build backend to hatchling |
| `poetry.toml` | **Delete** |
| `poetry.lock` | **Delete** (replaced by `uv.lock`) |
| `uv.lock` | **New** (auto-generated by `uv lock`) |
| `Makefile` | `poetry run` → `uv run`, `poetry install` → `uv sync` |
| `Dockerfile` | Replace Poetry with uv, copy .venv instead of site-packages |
| `Dockerfile.gpu` | Same as Dockerfile |
| `.github/workflows/test.yml` | Replace Poetry action with uv action |
| `.github/workflows/eval-tests.yml` | Replace Poetry action with uv action |
| `.github/workflows/lint.yml` | Optional: use `uvx` for tool installs |
| `README.md` | Update install/usage instructions |

## Risks and Mitigations

1. **Dependency resolution differences**: uv's resolver may produce a different lockfile. Run tests after migration to verify.
2. **PyTorch index**: torch often needs a custom index URL. May need `[tool.uv.sources]` or `[[tool.uv.index]]` to configure PyTorch's CPU/GPU wheel index.
3. **`~=` vs `^` semantics**: Careful translation needed for `ragas ~0.1.0` → `ragas>=0.1.0,<0.2.0` and `rouge-score ~0.1.2` → `rouge-score>=0.1.2,<0.2.0`.
4. **hatchling build backend**: Need to verify `packages = ["src/rainrag", "src/rainrag_eval"]` produces the same wheel layout as poetry-core.

## Execution Order

1. Convert `pyproject.toml` (Step 1)
2. Generate lock and remove Poetry files (Step 2)
3. Verify `uv sync && uv run pytest` passes locally
4. Update Makefile (Step 3)
5. Update Dockerfiles (Steps 4–5)
6. Update CI workflows (Step 6)
7. Update docs (Step 9)
8. Final verification: full test suite, Docker build, lint
