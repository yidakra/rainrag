# Build stage
FROM python:3.10-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev, no project itself yet)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src/ /app/src/
COPY config.yaml /app/config.yaml
COPY README.md /app/README.md

# Install the project itself
RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code and config
COPY src/ /app/src/
COPY config.yaml /app/config.yaml
COPY README.md /app/README.md
COPY pyproject.toml /app/

# Create necessary directories
RUN mkdir -p /data/archive /data/rainrag /data/embeddings /data/logs

# Use the venv Python
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

# Set the entrypoint
ENTRYPOINT ["rainrag"]

# Default command (can be overridden)
CMD ["--help"]
