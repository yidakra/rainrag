# Build stage
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Set working directory
WORKDIR /app

# Copy Poetry configuration
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --without dev --no-interaction --no-ansi

# Runtime stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ /app/src/
COPY config.yaml /app/config.yaml
COPY README.md /app/README.md

# Create necessary directories
RUN mkdir -p /data/archive /data/rainrag /data/embeddings /data/logs

# Set Python path
ENV PYTHONPATH=/app/src

# Install the package
COPY pyproject.toml /app/
RUN pip install -e .

# Set the entrypoint
ENTRYPOINT ["rainrag"]

# Default command (can be overridden)
CMD ["--help"]
