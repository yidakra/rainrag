#!/bin/bash
# Run tests in Docker container to avoid local environment issues

set -e

echo "🏗️  Building test image..."
docker build -t rainrag:test -f Dockerfile .

echo "🧪 Running tests in Docker container..."
docker run --rm --entrypoint python rainrag:test -m pytest "$@"