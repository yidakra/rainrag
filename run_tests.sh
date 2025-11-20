#!/bin/bash
# Quick script to run tests with proper environment

cd /home/user/rainrag

echo "Installing dependencies..."
poetry install --no-interaction

echo ""
echo "Running tests..."
poetry run pytest tests/ -v --tb=short

echo ""
echo "Test run complete!"
