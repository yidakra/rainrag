.PHONY: help install clean test test-unit test-integration test-cov format lint docker-build docker-push helm-install helm-uninstall qdrant-start qdrant-stop api streamlit up down api-bg streamlit-bg

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with Poetry
	poetry install

clean: ## Clean up generated files and caches
	rm -rf data/*.jsonl
	rm -rf embeddings/*.npy embeddings/*.jsonl
	rm -rf logs/*.log
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

test: ## Run all tests
	poetry run pytest

test-unit: ## Run unit tests only
	poetry run pytest tests/unit -v

test-integration: ## Run integration tests only
	poetry run pytest tests/integration -v

test-cov: ## Run tests with coverage report
	poetry run pytest --cov=src/rainrag --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

format: ## Format code with black
	poetry run black src/
	poetry run ruff check --fix src/

lint: ## Run linters
	poetry run ruff check src/
	poetry run mypy src/

# Docker commands
docker-build: ## Build Docker image
	docker build -t rainrag:latest .

docker-push: ## Push Docker image to registry (set REGISTRY variable)
	docker tag rainrag:latest $(REGISTRY)/rainrag:latest
	docker push $(REGISTRY)/rainrag:latest

# Local development
qdrant-start: ## Start local Qdrant instance
	docker run -d --name rainrag-qdrant \
		-p 6333:6333 -p 6334:6334 \
		-v $(PWD)/qdrant_storage:/qdrant/storage \
		qdrant/qdrant:v1.7.4

qdrant-stop: ## Stop local Qdrant instance
	docker stop rainrag-qdrant || true
	docker rm rainrag-qdrant || true

qdrant-logs: ## View Qdrant logs
	docker logs -f rainrag-qdrant

# Web frontend commands
api: ## Start FastAPI backend server
	@echo "Starting FastAPI backend at http://localhost:8001"
	@echo "API docs: http://localhost:8001/docs"
	poetry run python -m uvicorn rainrag.api:app --host 0.0.0.0 --port 8001 --reload

streamlit: ## Start Streamlit frontend
	@echo "Starting Streamlit frontend at http://localhost:7860"
	@echo "Note: Make sure API is running (make api) or set RAINRAG_API_URL"
	poetry run streamlit run app.py --server.address 0.0.0.0 --server.port 7860

up: qdrant-start api-bg streamlit-bg ## Start all services (Qdrant, API, Streamlit)
	@echo ""
	@echo "All services started:"
	@echo "  - Qdrant:    http://localhost:6333"
	@echo "  - API:       http://localhost:8001"
	@echo "  - Streamlit: http://localhost:7860"

down: qdrant-stop ## Stop all services
	@pkill -f "uvicorn rainrag.api" || true
	@pkill -f "streamlit run app.py" || true
	@echo "All services stopped"

api-bg: ## Start API in background
	@poetry run python -m uvicorn rainrag.api:app --host 0.0.0.0 --port 8001 > /tmp/rainrag-api.log 2>&1 &
	@sleep 2
	@echo "API started in background (logs: /tmp/rainrag-api.log)"

streamlit-bg: ## Start Streamlit in background
	@poetry run streamlit run app.py --server.address 0.0.0.0 --server.port 7860 > /tmp/rainrag-streamlit.log 2>&1 &
	@sleep 2
	@echo "Streamlit started in background (logs: /tmp/rainrag-streamlit.log)"

# CLI shortcuts
ingest: ## Run ingestion pipeline
	poetry run rainrag ingest

embed: ## Run embedding generation
	poetry run rainrag embed

index: ## Run indexing pipeline
	poetry run rainrag index

pipeline: ## Run full pipeline
	poetry run rainrag pipeline

info: ## Show system info
	poetry run rainrag info

# Helm commands
helm-install: ## Install Helm chart
	helm install rainrag ./helm/rainrag

helm-upgrade: ## Upgrade Helm deployment
	helm upgrade rainrag ./helm/rainrag

helm-uninstall: ## Uninstall Helm deployment
	helm uninstall rainrag

helm-lint: ## Lint Helm chart
	helm lint ./helm/rainrag

helm-template: ## Show rendered Helm templates
	helm template rainrag ./helm/rainrag

# Development setup
setup-dev: install qdrant-start ## Set up development environment
	@echo "Development environment ready!"
	@echo "Run 'poetry shell' to activate the virtual environment"
	@echo "Run 'make pipeline' to test the full pipeline"

# Cleanup everything
nuke: clean qdrant-stop ## Nuclear cleanup - remove all data and containers
	rm -rf qdrant_storage/
	@echo "All data and containers removed!"
