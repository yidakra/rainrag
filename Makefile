.PHONY: help install clean test test-unit test-integration test-cov format lint docker-build docker-push helm-install helm-uninstall qdrant-start qdrant-stop api streamlit up down api-bg streamlit-bg vllm-mistral vllm-gemma vllm-gptoss vllm-mistral-bg vllm-gemma-bg vllm-gptoss-bg vllm-start vllm-stop vllm-logs

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
	@if docker ps -a --format '{{.Names}}' | grep -q '^rainrag-qdrant$$'; then \
		if docker ps --format '{{.Names}}' | grep -q '^rainrag-qdrant$$'; then \
			echo "Qdrant is already running"; \
		else \
			echo "Starting existing Qdrant container"; \
			docker start rainrag-qdrant; \
		fi \
	else \
		echo "Creating and starting Qdrant container"; \
		docker run -d --name rainrag-qdrant \
			-p 6333:6333 -p 6334:6334 \
			-v $(PWD)/qdrant_storage:/qdrant/storage \
			qdrant/qdrant:v1.12.1; \
	fi
	@echo "Waiting for Qdrant to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -s http://localhost:6333/readyz > /dev/null 2>&1; then \
			echo "Qdrant is ready"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "Warning: Qdrant may not be ready yet"; \
		fi; \
		sleep 1; \
	done

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

up: qdrant-start ## Start all services (Qdrant, API, Streamlit)
	@echo "Starting API and Streamlit..."
	@$(MAKE) api-bg
	@$(MAKE) streamlit-bg
	@echo ""
	@echo "=== All Services Started ==="
	@echo ""
	@echo "Infrastructure:"
	@echo "  - Qdrant:              http://localhost:6333"
	@echo ""
	@echo "Application:"
	@echo "  - API:                 http://localhost:8001 (docs: /docs)"
	@echo "  - Streamlit UI:        http://localhost:7860"
	@echo ""
	@echo "Logs:"
	@echo "  - API:       /tmp/rainrag-api.log"
	@echo "  - Streamlit: /tmp/rainrag-streamlit.log"
	@echo ""
	@echo "Note: Set MISTRAL_API_KEY environment variable or in config.yaml"

down: qdrant-stop ## Stop all services
	@pkill -f "[u]vicorn rainrag.api" || true
	@pkill -f "[s]treamlit run app.py" || true
	@echo "All services stopped"

api-bg: ## Start API in background
	@cd $(PWD) && poetry run python -m uvicorn rainrag.api:app --host 0.0.0.0 --port 8001 > /tmp/rainrag-api.log 2>&1 &
	@echo "Waiting for API to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -s http://localhost:8001/health > /dev/null 2>&1; then \
			echo "API is ready"; \
			break; \
		fi; \
		if [ $$i -eq 30 ]; then \
			echo "Warning: API may not be ready yet"; \
		fi; \
		sleep 1; \
	done
	@if pgrep -f "uvicorn rainrag.api" > /dev/null; then \
		echo "API started in background (logs: /tmp/rainrag-api.log)"; \
	else \
		echo "API failed to start. Check /tmp/rainrag-api.log"; \
		exit 1; \
	fi

streamlit-bg: ## Start Streamlit in background
	@cd $(PWD) && poetry run streamlit run app.py --server.address 0.0.0.0 --server.port 7860 > /tmp/rainrag-streamlit.log 2>&1 &
	@sleep 3
	@if pgrep -f "streamlit run app.py" > /dev/null; then \
		echo "Streamlit started in background (logs: /tmp/rainrag-streamlit.log)"; \
	else \
		echo "Streamlit failed to start. Check /tmp/rainrag-streamlit.log"; \
		exit 1; \
	fi


# Model management
download-models: ## Download and cache required models (requires internet)
	@echo "Downloading embedding models..."
	@echo "This requires internet access and may take several minutes"
	poetry run python scripts/download_models.py

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
