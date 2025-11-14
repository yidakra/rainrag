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

up: qdrant-start vllm-start api-bg streamlit-bg ## Start all services (Qdrant, vLLM, API, Streamlit)
	@echo ""
	@echo "=== All Services Started ==="
	@echo ""
	@echo "Infrastructure:"
	@echo "  - Qdrant:              http://localhost:6333"
	@echo ""
	@echo "LLM Models (vLLM):"
	@echo "  - Mistral Small 24B:   http://localhost:8000"
	@echo "  - Gemma 2 27B:         http://localhost:8002"
	@echo "  - GPT-OSS 20B:         http://localhost:8003"
	@echo ""
	@echo "Application:"
	@echo "  - API:                 http://localhost:8001 (docs: /docs)"
	@echo "  - Streamlit UI:        http://localhost:7860"
	@echo ""
	@echo "Logs:"
	@echo "  - API:       /tmp/rainrag-api.log"
	@echo "  - Streamlit: /tmp/rainrag-streamlit.log"
	@echo "  - vLLM:      Use 'make vllm-logs' to view all model logs"
	@echo ""
	@echo "Switch between models seamlessly in the Streamlit UI!"

down: vllm-stop qdrant-stop ## Stop all services
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

# vLLM model servers
vllm-mistral: ## Start Mistral vLLM server (foreground)
	@echo "Starting Mistral Small 3.2 24B on port 8000"
	poetry run python -m vllm.entrypoints.openai.api_server \
		--model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
		--host 0.0.0.0 \
		--port 8000 \
		--dtype auto

vllm-gemma: ## Start Gemma vLLM server (foreground)
	@echo "Starting Gemma 2 27B on port 8002"
	poetry run python -m vllm.entrypoints.openai.api_server \
		--model google/gemma-2-27b-it \
		--host 0.0.0.0 \
		--port 8002 \
		--dtype auto

vllm-gptoss: ## Start GPT-OSS vLLM server (foreground)
	@echo "Starting GPT-OSS 20B on port 8003"
	poetry run python -m vllm.entrypoints.openai.api_server \
		--model gpt-oss:20b \
		--host 0.0.0.0 \
		--port 8003 \
		--dtype auto

vllm-mistral-bg: ## Start Mistral vLLM server in background
	@poetry run python -m vllm.entrypoints.openai.api_server \
		--model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
		--host 0.0.0.0 \
		--port 8000 \
		--dtype auto > /tmp/rainrag-vllm-mistral.log 2>&1 &
	@echo "Mistral vLLM started on port 8000 (logs: /tmp/rainrag-vllm-mistral.log)"

vllm-gemma-bg: ## Start Gemma vLLM server in background
	@poetry run python -m vllm.entrypoints.openai.api_server \
		--model google/gemma-2-27b-it \
		--host 0.0.0.0 \
		--port 8002 \
		--dtype auto > /tmp/rainrag-vllm-gemma.log 2>&1 &
	@echo "Gemma vLLM started on port 8002 (logs: /tmp/rainrag-vllm-gemma.log)"

vllm-gptoss-bg: ## Start GPT-OSS vLLM server in background
	@poetry run python -m vllm.entrypoints.openai.api_server \
		--model gpt-oss:20b \
		--host 0.0.0.0 \
		--port 8003 \
		--dtype auto > /tmp/rainrag-vllm-gptoss.log 2>&1 &
	@echo "GPT-OSS vLLM started on port 8003 (logs: /tmp/rainrag-vllm-gptoss.log)"

vllm-start: vllm-mistral-bg vllm-gemma-bg vllm-gptoss-bg ## Start all 3 vLLM servers in background
	@sleep 5
	@echo ""
	@echo "All vLLM servers started:"
	@echo "  - Mistral Small 3.2 24B: http://localhost:8000"
	@echo "  - Gemma 2 27B:           http://localhost:8002"
	@echo "  - GPT-OSS 20B:           http://localhost:8003"
	@echo ""
	@echo "Logs available at:"
	@echo "  - /tmp/rainrag-vllm-mistral.log"
	@echo "  - /tmp/rainrag-vllm-gemma.log"
	@echo "  - /tmp/rainrag-vllm-gptoss.log"

vllm-stop: ## Stop all vLLM servers
	@pkill -f "vllm.entrypoints.openai.api_server" || true
	@echo "All vLLM servers stopped"

vllm-logs: ## Show logs from all vLLM servers
	@echo "=== Mistral Logs ==="
	@tail -20 /tmp/rainrag-vllm-mistral.log 2>/dev/null || echo "No Mistral logs found"
	@echo ""
	@echo "=== Gemma Logs ==="
	@tail -20 /tmp/rainrag-vllm-gemma.log 2>/dev/null || echo "No Gemma logs found"
	@echo ""
	@echo "=== GPT-OSS Logs ==="
	@tail -20 /tmp/rainrag-vllm-gptoss.log 2>/dev/null || echo "No GPT-OSS logs found"

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
