.PHONY: help install clean test test-unit test-integration test-cov format lint docker-build docker-push helm-install helm-uninstall qdrant-start qdrant-stop api streamlit up down api-bg streamlit-bg mcp mcp-http mcp-inspector

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
			qdrant/qdrant:v1.16.3; \
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
	@echo "Configuration:"
	@LLM_PROVIDER=$$(grep -A3 "^llm:" config.yaml | grep "^  provider:" | awk '{print $$2}' | tr -d '"'); \
	LLM_MODEL=""; \
	if [ "$$LLM_PROVIDER" = "mistral" ]; then \
		LLM_MODEL=$$(grep -A5 "^mistral:" config.yaml | grep "^  model_name:" | head -1 | awk '{print $$2}' | tr -d '"'); \
	elif [ "$$LLM_PROVIDER" = "openai" ]; then \
		LLM_MODEL=$$(grep -A5 "^openai:" config.yaml | grep "^  model_name:" | head -1 | awk '{print $$2}' | tr -d '"'); \
	elif [ "$$LLM_PROVIDER" = "claude" ]; then \
		LLM_MODEL=$$(grep -A5 "^claude:" config.yaml | grep "^  model_name:" | head -1 | awk '{print $$2}' | tr -d '"'); \
	elif [ "$$LLM_PROVIDER" = "gemini" ]; then \
		LLM_MODEL=$$(grep -A5 "^gemini:" config.yaml | grep "^  model_name:" | head -1 | awk '{print $$2}' | tr -d '"'); \
	fi; \
	EMBED_PROVIDER=$$(grep -A5 "^embedding:" config.yaml | grep "^  provider:" | awk '{print $$2}' | tr -d '"'); \
	EMBED_MODEL=""; \
	if [ "$$EMBED_PROVIDER" = "local" ]; then \
		EMBED_MODEL=$$(grep -A5 "^embedding:" config.yaml | grep "^  model_name:" | awk '{print $$2}' | tr -d '"'); \
	elif [ "$$EMBED_PROVIDER" = "mistral" ]; then \
		EMBED_MODEL="mistral-embed"; \
	elif [ "$$EMBED_PROVIDER" = "openai" ]; then \
		EMBED_MODEL=$$(grep -A5 "^openai:" config.yaml | grep "^  embedding_model:" | awk '{print $$2}' | tr -d '"'); \
	elif [ "$$EMBED_PROVIDER" = "gemini" ]; then \
		EMBED_MODEL=$$(grep -A5 "^gemini:" config.yaml | grep "^  embedding_model:" | awk '{print $$2}' | tr -d '"'); \
	fi; \
	echo "  - LLM Provider:        $$LLM_PROVIDER ($$LLM_MODEL)"; \
	echo "  - Embedding Provider:  $$EMBED_PROVIDER ($$EMBED_MODEL)"
	@echo ""
	@echo "Logs:"
	@echo "  - API:       /tmp/rainrag-api.log"
	@echo "  - Streamlit: /tmp/rainrag-streamlit.log"
	@echo ""
	@echo "Note: Set MISTRAL_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or GOOGLE_API_KEY in .env file"

down: qdrant-stop ## Stop all services
	@pkill -f "[u]vicorn rainrag.api" || true
	@pkill -f "[s]treamlit run app.py" || true
	@echo "All services stopped"

api-bg: ## Start API in background
	@cd $(PWD) && set -a && [ -f .env ] && . ./.env || true && set +a && \
		poetry run python -m uvicorn rainrag.api:app --host 0.0.0.0 --port 8001 > /tmp/rainrag-api.log 2>&1 &
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
	@cd $(PWD) && set -a && [ -f .env ] && . ./.env || true && set +a && \
		poetry run streamlit run app.py --server.address 0.0.0.0 --server.port 7860 > /tmp/rainrag-streamlit.log 2>&1 &
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

# MCP Server commands
mcp: ## Run MCP server (default stdio transport for Claude Desktop/Cursor)
	@echo "Starting MCP server with stdio transport..."
	@echo "Connect from Claude Desktop or Cursor"
	poetry run rainrag mcp

mcp-http: ## Run MCP server with HTTP transport
	@echo "Starting MCP server at http://localhost:8000/mcp"
	@echo "Use for remote connections or ChatGPT integration"
	poetry run rainrag mcp --transport streamable-http --port 8000

mcp-inspector: ## Run MCP server with inspector for testing
	@echo "=== MCP Inspector Setup ==="
	@echo ""
	@echo "1. Install inspector (if not already installed):"
	@echo "   npm install -g @modelcontextprotocol/inspector"
	@echo ""
	@echo "2. Starting MCP server on http://localhost:8000/mcp"
	@echo ""
	@echo "3. In another terminal, run:"
	@echo "   npx @modelcontextprotocol/inspector"
	@echo ""
	@echo "4. Connect to: http://localhost:8000/mcp"
	@echo ""
	poetry run rainrag mcp --transport streamable-http --port 8000

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
