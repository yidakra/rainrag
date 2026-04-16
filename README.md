# RainRAG

**Retrieval-Augmented Generation (RAG) Pipeline for VTT Subtitle Processing**

RainRAG is a modular, open-source backend system for building a semantic search engine over VTT subtitle files. It uses state-of-the-art multilingual embeddings and vector search to enable efficient retrieval of broadcast transcripts in Russian and English. Supports multiple LLM providers including Mistral AI, OpenAI, Anthropic Claude, and Google Gemini.

## Features

- **Multilingual**: Supports Russian and English subtitles with `intfloat/multilingual-e5-large` embeddings
- **Modular Pipeline**: Separate stages for ingestion, embedding, and indexing
- **Production Ready**: Includes Helm charts for Kubernetes deployment
- **Type-safe**: Fully type-annotated Python code with Pydantic models
- **CLI Interface**: Easy-to-use command-line interface powered by Typer
- **Vector Search**: Uses Qdrant for efficient similarity search
- **Web UI**: Streamlit-based chat interface with FastAPI backend
- **MCP Server**: Deploy as Model Context Protocol server for Claude Desktop, ChatGPT, and Cursor integration
- **Video Playback**: Inline video player for retrieved content
- **Subtitle Access**: Download and view VTT files directly in the UI
- **Network Access**: Accessible from other devices on the same network with optional token authentication
- **Multi-Provider LLM**: Choose from Mistral AI, OpenAI (GPT-4/ChatGPT), Anthropic Claude, or Google Gemini
- **Flexible Embeddings**: Local model or API-based (Mistral, OpenAI, Gemini)
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching using Reciprocal Rank Fusion
- **Temporal Context**: Automatic detection of "recent" queries with time-decay boosting for relevant results
- **Web Metadata**: Enrich transcripts with titles, accurate dates, descriptions, and URLs from web sources
- **Reranking**: Cohere Rerank API integration for 10-15% accuracy improvement
- **Related Chunks**: Discover similar content and explore related video segments
- **Chunk Overlap**: 30-second overlaps between chunks to prevent information loss at boundaries

## Architecture

### Data Pipeline

```
┌─────────────────┐
│  VTT Files      │
│  (Archive)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Ingestion      │ Parse & clean VTT files
│  (ingest.py)    │ Detect language
│                 │ Load web metadata (optional)
└────────┬────────┘ Output: docs.jsonl
         │
         ▼
┌─────────────────┐
│  Embedding      │ Load multilingual-e5-large
│  (embed.py)     │ Generate embeddings
└────────┬────────┘ Cache: embeddings/*.npy
         │
         ▼
┌─────────────────┐
│  Indexing       │ Connect to Qdrant
│  (index.py)     │ Create collection
└────────┬────────┘ Index vectors + metadata
         │
         ▼
┌─────────────────┐
│  Qdrant         │ Vector Database
│  (Local)        │ Semantic Search
└─────────────────┘
```

### Query Architecture (Web UI)

```
┌─────────────────┐
│   Streamlit     │ Chat Interface
│   Frontend      │ (Port 7860)
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│   FastAPI       │ Query API
│   Backend       │ (Port 8001)
└────────┬────────┘
         │
         ├──────────┐
         │          │
         ▼          ▼
   ┌─────────┐  ┌──────────────────────┐
   │ Qdrant  │  │   LLM Providers      │
   │ Search  │  │ ┌──────────────────┐ │
   └─────────┘  │ │ Mistral AI       │ │
                │ │ OpenAI/ChatGPT   │ │
                │ │ Anthropic Claude │ │
                │ │ Google Gemini    │ │
                │ └──────────────────┘ │
                └──────────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
           Generated Answer
           + Context Chunks
```

## Quick Start

*Note: backup and restore instructions have been moved later in this document under the **Troubleshooting** section.*

Create a `.env` file by copying the template: `cp .env.example .env` (you can then edit it).

Required R2 variables are configured via `.env` (see `.env.example`):

- `R2_ACCOUNT_ID`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`

Optional:

- `R2_PREFIX` (defaults to `embeddings`)
- `R2_QDRANT_PREFIX` (defaults to `qdrant`)
- `QDRANT_URL` (defaults to `http://localhost:6333`)
- `QDRANT_COLLECTION` (defaults to `broadcast_transcripts`)
- `QDRANT_SNAPSHOT_NAME` (restore a specific snapshot)

### Prerequisites

- Python 3.10+
- uv (for dependency management)
- Docker (optional, for containerized deployment)
- Kubernetes/Minikube (optional, for Helm deployment)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/rainrag.git
cd rainrag
```

2. **Install dependencies with uv**

```bash
uv sync
```

3. **Run commands in the project environment**

```bash
# uv automatically manages the virtual environment per command
uv run rainrag --help
```

4. **Download required models** (only needed if using local embeddings)

```bash
# Download and cache the embedding model
make download-models

# Or manually:
uv run python scripts/download_models.py
```

**Note:** This step downloads the `multilingual-e5-large` embedding model (~2GB) and caches it locally. This is only required if you're using `provider: "local"` in your embedding configuration. If you're using `provider: "mistral"` for Mistral API embeddings, you can skip this step.

### Configuration

1. **Set up API Keys**

RainRAG supports multiple LLM and embedding providers. Set up the provider(s) you want to use:

```bash
# Mistral AI (default recommended)
export MISTRAL_API_KEY=your_mistral_key

# OpenAI (for GPT-4, ChatGPT, or embeddings)
export OPENAI_API_KEY=your_openai_key

# Anthropic Claude
export ANTHROPIC_API_KEY=your_claude_key

# Google Gemini
export GOOGLE_API_KEY=your_gemini_key

# Cohere (reranker)
export COHERE_API_KEY=your_cohere_key
```

Or add them to your `.env` file:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Getting API Keys:**
- **Mistral**: [console.mistral.ai](https://console.mistral.ai/) - See [docs/MISTRAL_SETUP.md](docs/MISTRAL_SETUP.md)
- **OpenAI**: [platform.openai.com](https://platform.openai.com/) - See [docs/OPENAI_SETUP.md](docs/OPENAI_SETUP.md)
- **Claude**: [console.anthropic.com](https://console.anthropic.com/) - See [docs/CLAUDE_SETUP.md](docs/CLAUDE_SETUP.md)
- **Gemini**: [makersuite.google.com](https://makersuite.google.com/) - See [docs/GEMINI_SETUP.md](docs/GEMINI_SETUP.md)

2. **Edit `config.yaml`** to customize paths and select your providers:

```yaml
paths:
  archive_root: "/path/to/your/vtt/files"
  docs_output: "./data/docs.jsonl"
  embeddings_cache: "./embeddings"

# Embedding configuration
embedding:
  provider: "mistral"  # Options: "local", "mistral", "openai", "gemini"
  model_name: "intfloat/multilingual-e5-large"
  # Optional prefix added to texts before embedding (e.g. "passage: " for E5).
  # Leave empty to auto-detect from the model name.
  prefix: ""
  device: "cuda"  # or "cpu" (only used with local provider)
  batch_size: 32

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "broadcast_transcripts"

# LLM provider selection
llm:
  provider: "mistral"  # Options: "mistral", "openai", "claude", "gemini"

# Mistral AI configuration
mistral:
  api_key: ""  # Leave empty to use MISTRAL_API_KEY env var
  model_name: "mistral-small-latest"
  max_tokens: 512
  temperature: 0.3

# OpenAI configuration
openai:
  api_key: ""  # Leave empty to use OPENAI_API_KEY env var
  model_name: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  max_tokens: 512
  temperature: 0.3

# Anthropic Claude configuration
claude:
  api_key: ""  # Leave empty to use ANTHROPIC_API_KEY env var
  model_name: "claude-haiku-4-5-20251001"
  max_tokens: 512
  temperature: 0.3

# Google Gemini configuration
gemini:
  api_key: ""  # Leave empty to use GOOGLE_API_KEY env var
  model_name: "gemini-2.5-flash"
  embedding_model: "models/text-embedding-004"
  max_tokens: 512
  temperature: 0.3
```

**See [docs/PROVIDER_COMPARISON.md](docs/PROVIDER_COMPARISON.md) for help choosing the right provider for your needs.**

### Two-Stage Retrieval

RainRAG implements two-stage retrieval (Zhai & Lafferty, [SIGIR 2002](https://dl.acm.org/doi/10.1145/564376.564386)) to improve recall on broadcast-transcript corpora, where user queries are typically formal or terse but the source material is informal spoken language.

**Stage 1 – Corpus smoothing**: handled by `hybrid_search` (BM25 + vector).

**Stage 2 – Query-side smoothing**: configured under `two_stage` in `config.yaml`.

```yaml
two_stage:
  enabled: true          # Master switch

  # 2a – LLM query rewriting
  # Rewrites the user query into transcript-register variants before retrieval.
  # Addresses vocabulary mismatch between formal queries and spoken transcripts.
  query_rewrite_enabled: true
  query_rewrite_variants: 2        # Rewrites generated; original is always included (3 total)
  query_rewrite_temperature: 0.7   # Higher → more diverse paraphrases

  # 2b – HyDE (Hypothetical Document Embedding)
  # Generates a hypothetical transcript passage and blends its embedding with the query.
  # More expensive (1 extra LLM call + 1 embed). Enable for highest recall.
  hyde_enabled: false
  hyde_alpha: 0.5        # 0.0 = raw query only, 1.0 = HyDE only
  hyde_temperature: 0.7  # Higher → more varied hypothetical passages
```

**Temperature design note:** `query_rewrite_temperature` and `hyde_temperature` are intentionally separate from the provider `temperature` setting used for final answer generation. Answer generation uses a low temperature (e.g., 0.3) for deterministic, source-grounded journalist output. The rewrite and HyDE calls use a higher temperature (default 0.7) to produce meaningfully diverse paraphrases and hypothetical passages — which is the whole point of these techniques.

### Choosing an Embedding Provider

RainRAG supports four embedding providers:

**Local Embeddings (`provider: "local"`)**
- Uses `intfloat/multilingual-e5-large` model running locally
- Requires downloading ~2GB model (one-time setup)
- Requires GPU/CPU resources to run the model
- Free to use (no API costs)
- 1024 dimensions
- Best for: High-volume queries, air-gapped environments, or when you have sufficient compute resources

**Mistral API Embeddings (`provider: "mistral"`)** - **Recommended for most users**
- Uses Mistral's `mistral-embed` model via API
- No local model download required
- Minimal compute resources needed
- Requires Mistral API key and incurs API costs
- 1024 dimensions
- Best for: Quick setup, limited compute resources, or testing

**OpenAI API Embeddings (`provider: "openai"`)**
- Uses OpenAI's `text-embedding-3-small` or `text-embedding-3-large`
- No local model download required
- Minimal compute resources needed
- Requires OpenAI API key and incurs API costs
- 1536 dimensions (small) or 3072 dimensions (large)
- Best for: Integration with existing OpenAI workflows

**Google Gemini API Embeddings (`provider: "gemini"`)**
- Uses Gemini's `models/text-embedding-004`
- No local model download required
- Minimal compute resources needed
- Requires Google API key and incurs API costs
- 768 dimensions
- Best for: Cost-effective embedding generation

To change embedding providers, update the `provider` field in your `config.yaml` embedding section and make sure the `qdrant.vector_size` matches the embedding dimensions.

**Important:** If you switch embedding providers after indexing, you must re-run the entire pipeline (`rainrag embed` and `rainrag index`) because the embedding dimensions differ between providers.

### Running Qdrant Locally

Start a local Qdrant instance using Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.16.3
```

## Usage

### CLI Commands

RainRAG provides a comprehensive CLI interface:

#### 1. Ingest VTT Files

Parse and clean VTT files from your archive:

```bash
rainrag ingest
```

This will:
- Recursively find all `.vtt` files
- Extract and clean transcript text
- Detect language (ru/en) from file paths
- Save to `data/docs.jsonl`

#### 2. Generate Embeddings

Create vector embeddings for all documents:

```bash
rainrag embed
```

This will:
- Load the multilingual-e5-large model
- Generate embeddings for all documents
- Cache embeddings locally in `embeddings/`

To force regeneration (ignore cache):

```bash
rainrag embed --force
```

#### 3. Index into Qdrant

Upload embeddings to the vector database:

```bash
rainrag index
```

To recreate the collection (delete existing data):

```bash
rainrag index --recreate
```

#### 4. Run Full Pipeline

Execute all three steps in sequence:

```bash
rainrag pipeline
```

Options:
- `--skip-ingest`: Skip the ingestion step
- `--skip-embed`: Skip the embedding step
- `--recreate-index`: Recreate the Qdrant collection

Example:

```bash
# Run full pipeline from scratch
rainrag pipeline --recreate-index

# Only embed and index (if ingestion already done)
rainrag pipeline --skip-ingest
```

#### 5. Query from CLI

Ask questions directly from the command line:

```bash
rainrag ask "What topics were discussed in the latest episode?"
```

Options:
- `--language`: Response language (`en` or `ru`, default: `en`)
- `--top-k`: Number of context documents to retrieve (default: 5)
- `--verbose`: Show retrieved context documents

Examples:

```bash
# English query
rainrag ask "Explain the main points about energy policy"

# Russian query
rainrag ask "О чём говорили в выпуске про энергетику?" --language ru

# Retrieve more context
rainrag ask "What was discussed about AI?" --top-k 10

# Show context documents
rainrag ask "Tell me about the interview" --verbose
```

#### 6. View System Info

Display configuration and collection statistics:

```bash
rainrag info
```

#### 7. Run MCP Server

Deploy RainRAG as an MCP (Model Context Protocol) server for integration with AI assistants like Claude Desktop, ChatGPT, and Cursor:

```bash
# Run with default settings (stdio transport for local tools)
rainrag mcp

# Run with HTTP transport for remote connections
rainrag mcp --transport streamable-http --port 8000
```

The MCP server exposes your RAG system as tools that AI assistants can use:
- `query_rag`: Full RAG pipeline (retrieve + generate answer)
- `retrieve_documents`: Retrieval only (no LLM generation)

**Setup with Claude Desktop:**

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "rainrag": {
      "command": "rainrag",
      "args": ["mcp", "--config", "/absolute/path/to/config.yaml"]
    }
  }
}
```

Restart Claude Desktop and you can now query your video transcripts directly from Claude!

**For detailed setup instructions** including Cursor and ChatGPT integration, see [docs/MCP_SETUP.md](docs/MCP_SETUP.md).

### Python API

You can also use RainRAG as a Python library:

```python
from rainrag.config import load_config
from rainrag.ingest import Ingester
from rainrag.embed import Embedder
from rainrag.index import QdrantIndexer

# Load configuration
config = load_config("config.yaml")

# Run ingestion
ingester = Ingester(config)
doc_count = ingester.ingest()

# Generate embeddings
embedder = Embedder(config)
embeddings, documents = embedder.embed()

# Index into Qdrant
indexer = QdrantIndexer(config)
indexer.connect()
indexer.create_collection()
indexer.index_documents(embeddings, documents)

# Search
query_embedding = embedder.model.encode("query: your search text")
results = indexer.search(query_embedding, top_k=5)
```

## Web Interface

RainRAG includes a complete web-based query interface with:
- **Streamlit Frontend**: Modern chat-style UI for asking questions
- **FastAPI Backend**: RESTful API for query processing
- **Multilingual Support**: Interface available in Russian and English
- **Network Access**: Accessible over LAN or Internet via DNS/reverse proxy (for example `https://rag.tvrain.tv`)

### Prerequisites

- Qdrant running locally (`make qdrant-start` starts the bundled container)
- API keys set for your chosen providers (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)
- `config.yaml` updated with the desired `llm.provider` and `embedding.provider`

Start everything locally:
```bash
make up
```
This brings up Qdrant, the FastAPI backend on port 8001, and the Streamlit UI on port 7860. Use `make down` to stop.

### Quick Start (Local Development)

#### Option 1: Using Make (Recommended)

Start all services with a single command:

```bash
# Start Qdrant, API, and Streamlit
make up

# Services will be available at:
# - Qdrant:    http://localhost:6333
# - API:       http://localhost:8001
# - Streamlit: http://localhost:7860
```

Stop all services:

```bash
make down
```

#### Option 2: Manual Start

1. **Start Qdrant**:
```bash
make qdrant-start
```

2. **Start the FastAPI backend** (in a new terminal):
```bash
make api
# API will be available at http://localhost:8001
# API docs at http://localhost:8001/docs
```

3. **Start the Streamlit frontend** (in another terminal):
```bash
make streamlit
# Frontend will be available at http://localhost:7860
```

#### Option 3: Using Docker Compose

```bash
# Build the image first (CPU)
docker buildx build --load -t rainrag:latest -f Dockerfile .

# Build the image first (GPU)
docker buildx build --load -t rainrag:gpu -f Dockerfile.gpu .

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

#
# NOTE: The compose file expects a `./secrets` directory containing API key files.
# Create the directory and key files before running `docker-compose up`.
#
#   mkdir -p ./secrets
#   make secrets   # or ./create-secrets.sh
#
# Then edit the generated files (e.g. `./secrets/mistral_api_key.txt`) and
# populate them with your provider API keys.
#
# For missing keys
#   - `./secrets/mistral_api_key.txt` and `./secrets/cohere_api_key.txt` are
#     required for Mistral/Cohere provider use.
#   - Either create and populate these files, or comment out the corresponding
#     `mistral_api_key`/`cohere_api_key` secret section in `docker-compose.yaml`.
#   - Other provider keys are optional and only needed if configured.
#
# Default providers (used out of the box):
#   - Mistral (LLM)
#   - Cohere (reranking)
#
# Optional providers (only needed if you configure them in `config.yaml`):
#   - OpenAI (GPT / embeddings)
#   - Anthropic (Claude)
#   - Google Gemini
#
# For safety, keep the secrets directory out of source control:
#   - The repo includes `secrets/*.txt.example` templates you can copy
#   - Your real secret files should be named `*.txt` and are ignored by git
#
# Recommended permissions (production):
#   chmod 600 ./secrets/*.txt
#
# Docker Compose mounts these as secrets and sets corresponding
# *_API_KEY_FILE environment variables in the containers.
```

### Network Access Configuration

To make the Streamlit interface accessible from other devices on your network:

1. **Find your machine's IP address**:

```bash
# Linux/macOS
ip addr show | grep "inet "
# or
ifconfig | grep "inet "

# Example output: 192.168.1.100
```

2. **Configure firewall** (if needed):

```bash
# Linux (ufw)
sudo ufw allow 7860/tcp
sudo ufw allow 8001/tcp

# macOS
# Go to System Preferences > Security & Privacy > Firewall > Firewall Options
# Allow incoming connections for the ports
```

3. **Access from another device**:

Open a browser on any device on the same network and navigate to:
```
http://192.168.1.100:7860
```
(Replace `192.168.1.100` with your machine's IP)

### Authentication Setup

To enable token-based authentication:

1. **Create a `.env` file**:

```bash
cp .env.example .env
```

2. **Edit `.env` and set your token**:

```bash
RAINRAG_AUTH_TOKEN=your_secret_token_here
STREAMLIT_AUTH_TOKEN=your_secret_token_here
```

3. **Start services with authentication**:

```bash
# Load environment variables
export $(cat .env | xargs)

# Start services
make up
```

Users will be prompted to enter the token when accessing the Streamlit interface.

### Using the Web UI

#### Sidebar Controls

1. **Language Selection**: Switch between Russian (Русский 🇷🇺) and English (English 🇬🇧)
2. **Context Chunks**: Adjust how many relevant transcript chunks to retrieve (1-10)
3. **Model Selection**: Seamlessly switch between LLM models:
   - Mistral Small 3.2 24B (Port 8000)
   - Gemma 2 27B (Port 8002)
   - GPT-OSS 20B (Port 8003)
   - Changes apply instantly - each model runs on its own vLLM instance
   - See [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) for running all 3 models simultaneously

#### Asking Questions

- Type your question in Russian or English in the chat input
- Questions can be about any topic covered in the indexed video transcripts

#### Viewing Results

- **Answer**: The assistant's response appears in a message bubble
- **Context**: Expand "Retrieved Context Chunks" to see source material
- **Video Player**: Videos display in a 2/3-width player on the left
- **Subtitles**: VTT files appear in a scrollable viewer on the right (1/3 width)
  - Language selector for videos with multiple subtitle languages (en/ru)
  - Download button to save VTT files locally
- **Date filters**: Optional date range filter (the picker is constrained to the archive date span from `RAINRAG_DOCS_PATH` / `./data/docs.jsonl`)
- **Metadata**: Each chunk displays filename, relevance score, language, and timecodes (when available)

#### Example Queries (Russian)

```
Какие темы обсуждались в последних выпусках?
Расскажи о политических новостях
Что говорили о экономике?
```

#### Example Queries (English)

```
What topics were discussed in recent episodes?
Tell me about political news
What was said about the economy?
```

### API Usage

The FastAPI backend can also be used directly:

#### Check API Health

```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "model_loaded": true,
  "vllm_model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
  "qdrant_collection": "broadcast_transcripts"
}
```

#### Submit a Query

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What topics were discussed?",
    "language": "en",
    "top_k": 3
  }'
```

Response:
```json
{
  "answer": "Based on the transcripts...",
  "context": [
    {
      "text": "...",
      "filename": "episode_001.vtt",
      "language": "en",
      "score": 0.89,
      "rank": 1,
      "video_url": "/video/path/to/episode_001.mp4",
      "vtt_url": "/vtt/path/to/episode_001.vtt"
    }
  ],
  "question": "What topics were discussed?",
  "num_documents": 3
}
```

#### With Authentication

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token_here" \
  -d '{
    "question": "Какие темы обсуждались?",
    "language": "ru",
    "top_k": 5
  }'
```

#### Serve Video Files

```bash
curl http://localhost:8001/video/path/to/episode.mp4 -o episode.mp4
```

#### Serve VTT Files

```bash
curl http://localhost:8001/vtt/path/to/episode.vtt -o episode.vtt
```

### Advanced Configuration

#### Custom API URL

If running the API on a different machine:

```bash
export RAINRAG_API_URL=http://192.168.1.200:8001
streamlit run app.py --server.address 0.0.0.0 --server.port 7860
```

#### Reverse Proxy Setup (HTTPS)

For production deployment with HTTPS, use Nginx or Caddy:

**Caddyfile example**:
```
rainrag.yourdomain.com {
    reverse_proxy localhost:7860
}

api.rainrag.yourdomain.com {
    reverse_proxy localhost:8001
}
```

**Nginx example**:
```nginx
server {
    listen 80;
    server_name rainrag.yourdomain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

server {
    listen 80;
    server_name api.rainrag.yourdomain.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
    }
}
```

## Docker Deployment

### Build the Docker Image

```bash
# CPU image
docker buildx build --load -t rainrag:latest -f Dockerfile .

# GPU image
docker buildx build --load -t rainrag:gpu -f Dockerfile.gpu .
```

### Run with Docker

```bash
# Run ingestion
docker run --rm \
  -v /path/to/vtt/files:/data/archive \
  -v $(pwd)/data:/data/rainrag \
  -v $(pwd)/embeddings:/data/embeddings \
  rainrag:latest ingest

# Run ingestion on GPU
docker run --rm --gpus all \
  -v /path/to/vtt/files:/data/archive \
  -v $(pwd)/data:/data/rainrag \
  -v $(pwd)/embeddings:/data/embeddings \
  rainrag:gpu ingest

# Run full pipeline
docker run --rm \
  -v /path/to/vtt/files:/data/archive \
  -v $(pwd)/data:/data/rainrag \
  -v $(pwd)/embeddings:/data/embeddings \
  --network host \
  rainrag:latest pipeline

# Run full pipeline on GPU
docker run --rm --gpus all \
  -v /path/to/vtt/files:/data/archive \
  -v $(pwd)/data:/data/rainrag \
  -v $(pwd)/embeddings:/data/embeddings \
  --network host \
  rainrag:gpu pipeline

# If you need to supply API keys or other secrets with a standalone
# `docker run` invocation, you can mount a local directory of key files
# and then set the corresponding *_API_KEY_FILE environment variables
# (for example, `-v $(pwd)/secrets:/run/secrets:ro` plus
# `-e OPENAI_API_KEY_FILE=/run/secrets/openai_api_key.txt`). See
# `docker-compose.yaml` for the Compose-based pattern which automatically
# exposes `./secrets/*.txt` as Docker secrets.
```

## Kubernetes Deployment with Helm

### Prerequisites

- Kubernetes cluster (Minikube for local development)
- Helm 3.x installed

### Deploy with Helm

1. **Start Minikube** (for local development)

```bash
minikube start
```

2. **Create necessary directories on the host** (for Minikube)

```bash
minikube ssh
sudo mkdir -p /data/archive /data/rainrag /data/embeddings
exit
```

3. **Install the Helm chart**

```bash
helm install rainrag ./helm/rainrag
```

### Customizing the Deployment

Create a custom `values.yaml`:

```yaml
# custom-values.yaml

qdrant:
  persistence:
    size: 20Gi
  resources:
    limits:
      memory: "4Gi"

ingestion:
  job:
    cron: true
    schedule: "0 2 * * *"  # Daily at 2 AM

  volumes:
    archive:
      hostPath: /mnt/archive

  config:
    embedding:
      device: "cuda"
      batch_size: 64
```

Install with custom values:

```bash
helm install rainrag ./helm/rainrag -f custom-values.yaml
```

### Helm Commands

```bash
# View deployment status
helm status rainrag

# Upgrade deployment
helm upgrade rainrag ./helm/rainrag

# Uninstall
helm uninstall rainrag

# View logs
kubectl logs -l app.kubernetes.io/name=rainrag-qdrant      # Qdrant
kubectl logs -l app.kubernetes.io/component=ingestion      # Ingestion job
kubectl logs -l app.kubernetes.io/component=api            # API backend
kubectl logs -l app.kubernetes.io/component=streamlit      # Streamlit frontend

# Port forward Streamlit to access locally
kubectl port-forward svc/rainrag-streamlit 7860:7860

# Access at http://localhost:7860
```

### Enabling Web UI in Kubernetes

The Helm chart includes FastAPI and Streamlit deployments. To enable them:

```yaml
# custom-values.yaml

# Enable API backend
api:
  enabled: true
  replicas: 1
  authToken: "your_secret_token"  # Optional
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"

# Enable Streamlit frontend
streamlit:
  enabled: true
  replicas: 1
  service:
    type: NodePort  # or LoadBalancer for cloud
    nodePort: 30786  # Fixed port for easy access
  authToken: "your_secret_token"  # Should match api.authToken
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
```

Then install/upgrade:

```bash
helm upgrade --install rainrag ./helm/rainrag -f custom-values.yaml
```

Access the Streamlit UI:
- **NodePort**: `http://<node-ip>:30786`
- **LoadBalancer**: Check external IP with `kubectl get svc rainrag-streamlit`
- **Port Forward**: `kubectl port-forward svc/rainrag-streamlit 7860:7860`

## Project Structure

```
rainrag/
├── src/
│   └── rainrag/
│       ├── __init__.py
│       ├── cli.py          # CLI interface
│       ├── config.py       # Configuration management
│       ├── ingest.py       # VTT parsing and ingestion
│       ├── embed.py        # Embedding generation
│       ├── index.py        # Qdrant indexing
│       ├── query.py        # RAG query engine
│       └── api.py          # FastAPI backend
├── app.py                  # Streamlit frontend
├── helm/
│   └── rainrag/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── qdrant-deployment.yaml
│           ├── qdrant-service.yaml
│           ├── qdrant-pvc.yaml
│           ├── vllm-deployment.yaml
│           ├── vllm-service.yaml
│           ├── api-deployment.yaml
│           ├── api-service.yaml
│           ├── api-secret.yaml
│           ├── streamlit-deployment.yaml
│           ├── streamlit-service.yaml
│           ├── streamlit-secret.yaml
│           ├── configmap.yaml
│           └── ingestion-job.yaml
├── data/                   # Output directory
├── embeddings/             # Embedding cache
├── logs/                   # Application logs
├── config.yaml             # Configuration file
├── docker-compose.yaml     # Docker Compose setup
├── .env.example            # Environment variables template
├── pyproject.toml          # Project dependencies and metadata
├── Dockerfile              # Container image
├── Makefile                # Development commands
└── README.md               # This file
```

## Configuration Reference

### Paths

- `archive_root`: Root directory containing VTT files
- `docs_output`: Output path for parsed documents (JSONL)
- `embeddings_cache`: Directory for cached embeddings
- `video_root`: Root directory containing video files (defaults to archive_root if not specified)

### Embedding

- `model_name`: HuggingFace model identifier
- `batch_size`: Batch size for embedding generation
- `max_seq_length`: Maximum sequence length for the model
- `device`: `"cuda"` or `"cpu"`
- `normalize_embeddings`: Whether to L2-normalize embeddings

### Qdrant

- `host`: Qdrant server hostname
- `port`: Qdrant server port (default: 6333)
- `collection_name`: Name of the vector collection
- `vector_size`: Embedding dimension (1024 for multilingual-e5-large)
- `distance`: Distance metric (`"Cosine"`, `"Euclidean"`, or `"Dot"`)
- `recreate_collection`: Auto-recreate collection on indexing

### Processing

- `num_workers`: Number of parallel workers for file processing
- `max_file_size`: Maximum VTT file size in bytes
- `min_text_length`: Minimum text length to process (characters)

### Video

- `enabled`: Enable video file serving (default: true)
- `extensions`: List of supported video file extensions
- `vtt_extensions`: List of supported VTT file extensions

### Web Metadata

- `enabled`: Enable loading of web metadata from JSON files (default: false)
- `path`: Path to directory containing web metadata JSON files (default: "./web_metadata")
- `min_content_length`: Minimum content length for web description text (default: 10)
- `require_web_metadata`: Controls ingestion behavior for missing web metadata (default: false)
  - `true`: Only ingest videos that have matching web metadata JSON files; skip videos without metadata entirely
  - `false`: Ingest all videos regardless of metadata availability; populate web-related fields (`web_title`, `web_date`, `web_description`, `web_url`) with `null`/`None` values when metadata is missing

### Chunking

- `enabled`: Enable automatic chunking of VTT files (default: true)
- `strategy`: Chunking strategy - `"time"` (time-based only), `"token"` (token-based only), or `"hybrid"` (time-based with token validation, recommended)
- `chunk_duration_seconds`: Duration of each time-based chunk in seconds (default: 300 = 5 minutes)
- `overlap_seconds`: Overlap between adjacent chunks in seconds (default: 30). Prevents information loss at boundaries.
- `min_chunk_tokens`: Minimum tokens per chunk (default: 50, chunks smaller than this may be merged)
- `max_chunk_tokens`: Maximum tokens per chunk (auto-detected from embedding model if not set)
- `token_buffer`: Safety buffer to reserve for special tokens (default: 50)

## VTT Chunking and Timestamp Utilization

RainRAG automatically chunks long VTT subtitle files into smaller segments optimized for embedding model token limits. This prevents truncation and ensures all content is searchable.

### Why Chunking?

Without chunking, a 2-hour video transcript would be embedded as a single document and truncated at the embedding model's token limit:
- **Without chunking**: 2-hour video (120,000 characters) → truncated to first 512 tokens (~1,800 characters) → 98.5% of content lost
- **With chunking**: 2-hour video → split into 24 chunks of 5 minutes each → 100% of content embedded and searchable

### Chunking Strategies

RainRAG supports three chunking strategies:

#### 1. Time-based Chunking (`strategy: "time"`)

Splits transcripts by video timestamp into fixed-duration segments (default: 5 minutes).

**Advantages:**
- Preserves semantic coherence (conversations/topics don't get split mid-sentence)
- Timestamps allow precise video seeking
- Natural boundaries for video content

**Configuration:**
```yaml
chunking:
  enabled: true
  strategy: "time"
  chunk_duration_seconds: 300  # 5 minutes
```

#### 2. Token-based Chunking (`strategy: "token"`)

Splits transcripts purely by token count to maximize embedding model utilization.

**Advantages:**
- Guarantees chunks fit within model limits
- Maximizes token usage for each chunk

**Disadvantages:**
- May split sentences or conversations unnaturally
- Less semantic coherence

**Configuration:**
```yaml
chunking:
  enabled: true
  strategy: "token"
  min_chunk_tokens: 50
  max_chunk_tokens: 462  # or auto-detected
```

#### 3. Hybrid Chunking (`strategy: "hybrid"`) **[Recommended]**

Combines time-based and token-based approaches: creates time-based chunks first, then validates they fit within token limits. Oversized chunks are split by token count while respecting subtitle cue boundaries.

**Advantages:**
- Best of both worlds: semantic coherence + token safety
- Adapts to different embedding models automatically
- Prevents both truncation and wasted token capacity

**Configuration:**
```yaml
chunking:
  enabled: true
  strategy: "hybrid"  # Default
  chunk_duration_seconds: 300  # 5 minutes
  overlap_seconds: 30  # NEW: 30-second overlap between chunks
  min_chunk_tokens: 50
  max_chunk_tokens: null  # Auto-detect from embedding model
  token_buffer: 50  # Safety margin for special tokens
```

### Chunk Overlap (Prevents Information Loss)

By default, RainRAG creates **30-second overlaps** between adjacent chunks to prevent critical information from being lost at chunk boundaries.

**Without overlap:**
```
Chunk 1: 00:00:00 - 00:05:00  |
Chunk 2:              00:05:00 - 00:10:00
                      ^ Hard boundary - conversation split here!
```

**With 30-second overlap (default):**
```
Chunk 1: 00:00:00 - 00:05:00     |
Chunk 2:           00:04:30 - 00:10:00
                   ^^^^^^^^ 30s overlap - conversation preserved!
```

**Why this matters:**
- Conversations/topics that span the 5-minute boundary stay intact
- Search quality improves because context isn't artificially split
- Small cost: ~10% more chunks (e.g., 26 chunks instead of 24 for 2-hour video)

**Configuration:**
```yaml
chunking:
  overlap_seconds: 30  # Default: 30 seconds
  # Set to 0 to disable overlap (not recommended)
  # Set to 60 for 1-minute overlap (more context, more chunks)
```

### Model-Aware Token Limits

RainRAG automatically adapts chunk sizes based on your embedding model's capabilities:

| Embedding Model | Token Limit | Effective Chunk Size (with buffer) |
|----------------|-------------|-------------------------------------|
| `intfloat/multilingual-e5-large` | 512 | 462 tokens (~1,600 chars) |
| `text-embedding-3-small` (OpenAI) | 8,191 | 8,141 tokens (~28,000 chars) |
| `text-embedding-3-large` (OpenAI) | 8,191 | 8,141 tokens (~28,000 chars) |
| `models/text-embedding-004` (Gemini) | 2,048 | 1,998 tokens (~7,000 chars) |

**This means:**
- **E5 local model**: 5-minute chunks → ~24 chunks per 2-hour video
- **OpenAI embedding**: 5-minute chunks stay intact (no splitting needed) → ~24 chunks per 2-hour video
- **Gemini embedding**: 5-minute chunks may occasionally be split → ~24-30 chunks per 2-hour video

You can override auto-detection by setting `max_chunk_tokens` explicitly in `config.yaml`.

### Cross-Language Video Identification

All language versions of the same video (e.g., `episode_001.en.vtt` and `episode_001.ru.vtt`) share the same `video_id`, allowing RainRAG to:
- Group multilingual results together in the UI
- Show both Russian and English subtitles for the same video
- Enable cross-language search (search in English, find Russian content and vice versa)

### Chunk Metadata

Each chunk includes rich metadata:
- `is_chunk`: Boolean indicating if document is a chunk
- `chunk_index`: Position in the sequence (0-based)
- `total_chunks`: Total number of chunks for this video
- `start_time_seconds`: Start timestamp in seconds
- `end_time_seconds`: End timestamp in seconds
- `start_time`: Human-readable start time (HH:MM:SS)
- `end_time`: Human-readable end time (HH:MM:SS)
- `video_id`: Unique identifier shared across language versions

This metadata enables:
- Precise video seeking to relevant segments
- Progress tracking (e.g., "Chunk 3 of 24")
- Time-range filtering

### Token Estimation

Since loading full tokenizers during ingestion is slow, RainRAG uses fast character-to-token ratio estimation:
- **English**: ~4 characters per token
- **Russian (Cyrillic)**: ~2.5 characters per token
- **Chinese**: ~1.5 characters per token
- **Default**: ~3.5 characters per token

This provides 95%+ accuracy without the overhead of loading embedding model tokenizers.

## Web Metadata Integration

RainRAG can enrich video transcripts with additional metadata from web sources (titles, accurate dates, descriptions, URLs).

### File Naming and Matching

RainRAG matches web metadata JSON files to VTT/video files using the video hash derived from the VTT filename:

1. **Video Hash Extraction**: For a VTT file named `episode_001.en.vtt`, the video hash is `episode_001` (filename stem without extension)
2. **JSON Filename**: The corresponding metadata file must be named `{video_hash}.json` (e.g., `episode_001.json`)
3. **Hash Algorithm**: The video hash is typically computed as SHA-256 of the original video filename:

```bash
# Example: Generate hash for episode_001.mp4
echo -n "episode_001.mp4" | sha256sum
d9f2eb1b7c0db375ce4456cdd6401e4a831f91e2badb9011534357b3792707f4  -

# Use first 16 characters for short hash: d9f2eb1b7c0db375
```

The hash can be derived from either the video filename (e.g., `episode_001.mp4`) or VTT filename (e.g., `episode_001.en.vtt`) - both should produce the same stem.

### How It Works

1. **Metadata Files**: Place JSON files in the `web_metadata/` directory, named after video hashes (e.g., `episode_001.json`)
2. **Automatic Loading**: During ingestion, RainRAG looks for matching metadata files for each VTT file
3. **Field Enrichment**: Web metadata fields are added to documents with the `web_` prefix

### Field Mapping

JSON fields are mapped to document metadata keys as follows:

| JSON Field | Document Key | Required | Description |
|------------|--------------|----------|-------------|
| `name` | `web_title` | Optional | Video title from web source |
| `date_active_start` | `web_date` | Optional | Publication date (ISO format, parsed to YYYY-MM-DD) |
| `date_active_start` | `web_date_ts` | Optional | Publication timestamp (Unix timestamp) |
| `url` | `web_url` | Optional | Original web URL |
| `preview_text` + `detail_text` | `web_description` | Required* | Combined description (HTML-decoded, concatenated with space) |

**Notes**:
- `detail_text` must be non-empty (after HTML decoding and whitespace trimming) for the metadata to be considered valid
- `preview_text` is optional and prepended to `detail_text` if present
- HTML entities in `preview_text` and `detail_text` are automatically decoded (`&nbsp;` → ` `, `&amp;` → `&`)
- Dates are parsed from ISO format (e.g., `2024-01-15T10:30:00Z`) to YYYY-MM-DD and Unix timestamp
- Invalid dates are logged as warnings and ignored

### Topics Covered by Web Metadata

The web metadata covers episodes discussing concrete topics such as:

- 30 тыс погибли на протестах в Иране
- 9 человек умерли в ПНИ
- Telegram частично блокируют
- Wildberries откроет свои отели
- Z-военкоры хвалят атаку США на Венесуэлу
- «Госуслуги» зовут в «университет спецназа»
- «Монахини-шпионки» ГРУ
- «Путин залег на дно»: куда он исчез после ареста Мадуро и что с Кадыровым | Белковский
- «СВО» сравнялась с ВОВ
- «Свошник» задушил женщину
- «Совет мира» с Путиным и Лукашенко
- «Узкие» против Гудкова
- «Чебурашка» против «Буратино» в прокате
- «Шереметьево» купило «Домодедово»
- Автор «Дозоров» зетнулся
- Агутин поет для военных
- Адам Кадыров поправляется
- Актер в РФ выступил против войны
- Аресты протестующих в Петербурге
- Астронавты полетят на Луну

These represent the specific subjects and headlines from episodes that have web metadata enrichment, enabling targeted retrieval of videos on these topics.

### Error Handling

- **Malformed JSON**: Files with invalid JSON are skipped with debug logging
- **Missing Files**: Videos without matching metadata get `None` values for all `web_*` fields
- **Invalid Dates**: Unparseable `date_active_start` values are logged and ignored
- **Empty Content**: Videos with empty/whitespace-only `detail_text` are skipped entirely
- **Config Toggle**: The `web_metadata.require_web_metadata` setting controls whether missing metadata causes ingestion to fail or continue with `None` values

### Two Ingestion Modes

#### Mode 1: Ingest All Videos (Default)
```yaml
web_metadata:
  enabled: true
  require_web_metadata: false  # Default
```
- Processes all VTT files regardless of metadata availability
- Videos without metadata get `None` values for web fields
- Useful for gradual enrichment of existing archives

#### Mode 2: Curated Dataset Only
```yaml
web_metadata:
  enabled: true
  require_web_metadata: true
```
- Only ingests videos that have corresponding web metadata files
- Skips videos without metadata entirely
- Useful for creating high-quality, curated datasets

### Metadata File Format

Each JSON file should contain web-scraped metadata:

```json
{
  "name": "Video Title Here",
  "date_active_start": "2024-01-15T10:30:00Z",
  "url": "https://example.com/video/123",
  "preview_text": "Short preview text...",
  "detail_text": "Full description with HTML entities like &nbsp; and &amp;"
}
```

RainRAG automatically:
- HTML-decodes content (`&nbsp;` → ` `, `&amp;` → `&`)
- Parses ISO dates to timestamps
- Validates content length (configurable minimum)

### Benefits

- **Accurate Dates**: Web publication dates instead of file modification times
- **Rich Context**: Titles and descriptions improve search relevance
- **Source Attribution**: URLs link back to original content
- **Flexible Deployment**: Choose between comprehensive or curated indexing

## Hybrid Search (Vector + BM25)

RainRAG supports **hybrid search** that combines vector similarity with BM25 keyword matching for improved search quality.

### Why Hybrid Search?

**Vector search alone** (semantic similarity):
- ✅ Great for conceptual matches ("video about energy" → finds "electricity", "power", "renewables")
- ❌ Can miss exact phrases or entity names

**BM25 keyword search**:
- ✅ Catches exact phrases and entity names ("Vladimir Putin" → finds exact mentions)
- ❌ Misses semantic variations ("car" doesn't match "automobile")

**Hybrid = Best of Both:**
- Semantic understanding + exact keyword matching
- 10-20% better accuracy for queries with specific names/terms
- No re-indexing required (builds BM25 from existing Qdrant data)

### Configuration

```yaml
hybrid_search:
  enabled: true  # Enable hybrid search
  bm25_weight: 0.3  # Weight for BM25 scores (0.0-1.0, vector weight = 1 - bm25_weight)
  top_k_multiplier: 3  # Retrieve 3x candidates before reranking
  fusion_method: "rrf"  # Score fusion: "rrf" or "weighted"
  rrf_k: 60  # RRF constant (standard from literature)
```

### How It Works

1. **Retrieve More Candidates:**
   - Vector search: Retrieve top-k × 3 documents (e.g., 15 for top-5)
   - BM25 search: Retrieve top-k × 3 documents

2. **Fuse Scores:**
   - **RRF (Reciprocal Rank Fusion)** - default, research-proven:
     ```
     RRF_score = Σ (1 / (k + rank)) for each result list
     ```
   - **Weighted Sum** - customizable weights:
     ```
     Combined_score = (1 - bm25_weight) × vector_score + bm25_weight × bm25_score
     ```

3. **Return Top-K:**
   - Sort fused results
   - Return top-k final results

### Usage Example

```bash
# Enable hybrid search in config.yaml
hybrid_search:
  enabled: true

# Restart the query engine
rainrag ask "What did Putin say about energy policy?"

# Results will combine:
# - Semantic matches (discussions about power/electricity/renewables)
# - Exact keyword matches (mentions of "Putin" and "energy")
```

### Performance Impact

- **Latency:** +50-100ms (BM25 indexing at startup, minimal query overhead)
- **Memory:** +~10MB per 1000 documents (BM25 index)
- **Accuracy:** +10-20% for keyword-heavy queries

### Recommended Settings

**General Use (Balanced):**
```yaml
fusion_method: "rrf"  # Rank-based fusion, no tuning needed
top_k_multiplier: 3
```

**Keyword-Heavy Queries (names, locations, etc.):**
```yaml
fusion_method: "weighted"
bm25_weight: 0.4  # Higher weight for keywords
```

**Mostly Semantic Queries:**
```yaml
fusion_method: "weighted"
bm25_weight: 0.2  # Lower weight for keywords
```

## Temporal Context Enhancement

RainRAG automatically detects temporal keywords in queries and applies time-aware boosting to prioritize recent content when appropriate.

### How It Works

1. **Temporal Keyword Detection**: Automatically identifies queries asking for "recent", "latest", "last week", etc.
   - English: recent, recently, latest, last, new, current, today, yesterday, this week/month/year
   - Russian: недавн, последн, новый, свежий, актуальн, сегодня, вчера

2. **Automatic Date Filtering**: For "recent" queries, automatically filters to last 30 days
   - Query: "What are the latest developments?"
   - System: Applies date filter for last 30 days automatically

3. **Time-Decay Boosting**: Adjusts relevance scores based on document recency
   - Formula: `boost = 1.0 / (1.0 + age_days / 30.0)`
   - Recent documents get higher scores for temporal queries
   - Non-temporal queries are unaffected

### Examples

**Temporal Query (automatic boost):**
```python
# User asks: "What are the latest news about energy prices?"
# System automatically:
# 1. Detects "latest" keyword
# 2. Filters to last 30 days
# 3. Boosts recent documents
response = query_engine.query("What are the latest news about energy prices?")
```

**Explicit Date Filter (manual control):**
```python
# Override automatic detection with explicit dates
response = query_engine.query(
    question="What happened in the energy sector?",
    date_from="2024-01-01",
    date_to="2024-12-31"
)
```

### Configuration

Time-decay boosting is automatically applied when temporal keywords are detected. The decay factor (30 days) is currently hardcoded but can be adjusted in `src/rainrag/query.py:_apply_time_decay_boost()`.

## Related Chunks Discovery

Find similar or related chunks based on vector similarity. Useful for exploring related content or finding "more from this video".

### API Endpoint

```python
POST /related-chunks
{
  "chunk_id": "episode_001.en.vtt_chunk_3",
  "top_k": 5,
  "same_video_only": false
}
```

**Response:**
```json
{
  "chunk_id": "episode_001.en.vtt_chunk_3",
  "num_related": 5,
  "related_chunks": [
    {
      "text": "Related content...",
      "score": 0.92,
      "video_url": "/video/episode_001.mp4#t=350",
      ...
    }
  ]
}
```

### Python API

```python
from rainrag.query import RAGQueryEngine
from rainrag.config import load_config

config = load_config()
engine = RAGQueryEngine(config)
engine.initialize()

# Find similar chunks
related = engine.find_related_chunks(
    chunk_id="episode_001.en.vtt_chunk_3",
    top_k=5,
    same_video_only=False  # Set True to only find chunks from same video
)

# Explore more from same video
more_from_video = engine.find_related_chunks(
    chunk_id="episode_001.en.vtt_chunk_3",
    top_k=10,
    same_video_only=True
)
```

### Use Cases

1. **Content Discovery**: "More like this" functionality
2. **Video Exploration**: Browse related segments from the same video
3. **Topic Clustering**: Find all chunks about a specific topic
4. **Quality Assurance**: Verify similar content across videos

## Reranking

RainRAG supports reranking retrieved documents using Cohere Rerank API for improved relevance.

### Configuration

```yaml
# config.yaml
reranker:
  enabled: true          # Enable reranking
  provider: "cohere"     # Currently only Cohere supported
  top_n: 5              # Return top 5 after reranking
  initial_k: 20         # Retrieve 20 candidates before reranking

cohere:
  api_key: "your-cohere-api-key"
  model_name: "rerank-v3.5"  # Options: rerank-v3.5, rerank-english-v3.0, rerank-multilingual-v3.0
```

### Benefits

- **10-15% accuracy improvement** over vector search alone
- **Cross-encoder architecture** better understands query-document relevance
- **Multilingual support** with rerank-multilingual-v3.0
- **Works with hybrid search** for best results

### Performance Impact

- Adds ~200-500ms latency per query
- Cost: ~$0.001 per 1000 rerank operations
- Recommended for production use with top_n=3-5

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run ruff check src/
```

### Type Checking

```bash
uv run mypy src/
```

## Performance Considerations

### GPU Acceleration

For faster embedding generation, use a CUDA-enabled GPU:

```yaml
embedding:
  device: "cuda"
  batch_size: 64  # Increase batch size with GPU
```

### Batch Processing

For large archives, consider processing in batches:

```bash
# Process and embed incrementally
rainrag ingest
rainrag embed
rainrag index

# Later, with new files
rainrag ingest  # Appends to docs.jsonl
rainrag embed --force  # Regenerate all embeddings
rainrag index --recreate  # Rebuild index
```

> Note: Incremental processing is now disabled by default in `config.yaml` (`incremental.enabled: false`).
> Enable it explicitly to use manifest-driven delta ingestion and incremental indexing.
> `incremental.manifest_path` and `incremental.alias_swap` are still available for configured incremental flows.

### Caching

Embeddings are automatically cached. To skip regeneration:

```bash
# First run (generates embeddings)
rainrag embed

# Subsequent runs (uses cache)
rainrag index  # Uses cached embeddings
```

## Troubleshooting

### Out of Memory

Reduce batch size in `config.yaml`:

```yaml
embedding:
  batch_size: 16  # Reduce from 32
```

If the full pipeline gets killed, run the stages separately to isolate which step needs tuning:

```bash
rainrag ingest
rainrag embed --force
rainrag index --recreate
```

### Qdrant Connection Failed

Ensure Qdrant is running:

```bash
docker ps | grep qdrant

# If not running, start it
docker run -p 6333:6333 qdrant/qdrant:v1.16.3
```

### Back Up / Restore Indexed Database (Qdrant) Across Servers

You can reuse the indexed DB on another server by backing up **Qdrant collection snapshots** and restoring them.  When migrating, don’t forget to also copy the embeddings cache (`embeddings/` path configured in `config.yaml` or via the `EMBEDDINGS_CACHE` env) so your vectors remain in sync with the database.

RainRAG includes helper scripts for the Qdrant portion (the corresponding Makefile targets will first verify that Qdrant is reachable):

```bash
# Backup current Qdrant collection snapshot to Cloudflare R2
# (equivalent to the Makefile target `make backup-qdrant-r2` which checks Qdrant availability)
./scripts/backup_qdrant_r2.sh

# Restore latest snapshot from R2 into local Qdrant
# (equivalent to the Makefile target `make restore-qdrant-r2` which checks Qdrant availability)
./scripts/restore_qdrant_r2.sh
```


### Embeddings Cache Backup/Restore

RainRAG also provides simple commands to back up or restore the embeddings cache directory. These are independent of the Qdrant snapshot tools but often used together when migrating data.

```bash
make backup-embeddings-r2   # push `embeddings/` to Cloudflare R2
make restore-embeddings-r2  # pull `embeddings/` from Cloudflare R2
```

You can control the R2 location with the same environment variables shown earlier (e.g. `R2_BUCKET`, `R2_PREFIX`).


Optional variables (in `.env`):

- `QDRANT_URL` (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` (default: `broadcast_transcripts`)
- `R2_QDRANT_PREFIX` (default: `qdrant`)
- `QDRANT_SNAPSHOT_NAME` (optional explicit snapshot file for restore)

Notes:

- Snapshot/restore preserves vectors + payload metadata, so indexing does not need to be rerun.
- You must also preserve the embeddings cache (default `embeddings/` or configured via `config.yaml`) when moving servers; the cache should match the points in Qdrant.
- Keep `config.yaml` collection name aligned with `QDRANT_COLLECTION`.
- For best consistency, avoid writes during backup.

### Queries Return No Results (Empty Collection)

If you get answers with no retrieved videos/VTTs, your collection is likely empty.

1) Check:

```bash
rainrag info
```

2) If `points_count` is `0`, repopulate it:

```bash
rainrag pipeline --recreate-index
```

### Date Filters Causing Errors / No Matches

- Date filtering depends on documents having valid `date` metadata and a numeric `date_ts` for safe range filtering.
- After updating RainRAG, **rerun the full pipeline** (don’t use `--skip-ingest`) so `date_ts` is populated end-to-end:

```bash
rainrag pipeline --recreate-index
```

### VTT Parsing Issues

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
```

Then check logs for detailed parsing information.

### Editor Shows “Import … could not be resolved”

If runtime works but the editor shows missing imports, ensure the editor uses the project uv-managed environment.

- Recommended: use an in-project venv (`.venv`) and point your editor/type-checker at it.
- This repo includes `pyrightconfig.json` and `.vscode/settings.json` to help Cursor/VS Code pick up `.venv`.
- Ensure local Python builds/install use `uv sync` or `pip install -e .` from repo root so runtime imports match type checker lookup (package root is `src/`).

## Roadmap

- [ ] Add support for additional subtitle formats (SRT, ASS)
- [x] Implement document summarization with local LLMs (Mistral via vLLM)
- [x] Build a web UI for querying the index (Streamlit + FastAPI)
- [ ] Add multi-node Qdrant support for horizontal scaling
- [ ] Implement incremental indexing (delta updates)
- [ ] Make embedding batch_size runtime-configurable (configure per available VRAM)
- [ ] Add metrics and monitoring (Prometheus/Grafana)
- [ ] Add VPN access configuration guide
- [ ] Implement query history persistence
- [ ] Add export functionality for query results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Qdrant](https://qdrant.tech/) for the vector database
- [sentence-transformers](https://www.sbert.net/) for embedding models
- [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) for multilingual embeddings

## Support

For issues and questions, please open an issue on GitHub.
