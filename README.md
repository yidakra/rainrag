# RainRAG

**Local-first Retrieval-Augmented Generation (RAG) Pipeline for VTT Subtitle Processing**

RainRAG is a modular, open-source backend system for building a semantic search engine over VTT subtitle files. It uses state-of-the-art multilingual embeddings and vector search to enable efficient retrieval of broadcast transcripts in Russian and English.

## Features

- **Local-first**: No external API calls, all processing runs locally
- **Multilingual**: Supports Russian and English subtitles with `intfloat/multilingual-e5-large` embeddings
- **Modular Pipeline**: Separate stages for ingestion, embedding, and indexing
- **Production Ready**: Includes Helm charts for Kubernetes deployment
- **Type-safe**: Fully type-annotated Python code with Pydantic models
- **CLI Interface**: Easy-to-use command-line interface powered by Typer
- **Vector Search**: Uses Qdrant for efficient similarity search
- **Web UI**: Streamlit-based chat interface with FastAPI backend
- **Video Playback**: Inline video player for retrieved content
- **Subtitle Access**: Download and view VTT files directly in the UI
- **Network Access**: Accessible from other devices on the same network with optional token authentication
- **LLM Integration**: Query interface powered by Mistral-Small via vLLM

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
   ┌─────────┐  ┌─────────┐
   │ Qdrant  │  │  vLLM   │
   │ Search  │  │ Mistral │
   └─────────┘  └─────────┘
         │          │
         └────┬─────┘
              ▼
        Generated Answer
        + Context Chunks
```

## Quick Start

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Docker (optional, for containerized deployment)
- Kubernetes/Minikube (optional, for Helm deployment)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/rainrag.git
cd rainrag
```

2. **Install dependencies with Poetry**

```bash
poetry install
```

3. **Activate the virtual environment**

```bash
poetry shell
```

4. **Download required models** (requires internet connection)

```bash
# Download and cache the embedding model
make download-models

# Or manually:
poetry run python scripts/download_models.py
```

**Important:** This step downloads the `multilingual-e5-large` embedding model (~2GB) and caches it locally. This is required before you can run RainRAG. You only need to do this once.

### Configuration

Edit `config.yaml` to customize paths and settings:

```yaml
paths:
  archive_root: "/path/to/your/vtt/files"
  docs_output: "./data/docs.jsonl"
  embeddings_cache: "./embeddings"

embedding:
  model_name: "intfloat/multilingual-e5-large"
  device: "cuda"  # or "cpu"
  batch_size: 32

qdrant:
  host: "localhost"
  port: 6333
  collection_name: "broadcast_transcripts"
```

### Running Qdrant Locally

Start a local Qdrant instance using Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.7.4
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

#### 5. View System Info

Display configuration and collection statistics:

```bash
rainrag info
```

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
- **Network Access**: Accessible from other devices on your LAN

### Prerequisites

Before starting the web interface, you need running LLM server(s) to generate answers:

**See [MULTI_MODEL_SETUP.md](MULTI_MODEL_SETUP.md) for running all 3 models simultaneously (recommended)**
**See [VLLM_SETUP.md](VLLM_SETUP.md) for single vLLM server configuration**
**See [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for detailed model-specific configurations**

Quick single-model startup:
```bash
# Mistral on port 8000 (default)
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000
```

Multi-model setup (run all 3 for seamless switching):
```bash
# Terminal 1: Mistral on port 8000
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 --port 8000

# Terminal 2: Gemma on port 8002
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it --port 8002

# Terminal 3: GPT-OSS on port 8003
python -m vllm.entrypoints.openai.api_server \
  --model gpt-oss:20b --port 8003
```

**Or use `make up` to start everything at once!**

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
# Build the image first
docker build -t rainrag:latest .

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
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
- **Text Preview**: Long text excerpts are collapsed with "Show full text" expansion
- **Metadata**: Each chunk displays filename, relevance score, and language

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
docker build -t rainrag:latest .
```

### Run with Docker

```bash
# Run ingestion
docker run --rm \
  -v /path/to/vtt/files:/data/archive \
  -v $(pwd)/data:/data/rainrag \
  -v $(pwd)/embeddings:/data/embeddings \
  rainrag:latest ingest

# Run full pipeline
docker run --rm \
  -v /path/to/vtt/files:/data/archive \
  -v $(pwd)/data:/data/rainrag \
  -v $(pwd)/embeddings:/data/embeddings \
  --network host \
  rainrag:latest pipeline
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
├── pyproject.toml          # Poetry dependencies
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

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black src/
poetry run ruff check src/
```

### Type Checking

```bash
poetry run mypy src/
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

### Qdrant Connection Failed

Ensure Qdrant is running:

```bash
docker ps | grep qdrant

# If not running, start it
docker run -p 6333:6333 qdrant/qdrant:v1.7.4
```

### VTT Parsing Issues

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
```

Then check logs for detailed parsing information.

## Roadmap

- [ ] Add support for additional subtitle formats (SRT, ASS)
- [x] Implement document summarization with local LLMs (Mistral via vLLM)
- [x] Build a web UI for querying the index (Streamlit + FastAPI)
- [ ] Add multi-node Qdrant support for horizontal scaling
- [ ] Implement incremental indexing (delta updates)
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