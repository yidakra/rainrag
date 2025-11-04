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

## Architecture

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

# View Qdrant logs
kubectl logs -l app.kubernetes.io/name=rainrag-qdrant

# View ingestion job logs
kubectl logs -l app.kubernetes.io/component=ingestion
```

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
│       └── index.py        # Qdrant indexing
├── helm/
│   └── rainrag/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── qdrant-deployment.yaml
│           ├── qdrant-service.yaml
│           ├── qdrant-pvc.yaml
│           ├── configmap.yaml
│           └── ingestion-job.yaml
├── data/                   # Output directory
├── embeddings/             # Embedding cache
├── logs/                   # Application logs
├── config.yaml             # Configuration file
├── pyproject.toml          # Poetry dependencies
├── Dockerfile              # Container image
└── README.md              # This file
```

## Configuration Reference

### Paths

- `archive_root`: Root directory containing VTT files
- `docs_output`: Output path for parsed documents (JSONL)
- `embeddings_cache`: Directory for cached embeddings

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
- [ ] Implement document summarization with local LLMs
- [ ] Build a web UI for querying the index
- [ ] Add multi-node Qdrant support for horizontal scaling
- [ ] Implement incremental indexing (delta updates)
- [ ] Add metrics and monitoring (Prometheus/Grafana)

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