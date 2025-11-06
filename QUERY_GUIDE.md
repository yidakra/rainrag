# RainRAG Query Layer Guide

This guide explains how to use the new querying functionality added to RainRAG, which allows you to ask natural language questions about your video transcripts.

## Architecture Overview

The query layer adds the following components to the existing RAG pipeline:

```
User Question
    ↓
1. Query Embedding (multilingual-e5-large)
    ↓
2. Vector Search (Qdrant)
    ↓
3. Context Building (Top-K relevant chunks)
    ↓
4. Answer Generation (Mistral-Small-3.2-24B-Instruct via vLLM)
    ↓
Natural Language Answer
```

## Prerequisites

Before using the query functionality, you need:

1. **Indexed transcripts**: Run the full pipeline to ingest and index your VTT files
   ```bash
   rainrag pipeline
   ```

2. **Running Qdrant**: Your Qdrant vector store must be running
   ```bash
   # Using Helm
   helm install rainrag ./helm/rainrag

   # Or locally with Docker
   docker run -p 6333:6333 qdrant/qdrant:v1.7.4
   ```

3. **Running vLLM server**: The Mistral model must be served via vLLM
   ```bash
   # Using Helm (with GPU)
   helm upgrade rainrag ./helm/rainrag --set vllm.enabled=true

   # Or locally with Docker
   docker run --gpus all -p 8000:8000 \
     vllm/vllm-openai:latest \
     --model mistralai/Mistral-Small-3.2-24B-Instruct \
     --port 8000
   ```

## CLI Usage

### Basic Query

Ask a question and get an answer:

```bash
rainrag ask "О чём говорили в выпуске про энергетику?"
```

### Verbose Mode

Show retrieved documents and sources:

```bash
rainrag ask "What were the main topics discussed?" --verbose
```

### Custom Number of Retrieved Documents

By default, the system retrieves 5 documents. You can change this:

```bash
rainrag ask "Tell me about the energy episode" --top-k 10
```

### Using a Different Config File

```bash
rainrag ask "What is this about?" --config /path/to/config.yaml
```

## Configuration

### Local Configuration (config.yaml)

The query system is configured in `config.yaml`:

```yaml
vllm:
  host: "localhost"
  port: 8000
  model_name: "mistralai/Mistral-Small-3.2-24B-Instruct"
  max_tokens: 512        # Maximum length of generated answer
  temperature: 0.3       # Lower = more focused, higher = more creative
  top_k: 5              # Number of documents to retrieve
```

### Kubernetes/Helm Configuration

When deploying with Helm, configure in `helm/rainrag/values.yaml`:

```yaml
vllm:
  enabled: true
  modelName: "mistralai/Mistral-Small-3.2-24B-Instruct"
  replicas: 1

  # GPU settings
  gpuMemoryUtilization: 0.9
  maxModelLen: 8192
  tensorParallelSize: 1

  resources:
    limits:
      nvidia.com/gpu: 1
      memory: "32Gi"
```

## Python API

You can also use the query engine directly in Python:

```python
from rainrag.config import load_config
from rainrag.query import RAGQueryEngine

# Load configuration
config = load_config("config.yaml")

# Initialize the query engine
engine = RAGQueryEngine(config)
engine.initialize()

# Ask a question
result = engine.query(
    question="О чём говорили в выпуске про энергетику?",
    top_k=5
)

# Access the results
print(f"Answer: {result['answer']}")
print(f"Retrieved {result['num_documents']} documents")

for doc in result['retrieved_documents']:
    print(f"- {doc['path']} (score: {doc['score']:.4f})")
```

## How It Works

### 1. Query Embedding

The user's question is embedded using the same model (`intfloat/multilingual-e5-large`) that was used to embed the documents. This ensures semantic similarity matching works correctly.

The query is prefixed with `"query: "` (an E5 model requirement) to optimize retrieval performance.

### 2. Vector Search

The query embedding is used to search the Qdrant vector store for the most semantically similar document chunks. The search uses cosine similarity by default.

### 3. Context Building

The top-K most relevant documents are retrieved and formatted into a context string that includes:
- The text of each document
- Source information (file path)
- Ranking information

### 4. Answer Generation

A prompt is constructed that includes:
- The retrieved context
- The user's question
- Instructions for the model

This prompt is sent to the vLLM server running Mistral-Small-3.2-24B-Instruct, which generates a natural language answer.

## Performance Tuning

### Retrieval Quality

- **Increase top_k**: Retrieve more documents for broader context (may be slower)
- **Decrease top_k**: Faster queries with more focused context

### Answer Quality

- **Temperature** (0.0 - 1.0):
  - Lower (0.1-0.3): More deterministic, factual answers
  - Higher (0.7-1.0): More creative, varied responses

- **Max Tokens**:
  - Lower (256-512): Shorter, concise answers
  - Higher (1024+): More detailed, comprehensive answers

### Speed Optimization

1. **vLLM Settings**:
   - Increase `gpu-memory-utilization` (0.8-0.95)
   - Use tensor parallelism for multiple GPUs
   - Enable quantization for faster inference

2. **Model Selection**:
   - Smaller models (7B-13B) are faster but less capable
   - Larger models (24B+) provide better answers but require more GPU memory

## Troubleshooting

### "Cannot connect to vLLM server"

Ensure vLLM is running and accessible:

```bash
curl http://localhost:8000/v1/models
```

### "Cannot connect to Qdrant"

Verify Qdrant is running:

```bash
curl http://localhost:6333/collections
```

### Empty or Poor Quality Answers

1. Check if documents are indexed:
   ```bash
   rainrag info
   ```

2. Try increasing `top_k` to retrieve more context:
   ```bash
   rainrag ask "your question" --top-k 10
   ```

3. Ensure your question is in the same language as most documents

### Out of Memory Errors (vLLM)

- Reduce `gpu-memory-utilization` to 0.8 or lower
- Reduce `max-model-len` (e.g., 4096 instead of 8192)
- Use a smaller model variant
- Enable tensor parallelism across multiple GPUs

## Example Workflow

Complete workflow from VTT files to answering questions:

```bash
# 1. Ingest VTT files
rainrag ingest

# 2. Generate embeddings
rainrag embed

# 3. Index in Qdrant
rainrag index

# 4. Start vLLM (in another terminal or via Helm)
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-Small-3.2-24B-Instruct

# 5. Ask questions!
rainrag ask "О чём говорили в выпуске про энергетику?" --verbose
```

## Advanced: Custom Prompts

If you want to customize the prompt template, edit `src/rainrag/query.py` in the `build_prompt()` method:

```python
def build_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
    # Customize this prompt to change how the model responds
    prompt = f"""You are an assistant that helps users understand video transcripts...

Context from video transcripts:
{context}

User Question: {query}

Answer:"""

    return prompt
```

## Kubernetes Deployment

Deploy the complete system with Helm:

```bash
# Install with vLLM enabled
helm install rainrag ./helm/rainrag \
  --set vllm.enabled=true \
  --set vllm.modelName="mistralai/Mistral-Small-3.2-24B-Instruct"

# Check deployment status
kubectl get pods
kubectl logs -f deployment/rainrag-vllm

# Test the query endpoint
kubectl port-forward svc/rainrag-vllm 8000:8000
curl http://localhost:8000/v1/models
```

## Next Steps

- Integrate the query API into a web interface (e.g., Streamlit)
- Add support for streaming responses
- Implement query history and caching
- Add multi-turn conversation support
- Create REST API wrapper for the query engine

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [E5 Embedding Model](https://huggingface.co/intfloat/multilingual-e5-large)
- [Mistral Models](https://docs.mistral.ai/)
