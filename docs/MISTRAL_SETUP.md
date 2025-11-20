# Mistral AI Setup Guide

This guide will help you set up Mistral AI as your LLM and/or embedding provider for RainRAG.

## Why Mistral AI?

Mistral AI is the **default recommended provider** for RainRAG because:

- ✅ **Excellent multilingual support** - Strong performance on Russian and English
- ✅ **Cost-effective** - Competitive pricing for both embeddings and LLM
- ✅ **Fast** - Low latency API responses
- ✅ **Flexible** - Multiple model sizes (Small, Medium, Large)
- ✅ **Privacy-friendly** - European company with strong data protection

## Prerequisites

- Mistral AI account
- Credit card for API billing (free tier available)

## Step 1: Get Your API Key

1. Go to [Mistral AI Console](https://console.mistral.ai/)
2. Sign up or log in
3. Navigate to **API Keys** section
4. Click **Create new key**
5. Give it a name (e.g., "RainRAG")
6. Copy the API key (you won't be able to see it again!)

## Step 2: Configure RainRAG

### Option A: Environment Variable (Recommended)

Add to your `.env` file:

```bash
MISTRAL_API_KEY=your_api_key_here
```

Or export directly:

```bash
export MISTRAL_API_KEY=your_api_key_here
```

### Option B: Direct Configuration

Edit `config.yaml`:

```yaml
mistral:
  api_key: "your_api_key_here"  # Not recommended - use env var instead
```

## Step 3: Select Mistral as Provider

### For LLM (Answer Generation)

Edit `config.yaml`:

```yaml
llm:
  provider: "mistral"

mistral:
  model_name: "mistral-small-latest"  # Recommended for most users
  max_tokens: 512
  temperature: 0.3
```

### For Embeddings (Vector Search)

Edit `config.yaml`:

```yaml
embedding:
  provider: "mistral"

qdrant:
  vector_size: 1024  # Mistral embeddings are 1024-dimensional
```

## Available Models

### LLM Models

| Model | Use Case | Performance | Cost |
|-------|----------|-------------|------|
| `mistral-small-latest` | **Recommended** - Fast, accurate for most tasks | ⭐⭐⭐⭐ | $ |
| `mistral-medium-latest` | Complex reasoning, longer context | ⭐⭐⭐⭐⭐ | $$ |
| `mistral-large-latest` | Maximum capability, enterprise use | ⭐⭐⭐⭐⭐ | $$$ |

### Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `mistral-embed` | 1024 | Standard embeddings (automatic when using `provider: "mistral"`) |

## Step 4: Test Your Setup

### Test CLI Query

```bash
rainrag ask "What is machine learning?" --verbose
```

You should see:
- "Using provider: mistral" in the output
- A generated answer
- Context documents retrieved from Qdrant

### Test Python API

```python
from rainrag.config import load_config
from rainrag.query import RAGQueryEngine

config = load_config("config.yaml")
engine = RAGQueryEngine(config)
engine.initialize()

result = engine.query(
    question="What is artificial intelligence?",
    top_k=5,
    language="en"
)

print(result["answer"])
```

## Configuration Examples

### Full Mistral Configuration

```yaml
# Use Mistral for everything (LLM + embeddings)
embedding:
  provider: "mistral"

llm:
  provider: "mistral"

mistral:
  api_key: ""  # Use MISTRAL_API_KEY env var
  model_name: "mistral-small-latest"
  max_tokens: 512
  temperature: 0.3
  top_k: 5

qdrant:
  vector_size: 1024  # For Mistral embeddings
```

### Mistral LLM + Local Embeddings

```yaml
# Use local embeddings (free) + Mistral LLM (paid)
embedding:
  provider: "local"
  model_name: "intfloat/multilingual-e5-large"
  device: "cuda"

llm:
  provider: "mistral"

mistral:
  api_key: ""
  model_name: "mistral-small-latest"

qdrant:
  vector_size: 1024  # For local embeddings
```

## Pricing

As of 2025:
- **mistral-small-latest**: ~$0.002 per 1K tokens (input), ~$0.006 per 1K tokens (output)
- **mistral-embed**: ~$0.0001 per 1K tokens

**Example cost calculation**:
- 10,000 documents × 500 tokens each = 5M tokens
- Embedding cost: 5M × $0.0001/1K = ~$0.50
- 100 queries × 2K tokens = 200K tokens
- Query cost: 200K × $0.004/1K = ~$0.80
- **Total: ~$1.30** for this workload

See [Mistral pricing](https://mistral.ai/pricing/) for current rates.

## Troubleshooting

### Error: "Invalid API key"

**Solution**: Check that your API key is correctly set:

```bash
echo $MISTRAL_API_KEY  # Should print your key
```

### Error: "Rate limit exceeded"

**Solution**: You're making too many requests. Options:
1. Wait a few seconds and retry
2. Upgrade to a paid tier for higher limits
3. Add retry logic with exponential backoff

### Error: "Model not found"

**Solution**: Check model name spelling:

```yaml
mistral:
  model_name: "mistral-small-latest"  # Correct
  # NOT "mistral-small" or "mistral-small-3.2"
```

### Slow responses

**Possible causes**:
1. Network latency - Check your internet connection
2. Model is large - Try `mistral-small-latest` instead of `mistral-large-latest`
3. Long prompts - Reduce `top_k` to retrieve fewer documents

## Advanced Configuration

### Adjusting Temperature

```yaml
mistral:
  temperature: 0.1  # More deterministic, focused
  # vs
  temperature: 0.7  # More creative, varied
```

### Increasing Max Tokens

```yaml
mistral:
  max_tokens: 1024  # For longer answers
```

### Optimizing for Cost

1. Use `mistral-small-latest` (cheapest)
2. Reduce `top_k` (fewer context docs = fewer input tokens)
3. Reduce `max_tokens` (shorter answers)

### Optimizing for Quality

1. Use `mistral-large-latest` (best model)
2. Increase `top_k` (more context)
3. Increase `max_tokens` (more detailed answers)
4. Adjust `temperature` to 0.3-0.5 for balanced responses

## Next Steps

- See [PROVIDER_COMPARISON.md](PROVIDER_COMPARISON.md) to compare with other providers
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See main [README.md](../README.md) for full usage guide

## Support

- Mistral Documentation: https://docs.mistral.ai/
- Mistral Discord: https://discord.gg/mistralai
- RainRAG Issues: https://github.com/yourusername/rainrag/issues
