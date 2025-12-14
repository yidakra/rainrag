# OpenAI Setup Guide

This guide will help you set up OpenAI (GPT-4, ChatGPT) as your LLM and/or embedding provider for RainRAG.

## Why OpenAI?

OpenAI is a great choice for RainRAG because:

- ✅ **State-of-the-art models** - GPT-4o and o1 are among the best LLMs available
- ✅ **Widely adopted** - Extensive documentation and community support
- ✅ **Reliable** - High uptime and consistent performance
- ✅ **Advanced embeddings** - text-embedding-3 models offer excellent quality
- ✅ **Familiar** - Many users already have OpenAI accounts

## Prerequisites

- OpenAI account
- Credit card for API billing (no free tier for API)
- $5 minimum initial credit

## Step 1: Get Your API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to **API keys** (https://platform.openai.com/api-keys)
4. Click **Create new secret key**
5. Give it a name (e.g., "RainRAG")
6. Copy the API key (you won't be able to see it again!)

**Note**: You'll need to add billing information and credits before you can make API calls.

## Step 2: Configure RainRAG

### Option A: Environment Variable (Recommended)

Add to your `.env` file:

```bash
OPENAI_API_KEY=sk-your_api_key_here
```

Or export directly:

```bash
export OPENAI_API_KEY=sk-your_api_key_here
```

### Option B: Direct Configuration

Edit `config.yaml`:

```yaml
openai:
  api_key: "sk-your_api_key_here"  # Not recommended - use env var instead
```

## Step 3: Select OpenAI as Provider

### For LLM (Answer Generation)

Edit `config.yaml`:

```yaml
llm:
  provider: "openai"

openai:
  model_name: "gpt-4o-mini"  # Recommended for cost/performance balance
  max_tokens: 512
  temperature: 0.3
```

### For Embeddings (Vector Search)

Edit `config.yaml`:

```yaml
embedding:
  provider: "openai"

openai:
  embedding_model: "text-embedding-3-small"  # 1536 dimensions

qdrant:
  vector_size: 1536  # Must match embedding model
```

**Important**: OpenAI embeddings have different dimensions than Mistral/local:
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-3-large`: 3072 dimensions

You must update `qdrant.vector_size` to match!

## Available Models

### LLM Models

| Model | Use Case | Performance | Cost | Context |
|-------|----------|-------------|------|---------|
| `gpt-4o-mini` | **Recommended** - Best value | Great | $ | 128K tokens |
| `gpt-4o` | High quality, multimodal | Excellent | $$$ | 128K tokens |
| `gpt-4-turbo` | Previous generation | Great | $$$ | 128K tokens |
| `gpt-3.5-turbo` | Legacy, cheaper | Good | $ | 16K tokens |
| `o1-preview` | Advanced reasoning | Excellent | $$$$ | 128K tokens |
| `o1-mini` | Reasoning, cost-effective | Great | $$ | 128K tokens |

### Embedding Models

| Model | Dimensions | Performance | Cost |
|-------|------------|-------------|------|
| `text-embedding-3-small` | 1536 | ⭐⭐⭐⭐ | $ |
| `text-embedding-3-large` | 3072 | ⭐⭐⭐⭐⭐ | $$ |
| `text-embedding-ada-002` | 1536 | ⭐⭐⭐ (legacy) | $ |

## Step 4: Test Your Setup

### Test CLI Query

```bash
rainrag ask "What is machine learning?" --verbose
```

You should see:
- "Using provider: openai" in the output
- A generated answer from GPT
- Context documents retrieved from Qdrant

### Test Python API

```python
from rainrag.config import load_config
from rainrag.query import RAGQueryEngine

config = load_config("config.yaml")
engine = RAGQueryEngine(config)
engine.initialize()

result = engine.query(
    question="Explain neural networks",
    top_k=5,
    language="en"
)

print(result["answer"])
```

## Configuration Examples

### OpenAI LLM + OpenAI Embeddings

```yaml
embedding:
  provider: "openai"

llm:
  provider: "openai"

openai:
  api_key: ""  # Use OPENAI_API_KEY env var
  model_name: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"
  max_tokens: 512
  temperature: 0.3

qdrant:
  vector_size: 1536  # For text-embedding-3-small
```

### OpenAI LLM + Local Embeddings (Cost Optimization)

```yaml
# Use free local embeddings + paid OpenAI LLM
embedding:
  provider: "local"
  model_name: "intfloat/multilingual-e5-large"

llm:
  provider: "openai"

openai:
  api_key: ""
  model_name: "gpt-4o-mini"

qdrant:
  vector_size: 1024  # For local embeddings
```

### High-Quality Setup (GPT-4o + Large Embeddings)

```yaml
embedding:
  provider: "openai"

llm:
  provider: "openai"

openai:
  api_key: ""
  model_name: "gpt-4o"  # Best quality
  embedding_model: "text-embedding-3-large"  # Best embeddings
  max_tokens: 1024
  temperature: 0.3

qdrant:
  vector_size: 3072  # For text-embedding-3-large
```

## Pricing

As of January 2025:

**LLM Pricing** (per 1M tokens):
- `gpt-4o-mini`: $0.150 (input) / $0.600 (output)
- `gpt-4o`: $2.50 (input) / $10.00 (output)
- `o1-mini`: $3.00 (input) / $12.00 (output)
- `o1-preview`: $15.00 (input) / $60.00 (output)

**Embedding Pricing** (per 1M tokens):
- `text-embedding-3-small`: $0.020
- `text-embedding-3-large`: $0.130

**Example cost calculation** (with gpt-4o-mini + small embeddings):
- 10,000 documents × 500 tokens each = 5M tokens
- Embedding cost: 5M × $0.020 / 1M = $0.10
- 100 queries × 2K tokens input + 500 tokens output = 250K tokens
- Query cost: (200K × $0.150 / 1M) + (50K × $0.600 / 1M) = $0.06
- **Total: ~$0.16** for this workload

See [OpenAI pricing](https://openai.com/api/pricing/) for current rates.

## Troubleshooting

### Error: "Incorrect API key provided"

**Solution**: Check your API key:

```bash
echo $OPENAI_API_KEY  # Should start with "sk-"
```

Make sure you copied the entire key.

### Error: "You exceeded your current quota"

**Solution**: You need to add credits to your OpenAI account:
1. Go to [Billing](https://platform.openai.com/account/billing)
2. Add payment method
3. Add credits ($5 minimum)

### Error: "Rate limit reached"

**Solution**: You're making too many requests:
1. Check your [rate limits](https://platform.openai.com/account/rate-limits)
2. Implement exponential backoff
3. Upgrade to higher tier if needed

### Error: "Model not found"

**Solution**: Check model name:

```yaml
openai:
  model_name: "gpt-4o-mini"  # Correct
  # NOT "gpt-4-mini" or "gpt4o-mini"
```

### Slow embedding generation

**Solution**: OpenAI API can be slower for large batches:
1. Use `text-embedding-3-small` (faster than large)
2. Consider local embeddings for large datasets
3. Check network latency

## Best Practices

### Cost Optimization

1. **Use gpt-4o-mini** for most queries (10-40x cheaper than gpt-4o)
2. **Use text-embedding-3-small** (6.5x cheaper than large)
3. **Reduce top_k** to minimize input tokens
4. **Set appropriate max_tokens** (don't use more than needed)
5. **Consider local embeddings** for large datasets (one-time generation)

### Quality Optimization

1. **Use gpt-4o or o1-preview** for complex queries
2. **Use text-embedding-3-large** for better retrieval
3. **Increase top_k** for more context
4. **Adjust temperature**:
   - `0.0-0.3`: Focused, deterministic
   - `0.3-0.7`: Balanced
   - `0.7-1.0`: Creative, varied

### Monitoring Usage

Track your costs in [OpenAI Dashboard](https://platform.openai.com/usage):
1. Set usage limits to avoid surprises
2. Enable email alerts
3. Monitor daily spend

## Advanced Configuration

### Using Different Models for Different Tasks

You can switch models by editing `config.yaml`:

```yaml
openai:
  model_name: "gpt-4o-mini"  # For general queries
  # Change to "gpt-4o" for complex analysis
```

### Optimizing Temperature

```yaml
openai:
  temperature: 0.1  # Very focused, consistent
  # vs
  temperature: 0.5  # Balanced creativity
```

### Handling Long Documents

OpenAI models have large context windows (128K tokens):

```yaml
openai:
  model_name: "gpt-4o"
  max_tokens: 2048  # For longer answers

# Retrieve more context
mistral:
  top_k: 10  # More documents in context
```

## Switching from Other Providers

### From Mistral to OpenAI

If you're already using Mistral embeddings and want to switch to OpenAI:

1. **Update config.yaml**:
```yaml
embedding:
  provider: "openai"

llm:
  provider: "openai"

qdrant:
  vector_size: 1536  # Change from 1024
```

2. **Re-run pipeline** (embeddings are incompatible):
```bash
rainrag embed --force
rainrag index --recreate
```

3. **Test**:
```bash
rainrag ask "test question" --verbose
```

## Next Steps

- See [PROVIDER_COMPARISON.md](PROVIDER_COMPARISON.md) to compare with other providers
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See main [README.md](../README.md) for full usage guide

## Support

- OpenAI Documentation: https://platform.openai.com/docs
- OpenAI Community: https://community.openai.com/
- RainRAG Issues: https://github.com/yourusername/rainrag/issues
