# Anthropic Claude Setup Guide

This guide will help you set up Anthropic Claude as your LLM provider for RainRAG.

## Why Claude?

Anthropic Claude is an excellent choice for RainRAG because:

- ✅ **Safety-focused** - Built with responsible AI principles
- ✅ **Long context** - 200K token context window
- ✅ **High quality** - Excellent reasoning and analysis
- ✅ **Reliable** - Consistent, predictable outputs
- ✅ **Multilingual** - Strong support for Russian and English

**Note**: Claude does not provide embedding models, so you'll need to use another provider (Mistral, OpenAI, Gemini, or local) for embeddings.

## Prerequisites

- Anthropic account
- Credit card for API billing
- $5 minimum initial credit

## Step 1: Get Your API Key

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Navigate to **API Keys**
4. Click **Create Key**
5. Give it a name (e.g., "RainRAG")
6. Copy the API key (starts with `sk-ant-`)

## Step 2: Configure RainRAG

### Environment Variable (Recommended)

Add to your `.env` file:

```bash
ANTHROPIC_API_KEY=sk-ant-your_api_key_here
```

Or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-your_api_key_here
```

## Step 3: Select Claude as LLM Provider

Edit `config.yaml`:

```yaml
# Choose embedding provider (Claude doesn't provide embeddings)
embedding:
  provider: "mistral"  # or "local", "openai", "gemini"

# Select Claude for LLM
llm:
  provider: "claude"

claude:
  api_key: ""  # Use ANTHROPIC_API_KEY env var
  model_name: "claude-haiku-4-5-20251001"  # Recommended
  max_tokens: 512
  temperature: 0.3
```

## Available Models

| Model | Use Case | Performance | Cost | Context |
|-------|----------|-------------|------|---------|
| `claude-haiku-4-5-20251001` | **Recommended** - Fast, efficient | Great | $ | 200K |
| `claude-sonnet-4-5-20250514` | Balanced quality/cost | Excellent | $$ | 200K |
| `claude-opus-4-20250514` | Maximum capability | Excellent | $$$ | 200K |

## Step 4: Test Your Setup

```bash
rainrag ask "What is machine learning?" --verbose
```

You should see:
- "Using provider: claude" in the output
- A generated answer from Claude
- Context documents retrieved from Qdrant

## Configuration Examples

### Claude LLM + Mistral Embeddings (Recommended)

```yaml
embedding:
  provider: "mistral"

llm:
  provider: "claude"

mistral:
  api_key: ""

claude:
  api_key: ""
  model_name: "claude-haiku-4-5-20251001"

qdrant:
  vector_size: 1024  # For Mistral embeddings
```

### Claude LLM + Local Embeddings (Cost Optimization)

```yaml
embedding:
  provider: "local"
  model_name: "intfloat/multilingual-e5-large"
  device: "cuda"

llm:
  provider: "claude"

claude:
  api_key: ""
  model_name: "claude-haiku-4-5-20251001"

qdrant:
  vector_size: 1024  # For local embeddings
```

### High-Quality Setup (Opus)

```yaml
embedding:
  provider: "openai"

llm:
  provider: "claude"

openai:
  embedding_model: "text-embedding-3-large"

claude:
  api_key: ""
  model_name: "claude-opus-4-20250514"  # Best quality
  max_tokens: 2048

qdrant:
  vector_size: 3072  # For OpenAI large embeddings
```

## Pricing

As of January 2025 (per million tokens):

| Model | Input | Output |
|-------|-------|--------|
| Haiku | $0.80 | $4.00 |
| Sonnet | $3.00 | $15.00 |
| Opus | $15.00 | $75.00 |

**Example cost** (100 queries with Haiku):
- Input: 200K tokens × $0.80/1M = $0.16
- Output: 50K tokens × $4.00/1M = $0.20
- **Total: ~$0.36**

See [Anthropic pricing](https://www.anthropic.com/pricing) for current rates.

## Troubleshooting

### Error: "Invalid API key"

Check your key starts with `sk-ant-`:

```bash
echo $ANTHROPIC_API_KEY
```

### Error: "Credit balance too low"

Add credits to your Anthropic account:
1. Go to [Billing](https://console.anthropic.com/settings/billing)
2. Add payment method
3. Purchase credits

### Error: "Rate limit exceeded"

Wait and retry, or upgrade your tier for higher limits.

## Best Practices

### Cost Optimization

1. Use **Haiku** for most queries (5x cheaper than Sonnet)
2. Use **local or Mistral embeddings** (Claude doesn't provide them)
3. Reduce `max_tokens` to needed length
4. Reduce `top_k` to minimize input tokens

### Quality Optimization

1. Use **Opus** for complex analysis
2. Use **Sonnet** for balanced quality/cost
3. Leverage 200K context window for more documents
4. Adjust temperature: 0.0-0.3 for focused, 0.3-0.7 for balanced

## Next Steps

- See [PROVIDER_COMPARISON.md](PROVIDER_COMPARISON.md) to compare providers
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See main [README.md](../README.md) for full usage guide

## Support

- Anthropic Documentation: https://docs.anthropic.com/
- Anthropic Support: https://support.anthropic.com/
- RainRAG Issues: https://github.com/yourusername/rainrag/issues
