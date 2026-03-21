# Google Gemini Setup Guide

This guide will help you set up Google Gemini as your LLM and/or embedding provider for RainRAG.

## Why Gemini?

Google Gemini is an excellent choice for RainRAG because:

- ✅ **Free tier available** - Generous free quota for testing
- ✅ **Fast** - Low latency responses
- ✅ **Cost-effective** - Competitive pricing
- ✅ **Both LLM and embeddings** - Complete solution in one provider
- ✅ **Multilingual** - Strong support for many languages including Russian

## Prerequisites

- Google account
- Google AI Studio access (free)

## Step 1: Get Your API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Get API key** or **Create API key**
4. Choose to create a key in a new or existing Google Cloud project
5. Copy the API key

**Note**: Gemini offers a generous free tier, perfect for testing!

## Step 2: Configure RainRAG

### Environment Variable (Recommended)

Add to your `.env` file:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

Or export directly:

```bash
export GOOGLE_API_KEY=your_gemini_api_key_here
```

## Step 3: Select Gemini as Provider

### For Both LLM and Embeddings

Edit `config.yaml`:

```yaml
# Use Gemini for embeddings
embedding:
  provider: "gemini"

# Use Gemini for LLM
llm:
  provider: "gemini"

gemini:
  api_key: ""  # Use GOOGLE_API_KEY env var
  model_name: "gemini-2.5-flash"  # Recommended
  embedding_model: "models/text-embedding-004"
  max_tokens: 512
  temperature: 0.3

qdrant:
  vector_size: 768  # Gemini embeddings (models/text-embedding-004) are 768-dimensional
```

### For LLM Only (with other embedding provider)

```yaml
embedding:
  provider: "mistral"  # or "local", "openai"

llm:
  provider: "gemini"

gemini:
  api_key: ""
  model_name: "gemini-2.5-flash"

qdrant:
  vector_size: 1024  # Depends on embedding provider
```

## Available Models

### LLM Models

| Model | Use Case | Performance | Cost | Context |
|-------|----------|-------------|------|---------|
| `gemini-2.5-flash` | **Recommended** - Fast, efficient | Great | $ | 1M tokens |
| `gemini-2.5-pro` | High quality, complex tasks | Excellent | $$ | 2M tokens |
| `gemini-2.5-flash-exp` | Experimental, latest features | Great | $ | 1M tokens |

### Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `models/text-embedding-004` | 768 | Standard embeddings (recommended) |
| `models/embedding-001` | 768 | Legacy model |

## Step 4: Test Your Setup

```bash
rainrag ask "What is artificial intelligence?" --verbose
```

You should see:
- "Using provider: gemini" in the output
- A generated answer
- Context documents retrieved from Qdrant

## Configuration Examples

### Full Gemini Setup (LLM + Embeddings)

```yaml
embedding:
  provider: "gemini"

llm:
  provider: "gemini"

gemini:
  api_key: ""  # Use GOOGLE_API_KEY env var
  model_name: "gemini-2.5-flash"
  embedding_model: "models/text-embedding-004"
  max_tokens: 512
  temperature: 0.3

qdrant:
  vector_size: 768  # For Gemini embeddings (models/text-embedding-004)
```

### Gemini LLM + Local Embeddings (Free Embeddings)

```yaml
embedding:
  provider: "local"
  model_name: "intfloat/multilingual-e5-large"
  device: "cuda"

llm:
  provider: "gemini"

gemini:
  api_key: ""
  model_name: "gemini-2.5-flash"

qdrant:
  vector_size: 1024  # For local embeddings
```

### High-Quality Setup (Pro Model)

```yaml
embedding:
  provider: "gemini"

llm:
  provider: "gemini"

gemini:
  api_key: ""
  model_name: "gemini-2.5-pro"  # Best quality
  embedding_model: "models/text-embedding-004"
  max_tokens: 2048
  temperature: 0.3

qdrant:
  vector_size: 768
```

## Pricing

### Free Tier (as of January 2025)

- **15 requests per minute** (RPM)
- **1 million tokens per minute** (TPM)
- **1,500 requests per day** (RPD)
- **Gemini 2.5 Flash**: Free for up to 10M tokens/month

### Paid Tier (Google Cloud)

**Gemini Flash** (per 1M tokens):
- Input: $0.075
- Output: $0.30

**Gemini Pro** (per 1M tokens):
- Input: $1.25
- Output: $5.00

**Embeddings**:
- $0.00001 per 1K characters (very cheap!)

**Example cost** with paid tier (100 queries):
- Input: 200K tokens × $0.075/1M = $0.015
- Output: 50K tokens × $0.30/1M = $0.015
- Embeddings: ~$0.005
- **Total: ~$0.035** (very affordable!)

See [Gemini pricing](https://ai.google.dev/pricing) for current rates.

## Troubleshooting

### Error: "API key not valid"

Check your API key:

```bash
echo $GOOGLE_API_KEY
```

Make sure you copied it correctly from Google AI Studio.

### Error: "Quota exceeded"

You've hit rate limits:
1. **Free tier**: Wait for quota to reset (per minute/day limits)
2. **Upgrade to paid**: Set up billing in Google Cloud
3. **Reduce request frequency**

### Error: "Resource exhausted"

You're making too many requests. Options:
1. Wait and retry
2. Implement exponential backoff
3. Upgrade to paid tier for higher limits

### Slow responses

Possible causes:
1. Network latency - Check connection
2. Large prompts - Reduce `top_k`
3. Use `gemini-2.5-flash` instead of `pro` for speed

## Best Practices

### Cost Optimization

1. **Use free tier** for development and testing
2. **Use Flash model** (15x cheaper than Pro)
3. **Use Gemini embeddings** (extremely cheap)
4. **Reduce max_tokens** to needed length
5. **Monitor usage** in Google Cloud Console

### Quality Optimization

1. **Use Pro model** for complex reasoning
2. **Increase top_k** for more context
3. **Leverage large context window** (1M-2M tokens)
4. **Adjust temperature**: 0.0-0.3 for focused, 0.3-0.7 for creative

### Free Tier Tips

The free tier is generous enough for:
- Development and testing
- Personal projects
- Low-volume production (< 100 queries/day)

**Staying within limits**:
- 15 RPM = 1 query every 4 seconds
- 1,500 RPD = Perfect for personal use

## Advanced Configuration

### Safety Settings

Gemini has built-in safety filters. If you get blocked responses:

```yaml
gemini:
  # These settings are in the API, not config.yaml
  # See Google AI documentation for safety settings
```

### Temperature Tuning

```yaml
gemini:
  temperature: 0.0  # Most deterministic
  # vs
  temperature: 1.0  # Most creative
```

## Switching from Other Providers

### From Mistral to Gemini

If switching embeddings:

1. **Update config.yaml**:
```yaml
embedding:
  provider: "gemini"

llm:
  provider: "gemini"

qdrant:
  vector_size: 768  # Change from 1024
```

2. **Re-run pipeline**:
```bash
rainrag embed --force
rainrag index --recreate
```

3. **Test**:
```bash
rainrag ask "test" --verbose
```

## Next Steps

- See [PROVIDER_COMPARISON.md](PROVIDER_COMPARISON.md) to compare providers
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See main [README.md](../README.md) for full usage guide

## Support

- Google AI Documentation: https://ai.google.dev/docs
- Google AI Studio: https://makersuite.google.com/
- RainRAG Issues: https://github.com/yourusername/rainrag/issues
