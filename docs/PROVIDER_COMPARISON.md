# LLM and Embedding Provider Comparison

This guide helps you choose the right provider(s) for your RainRAG deployment.

## Quick Decision Tree

```
Do you have GPU with 16GB+ VRAM?
├─ YES → Consider local embeddings (free, fast)
└─ NO  → Use API embeddings

What's your priority?
├─ Cost → Gemini (free tier) or Mistral
├─ Quality → Claude Opus or GPT-4o
├─ Speed → Gemini Flash or Mistral Small
└─ Privacy → Mistral (European) or local embeddings

What's your budget?
├─ Free/Testing → Gemini (free tier) + local embeddings
├─ Low ($5-20/month) → Mistral LLM + Mistral embeddings
├─ Medium ($20-100/month) → OpenAI GPT-4o-mini or Claude Haiku
└─ High → OpenAI GPT-4o or Claude Opus
```

## LLM Provider Comparison

| Feature | Mistral | OpenAI | Claude | Gemini |
|---------|---------|--------|--------|--------|
| **Best For** | Balanced | Wide adoption | Safety | Free tier |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | Fast | Fast | Medium | Very Fast |
| **Cost ($/1M tokens)** | $2-6 | $0.15-10 | $0.80-75 | $0-5 |
| **Context Window** | 32K | 128K | 200K | 1M-2M |
| **Free Tier** | No | No | No | Yes ✅ |
| **Multilingual** | Excellent | Good | Good | Excellent |
| **Privacy** | EU-based | US-based | US-based | US-based |

### Recommended Models by Provider

**Mistral** (Default Recommended):
- `mistral-small-latest`: Best value, fast, accurate
- `mistral-large-latest`: Maximum capability

**OpenAI**:
- `gpt-4o-mini`: Best cost/performance
- `gpt-4o`: Maximum quality
- `o1-mini`: Advanced reasoning

**Claude**:
- `claude-haiku-4-5-20251001`: Fast, economical
- `claude-sonnet-4-5-20250514`: Balanced
- `claude-opus-4-20250514`: Maximum quality

**Gemini**:
- `gemini-2.5-flash`: Fast, free tier
- `gemini-2.5-pro`: High quality

## Embedding Provider Comparison

| Feature | Local | Mistral | OpenAI | Gemini |
|---------|-------|---------|--------|--------|
| **Best For** | High volume | Balanced | Integration | Cost |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | Very Fast* | Fast | Fast | Fast |
| **Cost** | Free | $ | $$ | $ |
| **Setup** | Complex | Simple | Simple | Simple |
| **Dimensions** | 1024 | 1024 | 1536/3072 | 768 |
| **Requirements** | GPU/CPU | API key | API key | API key |

*Speed depends on your hardware

### Recommended: Mistral Embeddings

For most users, we recommend **Mistral embeddings** because:
- Good quality/cost balance
- Same provider as recommended LLM (simplicity)
- 1024 dimensions (standard)
- Fast API

## Cost Comparison (Example Workload)

**Scenario**: 10,000 documents, 100 queries/day

### Embedding Cost (One-Time)

| Provider | Model | Cost |
|----------|-------|------|
| Local | multilingual-e5-large | $0 (free) |
| Mistral | mistral-embed | ~$0.50 |
| OpenAI | text-embedding-3-small | ~$0.10 |
| OpenAI | text-embedding-3-large | ~$0.65 |
| Gemini | text-embedding-004 | ~$0.05 |

### Query Cost (Monthly, 3000 queries)

| Provider | Model | Monthly Cost |
|----------|-------|--------------|
| Mistral | mistral-small-latest | ~$3-5 |
| OpenAI | gpt-4o-mini | ~$1-2 |
| OpenAI | gpt-4o | ~$20-30 |
| Claude | claude-haiku | ~$5-8 |
| Claude | claude-opus | ~$50-80 |
| Gemini | gemini-2.5-flash | ~$0-2 (free tier!) |
| Gemini | gemini-2.5-pro | ~$15-25 |

### Total Monthly Cost (LLM + Embeddings)

**Budget Option** (Gemini):
- Embeddings: Gemini ($0.05 one-time)
- LLM: Gemini Flash (free or ~$2/month)
- **Total: ~$2/month**

**Recommended Option** (Mistral):
- Embeddings: Mistral ($0.50 one-time)
- LLM: Mistral Small (~$4/month)
- **Total: ~$4/month**

**Quality Option** (Mixed):
- Embeddings: OpenAI Large ($0.65 one-time)
- LLM: Claude Sonnet (~$15/month)
- **Total: ~$15/month**

**Premium Option** (OpenAI/Claude):
- Embeddings: OpenAI Large ($0.65 one-time)
- LLM: GPT-4o or Claude Opus (~$25-80/month)
- **Total: ~$25-80/month**

## Performance Comparison

### Speed (Latency for typical query)

| Provider | LLM | Embeddings |
|----------|-----|------------|
| Local | N/A | 100-500ms* |
| Mistral | 1-3s | 200-500ms |
| OpenAI | 1-4s | 100-300ms |
| Claude | 2-5s | N/A |
| Gemini | 0.5-2s | 200-400ms |

*Depends on hardware

### Quality (Subjective assessment for RAG tasks)

**LLM Quality**:
1. Claude Opus ⭐⭐⭐⭐⭐ (Best reasoning)
2. GPT-4o ⭐⭐⭐⭐⭐ (Best overall)
3. o1-preview ⭐⭐⭐⭐⭐ (Best complex reasoning)
4. Gemini Pro ⭐⭐⭐⭐ (Very good)
5. Claude Sonnet ⭐⭐⭐⭐ (Very good)
6. Mistral Large ⭐⭐⭐⭐ (Very good)
7. GPT-4o-mini ⭐⭐⭐⭐ (Good)
8. Claude Haiku ⭐⭐⭐⭐ (Good)
9. Mistral Small ⭐⭐⭐⭐ (Good)
10. Gemini Flash ⭐⭐⭐⭐ (Good)

**Embedding Quality**:
1. OpenAI text-embedding-3-large ⭐⭐⭐⭐⭐
2. OpenAI text-embedding-3-small ⭐⭐⭐⭐
3. Mistral mistral-embed ⭐⭐⭐⭐
4. Local multilingual-e5-large ⭐⭐⭐⭐
5. Gemini text-embedding-004 ⭐⭐⭐⭐

## Use Case Recommendations

### Personal Project / Testing

**Recommendation**: Gemini (free tier)

```yaml
embedding:
  provider: "gemini"
llm:
  provider: "gemini"
gemini:
  model_name: "gemini-2.5-flash"
  embedding_model: "models/text-embedding-004"
```

**Why**: Free tier is generous, easy setup, good quality.

### Small Business / Production (< 1000 queries/day)

**Recommendation**: Mistral LLM + Mistral Embeddings

```yaml
embedding:
  provider: "mistral"
llm:
  provider: "mistral"
mistral:
  model_name: "mistral-small-latest"
```

**Why**: Best cost/performance, single provider simplicity, EU privacy.

### Medium Business / Production (1000-10000 queries/day)

**Recommendation**: OpenAI LLM (mini) + Mistral Embeddings

```yaml
embedding:
  provider: "mistral"
llm:
  provider: "openai"
openai:
  model_name: "gpt-4o-mini"
```

**Why**: Excellent quality, affordable, scalable.

### Enterprise / High Quality Requirements

**Recommendation**: Claude Opus + OpenAI Large Embeddings

```yaml
embedding:
  provider: "openai"
llm:
  provider: "claude"
openai:
  embedding_model: "text-embedding-3-large"
claude:
  model_name: "claude-opus-4-20250514"
```

**Why**: Maximum quality, long context, reliable.

### High Volume / Cost Sensitive

**Recommendation**: Local Embeddings + Gemini or Mistral LLM

```yaml
embedding:
  provider: "local"
  model_name: "intfloat/multilingual-e5-large"
  device: "cuda"
llm:
  provider: "gemini"  # or "mistral"
gemini:
  model_name: "gemini-2.5-flash"
```

**Why**: Free embeddings, cheap LLM, no per-query embedding costs.

### Privacy-Focused / Regulated Industry

**Recommendation**: Local Embeddings + Mistral LLM

```yaml
embedding:
  provider: "local"
llm:
  provider: "mistral"
mistral:
  model_name: "mistral-small-latest"
```

**Why**: Embeddings stay local, Mistral is EU-based with strong privacy.

### Research / Experimentation

**Recommendation**: Mix and match, easy provider switching

```yaml
# Week 1: Test Gemini
llm:
  provider: "gemini"

# Week 2: Test OpenAI
llm:
  provider: "openai"

# Week 3: Test Claude
llm:
  provider: "claude"

# Compare results and costs
```

**Why**: RainRAG makes it easy to switch providers without code changes.

## Feature Comparison

| Feature | Mistral | OpenAI | Claude | Gemini |
|---------|---------|--------|--------|--------|
| Embeddings | ✅ | ✅ | ❌ | ✅ |
| Function calling | ✅ | ✅ | ✅ | ✅ |
| JSON mode | ✅ | ✅ | ✅ | ✅ |
| Vision | ❌ | ✅ | ✅ | ✅ |
| Code execution | ❌ | ✅ | ❌ | ✅ |
| Multilingual | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Russian support | Excellent | Good | Good | Excellent |
| English support | Excellent | Excellent | Excellent | Excellent |

## Switching Providers

### Switching LLM Only (Easy)

Just change the provider in `config.yaml`:

```yaml
llm:
  provider: "openai"  # Change from "mistral" to "openai"
```

No re-indexing needed!

### Switching Embeddings (Requires Re-indexing)

1. Update `config.yaml`:
```yaml
embedding:
  provider: "openai"  # Changed from "mistral"
qdrant:
  vector_size: 1536  # Changed from 1024
```

2. Re-run pipeline:
```bash
rainrag embed --force
rainrag index --recreate
```

3. Test:
```bash
rainrag ask "test" --verbose
```

## Summary Recommendations

| Your Situation | Recommended Setup |
|----------------|-------------------|
| Just testing | Gemini (free tier) for everything |
| Small production | Mistral for everything |
| Need best quality | Claude Opus + OpenAI embeddings |
| High volume | Local embeddings + Gemini/Mistral LLM |
| Privacy-focused | Local embeddings + Mistral LLM |
| Cost-optimized | Gemini or Mistral Small |
| Existing OpenAI user | OpenAI for everything |

## Next Steps

- Read detailed setup guides:
  - [MISTRAL_SETUP.md](MISTRAL_SETUP.md)
  - [OPENAI_SETUP.md](OPENAI_SETUP.md)
  - [CLAUDE_SETUP.md](CLAUDE_SETUP.md)
  - [GEMINI_SETUP.md](GEMINI_SETUP.md)
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- See main [README.md](../README.md) for full usage guide
