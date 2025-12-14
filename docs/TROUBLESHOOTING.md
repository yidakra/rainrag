# Troubleshooting Guide

This guide helps you diagnose and fix common issues with RainRAG.

## Table of Contents

- [API Key Issues](#api-key-issues)
- [Provider Errors](#provider-errors)
- [Embedding Issues](#embedding-issues)
- [Qdrant Connection Issues](#qdrant-connection-issues)
- [Performance Issues](#performance-issues)
- [Configuration Issues](#configuration-issues)
- [MCP Server Issues](#mcp-server-issues)

---

## API Key Issues

### "Invalid API key" / "Unauthorized"

**Symptoms**: Error messages about invalid or unauthorized API keys

**Diagnosis**:
```bash
# Check which provider is configured
grep "provider:" config.yaml

# Check if env var is set
echo $MISTRAL_API_KEY
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

**Solutions**:

1. **Verify API key is set correctly**:
```bash
# Make sure the env var for your provider is set
export MISTRAL_API_KEY=your_key_here

# Or add to .env file
echo "MISTRAL_API_KEY=your_key_here" >> .env
```

2. **Check key format**:
- Mistral: No specific prefix
- OpenAI: Starts with `sk-`
- Claude: Starts with `sk-ant-`
- Gemini: No specific prefix

3. **Verify key is active**:
- Log into provider console
- Check that key hasn't been revoked
- Verify billing is set up (for paid tiers)

### "Quota exceeded" / "Insufficient credits"

**Symptoms**: Errors about quotas or credits

**Solutions**:

1. **Check your balance**:
- Mistral: [console.mistral.ai](https://console.mistral.ai/)
- OpenAI: [platform.openai.com/account/billing](https://platform.openai.com/account/billing)
- Claude: [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing)
- Gemini: [console.cloud.google.com/billing](https://console.cloud.google.com/billing)

2. **Add credits** or **wait for quota reset**

3. **Switch to free tier** (Gemini only):
```yaml
llm:
  provider: "gemini"
gemini:
  model_name: "gemini-2.5-flash"
```

---

## Provider Errors

### "Rate limit exceeded"

**Symptoms**: `429 Too Many Requests` errors

**Solutions**:

1. **Wait and retry** - Most rate limits reset within 1 minute

2. **Reduce query frequency**:
```bash
# Add delay between queries
for q in "query1" "query2"; do
  rainrag ask "$q"
  sleep 5  # Wait 5 seconds between queries
done
```

3. **Upgrade tier** - Higher tiers have higher rate limits

4. **Switch to faster provider**:
```yaml
llm:
  provider: "gemini"  # Has generous free tier limits
```

### "Model not found"

**Symptoms**: Error about model name not being recognized

**Solutions**:

1. **Check model name spelling**:
```yaml
# Correct names
mistral:
  model_name: "mistral-small-latest"  # ✅
openai:
  model_name: "gpt-4o-mini"  # ✅
claude:
  model_name: "claude-haiku-4-5-20251001"  # ✅
gemini:
  model_name: "gemini-2.5-flash"  # ✅
```

2. **Check provider documentation** for current model names

3. **Try a different model**:
```yaml
# If specific model fails, try default for that provider
mistral:
  model_name: "mistral-small-latest"
```

### "Connection error" / "Timeout"

**Symptoms**: Network errors, timeouts

**Solutions**:

1. **Check internet connection**:
```bash
ping api.mistral.ai
ping api.openai.com
```

2. **Check firewall/proxy** - Make sure API endpoints aren't blocked

3. **Increase timeout** (if applicable in code)

4. **Try different network** - Corporate networks may block APIs

---

## Embedding Issues

### "Vector size mismatch"

**Symptoms**: Error when indexing: "Vector dimension mismatch"

**Diagnosis**:
```bash
# Check configured vector size
grep "vector_size:" config.yaml

# Check embedding provider
grep "provider:" config.yaml | head -1
```

**Solutions**:

**Provider-specific vector sizes**:
```yaml
# Local embeddings
embedding:
  provider: "local"
qdrant:
  vector_size: 1024  # ✅

# Mistral embeddings
embedding:
  provider: "mistral"
qdrant:
  vector_size: 1024  # ✅

# OpenAI small embeddings
embedding:
  provider: "openai"
openai:
  embedding_model: "text-embedding-3-small"
qdrant:
  vector_size: 1536  # ✅ Must match!

# OpenAI large embeddings
embedding:
  provider: "openai"
openai:
  embedding_model: "text-embedding-3-large"
qdrant:
  vector_size: 3072  # ✅ Must match!

# Gemini embeddings
embedding:
  provider: "gemini"
qdrant:
  vector_size: 768  # ✅
```

**If you changed embedding providers**:
```bash
# Must re-run pipeline with correct vector_size in config
rainrag embed --force
rainrag index --recreate
```

### "Embedding failed" / "CUDA out of memory"

**Symptoms**: Errors during embedding generation

**Solutions for local embeddings**:

1. **Use CPU instead of GPU**:
```yaml
embedding:
  provider: "local"
  device: "cpu"  # Changed from "cuda"
```

2. **Reduce batch size**:
```yaml
embedding:
  batch_size: 16  # Reduced from 32
```

3. **Switch to API embeddings** (no local compute needed):
```yaml
embedding:
  provider: "mistral"  # or "openai" or "gemini"
```

---

## Qdrant Connection Issues

### "Connection refused" / "Cannot connect to Qdrant"

**Symptoms**: Errors connecting to Qdrant at localhost:6333

**Diagnosis**:
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Check if port 6333 is in use
lsof -i :6333  # Linux/Mac
netstat -an | grep 6333  # Windows
```

**Solutions**:

1. **Start Qdrant**:
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.1
```

2. **Check correct host/port in config**:
```yaml
qdrant:
  host: "localhost"  # or IP address
  port: 6333
```

3. **If using remote Qdrant**:
```yaml
qdrant:
  host: "qdrant.example.com"
  port: 6333
```

### "Collection not found"

**Symptoms**: Error that collection doesn't exist

**Solutions**:

1. **Create collection by running index**:
```bash
rainrag index
```

2. **Recreate collection** (if corrupted):
```bash
rainrag index --recreate
```

3. **Check collection name matches**:
```yaml
qdrant:
  collection_name: "broadcast_transcripts"  # Must match across all commands
```

---

## Performance Issues

### Slow query responses

**Possible causes and solutions**:

1. **Slow LLM provider**:
```yaml
# Switch to faster model
llm:
  provider: "gemini"  # Gemini Flash is very fast
gemini:
  model_name: "gemini-2.5-flash"
```

2. **Too many context documents**:
```yaml
# Reduce top_k
mistral:
  top_k: 3  # Reduced from 5 or 10
```

```bash
# Or via CLI
rainrag ask "question" --top-k 3
```

3. **Large max_tokens**:
```yaml
# Reduce max output length
openai:
  max_tokens: 256  # Reduced from 512
```

4. **Network latency** - Check internet connection

### Slow embedding generation

**Solutions**:

1. **Use GPU for local embeddings**:
```yaml
embedding:
  provider: "local"
  device: "cuda"  # Much faster than CPU
```

2. **Increase batch size** (if you have enough RAM/VRAM):
```yaml
embedding:
  batch_size: 64  # Increased from 32
```

3. **Use API embeddings** (no local computation):
```yaml
embedding:
  provider: "mistral"  # or "openai" or "gemini"
```

### High costs

**Solutions**:

1. **Switch to cheaper models**:
```yaml
# Instead of GPT-4o ($10/1M tokens output)
llm:
  provider: "openai"
openai:
  model_name: "gpt-4o-mini"  # $0.60/1M tokens output

# Or use Gemini free tier
llm:
  provider: "gemini"
gemini:
  model_name: "gemini-2.5-flash"
```

2. **Reduce context size**:
```yaml
# Fewer documents = fewer input tokens
mistral:
  top_k: 3  # Instead of 10
```

3. **Reduce max_tokens**:
```yaml
# Shorter answers = fewer output tokens
openai:
  max_tokens: 256  # Instead of 512 or 1024
```

4. **Use local embeddings** (free):
```yaml
embedding:
  provider: "local"
```

---

## Configuration Issues

### "Config file not found"

**Symptoms**: `config.yaml` not found

**Solutions**:

1. **Create from example**:
```bash
cp config.yaml.example config.yaml
```

2. **Specify path explicitly**:
```bash
rainrag --config /path/to/config.yaml ingest
```

### "Invalid configuration"

**Symptoms**: Validation errors when loading config

**Solutions**:

1. **Check YAML syntax**:
```bash
# Validate YAML
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

2. **Check required fields are present**:
```yaml
paths:  # Required
  archive_root: "/path/to/vtt"
  docs_output: "./data/docs.jsonl"

embedding:  # Required
  provider: "mistral"

qdrant:  # Required
  host: "localhost"
  port: 6333

llm:  # Required
  provider: "mistral"
```

3. **Check provider-specific config matches selected provider**:
```yaml
llm:
  provider: "openai"  # If using OpenAI...

openai:  # ...must have openai config
  api_key: ""
  model_name: "gpt-4o-mini"
```

---

## MCP Server Issues

### "MCP server won't start"

**Symptoms**: Server fails to initialize or crashes immediately

**Diagnosis**:
```bash
# Check if Qdrant is running
curl http://localhost:6333

# Check config is valid
rainrag info

# Check logs
tail -f ./logs/rainrag.log
```

**Solutions**:

1. **Ensure Qdrant is running**:
```bash
# Start Qdrant with Docker
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:v1.12.1
```

2. **Verify config.yaml is valid**:
```bash
# Test config loading
python -c "from rainrag.config import load_config; load_config('config.yaml')"
```

3. **Check API keys are set** (if using API-based embeddings/LLM):
```bash
echo $MISTRAL_API_KEY
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $GOOGLE_API_KEY
```

4. **Install MCP dependencies**:
```bash
poetry install  # Ensure mcp[cli] is installed
```

### "Claude Desktop doesn't detect MCP server"

**Symptoms**: No server icon in Claude Desktop, server not appearing in tools

**Diagnosis**:
```bash
# Check Claude Desktop config path
# macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
# Windows: %APPDATA%\Claude\claude_desktop_config.json

# Verify rainrag command is in PATH
which rainrag

# Test server starts manually
rainrag mcp
```

**Solutions**:

1. **Use absolute paths in config**:
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

2. **Verify JSON syntax**:
```bash
# Validate JSON on macOS/Linux
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | python -m json.tool
```

3. **Check rainrag is in PATH**:
```bash
# If using poetry
poetry shell
which rainrag  # Should show path to rainrag

# Add to PATH if needed
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

4. **Restart Claude Desktop completely**:
- Quit Claude Desktop (not just close window)
- Wait 5 seconds
- Relaunch Claude Desktop

5. **Check Claude Desktop logs** (if available):
- macOS: `~/Library/Logs/Claude/`
- Windows: `%APPDATA%\Claude\logs\`

### "MCP queries timing out"

**Symptoms**: Queries take too long or timeout

**Solutions**:

1. **Use API-based embeddings instead of local**:
```yaml
embedding:
  provider: "mistral"  # or "openai", "gemini"
```

2. **Use faster LLM models**:
```yaml
llm:
  provider: "gemini"
gemini:
  model_name: "gemini-2.5-flash"  # Faster than pro
```

3. **Reduce context chunks**:
```yaml
mistral:
  top_k: 3  # Reduce from 5 to 3
```

4. **Ensure Qdrant is local**:
```yaml
qdrant:
  host: "localhost"  # Not remote host
  port: 6333
```

### "HTTP transport not accessible"

**Symptoms**: Can't connect to MCP server via HTTP

**Solutions**:

1. **Check firewall allows port**:
```bash
# Test if port is open
curl http://localhost:8000/mcp
```

2. **Bind to all interfaces if remote access needed**:
```bash
rainrag mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

3. **Verify no port conflicts**:
```bash
# Check if port is already in use
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

### "MCP tools not appearing in AI assistant"

**Symptoms**: Server connected but tools (query_rag, retrieve_documents) don't show up

**Solutions**:

1. **Verify server is initialized**:
```bash
# Check logs for initialization messages
tail -f ./logs/rainrag.log | grep "MCP server"
```

2. **Test with MCP Inspector**:
```bash
# Install inspector
npm install -g @modelcontextprotocol/inspector

# Run server with HTTP
rainrag mcp --transport streamable-http

# In another terminal
npx @modelcontextprotocol/inspector
# Connect to http://localhost:8000/mcp
```

3. **Ensure collection has data**:
```bash
# Check Qdrant has indexed documents
rainrag info

# Should show points count > 0
```

4. **Re-run pipeline if no data**:
```bash
rainrag pipeline
```

---

## Environment Variable Issues

### Environment variables not loading

**Symptoms**: API keys not being recognized despite being in `.env`

**Solutions**:

1. **Ensure `.env` is in project root**:
```bash
ls -la .env  # Should exist in same directory as config.yaml
```

2. **Check `.env` file format**:
```bash
# Correct format (no spaces around =)
MISTRAL_API_KEY=your_key_here
OPENAI_API_KEY=sk-your_key

# Incorrect format
MISTRAL_API_KEY = your_key_here  # ❌ Spaces around =
```

3. **Manually export vars** (if .env not working):
```bash
export MISTRAL_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
```

4. **Verify vars are loaded**:
```bash
echo $MISTRAL_API_KEY  # Should print your key
```

---

## Getting Help

If you're still stuck:

1. **Enable verbose logging**:
```bash
rainrag ask "question" --verbose
```

2. **Check logs**:
```bash
tail -f logs/rainrag.log
```

3. **Test with minimal config**:
```yaml
# Simplest possible config
embedding:
  provider: "gemini"  # Free tier
llm:
  provider: "gemini"  # Free tier
```

4. **Search existing issues**:
- [RainRAG GitHub Issues](https://github.com/yourusername/rainrag/issues)

5. **Create new issue** with:
- Error message (full output)
- Your `config.yaml` (remove API keys!)
- Steps to reproduce
- Provider being used
- Python/OS version

---

## Quick Fixes Checklist

**Before asking for help, try these**:

- [ ] Restart Qdrant: `docker restart <qdrant-container>`
- [ ] Verify API keys are set: `echo $MISTRAL_API_KEY`
- [ ] Check internet connection
- [ ] Try different provider: `llm: provider: "gemini"`
- [ ] Recreate Qdrant index: `rainrag index --recreate`
- [ ] Clear cache: `rm -rf embeddings/*` then `rainrag embed --force`
- [ ] Check logs: `tail logs/rainrag.log`
- [ ] Update dependencies: `poetry install`
- [ ] Restart services

---

## Provider-Specific Troubleshooting

See detailed provider guides:
- [MISTRAL_SETUP.md](MISTRAL_SETUP.md#troubleshooting)
- [OPENAI_SETUP.md](OPENAI_SETUP.md#troubleshooting)
- [CLAUDE_SETUP.md](CLAUDE_SETUP.md#troubleshooting)
- [GEMINI_SETUP.md](GEMINI_SETUP.md#troubleshooting)
