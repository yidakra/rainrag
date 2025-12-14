# MCP Server Setup Guide

This guide explains how to deploy RainRAG as an MCP (Model Context Protocol) server and integrate it with AI assistants like Claude Desktop, ChatGPT, and Cursor.

## What is MCP?

The Model Context Protocol (MCP) is an open standard introduced by Anthropic that enables AI applications to securely connect to external data sources and tools. By deploying RainRAG as an MCP server, you can query your video transcript database directly from AI assistants.

## Features

When deployed as an MCP server, RainRAG exposes:

### Tools

1. **query_rag**: Full RAG pipeline
   - Embeds your question
   - Retrieves relevant video transcript chunks
   - Generates a comprehensive answer using your configured LLM
   - Returns answer with context and metadata

2. **retrieve_documents**: Retrieval-only mode
   - Embeds your question
   - Retrieves relevant video transcript chunks
   - Returns documents without LLM generation (useful for seeing available context)

### Resources

- **config://current**: View current RainRAG configuration

## Prerequisites

1. **RainRAG Setup**: Complete the basic RainRAG setup (ingest, embed, index)
2. **Qdrant Running**: Ensure your Qdrant vector database is running
3. **MCP Package**: Install the MCP dependency (included in v0.1.0+)

## Installation

If you haven't already installed RainRAG:

```bash
# Clone and install
git clone https://github.com/yidakra/rainrag.git
cd rainrag
poetry install

# Or update existing installation to get MCP support
poetry install
```

## Configuration

The MCP server configuration is in `config.yaml`:

```yaml
mcp:
  # Transport protocol: "stdio", "streamable-http", or "sse"
  transport: "stdio"
  # Host for HTTP-based transports (ignored for stdio)
  host: "localhost"
  # Port for HTTP-based transports (ignored for stdio)
  port: 8000
```

### Transport Options

- **stdio**: Standard input/output (recommended for Claude Desktop, Cursor, and local tools)
- **streamable-http**: HTTP server (for remote connections and web-based integrations)
- **sse**: Server-Sent Events (alternative HTTP-based protocol)

## Running the MCP Server

### Basic Usage

```bash
# Run with default settings from config.yaml
rainrag mcp

# Run with custom config file
rainrag mcp --config /path/to/config.yaml

# Run with HTTP transport
rainrag mcp --transport streamable-http --port 8080

# Run with specific host and port
rainrag mcp --transport streamable-http --host 0.0.0.0 --port 9000
```

### Command Options

```
Options:
  -c, --config PATH     Path to configuration file [default: config.yaml]
  -t, --transport TEXT  Transport protocol (stdio, sse, streamable-http)
  -h, --host TEXT       Host for HTTP-based transports
  -p, --port INTEGER    Port for HTTP-based transports
  --help               Show this message and exit
```

## Integration with AI Assistants

### Claude Desktop

Claude Desktop (version 0.7.0+) has native MCP support.

1. **Create MCP Server Configuration**

   Add RainRAG to your Claude Desktop configuration file:

   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "rainrag": {
         "command": "rainrag",
         "args": ["mcp", "--config", "/absolute/path/to/rainrag/config.yaml"]
       }
     }
   }
   ```

   **Note**: Use absolute paths for the config file.

2. **Restart Claude Desktop**

3. **Verify Connection**

   In Claude Desktop, you should see a 🔌 icon or indicator showing the RainRAG server is connected.

4. **Use RainRAG**

   You can now ask Claude questions like:
   - "Use the query_rag tool to search for discussions about machine learning"
   - "What topics are covered in the video transcripts about AI?"
   - "Retrieve documents related to neural networks"

### Cursor

Cursor supports MCP servers for enhanced context.

Cursor typically reads MCP server configuration from `~/.cursor/mcp.json`.

1. **Add RainRAG Server**

   Create/edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "rainrag": {
      "command": "bash",
      "args": [
        "-lc",
        "cd /absolute/path/to/rainrag && poetry run rainrag mcp --config /absolute/path/to/rainrag/config.yaml"
      ]
    }
  }
}
```

2. **Restart Cursor**

3. **Use RainRAG**

   In Cursor Chat, ask it to call `retrieve_documents` or `query_rag`.

#### Important (stdio transport)

Cursor uses **stdio** transport. That means the MCP server must write **only JSON-RPC** to **stdout**.
Any extra prints/banners on stdout can break Cursor with JSON parse errors. RainRAG routes startup messages to stderr for stdio.

### ChatGPT (via HTTP)

ChatGPT (macOS app) uses “Connectors”. It typically requires an **HTTPS** URL and will reject private/LAN `http://…`
addresses with “Unsafe URL”. Use the HTTP transport plus an HTTPS tunnel.

1. **Run MCP Server with HTTP Transport (local)**

   ```bash
   # Bind locally; expose via tunnel (recommended)
   FASTMCP_TRANSPORT_SECURITY__ENABLE_DNS_REBINDING_PROTECTION=false \
     rainrag mcp --transport streamable-http --host 127.0.0.1 --port 8000
   ```

   Notes:
   - Disabling DNS-rebinding protection is **for tunneling/testing**; do not expose this directly to an untrusted network.
   - If you want LAN-only access (no tunnel), use `--host 0.0.0.0` and keep DNS protection enabled with an allowlist.

2. **Create an HTTPS tunnel (Cloudflare Quick Tunnel)**

   On the same machine:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

   It will print a URL like `https://<name>.trycloudflare.com`.

3. **Add connector in ChatGPT (Mac)**

   Use:
   - **MCP Server URL**: `https://<name>.trycloudflare.com/mcp`

4. **Test**

   Ask ChatGPT to call:
   - `retrieve_documents` with `question="Борис Кагарлицкий"`, `top_k=1`

### Testing with MCP Inspector

The MCP Inspector is a useful tool for testing your MCP server:

```bash
# Install MCP Inspector
npm install -g @modelcontextprotocol/inspector

# Run RainRAG with HTTP transport
rainrag mcp --transport streamable-http

# In another terminal, run the inspector
npx @modelcontextprotocol/inspector
```

Open your browser to the inspector URL and connect to `http://localhost:8000/mcp` to test the available tools.

## Usage Examples

Once integrated, you can use RainRAG from your AI assistant. Here are practical examples for different scenarios:

### Scenario 1: Finding Specific Topics (English)

**What you ask Claude/ChatGPT/Cursor:**
> "Use the query_rag tool to find out what was discussed about renewable energy in the videos"

**MCP Tool Call:**
```
Tool: query_rag
Arguments:
  question: "What topics were discussed about renewable energy?"
  language: "en"
  top_k: 5
```

**What you get back:**
- A comprehensive answer synthesized from the video transcripts
- Source citations with video filenames
- Relevance scores for each retrieved segment

### Scenario 2: Multilingual Search (Russian)

**What you ask:**
> "О чём говорили в последних видео про искусственный интеллект? Ответь по-русски."

**MCP Tool Call:**
```
Tool: query_rag
Arguments:
  question: "О чём говорили в видео про искусственный интеллект?"
  language: "ru"
  top_k: 5
```

**Result:**
- Answer in Russian
- Retrieves from both Russian and English transcripts
- Prioritizes based on semantic similarity

### Scenario 3: Getting Raw Context (No LLM Generation)

**What you ask:**
> "Use retrieve_documents to find 10 relevant transcript chunks about climate change without generating an answer"

**MCP Tool Call:**
```
Tool: retrieve_documents
Arguments:
  question: "climate change global warming environment"
  top_k: 10
```

**Result:**
- Just the retrieved document chunks
- No LLM processing (faster)
- Useful for examining raw context before asking detailed questions

### Scenario 4: Deep Dive with More Context

**What you ask:**
> "I need a detailed analysis of discussions about machine learning. Use 10 sources."

**MCP Tool Call:**
```
Tool: query_rag
Arguments:
  question: "What are all the topics covered regarding machine learning, including techniques, applications, and challenges?"
  language: "en"
  top_k: 10
```

**Result:**
- Comprehensive answer from 10 different transcript segments
- More context = better coverage but slower response

### Scenario 5: Checking System Configuration

**What you ask:**
> "What's the current RainRAG configuration?"

**MCP Resource:**
```
Resource: config://current
```

**Result:**
- Current embedding provider and model
- LLM provider and model
- Vector database details
- Collection information

### Scenario 6: Comparative Analysis

**What you ask:**
> "Compare what was said about solar energy versus nuclear energy in the videos"

**MCP Tool Calls (sequential):**
```
1. query_rag(question="solar energy advantages disadvantages", language="en", top_k=5)
2. query_rag(question="nuclear energy advantages disadvantages", language="en", top_k=5)
```

**Result:**
- The AI assistant will combine both results
- Provides comparative analysis based on video content

### Scenario 7: Time-Specific or Contextual Queries

**What you ask:**
> "What recent developments in AI were discussed?"

**MCP Tool Call:**
```
Tool: query_rag
Arguments:
  question: "recent developments artificial intelligence latest advances"
  language: "en"
  top_k: 7
```

**Note:** Results depend on your video transcript dates and content.

## Natural Language Examples

You don't need to explicitly call the tools - just ask naturally:

**Simple Questions:**
- "What do the videos say about quantum computing?"
- "Расскажи про обсуждение экономики в видео"
- "Summarize the main points about cryptocurrency"

**Complex Queries:**
- "What are the different perspectives on climate policy mentioned across multiple videos?"
- "Can you find all references to specific AI models or frameworks?"
- "What technical challenges were discussed regarding renewable energy implementation?"

**Follow-up Questions:**
- After getting an answer: "Can you provide more details about the second point?"
- "Which specific video discussed this in the most depth?"
- "Were there any counterarguments mentioned?"

## Tool Parameters Reference

### query_rag
- **question** (required): Your search query or question
- **language** (optional): "en" or "ru" - language for the response (default: "en")
- **top_k** (optional): Number of chunks to retrieve, 1-20 (default: 5)

### retrieve_documents
- **question** (required): Your search query
- **top_k** (optional): Number of chunks to retrieve (default: 5)

**Tips:**
- Use `top_k=3` for quick, focused answers
- Use `top_k=7-10` for comprehensive, detailed answers
- Use `retrieve_documents` first to see what context is available
- Specify `language="ru"` for Russian responses (works with both Russian and English transcripts)

## Troubleshooting

### MCP Server Won't Start

**Issue**: Server fails to initialize

**Solutions**:
- Verify Qdrant is running: `curl http://localhost:6333`
- Check your config.yaml is valid
- Ensure API keys are set (if using API-based embeddings or LLM)
- Check logs in `./logs/rainrag.log`

### Claude Desktop Not Detecting Server

**Issue**: No 🔌 icon or server indicator

**Solutions**:
- Verify the `claude_desktop_config.json` file syntax
- Use absolute paths (not relative like `./config.yaml`)
- Check that `rainrag` command is in your PATH
- Restart Claude Desktop completely
- Check Claude Desktop logs for connection errors

### Queries Returning No Results

**Issue**: Tools return empty or no documents

**Solutions**:
- Verify your Qdrant collection has indexed documents:
  ```bash
  rainrag info
  ```
- Re-run the indexing pipeline:
  ```bash
  rainrag pipeline
  ```
- Check the collection name in config.yaml matches Qdrant

### Slow Query Performance

**Issue**: Queries take a long time

**Solutions**:
- Use API-based embeddings (Mistral/OpenAI/Gemini) instead of local models for faster query embedding
- Reduce `top_k` value in config.yaml to retrieve fewer documents
- Ensure Qdrant is running locally (not remote) for fastest retrieval
- Use a faster LLM (e.g., `gemini-2.5-flash`, `gpt-4o-mini`, `mistral-small-latest`)

### HTTP Transport Issues

**Issue**: “Not Acceptable: Client must accept text/event-stream” or browser shows JSON error

**Solutions**:
- `/mcp` is not a “normal” endpoint; opening it in a browser will not work.
- A plain `curl http://localhost:8000/mcp` returns `406 Not Acceptable` (missing `Accept: text/event-stream`).
- The correct way to test is with an MCP client. Example (Python):

```bash
poetry run python - <<'PY'
import anyio
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

URL = "http://127.0.0.1:8000/mcp"

async def main():
    async with streamablehttp_client(URL) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print([t.name for t in tools.tools])

anyio.run(main)
PY
```

**Issue**: “Invalid Host header” (421) when accessed via LAN/tunnel

**Cause**: DNS rebinding protection rejects the request’s Host header.

**Solutions**:
- For LAN access: run with `--host 0.0.0.0` and ensure the Host is allowlisted.
- For HTTPS tunnels (e.g. `trycloudflare.com`): disable protection for testing:
  `FASTMCP_TRANSPORT_SECURITY__ENABLE_DNS_REBINDING_PROTECTION=false`

## Advanced Configuration

### Custom Embedding Provider

To use API-based embeddings for faster query times:

```yaml
embedding:
  provider: "mistral"  # or "openai", "gemini"
  # model_name only used for local provider
```

### Multiple Language Support

RainRAG supports both English and Russian. Specify the language in queries:

```python
# English
query_rag(question="What is discussed?", language="en")

# Russian
query_rag(question="О чём говорили?", language="ru")
```

### Adjusting Retrieval Context

Control the number of document chunks retrieved:

```yaml
mistral:
  top_k: 5  # Number of chunks to retrieve

openai:
  top_k: 7

claude:
  top_k: 10

gemini:
  top_k: 5
```

Or override in the query:
```python
query_rag(question="...", top_k=10)
```

## Security Considerations

### API Keys

- Store API keys in environment variables, not in config.yaml:
  ```bash
  export MISTRAL_API_KEY="your-key-here"
  export OPENAI_API_KEY="your-key-here"
  export ANTHROPIC_API_KEY="your-key-here"
  export GOOGLE_API_KEY="your-key-here"
  ```

### Network Exposure

- Use `stdio` transport for local-only access (most secure)
- If using HTTP transport, consider:
  - Running behind a reverse proxy with authentication
  - Using HTTPS/TLS encryption
  - Restricting to localhost (`127.0.0.1`) if not needed externally
  - Implementing rate limiting

### Data Privacy

- Video transcripts are stored locally in Qdrant
- When using API-based LLMs, retrieved context is sent to the API provider
- Consider data sensitivity when choosing LLM providers
- For maximum privacy, use local embeddings and local LLM (if available)

## Performance Optimization

### Expected Performance Metrics

**Typical Query Latency (End-to-End):**

| Configuration | Embedding | Retrieval | LLM Generation | Total |
|--------------|-----------|-----------|----------------|-------|
| **Fast** (API + Flash) | 0.5-1s | 0.1-0.3s | 1-2s | **2-4s** |
| **Balanced** (API + Standard) | 0.5-1s | 0.1-0.3s | 2-4s | **3-6s** |
| **Local** (Local model) | 2-5s | 0.1-0.3s | 2-4s | **5-10s** |
| **Quality** (API + Claude) | 0.5-1s | 0.1-0.3s | 4-8s | **5-10s** |

**Note:** First query may be slower due to model loading (local) or cold start.

### Optimization Strategies

#### Speed-Optimized Configuration

**Best for:** Interactive use, real-time queries, quick exploration

```yaml
embedding:
  provider: "mistral"  # Fast API-based embeddings

llm:
  provider: "gemini"

gemini:
  model_name: "gemini-2.5-flash"  # Fastest LLM
  max_tokens: 512
  top_k: 3  # Fewer documents = faster
```

**Expected latency:** 2-4 seconds per query

#### Quality-Optimized Configuration

**Best for:** Detailed analysis, comprehensive answers, research

```yaml
embedding:
  provider: "openai"

openai:
  embedding_model: "text-embedding-3-large"  # Higher quality embeddings

llm:
  provider: "claude"

claude:
  model_name: "claude-3-5-sonnet-20240620"  # Best quality
  max_tokens: 2048
  top_k: 10  # More context
```

**Expected latency:** 5-12 seconds per query

#### Balanced Configuration (Recommended)

**Best for:** General use, good mix of speed and quality

```yaml
embedding:
  provider: "mistral"

llm:
  provider: "openai"

openai:
  model_name: "gpt-4o-mini"  # Good balance
  max_tokens: 1024
  top_k: 5
```

**Expected latency:** 3-6 seconds per query

### Performance Tuning Tips

#### 1. Embedding Provider Choice

**Impact:** Affects initial query embedding time

- **Local** (`intfloat/multilingual-e5-large`): 2-5s first query, 1-2s subsequent
  - Pros: No API costs, no rate limits, works offline
  - Cons: Requires GPU/CPU resources, slower

- **Mistral API** (`mistral-embed`): 0.5-1s
  - Pros: Fast, reliable, good quality
  - Cons: API costs, requires internet

- **OpenAI API** (`text-embedding-3-small`): 0.3-0.7s
  - Pros: Very fast, excellent quality
  - Cons: Higher API costs

- **Gemini API** (`text-embedding-004`): 0.4-0.8s
  - Pros: Fast, cost-effective
  - Cons: Requires Google Cloud setup

**Recommendation:** Use Mistral or OpenAI API for MCP integration (better UX)

#### 2. LLM Provider Choice

**Impact:** Affects answer generation time

| Provider | Model | Speed | Quality | Cost |
|----------|-------|-------|---------|------|
| Gemini | `gemini-2.5-flash` | ⚡⚡⚡ Fastest | ⭐⭐⭐ Good | 💰 Cheap |
| OpenAI | `gpt-4o-mini` | ⚡⚡ Fast | ⭐⭐⭐⭐ Great | 💰💰 Medium |
| Mistral | `mistral-small-latest` | ⚡⚡ Fast | ⭐⭐⭐ Good | 💰💰 Medium |
| OpenAI | `gpt-4o` | ⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | 💰💰💰 High |
| Claude | `claude-3-5-sonnet` | ⚡ Moderate | ⭐⭐⭐⭐⭐ Excellent | 💰💰💰 High |

#### 3. Context Size (`top_k`)

**Impact:** Affects both retrieval time and LLM processing time

- `top_k: 3` - Quick answers, focused context (~2-3s total)
- `top_k: 5` - Balanced (default) (~3-6s total)
- `top_k: 7-10` - Comprehensive answers (~5-10s total)
- `top_k: 15-20` - Maximum coverage (~8-15s total)

**Guideline:**
- Simple factual queries: 3-5
- Complex analysis: 7-10
- Exhaustive research: 10-20

#### 4. Infrastructure Optimization

**Qdrant Location:**
- **Local** (same machine): 0.1-0.3s retrieval ✅
- **Local network**: 0.3-1s retrieval
- **Remote/Cloud**: 1-3s retrieval ❌

**MCP Server Location:**
- **Local** (stdio): Best for Claude Desktop/Cursor
- **Local network** (HTTP): Good for team use
- **Cloud** (HTTP): Slower but accessible anywhere

### Monitoring Performance

**Check query logs:**
```bash
tail -f ./logs/rainrag.log | grep "MCP"
```

**Test with MCP Inspector:**
```bash
# Time a query
time curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"tool":"query_rag","args":{"question":"test"}}'
```

### Troubleshooting Slow Queries

**If queries take >10 seconds:**

1. **Check embedding provider** - Switch from local to API
2. **Check Qdrant location** - Ensure it's running locally
3. **Reduce `top_k`** - Try 3 instead of 5
4. **Check network latency** - Test API response times
5. **Review LLM choice** - Use Flash/Mini models

**If queries take >30 seconds:**

- Likely a timeout or configuration issue
- Check logs: `tail -f ./logs/rainrag.log`
- Verify Qdrant is accessible: `curl http://localhost:6333`
- Check API keys are valid

## Next Steps

- **Customize Prompts**: Modify system messages in `src/rainrag/query.py`
- **Add More Tools**: Extend `src/rainrag/mcp_server.py` with additional MCP tools
- **Monitor Usage**: Check logs in `./logs/rainrag.log`
- **Scale Up**: Deploy with Docker/Kubernetes for production use (see main README)

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Claude Desktop MCP Guide](https://www.anthropic.com/docs/model-context-protocol)
- [RainRAG Documentation](https://github.com/yidakra/rainrag)

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `./logs/rainrag.log`
3. Open an issue on [GitHub](https://github.com/yidakra/rainrag/issues)
