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

1. **Open Cursor Settings**

   Go to Settings → Features → MCP Servers

2. **Add RainRAG Server**

   Add the following configuration:

   ```json
   {
     "rainrag": {
       "command": "rainrag",
       "args": ["mcp", "--config", "/absolute/path/to/rainrag/config.yaml"]
     }
   }
   ```

3. **Restart Cursor**

4. **Use RainRAG**

   In the chat or command palette, you can now use RainRAG tools to query your video transcripts.

### ChatGPT (via HTTP)

For ChatGPT or other web-based assistants, use the HTTP transport:

1. **Run MCP Server with HTTP Transport**

   ```bash
   rainrag mcp --transport streamable-http --host 0.0.0.0 --port 8000
   ```

2. **Configure Custom Action/Plugin**

   The server will be available at:
   ```
   http://your-server:8000/mcp
   ```

   **Note**: MCP HTTP integration with ChatGPT may require additional setup or plugins as ChatGPT's MCP support is still evolving. Check OpenAI's documentation for the latest integration methods.

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

Once integrated, you can use RainRAG from your AI assistant:

### Full RAG Query

```
Use query_rag with question "What are the main topics discussed in videos about renewable energy?" and language "en"
```

### Retrieval Only

```
Use retrieve_documents to find 10 transcript chunks related to "climate change"
```

### Russian Language Query

```
Use query_rag with question "О чём говорили в видео про искусственный интеллект?" and language "ru"
```

### Check Configuration

```
Show me the config://current resource
```

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

**Issue**: Can't connect to HTTP server

**Solutions**:
- Check firewall rules allow connections on the specified port
- Verify the server is running: `curl http://localhost:8000/mcp`
- Try binding to all interfaces: `--host 0.0.0.0`
- Check for port conflicts with other services

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

### For Fastest Queries

1. **Use API-based embeddings**: Mistral/OpenAI/Gemini embeddings are faster than local models
2. **Use fast LLMs**: `gemini-2.5-flash`, `gpt-4o-mini`, or `mistral-small-latest`
3. **Reduce context**: Lower `top_k` to retrieve fewer documents
4. **Local Qdrant**: Run Qdrant on the same machine as the MCP server

### For Best Quality

1. **More context**: Increase `top_k` to 7-10 documents
2. **Better LLMs**: Use `claude-3-5-sonnet`, `gpt-4o`, or `gemini-2.5-pro`
3. **Higher max_tokens**: Increase for longer, more detailed answers

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
