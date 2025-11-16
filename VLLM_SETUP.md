# vLLM Server Setup for Chat Completions

## Current Configuration

- vLLM version: 0.11.0 (supports chat completions API)
- Model: mistralai/Mistral-Small-3.2-24B-Instruct-2506
- Config: `use_chat_completions: true` in config.yaml

## Starting vLLM Server

The vLLM server must be running for the RAG system to generate answers. Here's how to start it:

### Basic Command

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096
```

### With GPU Optimization

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

### For Multi-GPU

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096
```

## Verifying Chat Completions API

Test that the chat completions endpoint is available:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Say hello in Russian"}
    ],
    "max_tokens": 50
  }'
```

Expected response:
```json
{
  "id": "cmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Здравствуйте!"
      },
      "finish_reason": "stop"
    }
  ]
}
```

## Troubleshooting

### Chat API Returns 404

If you get a 404 error on `/v1/chat/completions`, check:

1. **vLLM version**: Ensure vLLM >= 0.2.0
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   ```

2. **Server startup logs**: Check if the server started successfully
   
3. **Port conflicts**: Make sure port 8000 is not in use by another service

### Automatic Fallback

The system is configured with automatic fallback:
- First tries `/v1/chat/completions` (best for instruction following)
- Falls back to `/v1/completions` with Mistral instruction template if chat API unavailable
- Both methods should work, but chat completions gives better results

### Performance Tips

1. **GPU Memory**: Adjust `--gpu-memory-utilization` (default 0.9) if you get OOM errors
2. **Context Length**: Reduce `--max-model-len` if the model doesn't fit in memory
3. **Batch Size**: Add `--max-num-seqs 32` for better throughput with multiple requests

## Integration with RAG System

Once the vLLM server is running:

1. Start the FastAPI backend: `make api` or `poetry run uvicorn rainrag.api:app --host 0.0.0.0 --port 8001`
2. Start the Streamlit UI: `make streamlit` or `poetry run streamlit run app.py --server.port 8501`
3. Access the UI at http://localhost:8501

The system will automatically use the chat completions API for better Russian language support.
