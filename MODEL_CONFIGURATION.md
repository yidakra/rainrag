# Model Configuration Guide

RainRAG supports multiple LLM models through vLLM or Ollama. The system automatically detects the appropriate chat template based on the model name, or you can specify it manually.

## Supported Models

### Mistral Models
- **Model**: `mistralai/Mistral-Small-3.2-24B-Instruct-2506`
- **Template**: Mistral (`<s>[INST]...[/INST]`)
- **Status**: ✅ Tested and working

### Gemma Models
- **Model**: `gemma3:27b`
- **Template**: Gemma (`<start_of_turn>...<end_of_turn>`)
- **Status**: Configured and ready

### ChatML/GPT Models
- **Model**: `gpt-oss:20b` or any ChatML-based model
- **Template**: ChatML (`<|im_start|>...<|im_end|>`)
- **Status**: Configured and ready

### Generic Models
- **Model**: Any other instruct-tuned model
- **Template**: Simple format (System: ... User: ...)
- **Status**: Fallback option

## Configuration

Edit `config.yaml` to specify your model:

### Example 1: Mistral (Default)

```yaml
vllm:
  host: "localhost"
  port: 8000
  model_name: "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  max_tokens: 512
  temperature: 0.3
  top_k: 5
  use_chat_completions: true
  chat_template: "auto"  # Auto-detects from model name
```

### Example 2: Gemma 3 27B

```yaml
vllm:
  host: "localhost"
  port: 8000
  model_name: "gemma3:27b"
  max_tokens: 512
  temperature: 0.3
  top_k: 5
  use_chat_completions: true
  chat_template: "auto"  # Will detect "gemma" template
```

### Example 3: GPT-OSS 20B

```yaml
vllm:
  host: "localhost"
  port: 8000
  model_name: "gpt-oss:20b"
  max_tokens: 512
  temperature: 0.3
  top_k: 5
  use_chat_completions: true
  chat_template: "auto"  # Will detect "chatml" template
```

### Manual Template Override

If auto-detection doesn't work, specify the template manually:

```yaml
vllm:
  model_name: "custom-model"
  chat_template: "gemma"  # Options: mistral, gemma, chatml, generic
```

## Starting vLLM for Different Models

### Mistral

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

### Gemma 3 27B

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --tensor-parallel-size 2  # For multi-GPU
```

Or if using Ollama:

```bash
# Pull the model first
ollama pull gemma3:27b

# Ollama automatically serves OpenAI-compatible API
# Default URL: http://localhost:11434/v1
```

Update config.yaml for Ollama:
```yaml
vllm:
  host: "localhost"
  port: 11434  # Ollama's default port
  model_name: "gemma3:27b"
```

### GPT-OSS 20B

If using through Ollama:

```bash
ollama pull gpt-oss:20b
```

If using through vLLM (assuming model is available):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model path/to/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
```

## Chat Templates Explained

### Mistral Format
```
<s>[INST] {system_message}

{user_message}

{russian_instruction} [/INST]
```

### Gemma Format
```
<start_of_turn>user
{system_message}

{user_message}

{russian_instruction}<end_of_turn>
<start_of_turn>model
```

### ChatML Format
```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}

{russian_instruction}<|im_end|>
<|im_start|>assistant
```

### Generic Format
```
System: {system_message}

User: {user_message}

{russian_instruction}
```

## Testing Different Models

After switching models, test the setup:

```bash
# Restart the API service
make down
make up

# Try a test query in Russian
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "гей-парад Киев",
    "language": "ru",
    "top_k": 3
  }'
```

The response should be in Russian regardless of which model you use.

## Troubleshooting

### Template Not Working

If the auto-detected template doesn't work:
1. Check the model's documentation for its chat format
2. Try each template manually: `chat_template: "mistral"`, `chat_template: "gemma"`, etc.
3. Check vLLM logs for template errors

### Model Not Responding in Russian

1. Verify `language: "ru"` is set in the request
2. Check that the system message contains Russian instructions
3. Try increasing `temperature` to 0.5 for more creative responses
4. Some models may need more explicit prompting

### Ollama Port Conflicts

If using Ollama on port 11434 with vLLM on port 8000:
- Update `vllm.port` in config.yaml to match your server
- Ollama default: `port: 11434`
- vLLM default: `port: 8000`
