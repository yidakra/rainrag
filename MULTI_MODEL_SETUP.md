# Multi-Model Setup Guide

This guide explains how to run all 3 supported LLM models simultaneously for seamless switching in the RainRAG UI.

## Overview

RainRAG supports running multiple vLLM instances on different ports, allowing you to switch between models instantly without any downtime or configuration changes.

## Supported Models and Ports

| Model | Display Name | Port | Chat Template |
|-------|-------------|------|---------------|
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Mistral Small 3.2 24B | 8000 | mistral |
| `google/gemma-2-27b-it` | Gemma 2 27B | 8002 | gemma |
| `gpt-oss:20b` | GPT-OSS 20B | 8003 | chatml |

**Note:** Port 8001 is reserved for the RainRAG API server.

## Quick Start: Run All 3 Models

### Option 1: Using Screen/Tmux (Recommended for Development)

Run each vLLM instance in a separate screen/tmux session:

```bash
# Terminal 1: Mistral (Port 8000)
screen -S vllm-mistral
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
# Detach: Ctrl+A, D

# Terminal 2: Gemma (Port 8002)
screen -S vllm-gemma
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype auto
# Detach: Ctrl+A, D

# Terminal 3: GPT-OSS (Port 8003)
screen -S vllm-gptoss
python -m vllm.entrypoints.openai.api_server \
  --model gpt-oss:20b \
  --host 0.0.0.0 \
  --port 8003 \
  --dtype auto
# Detach: Ctrl+A, D
```

To reattach to any session:
```bash
screen -r vllm-mistral   # or vllm-gemma, vllm-gptoss
screen -ls               # list all sessions
```

### Option 2: Using systemd (Recommended for Production)

Create service files for each model:

**1. Create `/etc/systemd/system/vllm-mistral.service`:**

```ini
[Unit]
Description=vLLM Server - Mistral Small 3.2 24B
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser
Environment="PATH=/home/youruser/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**2. Create `/etc/systemd/system/vllm-gemma.service`:**

```ini
[Unit]
Description=vLLM Server - Gemma 2 27B
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser
Environment="PATH=/home/youruser/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --host 0.0.0.0 \
  --port 8002 \
  --dtype auto
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**3. Create `/etc/systemd/system/vllm-gptoss.service`:**

```ini
[Unit]
Description=vLLM Server - GPT-OSS 20B
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser
Environment="PATH=/home/youruser/.local/bin:/usr/bin"
ExecStart=/usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model gpt-oss:20b \
  --host 0.0.0.0 \
  --port 8003 \
  --dtype auto
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**4. Enable and start the services:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm-mistral vllm-gemma vllm-gptoss
sudo systemctl start vllm-mistral vllm-gemma vllm-gptoss

# Check status
sudo systemctl status vllm-mistral
sudo systemctl status vllm-gemma
sudo systemctl status vllm-gptoss

# View logs
sudo journalctl -u vllm-mistral -f
sudo journalctl -u vllm-gemma -f
sudo journalctl -u vllm-gptoss -f
```

### Option 3: Using Docker Compose

Create a `docker-compose.vllm.yaml` file:

```yaml
version: '3.8'

services:
  vllm-mistral:
    image: vllm/vllm-openai:latest
    container_name: vllm-mistral
    ports:
      - "8000:8000"
    command: >
      --model mistralai/Mistral-Small-3.2-24B-Instruct-2506
      --host 0.0.0.0
      --port 8000
      --dtype auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  vllm-gemma:
    image: vllm/vllm-openai:latest
    container_name: vllm-gemma
    ports:
      - "8002:8002"
    command: >
      --model google/gemma-2-27b-it
      --host 0.0.0.0
      --port 8002
      --dtype auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  vllm-gptoss:
    image: vllm/vllm-openai:latest
    container_name: vllm-gptoss
    ports:
      - "8003:8003"
    command: >
      --model gpt-oss:20b
      --host 0.0.0.0
      --port 8003
      --dtype auto
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

Start all instances:
```bash
docker-compose -f docker-compose.vllm.yaml up -d

# View logs
docker-compose -f docker-compose.vllm.yaml logs -f vllm-mistral
docker-compose -f docker-compose.vllm.yaml logs -f vllm-gemma
docker-compose -f docker-compose.vllm.yaml logs -f vllm-gptoss
```

## Verifying the Setup

Test each vLLM instance is running:

```bash
# Test Mistral (Port 8000)
curl http://localhost:8000/v1/models

# Test Gemma (Port 8002)
curl http://localhost:8002/v1/models

# Test GPT-OSS (Port 8003)
curl http://localhost:8003/v1/models
```

## Using with RainRAG

1. **Start RainRAG API:**
   ```bash
   make api
   ```

2. **Start Streamlit UI:**
   ```bash
   make streamlit
   ```

3. **Switch models in the UI:**
   - Open the sidebar
   - Select any model from the dropdown
   - Switch happens instantly - the API connects to the appropriate vLLM instance
   - No restart or config changes needed!

## Hardware Requirements

### GPU Memory Requirements (Approximate)

- **Mistral Small 3.2 24B**: ~48GB VRAM (2x A100 40GB or 1x A100 80GB)
- **Gemma 2 27B**: ~54GB VRAM (2x A100 40GB or 1x A100 80GB)
- **GPT-OSS 20B**: ~40GB VRAM (1x A100 40GB or 2x RTX 3090)

### Running with Limited GPUs

If you have limited GPU resources, you can:

**Option 1: Run models on separate GPUs**
```bash
# Mistral on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --port 8000

# Gemma on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --port 8002

# GPT-OSS on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --model gpt-oss:20b \
  --port 8003
```

**Option 2: Use tensor parallelism**
```bash
# Mistral on GPU 0,1 with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --port 8000 \
  --tensor-parallel-size 2

# Gemma on GPU 2,3
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-2-27b-it \
  --port 8002 \
  --tensor-parallel-size 2
```

**Option 3: Run only your preferred models**

You don't need to run all 3 models. Run only the ones you want to use:
- The UI will show all 3 in the dropdown
- Switching to an unavailable model will show a clear error message
- The other models will work seamlessly

## Troubleshooting

### Model fails to load
- Check GPU memory: `nvidia-smi`
- Verify model is downloaded: models are cached in `~/.cache/huggingface/hub`
- Check vLLM logs for specific errors

### Port already in use
```bash
# Find what's using the port
lsof -i :8000  # or 8001, 8002

# Kill the process
kill -9 <PID>
```

### Model switching fails in UI
- Verify target vLLM instance is running: `curl http://localhost:PORT/v1/models`
- Check API logs: `tail -f /tmp/rainrag-api.log`
- Ensure ports 8000-8002 are accessible from the API server

### Out of memory errors
- Use quantization: `--quantization awq` or `--quantization gptq`
- Enable CPU offloading: `--cpu-offload-gb 20`
- Use smaller batch sizes: `--max-num-seqs 128`
- Reduce max model length: `--max-model-len 2048`

## Performance Tips

1. **Enable flash attention** (if supported):
   ```bash
   pip install flash-attn --no-build-isolation
   ```

2. **Optimize batch size** for your GPU:
   ```bash
   --max-num-batched-tokens 4096
   --max-num-seqs 256
   ```

3. **Use quantization** for memory-constrained setups:
   ```bash
   --quantization awq  # or gptq, sq
   ```

4. **Enable prefix caching** for repeated queries:
   ```bash
   --enable-prefix-caching
   ```

## Next Steps

- See [MODEL_CONFIGURATION.md](MODEL_CONFIGURATION.md) for detailed model-specific configurations
- See [VLLM_SETUP.md](VLLM_SETUP.md) for vLLM installation and setup
- See [README.md](README.md) for full RainRAG documentation
