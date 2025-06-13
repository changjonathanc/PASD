# PASD-SDXL FastAPI Server

A minimalistic FastAPI server for PASD-SDXL that loads the model once in GPU memory and efficiently serves super-resolution requests.

## Features

- 🚀 **Single model loading** - Loads PASD-SDXL once at startup
- 🔥 **GPU memory resident** - Keeps model in GPU for fast inference
- ⚡ **Fast API** - REST API with automatic documentation
- 🎛️ **Configurable parameters** - Scale, steps, guidance, prompts
- 📊 **Health monitoring** - GPU memory usage and model status
- 🔄 **Memory management** - Automatic cleanup and optimization

## Quick Start

### 1. Install PASD Package

```bash
# Install PASD from this dev branch
pip install git+https://github.com/cccntu/PASD.git@dev

# Or clone and install locally
git clone -b dev https://github.com/cccntu/PASD.git
cd PASD
pip install -e .
```

### 2. Install API Dependencies

```bash
# Navigate to API directory
cd api/

# Install server requirements
pip install -r requirements.txt
```

### 3. Start Server

```bash
# Start server (loads model automatically)
python server.py

# Or with uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server will:
- Load PASD-SDXL model into GPU memory (takes ~2-3 minutes first time)
- Start accepting requests at `http://localhost:8000`
- Show automatic API docs at `http://localhost:8000/docs`

### 4. Use the API

#### Python Client

```bash
# Upscale an image (from api/ directory)
python client_example.py ../examples/dog.png --scale 2

# With custom parameters
python client_example.py ../examples/dog.png --scale 4 --steps 30 --prompt "ultra sharp, professional"
```

#### cURL

```bash
# Basic upscaling
curl -X POST "http://localhost:8000/upscale?scale=2" \
     -F "file=@examples/dog.png" \
     --output upscaled.png

# With parameters
curl -X POST "http://localhost:8000/upscale?scale=2&steps=30&guidance_scale=7.5&prompt=high%20quality" \
     -F "file=@input.jpg" \
     --output result.png
```

#### Python requests

```python
import requests

# Check server health
response = requests.get("http://localhost:8000/health")
print(response.json())

# Upscale image
with open("input.jpg", "rb") as f:
    files = {"file": ("input.jpg", f, "image/jpeg")}
    params = {"scale": 2, "steps": 25}
    response = requests.post("http://localhost:8000/upscale", files=files, params=params)

with open("output.png", "wb") as f:
    f.write(response.content)
```

## API Endpoints

### `GET /`
Basic health check and server info.

### `GET /health`
Detailed health check with GPU memory usage.

### `POST /upscale`
Main upscaling endpoint.

**Parameters:**
- `file` (required): Image file to upscale
- `scale` (1-4): Upscaling factor (default: 2)
- `steps` (10-100): Number of inference steps (default: 25)
- `guidance_scale` (1.0-20.0): Guidance scale (default: 7.0)
- `prompt`: Additional prompt text (default: "")
- `negative_prompt`: Negative prompt (default: "blurry, low quality, distorted")

**Response:**
- PNG image file
- Headers with original/output dimensions

## Performance

### Memory Usage
- **Model loading**: ~8GB VRAM (one-time)
- **Per request**: ~2-4GB VRAM (temporary)
- **Total recommended**: 12GB+ VRAM

### Speed (RTX 3090)
- **First request**: ~15-20 seconds (includes model warmup)
- **Subsequent requests**: ~8-12 seconds
- **2x upscale**: ~8 seconds
- **4x upscale**: ~15 seconds

## Configuration

### Environment Variables

```bash
# Server settings
export PASD_HOST=0.0.0.0
export PASD_PORT=8000
export PASD_WORKERS=1

# Model settings  
export PASD_DEVICE=cuda
export PASD_PRECISION=bf16
```

### Memory Optimization

For lower VRAM systems:

```python
# In server.py, modify these lines:
model_pipeline.enable_model_cpu_offload()  # Offload to CPU when not in use
model_pipeline.enable_sequential_cpu_offload()  # More aggressive offloading
```

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu22.04

WORKDIR /app
COPY . .

RUN pip install -e .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python", "server.py"]
```

### Systemd Service

```ini
[Unit]
Description=PASD-SDXL API Server
After=network.target

[Service]
Type=simple
User=pasd
WorkingDirectory=/opt/pasd
ExecStart=/opt/pasd/.venv/bin/python server.py
Restart=always
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

## Monitoring

### Logs
```bash
# View server logs
tail -f server.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Health Checks
```bash
# Simple health check
curl http://localhost:8000/health

# Detailed monitoring
curl -s http://localhost:8000/health | jq .gpu_memory_allocated
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size or enable CPU offloading
   - Use smaller scale factors
   - Monitor with `nvidia-smi`

2. **Model loading fails**
   - Check internet connection (downloads from HuggingFace)
   - Verify CUDA installation
   - Check disk space (~20GB needed)

3. **Slow performance**
   - Ensure GPU is being used (`nvidia-smi`)
   - Check VRAM usage
   - Consider model CPU offloading

### Debug Mode

```bash
# Start with debug logging
PYTHONPATH=. uvicorn server:app --host 0.0.0.0 --port 8000 --log-level debug
```