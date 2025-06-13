# Getting Started with PASD

Quick setup guide for PASD (Pixel-Aware Stable Diffusion) image super-resolution.

## Prerequisites

- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090 or better)
- Python 3.8+
- CUDA toolkit

## Installation

```bash
git clone https://github.com/yangxy/PASD.git
cd PASD
pip install -e .
pip install -r requirements-test.txt
```

## Download Models

**For 24GB VRAM (RTX 3090/4090):**
```bash
cd runs/pasd/
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd.zip
unzip pasd.zip
cd ../..
```

**For 80GB VRAM (A100/H100):**
```bash
cd runs/pasd_rrdb/
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_rrdb.zip
unzip pasd_rrdb.zip
cd ../..
```

## Quick Start

### Run Examples Script

```bash
# Automatic script with best quality settings
./run_examples.sh
```

### Manual Commands

**24GB VRAM (Best quality for RTX 3090/4090):**
```bash
python test_pasd.py --image_path examples/dog.png --upscale 2
```

**80GB VRAM (Highest quality for A100/H100):**
```bash
python test_pasd.py \
    --pasd_model_path "runs/pasd_rrdb/checkpoint-100000" \
    --image_path examples/dog.png \
    --upscale 4
```

**Process Multiple Images:**
```bash
python test_pasd.py --image_path examples/Set5/ --upscale 2
```

**Colorization:**
```bash
python test_pasd.py --image_path path/to/grayscale_image.png --control_type grayscale
```

## Optional: Higher Quality

**More denoising steps (slower but better):**
```bash
python test_pasd.py --image_path examples/dog.png --num_inference_steps 50 --upscale 2
```

**Custom prompts:**
```bash
python test_pasd.py --image_path examples/dog.png --prompt "professional photography, sharp details" --upscale 2
```

## Web Interface

```bash
python gradio_pasd.py
```
Open `http://127.0.0.1:7860` in your browser.

## Troubleshooting

**Model not found error**: Download the PASD models first (see Download Models section)

**Out of memory**: Your GPU doesn't have enough VRAM. Use a smaller `--upscale` value or try PASD Light:
```bash
# For GPUs with less than 24GB VRAM
cd runs/pasd_light/
wget https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_light.zip
unzip pasd_light.zip
cd ../..

python test_pasd.py --use_pasd_light --image_path examples/dog.png
```

Test with the images in `examples/` to verify everything works.