# Getting Started with PASD

This guide will help you set up and run PASD (Pixel-Aware Stable Diffusion) for image super-resolution and stylization.

## Prerequisites

- NVIDIA GPU with 8GB+ VRAM (24GB recommended for best performance)
- Python 3.8+
- CUDA toolkit installed

## Installation

1. **Clone and install the repository:**
```bash
git clone https://github.com/yangxy/PASD.git
cd PASD
pip install -e .
pip install -r requirements-test.txt
```

2. **Download base Stable Diffusion model:**
```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download SD1.5 model files
wget -O - https://github.com/yangxy/PASD/archive/main.tar.gz | tar xz --strip=1 "PASD-main/checkpoints"
```

You'll also need to download the full SD1.5 model from [HuggingFace](https://huggingface.co/runwayml/stable-diffusion-v1-5) and place it in `checkpoints/stable-diffusion-v1-5/`.

3. **Download PASD pretrained models:**

Download one or more of these models and extract to the `runs/` directory:
- [pasd](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd.zip) - Full model
- [pasd_light](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_light.zip) - Lightweight version
- [pasd_rrdb](https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_rrdb.zip) - With RRDB enhancement

## Quick Start

### Basic Image Super-Resolution

```bash
# Upscale a single image
python test_pasd.py --image_path examples/dog.png --upscale 2

# Process all images in a folder
python test_pasd.py --image_path examples/Set5/ --upscale 2
```

### Enhanced Results with Personalized Models

For better quality, use personalized models:

```bash
python test_pasd.py \
    --image_path examples/dog.png \
    --use_personalized_model \
    --upscale 2 \
    --output_dir output
```

### Memory-Efficient Version

If you have limited VRAM, use PASD Light:

```bash
python test_pasd.py \
    --image_path examples/dog.png \
    --use_pasd_light \
    --upscale 2
```

### Colorization

Convert grayscale images to color:

```bash
python test_pasd.py \
    --image_path path/to/grayscale_image.png \
    --control_type grayscale \
    --use_pasd_light
```

## Advanced Options

### Common Parameters

- `--upscale`: Scale factor (1, 2, 4)
- `--num_inference_steps`: Denoising steps (20 default, higher = better quality)
- `--guidance_scale`: Classifier-free guidance (9.0 default)
- `--conditioning_scale`: ControlNet strength (1.0 default)
- `--seed`: Random seed for reproducible results

### GPU Memory Management

For different GPU memory sizes:

**8-12GB VRAM:**
```bash
python test_pasd.py \
    --use_pasd_light \
    --decoder_tiled_size 128 \
    --encoder_tiled_size 512 \
    --latent_tiled_size 160
```

**16-20GB VRAM:**
```bash
python test_pasd.py \
    --decoder_tiled_size 192 \
    --encoder_tiled_size 768 \
    --latent_tiled_size 240
```

**24GB+ VRAM (default settings work well):**
```bash
python test_pasd.py  # Uses optimized defaults
```

## Web Interface

Launch the Gradio demo for easy experimentation:

```bash
python gradio_pasd.py
```

Then open your browser to the displayed URL (usually `http://127.0.0.1:7860`).

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce tile sizes or use `--use_pasd_light`
2. **xformers not available on macOS**: This is expected, the code will fall back gracefully
3. **Model not found**: Ensure you've downloaded the pretrained models to the correct directories

### Getting Help

- Check the [main README](README.md) for detailed documentation
- Issues can be reported on the [GitHub repository](https://github.com/yangxy/PASD/issues)
- For questions, contact: yangtao9009@gmail.com

## Example Results

The `samples/` directory contains example outputs showing:
- Real image super-resolution
- Old photo restoration  
- Personalized stylization
- Colorization results

Start with the provided test images in `examples/` to verify your setup is working correctly.