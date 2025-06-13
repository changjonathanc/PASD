# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
PASD (Pixel-Aware Stable Diffusion) is a research project for realistic image super-resolution and personalized stylization using stable diffusion. The project supports multiple tasks including realistic image SR, old photo restoration, personalized stylization, and colorization.

## Development Commands

### Installation
```bash
# Development setup
pip install -e .

# Testing dependencies  
pip install -r requirements-test.txt
```

### Training
```bash
# Train PASD model
bash ./train_pasd.sh

# Train PASD Light variant
python train_pasd.py --use_pasd_light
```

### Testing
```bash
# Basic PASD testing
python test_pasd.py

# PASD Light with personalized models
python test_pasd.py --use_pasd_light --use_personalized_model

# PASD SDXL testing
python test_pasd_sdxl.py

# Gradio demo
python gradio_pasd.py
```

## Architecture

### Core Components
- **Models**: Two variants available
  - `pasd/models/pasd/`: Full PASD model (UNet2D, ControlNet)
  - `pasd/models/pasd_light/`: Lightweight variant with attention optimization
- **Pipelines**: Custom diffusion pipelines in `pasd/pipelines/`
  - `pipeline_pasd.py`: Main SD1.5-based pipeline
  - `pipeline_pasd_sdxl.py`: SDXL-based pipeline
- **Data Loading**: Multiple dataset loaders in `pasd/dataloader/`
  - `webdatasets.py`: Recommended for training (WebDataset format)
  - `localdatasets.py`: Local dataset handling
  - `realesrgan.py`: Real-ESRGAN degradation pipeline

### Key Dependencies
- Built on HuggingFace Diffusers framework
- Uses Accelerate for distributed training
- WebDataset for efficient data loading
- xformers for memory optimization (not available on macOS)

### Directory Structure
- `checkpoints/`: Model weights and configs
  - `stable-diffusion-v1-5/`: Base SD1.5 model
  - `personalized_models/`: Custom models for stylization
- `runs/`: Training outputs and pretrained PASD models
- `examples/`: Test image datasets (Set5, Set14, RealSRSet)
- `pasd/annotator/`: Face detection and YOLO utilities
- `pasd/myutils/`: Utility functions (VAE hooks, color correction)

### Control Types
The system supports multiple control types via `--control_type`:
- `realisr`: Real image super-resolution
- `grayscale`: Colorization from grayscale input
- Face restoration and stylization modes

### Model Variants
Use `--use_pasd_light` flag to switch between:
- Full PASD: Higher quality, more memory
- PASD Light: Optimized attention, faster inference