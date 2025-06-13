#!/usr/bin/env python3
"""
FastAPI server for PASD-SDXL image super-resolution
Loads model once in GPU memory and serves requests efficiently
"""

import os
import io
import asyncio
from pathlib import Path
from typing import Optional
import logging

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
model_pipeline = None
device = None


async def load_model():
    """Load PASD-SDXL model into GPU memory"""
    global model_pipeline, device
    
    logger.info("🚀 Loading PASD-SDXL model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Import required modules
        from accelerate import Accelerator
        from diffusers import EulerDiscreteScheduler, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
        from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel
        from pasd.models.pasd.controlnet import ControlNetModel
        from pasd.pipelines.pipeline_pasd_sdxl import StableDiffusionXLControlNetPipeline
        
        # Initialize accelerator
        accelerator = Accelerator(mixed_precision="bf16" if device.type == "cuda" else "no")
        
        # Model paths
        pretrained_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        pasd_model_path = "yangtao9009/PASD-SDXL"
        
        # Load components
        logger.info("Loading scheduler...")
        scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
        
        logger.info("Loading text encoders...")
        text_encoder_1 = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
        
        logger.info("Loading tokenizers...")
        tokenizer_1 = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
        
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        
        logger.info("Loading PASD UNet and ControlNet...")
        unet = UNet2DConditionModel.from_pretrained(pasd_model_path, subfolder="checkpoint-200000/unet")
        controlnet = ControlNetModel.from_pretrained(pasd_model_path, subfolder="checkpoint-200000/controlnet")
        
        # Create pipeline
        logger.info("Creating pipeline...")
        model_pipeline = StableDiffusionXLControlNetPipeline(
            vae=vae,
            text_encoder=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer_1,
            tokenizer_2=tokenizer_2,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
        )
        
        # Move to device
        model_pipeline = model_pipeline.to(device)
        
        # Enable memory efficient attention if available
        try:
            model_pipeline.enable_xformers_memory_efficient_attention()
            logger.info("✅ XFormers memory efficient attention enabled")
        except:
            logger.info("⚠️  XFormers not available, using default attention")
        
        # Enable model CPU offload for better memory management
        if device.type == "cuda":
            model_pipeline.enable_model_cpu_offload()
            logger.info("✅ Model CPU offload enabled")
        
        logger.info("🎉 PASD-SDXL model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


async def unload_model():
    """Clean up model from memory"""
    global model_pipeline
    if model_pipeline is not None:
        del model_pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Model unloaded from memory")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model lifecycle"""
    # Startup
    await load_model()
    yield
    # Shutdown
    await unload_model()


# Create FastAPI app
app = FastAPI(
    title="PASD-SDXL Super-Resolution API",
    description="High-quality image super-resolution using PASD-SDXL",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PASD-SDXL Super-Resolution API",
        "status": "ready" if model_pipeline is not None else "loading",
        "device": str(device) if device else "unknown"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if model_pipeline is None:
        return {"status": "error", "message": "Model not loaded"}
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        **gpu_info
    }


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    scale: int = Query(2, ge=1, le=4, description="Upscaling factor (1-4)"),
    steps: int = Query(25, ge=10, le=100, description="Number of inference steps"),
    guidance_scale: float = Query(7.0, ge=1.0, le=20.0, description="Guidance scale"),
    conditioning_scale: float = Query(0.8, ge=0.1, le=1.0, description="ControlNet conditioning scale"),
    prompt: str = Query("", description="Additional prompt"),
    negative_prompt: str = Query("blurry, dirty, messy, frames, deformed, dotted, noise, raster lines, unclear, lowres, over-smoothed, painting, ai generated", description="Negative prompt")
):
    """
    Upscale an image using PASD-SDXL
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load image
        image_data = await file.read()
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        logger.info(f"Processing image: {input_image.size} -> {scale}x upscale")
        
        # Image preprocessing (matching test_pasd_sdxl.py logic)
        ori_width, ori_height = input_image.size
        
        # Apply upscaling first
        validation_image = input_image.resize((input_image.size[0] * scale, input_image.size[1] * scale))
        
        # Ensure dimensions are multiples of 8 (required for diffusion models)
        validation_image = validation_image.resize((
            validation_image.size[0] // 8 * 8, 
            validation_image.size[1] // 8 * 8
        ))
        
        # Prepare prompts (matching test_pasd_sdxl.py defaults)
        base_prompt = "photorealistic, clean, high-resolution, 8k"
        full_prompt = f"{base_prompt}, {prompt}" if prompt else base_prompt
        
        # Create args-like object for pipeline compatibility (matching test_pasd_sdxl.py defaults)
        class Args:
            def __init__(self):
                self.control_type = "realisr"
                self.conditioning_scale = conditioning_scale
                self.latent_tiled_size = 180
                self.latent_tiled_overlap = 8
                self.decoder_tiled_size = 512
                self.encoder_tiled_size = 2048
        
        args = Args()
        
        # Generate image (matching test_pasd_sdxl.py call)
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = model_pipeline(
                args,
                prompt=full_prompt,
                image=validation_image,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=conditioning_scale,
                guess_mode=False,
            )
        
        output_image = result.images[0]
        
        # Post-processing (matching test_pasd_sdxl.py)
        try:
            from pasd.myutils.wavelet_color_fix import wavelet_color_fix
            output_image = wavelet_color_fix(output_image, validation_image)
        except ImportError:
            logger.warning("Wavelet color fix not available, skipping")
        
        # Resize to final output size (matching test_pasd_sdxl.py logic)
        final_output = output_image.resize((ori_width * scale, ori_height * scale))
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        final_output.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        logger.info(f"✅ Image processed: {input_image.size} -> {final_output.size}")
        
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=upscaled_{scale}x_{file.filename}",
                "X-Original-Size": f"{input_image.width}x{input_image.height}",
                "X-Output-Size": f"{final_output.width}x{final_output.height}",
                "X-Scale-Factor": str(scale)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Don't reload in production
        log_level="info"
    )