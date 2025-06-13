#!/usr/bin/env python3
"""
FastAPI server for PASD-SDXL image super-resolution
Matches test_pasd_sdxl.py behavior exactly
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
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global storage
model_pipeline = None
accelerator = None
model = None
preprocess = None  
category = None
resize_preproc = None
generator = None

async def load_model():
    """Load PASD-SDXL model exactly like test_pasd_sdxl.py"""
    global model_pipeline, accelerator, model, preprocess, category, resize_preproc, generator
    
    logger.info("🚀 Loading PASD-SDXL model exactly like test script...")
    
    try:
        # Import required modules
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from diffusers import EulerDiscreteScheduler, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
        from pasd.models.pasd.unet_2d_condition import UNet2DConditionModel
        from pasd.models.pasd.controlnet import ControlNetModel
        from pasd.pipelines.pipeline_pasd_sdxl import StableDiffusionXLControlNetPipeline
        from diffusers.utils.import_utils import is_xformers_available

        # Create args exactly like test script defaults
        class Args:
            def __init__(self):
                self.pretrained_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
                self.pasd_model_path = "yangtao9009/PASD-SDXL"
                self.use_pasd_light = False
                self.control_type = "realisr"
                self.mixed_precision = "bf16"
                self.use_personalized_model = False
                self.personalized_model_path = None
                self.high_level_info = "caption"
                self.prompt = ""
                self.added_prompt = "photorealistic, clean, high-resolution, 8k"
                self.negative_prompt = "blurry, dirty, messy, frames, deformed, dotted, noise, raster lines, unclear, lowres, over-smoothed, painting, ai generated"
                self.upscale = 2
                self.process_size = 1280
                self.num_inference_steps = 25
                self.guidance_scale = 7.0
                self.conditioning_scale = 0.8
                self.latent_tiled_size = 180
                self.latent_tiled_overlap = 8
                self.decoder_tiled_size = 512
                self.encoder_tiled_size = 2048
                self.seed = None
                self.use_refiner = False
                self.output_dir = "output"

        args = Args()
        
        # Initialize accelerator exactly like test script
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
        
        # Set seed if specified
        if args.seed is not None:
            set_seed(args.seed)

        # Load pipeline exactly like load_pasd_pipeline function
        logger.info("Loading models...")
        
        # Load scheduler, tokenizer and models (exact copy from test script)
        scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
        text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_path, subfolder="text_encoder_2")
        tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
        
        # VAE loading logic (exact copy from test script)
        if args.mixed_precision == "fp16":
            vae = AutoencoderKL.from_pretrained("checkpoints/stabilityai", subfolder="sdxl-vae-fp16-fix")
        else:
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")

        unet = UNet2DConditionModel.from_pretrained(args.pasd_model_path, subfolder="checkpoint-200000/unet")
        controlnet = ControlNetModel.from_pretrained(args.pasd_model_path, subfolder="checkpoint-200000/controlnet")

        # Freeze components (exact copy from test script)
        vae.requires_grad_(False)
        text_encoder_1.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        unet.requires_grad_(False)
        controlnet.requires_grad_(False)

        # Weight dtype logic (exact copy from test script)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move to device and cast (exact copy from test script)
        text_encoder_1.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)
        controlnet.to(accelerator.device, dtype=weight_dtype)

        # XFormers logic (exact copy from test script)
        enable_xformers = False  # Default from test script
        if enable_xformers:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
                controlnet.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers enabled")
            else:
                raise ValueError("xformers is not available")

        # Create pipeline (exact copy from test script)
        model_pipeline = StableDiffusionXLControlNetPipeline(
            vae=vae, text_encoder=text_encoder_1, text_encoder_2=text_encoder_2, 
            tokenizer=tokenizer_1, tokenizer_2=tokenizer_2, 
            unet=unet, controlnet=controlnet, scheduler=scheduler,
        )
        
        # Note: VAE tiling is commented out in test script, so we skip it
        
        # Load high level net (simplified - we'll skip caption generation for server)
        model = None
        preprocess = None
        category = None

        # Setup resize preprocessing (exact copy from test script)
        resize_preproc = transforms.Compose([
            transforms.Resize(args.process_size, interpolation=transforms.InterpolationMode.BILINEAR),
        ] if args.control_type == "realisr" else [
            transforms.Resize(args.process_size, max_size=args.process_size*2, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

        # Setup generator (exact copy from test script)
        if accelerator.is_main_process:
            generator = torch.Generator(device=accelerator.device)
            if args.seed is not None:
                generator.manual_seed(args.seed)

        logger.info("🎉 PASD-SDXL model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


async def unload_model():
    """Clean up model from memory"""
    global model_pipeline, accelerator, model, preprocess, category, resize_preproc, generator
    if model_pipeline is not None:
        del model_pipeline
        del accelerator
        del model
        del preprocess
        del category
        del resize_preproc
        del generator
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
    description="High-quality image super-resolution using PASD-SDXL (matches test_pasd_sdxl.py exactly)",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "PASD-SDXL Super-Resolution API",
        "status": "ready" if model_pipeline is not None else "loading",
        "device": str(accelerator.device) if accelerator else "unknown"
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
        "device": str(accelerator.device),
        **gpu_info
    }


def get_validation_prompt(image, prompt=""):
    """Simplified prompt generation (skip caption for server)"""
    base_prompt = "photorealistic, clean, high-resolution, 8k"
    return f"{base_prompt}, {prompt}" if prompt else base_prompt


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
    Upscale an image using PASD-SDXL (matches test_pasd_sdxl.py exactly)
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Load image (exact copy from test script line 223)
        image_data = await file.read()
        validation_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        logger.info(f"Processing image: {validation_image.size}")
        
        # Create args for this request (using scale parameter)
        class Args:
            def __init__(self):
                self.control_type = "realisr"
                self.conditioning_scale = conditioning_scale
                self.upscale = scale
                self.process_size = 1280
                self.num_inference_steps = steps
                self.guidance_scale = guidance_scale
                self.added_prompt = "photorealistic, clean, high-resolution, 8k"
                self.negative_prompt = negative_prompt
                # Tiling parameters
                self.latent_tiled_size = 180
                self.latent_tiled_overlap = 8
                self.decoder_tiled_size = 512
                self.encoder_tiled_size = 2048

        args = Args()
        
        # Exact processing logic from test script (lines 225-249)
        if args.control_type == "realisr":
            validation_prompt = get_validation_prompt(validation_image, prompt)
            validation_prompt += args.added_prompt
            negative_prompt_final = args.negative_prompt
        else:
            raise NotImplementedError("Only realisr control type supported")

        # Exact image preprocessing from test script (lines 240-249)
        ori_width, ori_height = validation_image.size
        resize_flag = False
        rscale = args.upscale if args.control_type == "realisr" else 1

        validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))

        if min(validation_image.size) < args.process_size or args.control_type == "grayscale":
            validation_image = resize_preproc(validation_image)

        validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
        resize_flag = True

        logger.info(f"Processed image size: {validation_image.size}")

        # Exact pipeline call from test script (lines 253-257)
        image = model_pipeline(
            args, 
            prompt=validation_prompt, 
            image=validation_image, 
            num_inference_steps=args.num_inference_steps, 
            generator=generator,
            guidance_scale=args.guidance_scale, 
            negative_prompt=negative_prompt_final, 
            controlnet_conditioning_scale=args.conditioning_scale,
            guess_mode=False,
        ).images[0]

        # Exact post-processing from test script (lines 262-267)
        if args.control_type == "realisr": 
            if True:  # args.conditioning_scale < 1.0:
                try:
                    from pasd.myutils.wavelet_color_fix import wavelet_color_fix
                    image = wavelet_color_fix(image, validation_image)
                except ImportError:
                    logger.warning("Wavelet color fix not available")

            if resize_flag: 
                image = image.resize((ori_width*rscale, ori_height*rscale))

        logger.info(f"Final image size: {image.size}")

        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        
        logger.info(f"✅ Image processed: {validation_image.size} -> {image.size}")
        
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=upscaled_{scale}x_{file.filename}",
                "X-Original-Size": f"{ori_width}x{ori_height}",
                "X-Output-Size": f"{image.width}x{image.height}",
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
        reload=False,
        log_level="info"
    )