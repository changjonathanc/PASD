#!/usr/bin/env python3
"""
Example client for PASD-SDXL API server
"""

import requests
from pathlib import Path


def upscale_image(
    image_path: str,
    server_url: str = "http://localhost:8000",
    scale: int = 2,
    steps: int = 25,
    guidance_scale: float = 7.0,
    prompt: str = "",
    output_path: str = None
):
    """
    Send image to PASD-SDXL server for upscaling
    """
    
    # Check if server is healthy
    try:
        health_response = requests.get(f"{server_url}/health")
        if health_response.status_code != 200:
            print(f"❌ Server not healthy: {health_response.text}")
            return False
        print("✅ Server is healthy")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False
    
    # Prepare image file
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Prepare output path
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_upscaled_{scale}x.png"
    
    print(f"🚀 Uploading {image_path.name} for {scale}x upscaling...")
    
    # Send request
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            params = {
                "scale": scale,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "prompt": prompt,
            }
            
            response = requests.post(
                f"{server_url}/upscale",
                files=files,
                params=params,
                timeout=300  # 5 minute timeout
            )
        
        if response.status_code == 200:
            # Save result
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            # Print info from headers
            original_size = response.headers.get("X-Original-Size", "unknown")
            output_size = response.headers.get("X-Output-Size", "unknown")
            
            print(f"✅ Success!")
            print(f"   Original: {original_size}")
            print(f"   Output: {output_size}")
            print(f"   Saved: {output_path}")
            return True
        else:
            print(f"❌ Server error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PASD-SDXL API Client")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--scale", type=int, default=2, choices=[1,2,3,4], help="Upscale factor")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--prompt", default="", help="Additional prompt")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    success = upscale_image(
        image_path=args.image,
        server_url=args.server,
        scale=args.scale,
        steps=args.steps,
        guidance_scale=args.guidance,
        prompt=args.prompt,
        output_path=args.output
    )
    
    exit(0 if success else 1)