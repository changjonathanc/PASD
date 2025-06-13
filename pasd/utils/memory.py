"""Memory management utilities for PASD."""

import gc
import torch
import logging

logger = logging.getLogger(__name__)


def cleanup_memory():
    """Clean up GPU and system memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info():
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
    
    allocated = torch.cuda.memory_allocated() // 1024**2  # MB
    reserved = torch.cuda.memory_reserved() // 1024**2   # MB
    total = torch.cuda.get_device_properties(0).total_memory // 1024**2  # MB
    free = total - reserved
    
    return {
        "allocated": allocated,
        "reserved": reserved, 
        "free": free,
        "total": total
    }


def log_memory_usage(stage=""):
    """Log current memory usage."""
    if torch.cuda.is_available():
        mem_info = get_gpu_memory_info()
        logger.info(f"{stage} GPU Memory: {mem_info['allocated']}MB allocated, "
                   f"{mem_info['free']}MB free, {mem_info['total']}MB total")


def get_optimal_tile_size(image_height, image_width, base_tile_size=512):
    """Calculate optimal tile size based on image dimensions and available memory."""
    if not torch.cuda.is_available():
        return base_tile_size
    
    mem_info = get_gpu_memory_info()
    available_mb = mem_info['free']
    
    # Estimate memory needed per tile (rough approximation)
    bytes_per_pixel = 16  # fp16 with multiple channels and intermediate tensors
    pixels_per_tile = base_tile_size * base_tile_size
    mb_per_tile = (pixels_per_tile * bytes_per_pixel) // 1024**2
    
    # Use conservative estimate (50% of available memory)
    safe_available = available_mb * 0.5
    max_tiles = max(1, int(safe_available // mb_per_tile))
    
    # Calculate optimal tile size based on image dimensions
    total_pixels = image_height * image_width
    if total_pixels <= pixels_per_tile:
        return min(base_tile_size, max(image_height, image_width))
    
    # Adjust tile size if we have limited memory
    if max_tiles < 4:  # If we can't handle at least 4 tiles, reduce tile size
        reduction_factor = 0.75
        return int(base_tile_size * reduction_factor)
    
    return base_tile_size