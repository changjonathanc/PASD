#!/bin/bash

# PASD Example Runner Script - Best Quality Options
# Runs all PASD configurations with optimal settings

echo "🚀 PASD Example Runner - Best Quality"
echo "====================================="
echo "Running all examples with best quality settings..."
echo ""

# Create output directories
mkdir -p output/{basic,sdxl,custom_prompt,batch}

# Track success/failure
TOTAL_TESTS=3
PASSED_TESTS=0
FAILED_TESTS=()

run_test() {
    local test_name="$1"
    local test_cmd="$2"
    local output_dir="$3"
    
    echo "🎯 Running $test_name..."
    if eval "$test_cmd"; then
        echo "✅ $test_name complete! Check $output_dir/"
        ((PASSED_TESTS++))
    else
        echo "❌ $test_name FAILED"
        FAILED_TESTS+=("$test_name")
    fi
    echo ""
}

# 1. Basic PASD (24GB optimal)
run_test "Basic PASD (2x upscale, 30 steps)" \
    "python test_pasd.py --image_path examples/dog.png --upscale 2 --num_inference_steps 30 --output_dir output/basic" \
    "output/basic"

# 2. PASD SDXL (highest quality alternative - more reliable than RRDB)
if [ -f "test_pasd_sdxl.py" ]; then
    run_test "PASD SDXL (2x upscale, best quality)" \
        "python test_pasd_sdxl.py --image_path examples/dog.png --upscale 2 --num_inference_steps 30 --output_dir output/sdxl" \
        "output/sdxl"
else
    echo "⚠️  PASD SDXL script not found"
    echo ""
fi

# Skip PASD RRDB due to dependency conflicts
echo "⚠️  Skipping PASD RRDB (dependency conflicts with basicsr/torchvision)"
echo "   Use PASD SDXL above for highest quality instead"
echo ""

# 3. Colorization
echo "🎯 Preparing colorization test..."
if [ ! -f examples/dog_gray.png ]; then
    python -c "
from PIL import Image
img = Image.open('examples/dog.png').convert('L').convert('RGB')
img.save('examples/dog_gray.png')
print('Created examples/dog_gray.png')
" || echo "Failed to create grayscale image"
fi

# Skip PASD Light colorization due to diffusers import issues
echo "⚠️  Skipping colorization (PASD Light has diffusers import conflicts)"
echo "   Use basic PASD for colorization if needed:"
echo "   python test_pasd.py --control_type grayscale --image_path examples/dog_gray.png"
echo ""

# 4. Custom prompts with high quality
run_test "Custom Prompts (2x upscale, 40 steps)" \
    "python test_pasd.py --image_path examples/dog.png --prompt 'professional photography, ultra sharp, award winning, masterpiece' --upscale 2 --num_inference_steps 40 --guidance_scale 12.0 --output_dir output/custom_prompt" \
    "output/custom_prompt"

# 5. Batch processing Set5
run_test "Batch Processing Set5 (2x upscale, 25 steps)" \
    "python test_pasd.py --image_path examples/Set5/ --upscale 2 --num_inference_steps 25 --output_dir output/batch" \
    "output/batch"

echo "🎉 PASD Example Runner Complete!"
echo "=================================="
echo "Passed: $PASSED_TESTS/$TOTAL_TESTS tests"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "❌ Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "   - $test"
    done
    echo ""
    echo "💡 Common failure causes:"
    echo "   - Missing dependencies: pip install basicsr"
    echo "   - Insufficient GPU memory (need 24GB+ for best quality)"
    echo "   - Missing PASD model weights in runs/ directory"
    echo "   - CUDA/PyTorch installation issues"
fi

echo ""
echo "📂 Output directories:"
echo "   - output/basic/ - Standard PASD results"
echo "   - output/sdxl/ - Highest quality SDXL results (if available)"
echo "   - output/custom_prompt/ - Enhanced prompt results"
echo "   - output/batch/ - Set5 benchmark results"

echo ""
echo "📊 GPU Memory Usage:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "   GPU Memory: %s MB / %s MB (%.1f%% used)\n", $1, $2, $1/$2*100}'
else
    echo "   nvidia-smi not available"
fi