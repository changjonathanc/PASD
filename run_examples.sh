#!/bin/bash

# PASD Example Runner Script
# This script demonstrates different PASD configurations and use cases

set -e  # Exit on any error

echo "🚀 PASD Example Runner"
echo "======================"

# Create output directories
mkdir -p output/{basic,light,rrdb,colorization,custom_prompt,batch}

echo ""
echo "📋 Available examples:"
echo "1. Basic PASD (24GB VRAM)"
echo "2. PASD Light (Lower memory)"  
echo "3. PASD RRDB (80GB VRAM, highest quality)"
echo "4. Colorization"
echo "5. Custom prompts"
echo "6. Batch processing"
echo "7. All examples"
echo ""

read -p "Select option (1-7): " choice

case $choice in
    1)
        echo "🎯 Running Basic PASD..."
        python test_pasd.py \
            --image_path examples/dog.png \
            --upscale 2 \
            --output_dir output/basic
        echo "✅ Basic PASD complete! Check output/basic/"
        ;;
    2)
        echo "🎯 Running PASD Light..."
        python test_pasd.py \
            --use_pasd_light \
            --image_path examples/dog.png \
            --upscale 2 \
            --output_dir output/light
        echo "✅ PASD Light complete! Check output/light/"
        ;;
    3)
        echo "🎯 Running PASD RRDB (highest quality)..."
        python test_pasd.py \
            --pasd_model_path "runs/pasd_rrdb/checkpoint-100000" \
            --image_path examples/dog.png \
            --upscale 4 \
            --num_inference_steps 30 \
            --output_dir output/rrdb
        echo "✅ PASD RRDB complete! Check output/rrdb/"
        ;;
    4)
        echo "🎯 Running Colorization..."
        # Check if we have a grayscale image, if not convert one
        if [ ! -f examples/dog_gray.png ]; then
            echo "Creating grayscale version of dog.png..."
            python -c "
from PIL import Image
img = Image.open('examples/dog.png').convert('L').convert('RGB')
img.save('examples/dog_gray.png')
print('Created examples/dog_gray.png')
"
        fi
        
        python test_pasd.py \
            --control_type grayscale \
            --image_path examples/dog_gray.png \
            --output_dir output/colorization
        echo "✅ Colorization complete! Check output/colorization/"
        ;;
    5)
        echo "🎯 Running with custom prompts..."
        python test_pasd.py \
            --image_path examples/dog.png \
            --prompt "professional photography, sharp details, award winning" \
            --upscale 2 \
            --num_inference_steps 30 \
            --output_dir output/custom_prompt
        echo "✅ Custom prompt complete! Check output/custom_prompt/"
        ;;
    6)
        echo "🎯 Running batch processing on Set5..."
        python test_pasd.py \
            --image_path examples/Set5/ \
            --upscale 2 \
            --output_dir output/batch
        echo "✅ Batch processing complete! Check output/batch/"
        ;;
    7)
        echo "🎯 Running ALL examples (this will take a while)..."
        echo ""
        
        # Basic PASD
        echo "1/6 Basic PASD..."
        python test_pasd.py \
            --image_path examples/dog.png \
            --upscale 2 \
            --output_dir output/basic
        
        # PASD Light  
        echo "2/6 PASD Light..."
        python test_pasd.py \
            --use_pasd_light \
            --image_path examples/dog.png \
            --upscale 2 \
            --output_dir output/light
        
        # PASD RRDB (if available)
        if [ -d "runs/pasd_rrdb/checkpoint-100000" ]; then
            echo "3/6 PASD RRDB..."
            python test_pasd.py \
                --pasd_model_path "runs/pasd_rrdb/checkpoint-100000" \
                --image_path examples/dog.png \
                --upscale 4 \
                --output_dir output/rrdb
        else
            echo "3/6 Skipping PASD RRDB (model not found)"
        fi
        
        # Colorization
        echo "4/6 Colorization..."
        if [ ! -f examples/dog_gray.png ]; then
            python -c "
from PIL import Image
img = Image.open('examples/dog.png').convert('L').convert('RGB')
img.save('examples/dog_gray.png')
"
        fi
        python test_pasd.py \
            --control_type grayscale \
            --image_path examples/dog_gray.png \
            --output_dir output/colorization
        
        # Custom prompts
        echo "5/6 Custom prompts..."
        python test_pasd.py \
            --image_path examples/dog.png \
            --prompt "professional photography, sharp details" \
            --upscale 2 \
            --output_dir output/custom_prompt
        
        # Batch processing
        echo "6/6 Batch processing..."
        python test_pasd.py \
            --image_path examples/Set5/ \
            --upscale 2 \
            --output_dir output/batch
        
        echo ""
        echo "🎉 ALL EXAMPLES COMPLETE!"
        echo "Check the following directories:"
        echo "- output/basic/ - Basic PASD results"
        echo "- output/light/ - PASD Light results"
        echo "- output/rrdb/ - PASD RRDB results (if available)"
        echo "- output/colorization/ - Colorization results"
        echo "- output/custom_prompt/ - Custom prompt results"
        echo "- output/batch/ - Batch processing results"
        ;;
    *)
        echo "❌ Invalid option. Please select 1-7."
        exit 1
        ;;
esac

echo ""
echo "🎉 Done! GPU memory usage summary:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "GPU Memory: %s MB / %s MB (%.1f%% used)\n", $1, $2, $1/$2*100}'
else
    echo "nvidia-smi not available"
fi