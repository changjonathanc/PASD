#!/bin/bash

# PASD Model Downloader
# Downloads all PASD models with smart resume/skip functionality

echo "🚀 PASD Model Downloader"
echo "========================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Download statistics
TOTAL_DOWNLOADS=0
COMPLETED_DOWNLOADS=0
SKIPPED_DOWNLOADS=0
FAILED_DOWNLOADS=0

download_and_extract() {
    local name="$1"
    local url="$2"
    local dir="$3"
    local zip_file="$4"
    local check_path="$5"
    
    echo -e "${BLUE}📦 $name${NC}"
    
    # Check if already extracted
    if [ -d "$check_path" ]; then
        echo -e "   ${GREEN}✅ Already downloaded and extracted${NC}"
        ((SKIPPED_DOWNLOADS++))
        echo ""
        return 0
    fi
    
    # Create directory if it doesn't exist
    mkdir -p "$dir"
    cd "$dir"
    
    # Check if zip already exists
    if [ -f "$zip_file" ]; then
        echo -e "   ${YELLOW}📁 Zip file exists, extracting...${NC}"
    else
        echo -e "   ${YELLOW}⬇️  Downloading...${NC}"
        if ! wget -q --show-progress "$url" -O "$zip_file"; then
            echo -e "   ${RED}❌ Download failed${NC}"
            ((FAILED_DOWNLOADS++))
            cd - > /dev/null
            echo ""
            return 1
        fi
    fi
    
    # Extract
    echo -e "   ${YELLOW}📂 Extracting...${NC}"
    if unzip -q "$zip_file"; then
        echo -e "   ${GREEN}✅ Complete${NC}"
        rm "$zip_file"  # Clean up zip file
        ((COMPLETED_DOWNLOADS++))
    else
        echo -e "   ${RED}❌ Extraction failed${NC}"
        ((FAILED_DOWNLOADS++))
        cd - > /dev/null
        echo ""
        return 1
    fi
    
    cd - > /dev/null
    echo ""
    return 0
}

echo "Checking available models..."
echo ""

# PASD Light (Recommended for most users)
((TOTAL_DOWNLOADS++))
download_and_extract \
    "PASD Light (Memory Efficient)" \
    "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_light.zip" \
    "runs/pasd_light" \
    "pasd_light.zip" \
    "runs/pasd_light/checkpoint-100000"

# PASD Full (Better quality)
((TOTAL_DOWNLOADS++))
download_and_extract \
    "PASD Full (High Quality)" \
    "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd.zip" \
    "runs/pasd" \
    "pasd.zip" \
    "runs/pasd/checkpoint-100000"

# PASD RRDB (Highest quality)
((TOTAL_DOWNLOADS++))
download_and_extract \
    "PASD RRDB (Highest Quality - Large Download)" \
    "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_rrdb.zip" \
    "runs/pasd_rrdb" \
    "pasd_rrdb.zip" \
    "runs/pasd_rrdb/checkpoint-100000"

# PASD Light RRDB (Alternative)
((TOTAL_DOWNLOADS++))
download_and_extract \
    "PASD Light RRDB (Balanced)" \
    "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/PASD/pasd_light_rrdb.zip" \
    "runs/pasd_light_rrdb" \
    "pasd_light_rrdb.zip" \
    "runs/pasd_light_rrdb/checkpoint-100000"

echo -e "${BLUE}🎉 Download Summary${NC}"
echo "=================="
echo -e "Total models: $TOTAL_DOWNLOADS"
echo -e "${GREEN}✅ Completed: $COMPLETED_DOWNLOADS${NC}"
echo -e "${YELLOW}⏭️  Skipped: $SKIPPED_DOWNLOADS${NC}"
echo -e "${RED}❌ Failed: $FAILED_DOWNLOADS${NC}"
echo ""

# Provide usage recommendations
if [ $COMPLETED_DOWNLOADS -gt 0 ] || [ $SKIPPED_DOWNLOADS -gt 0 ]; then
    echo -e "${BLUE}📋 Usage Recommendations${NC}"
    echo "========================"
    
    if [ -d "runs/pasd_light/checkpoint-100000" ]; then
        echo -e "${GREEN}🔧 For 8-16GB VRAM (GTX 1080, RTX 3070):${NC}"
        echo "   python test_pasd.py --use_pasd_light --image_path examples/dog.png"
        echo ""
    fi
    
    if [ -d "runs/pasd/checkpoint-100000" ]; then
        echo -e "${GREEN}🔧 For 24GB VRAM (RTX 3090/4090):${NC}"
        echo "   python test_pasd.py --image_path examples/dog.png --upscale 2"
        echo ""
    fi
    
    if [ -d "runs/pasd_rrdb/checkpoint-100000" ]; then
        echo -e "${GREEN}🔧 For 80GB VRAM (A100/H100):${NC}"
        echo "   python test_pasd.py --pasd_model_path runs/pasd_rrdb/checkpoint-100000 --upscale 4"
        echo ""
    fi
    
    echo -e "${BLUE}🚀 Run all examples:${NC}"
    echo "   ./run_examples.sh"
fi

if [ $FAILED_DOWNLOADS -gt 0 ]; then
    echo ""
    echo -e "${RED}⚠️  Some downloads failed. Check your internet connection and try again.${NC}"
fi

echo ""
echo -e "${BLUE}💾 Disk Usage:${NC}"
if command -v du &> /dev/null; then
    echo "   PASD Models: $(du -sh runs/ 2>/dev/null | cut -f1 || echo 'N/A')"
fi

echo -e "${BLUE}🎯 Next Steps:${NC}"
echo "   1. Run: ./run_examples.sh"
echo "   2. Or try: python test_pasd.py --image_path examples/dog.png"
echo "   3. Check output/ directory for results"