#!/bin/bash

# TubeTitler Demo Script
# This script demonstrates the complete workflow of TubeTitler

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Demo video - Tech talk on AI by Google DeepMind
VIDEO_ID="N5oZIO8pE40"
VIDEO_URL="https://www.youtube.com/watch?v=${VIDEO_ID}"

# Print header
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}       TubeTitler Demo                   ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg is not installed.${NC}"
    echo "Please install FFmpeg first. It's required for video processing."
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo
fi

# Check if model exists or will be downloaded
echo -e "${YELLOW}Checking pre-trained model...${NC}"
MODEL_DIR="models/checkpoints/title_generator"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo -e "${GREEN}Pre-trained model exists locally${NC}"
else
    echo -e "${YELLOW}Pre-trained model will be downloaded automatically during first run${NC}"
fi
echo

# Process the video
echo -e "${YELLOW}Processing video ${VIDEO_ID}${NC}"
echo "This step will download audio, transcribe it, extract frames, and generate CLIP embeddings."
echo
python -m media_magic.title_generator --video ${VIDEO_URL} --num_titles 1
echo

# Explore the processed data
echo -e "${YELLOW}Exploring processed data${NC}"
echo "Let's look at what data was generated:"
echo

# Get transcript path
TRANSCRIPT_PATH="data/raw/transcripts/${VIDEO_ID}.json"
if [ -f "$TRANSCRIPT_PATH" ]; then
    echo -e "${BLUE}Transcript excerpt:${NC}"
    python -c "
import json
try:
    with open('$TRANSCRIPT_PATH', 'r') as f:
        data = json.load(f)
        text = data.get('text', '')
        excerpt = text[:200] + '...' if len(text) > 200 else text
        print(excerpt)
except Exception as e:
    print(f'Error reading transcript: {str(e)}')
"
    echo
else
    echo -e "${YELLOW}Transcript not found. Check the temp directory for transcriptions.${NC}"
fi

# Get embeddings info
EMBEDDING_PATH="data/processed/embeddings/${VIDEO_ID}.json"
if [ -f "$EMBEDDING_PATH" ]; then
    echo -e "${BLUE}CLIP embeddings info:${NC}"
    python -c "
import json
try:
    with open('$EMBEDDING_PATH', 'r') as f:
        data = json.load(f)
        print(f'CLIP model: {data.get(\"clip_model\", \"Unknown\")}')
        embedding = data.get('clip_embedding_avg', [])
        print(f'Embedding dimensions: {len(embedding)}')
except Exception as e:
    print(f'Error reading embeddings: {str(e)}')
"
    echo
else
    echo -e "${YELLOW}Embeddings file not found. They may be stored in a different location.${NC}"
fi

# Generate titles with different creativity levels
echo -e "${YELLOW}Title generation with various creativity levels${NC}"
echo "Generating multiple title suggestions with different creativity levels..."
echo

# Run with different creativity levels
echo -e "${BLUE}Lower creativity (0.5):${NC}"
python -m media_magic.title_generator --video ${VIDEO_URL} --num_titles 1 --creativity 0.5
echo

echo -e "${BLUE}Higher creativity (0.9):${NC}"
python -m media_magic.title_generator --video ${VIDEO_URL} --num_titles 1 --creativity 0.9
echo

# Final message
echo -e "${GREEN}Demo completed!${NC}"
echo "You can generate your own titles using:"
echo "python -m media_magic.title_generator --video \"YOUR_YOUTUBE_URL\" --num_titles 3 --creativity 0.7"
echo
echo -e "${BLUE}=========================================${NC}" 