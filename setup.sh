#!/bin/bash

# TubeTitler Setup Script
# ----------------------
# This script sets up the environment for TubeTitler

set -e  # Exit on error

echo "========== TubeTitler Setup =========="
echo "This script will set up the TubeTitler environment."

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d " " -f 2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Detected Python $PYTHON_VERSION"

# Create or activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

source .venv/bin/activate
echo "Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install wheel setuptools

# Try installing requirements
echo "Installing main requirements..."
pip install -r requirements.txt

# Create directories
echo "Creating necessary directories..."
mkdir -p data/raw/videos
mkdir -p data/raw/thumbnails
mkdir -p data/raw/transcripts
mkdir -p data/raw/frames
mkdir -p data/processed/embeddings
mkdir -p models/checkpoints/title_generator

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg is not installed. Please install it for video processing."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  On macOS you can install it with: brew install ffmpeg"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  On Ubuntu/Debian: sudo apt-get install ffmpeg"
        echo "  On CentOS/RHEL: sudo yum install ffmpeg"
    fi
    echo "Please install ffmpeg and run this script again."
    exit 1
fi

# Verify whisper installation
echo "Verifying Whisper installation..."
python -c "
try:
    import whisper
    print('Whisper is installed correctly')
except ImportError:
    print('Warning: Whisper is not installed correctly.')
    print('   Try running: pip install --force-reinstall openai-whisper')
"

# Verify CLIP installation
echo "Verifying CLIP installation..."
python -c "
try:
    import clip
    print('CLIP is installed correctly')
except ImportError:
    print('Warning: CLIP is not installed correctly.')
    print('   Try running: pip install git+https://github.com/openai/CLIP.git')
"

# Download pre-trained model
echo "Checking for pre-trained model..."
MODEL_DIR="models/checkpoints/title_generator"
MODEL_FILE="$MODEL_DIR/pytorch_model.bin"

if [ -f "$MODEL_FILE" ]; then
    echo "Pre-trained model already exists"
else
    echo "Downloading pre-trained model..."
    # Python script to download the model
    python -c "
import os
import requests
from tqdm import tqdm
import zipfile

MODEL_URL = 'https://huggingface.co/AndrewGuenther/TubeTitler/resolve/main/title_generator.zip'
ZIP_PATH = 'models/checkpoints/title_generator.zip'

try:
    print('Downloading pre-trained model...')
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(ZIP_PATH, 'wb') as f, tqdm(
        total=total_size, 
        unit='B', 
        unit_scale=True, 
        unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)
    
    print('Extracting model files...')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('models/checkpoints')
    
    # Clean up
    os.remove(ZIP_PATH)
    print('Model downloaded and extracted successfully')
except Exception as e:
    print(f'Error downloading model: {str(e)}')
    print('You can still use TubeTitler, but you will need to manually download the model later.')
"
fi

# Create config.yaml if it doesn't exist
if [ ! -f "config.yaml" ]; then
    echo "Creating config.yaml template..."
    cat > config.yaml << EOL
api_keys:
  youtube: "YOUR_YOUTUBE_API_KEY"  # Optional, for search functionality

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  thumbnails: "data/raw/thumbnails"
  frames: "data/raw/frames"
  transcripts: "data/raw/transcripts"
  embeddings: "data/processed/embeddings"

models:
  whisper:
    model_size: "base"  # tiny, base, small, medium, large
    language: "en"
  clip:
    model_name: "ViT-B/32"

processing:
  max_frames_per_video: 3
EOL
    echo "Config file created. YouTube API key is optional, needed only for searching videos."
fi

echo
echo "Setup complete!"
echo
echo "Next steps:"
echo "1. Generate titles for a YouTube video: python -m bin.youtube_title_generator --video \"YOUR_YOUTUBE_URL\""
echo "2. Process a specific video: python -m engine.test_simple --video \"YOUR_YOUTUBE_URL\" --num_titles 5"
echo "3. Or process multiple videos from a search: python -m engine.pipeline --search \"python tutorial\" --max_videos 5"
echo 