# TubeTitler 🎬

TubeTitler is your tool designed for crafting killer YouTube titles that actually work. It analyzes what's said and shown in videos to generate titles that boost CTR without resorting to cheap clickbait.

## How It Works

Behind the scenes, TubeTitler uses a multimodal approach:

- **Audio transcription** via Whisper ASR captures spoken content
- **Visual analysis** using CLIP embeddings understands visual context
- **Title generation** with a custom-trained T5 model optimized for engagement

These components are combined through a fusion layer to produce titles that reflect the video’s essence while boosting viewer interest.

## Quick Start

```bash
# Generate titles for any YouTube video
python -m media_magic.title_generator --video "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Adjust creativity level for more variety
python -m media_magic.title_generator --video "YOUR_VIDEO_URL" --creativity 0.9
```

## Features

- **Creativity dial**: Adjust from conservative (0.1) to bold (1.0) title styles.
- **Length control**: =Options for concise headlines or detailed descriptions.
- **Multiple options**: Generate a bunch of titles and pick your favorite.
- **Fallback system**: Ensures usable titles even if the model struggles.
- **Multimodal understanding**: Titles that integrate audio and visual insights.

## Real-World Examples

Here's what TubeTitler can do with popular videos:

```
Video: Rick Astley - Never Gonna Give You Up
Generated: 
→ The Ultimate 80s Dance Moves You Can't Resist
→ Rick Astley's Iconic Song That Took Over The Internet
→ The Most Famous Meme Song Of All Time
```

## Setup

```bash
# Get the code
git clone https://github.com/yourusername/TubeTitler.git
cd TubeTitler

# Install dependencies
pip install -r requirements.txt

# Fire it up
python -m media_magic.title_generator --video "YOUR_VIDEO_URL"
```
**Note:**  
This project uses Git LFS to store the model file.  
Please run `brew install git-lfs && git lfs install` before cloning, or run `git lfs pull` after cloning to download the model.

## Tech Stack

TubeTitler's pipeline works in three phases:

- **Extraction**: Downloads audio and thumbnail using yt-dlp
- **Processing**: Transcribes with Whisper, encodes visuals with CLIP
- **Generation**: Fuses multimodal embeddings through a custom T5 model

The model was trained on a dataset of high-performing YouTube videos across multiple niches to understand engagement patterns.

## Advanced Usage

Run with `--help` to unlock all options:

```bash
python -m media_magic.title_generator --help
```