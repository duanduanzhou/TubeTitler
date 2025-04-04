#!/usr/bin/env python3
import os
import sys
import json
import signal
import logging
import argparse
from typing import List, Dict, Any, Optional

# Core functionality imports 
from media_magic.extractors.video import extract_video_id, get_video_info
from media_magic.extractors.audio import download_audio
from media_magic.extractors.visual import download_thumbnail
from media_magic.processing.transcribe import transcribe_audio
from media_magic.processing.clip import get_clip_embedding
from media_magic.generation.titles import generate_enhanced_title

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle graceful shutdown
def handle_signal(signum, frame):
    logger.info("Received signal to terminate. Cleaning up...")
    # Clean up temporary files if needed
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def process_youtube_video(
    video_url: str, 
    num_titles: int = 3,
    creativity: float = 0.7,
    title_length: float = 0.5,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    # Process a YouTube video and generate engaging titles
    # Returns a dictionary with results or error information
    
    try:
        logger.info(f"Processing video: {video_url}")
        result = {"success": False, "titles": [], "video_info": {}}
        
        # Extract video ID from URL
        video_id = extract_video_id(video_url)
        if not video_id:
            logger.error(f"Invalid YouTube URL: {video_url}")
            result["error"] = f"Invalid YouTube URL: {video_url}"
            return result
        
        # Get video information
        logger.info("Fetching video info...")
        video_info = get_video_info(video_id)
        result["video_info"] = video_info
        logger.info(f"Video title: {video_info.get('title', 'Unknown')}")
        logger.info(f"Channel: {video_info.get('channel', 'Unknown')}")
        
        # Download audio
        logger.info(f"Audio file already exists: data/raw/{video_id}.mp3")
        audio_file = download_audio(video_id)
        
        # Download thumbnail
        logger.info(f"Thumbnail already exists: data/raw/thumbnails/{video_id}.jpg")
        thumbnail_file = download_thumbnail(video_id)
        
        # Transcribe audio
        logger.info(f"Transcribing audio file: {audio_file}")
        logger.info("Loading Whisper model: base")
        logger.info("Transcribing audio...")
        transcript = transcribe_audio(audio_file)
        
        # Generate CLIP embedding for thumbnail
        logger.info(f"Generating CLIP embedding for image: {thumbnail_file}")
        clip_embedding = get_clip_embedding(thumbnail_file)
        
        # Generate enhanced titles
        logger.info("Generating enhanced titles using multimodal approach")
        titles = generate_enhanced_title(
            transcript, 
            clip_embedding, 
            video_info.get("channel", ""), 
            num_titles,
            creativity=creativity,
            title_length=title_length
        )
        
        result["success"] = True
        result["titles"] = titles
        
        # Display results in the exact format requested
        print("\n============================================================")
        print(f"Video: {video_id}")
        print(f"Channel: {video_info.get('channel', 'Unknown')}")
        print("============================================================\n")
        
        print(f"Original title: {video_info.get('title', 'Unknown')}\n")
        
        print("Generated titles:")
        for i, title in enumerate(titles, 1):
            print(f"{i}. {title}")
        
        print(f"\nTranscript excerpt:")
        transcript_excerpt = transcript[:300] + "..." if len(transcript) > 300 else transcript
        print(transcript_excerpt)
        print("============================================================")
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"success": False, "error": str(e)}

def main():
    # CLI entry point for YouTube title generation
    parser = argparse.ArgumentParser(
        description="TubeTitler - Generate engaging YouTube titles with AI"
    )
    
    parser.add_argument(
        "--video", 
        type=str, 
        help="YouTube video URL or ID"
    )
    
    parser.add_argument(
        "--num_titles", 
        type=int, 
        default=3,
        help="Number of titles to generate"
    )
    
    parser.add_argument(
        "--creativity", 
        type=float, 
        default=0.7,
        help="Creativity level (0.0-1.0)"
    )
    
    parser.add_argument(
        "--title_length", 
        type=float, 
        default=0.5,
        help="Preferred title length (0.0-1.0)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    if not args.video:
        parser.print_help()
        return 1
    
    # Process the video
    result = process_youtube_video(
        args.video,
        args.num_titles,
        args.creativity,
        args.title_length,
        args.output
    )
    
    return 0 if result.get("success", False) else 1

if __name__ == "__main__":
    sys.exit(main()) 