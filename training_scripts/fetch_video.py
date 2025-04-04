import os
import sys
import logging
import argparse
from pathlib import Path
import yt_dlp
import requests
import json

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config
from utils.youtube_api import YouTubeAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fetch_video_info(video_id):
    """Fetch video information from YouTube API
    @ Returns: Dictionary with video metadata"""
    try:
        # Initialize API
        api = YouTubeAPI()
        
        # Get video info
        info = api.get_video_info(video_id)
        
        return info
    except Exception as e:
        return {"success": False, "error": str(e)}

def download_video(video_id, mode="audio", output_dir=None):
    """Download YouTube video in specified mode (audio, video, or both)
    @ Returns: Dictionary with download results"""
    try:
        config = get_config()
        
        if output_dir is None:
            output_dir = config.get_path("raw_data")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up yt-dlp options based on mode
        if mode == "audio":
            output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
            options = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                'noplaylist': True,
                'progress_hooks': [],
            }
            expected_output = os.path.join(output_dir, f"{video_id}.mp3")
        elif mode == "video":
            output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
            options = {
                'format': 'best[height<=720]',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                'noplaylist': True,
                'progress_hooks': [],
            }
            expected_output = os.path.join(output_dir, f"{video_id}.mp4")
        else:  # both
            output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")
            options = {
                'format': 'best[height<=720]',
                'outtmpl': output_template,
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                'noplaylist': True,
                'progress_hooks': [],
            }
            expected_output = os.path.join(output_dir, f"{video_id}.mp4")
            
        # Skip download if file already exists
        if os.path.exists(expected_output):
            logger.info(f"File already exists: {expected_output}")
            return {
                "success": True,
                "video_id": video_id,
                "output_file": expected_output,
                "mode": mode,
                "message": "File already exists"
            }
        
        # Download video
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([video_url])
        
        # Download thumbnail
        download_thumbnail(video_id)
        
        # Check if download was successful
        if os.path.exists(expected_output):
            logger.info(f"Downloaded {mode} to {expected_output}")
            return {
                "success": True,
                "video_id": video_id,
                "output_file": expected_output,
                "mode": mode
            }
        else:
            return {
                "success": False,
                "video_id": video_id,
                "error": f"File not found after download: {expected_output}",
                "mode": mode
            }
            
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "mode": mode
        }

def download_thumbnail(video_id, output_dir=None):
    """Download video thumbnail at highest available resolution
    @ Returns: Dictionary with download results"""
    try:
        config = get_config()
        
        if output_dir is None:
            output_dir = config.get_path("thumbnails")
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{video_id}.jpg")
        
        # Skip if thumbnail already exists
        if os.path.exists(output_path):
            logger.info(f"Thumbnail already exists: {output_path}")
            return {
                "success": True,
                "video_id": video_id,
                "output_file": output_path,
                "message": "File already exists"
            }
        
        # Try different thumbnail qualities (best to worst)
        thumbnail_urls = [
            f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/sddefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
            f"https://img.youtube.com/vi/{video_id}/default.jpg",
        ]
        
        for url in thumbnail_urls:
            response = requests.get(url)
            if response.status_code == 200 and response.content:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded thumbnail to {output_path}")
                return {
                    "success": True,
                    "video_id": video_id,
                    "output_file": output_path
                }
        
        return {
            "success": False,
            "video_id": video_id,
            "error": "No valid thumbnail found"
        }
        
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e)
        }

def search_videos(query, max_results=10):
    """Search for YouTube videos using the API
    @ Returns: Dictionary with search results"""
    try:
        api = YouTubeAPI()
        # Search for videos
        results = api.search_videos(query, max_results)
        return results
    except Exception as e:
        return {"success": False, "error": str(e), "video_ids": []}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fetch YouTube videos")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_id", type=str, help="Download a specific video")
    group.add_argument("--search", type=str, help="Search for videos")
    
    parser.add_argument("--mode", type=str, choices=['audio', 'video', 'both'], 
                      default='audio', help="Download mode")
    parser.add_argument("--max_results", type=int, default=5, 
                      help="Maximum search results")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.video_id:
        # Download a single video
        result = download_video(args.video_id, args.mode, args.output_dir)
        
        if result.get("success", False):
            print(f"Successfully downloaded {args.video_id}")
            print(f"Output file: {result.get('output_file')}")
        else:
            print(f"Failed to download {args.video_id}: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    elif args.search:
        # Search for videos
        result = search_videos(args.search, args.max_results)
        
        if result.get("success", False):
            print(f"Found {len(result.get('video_ids', []))} videos for '{args.search}':")
            for i, video_id in enumerate(result.get('video_ids', []), 1):
                print(f"{i}. {video_id}")
        else:
            print(f"Search failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

if __name__ == "__main__":
    main()