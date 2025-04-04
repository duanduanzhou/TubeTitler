#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import subprocess
import argparse
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config
from training_scripts.fetch_video import fetch_video_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_thumbnail(video_id: str, resolution: str = "maxres") -> Dict[str, Any]:
    """Download YouTube video thumbnail at specified resolution
    @ Returns: Dictionary with thumbnail info"""
    
    result = {
        "success": False,
        "video_id": video_id,
        "thumbnail_path": None,
        "error": None
    }
    
    # Get config
    config = get_config()
    thumbnails_dir = config.get_path("thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    # Thumbnail URL patterns
    resolution_map = {
        "default": f"https://i.ytimg.com/vi/{video_id}/default.jpg",
        "medium": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        "high": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "standard": f"https://i.ytimg.com/vi/{video_id}/sddefault.jpg",
        "maxres": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
    }
    
    # Try to download the requested resolution, or fall back to lower ones
    resolutions_to_try = ["maxres", "standard", "high", "medium", "default"]
    
    if resolution in resolutions_to_try:
        # Start from the requested resolution
        start_index = resolutions_to_try.index(resolution)
        resolutions_to_try = resolutions_to_try[start_index:]
    
    try:
        import requests
        from PIL import Image
        
        for res in resolutions_to_try:
            thumbnail_url = resolution_map[res]
            thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}.jpg")
            
            response = requests.get(thumbnail_url, stream=True)
            if response.status_code == 200:
                with open(thumbnail_path, "wb") as f:
                    f.write(response.content)
                
                # Verify the image can be opened
                try:
                    Image.open(thumbnail_path)
                    result["thumbnail_path"] = thumbnail_path
                    result["success"] = True
                    logger.info(f"Downloaded {res} thumbnail for {video_id}")
                    break
                except Exception as e:
                    logger.warning(f"Downloaded thumbnail for {video_id} but image is invalid: {str(e)}")
                    continue
            else:
                logger.debug(f"Thumbnail not available at resolution {res} for {video_id}")
        
        if not result["success"]:
            result["error"] = "Failed to download thumbnail at any resolution"
            logger.error(result["error"])
        
        return result
            
    except Exception as e:
        result["error"] = f"Error downloading thumbnail: {str(e)}"
        logger.error(result["error"])
        return result

def extract_video_frames(video_id: str, num_frames: int = 3) -> Dict[str, Any]:
    """Extract specific number of frames from video using FFmpeg
    @ Returns: Dictionary with frame extraction results"""
    
    result = {
        "success": False,
        "video_id": video_id,
        "frame_paths": [],
        "error": None
    }
    
    # Get config
    config = get_config()
    raw_dir = config.get_path("raw_data")
    frames_dir = config.get_path("frames")
    video_frames_dir = os.path.join(frames_dir, video_id)
    
    # Create output directory
    os.makedirs(video_frames_dir, exist_ok=True)
    
    # Check if video file exists
    video_path = os.path.join(raw_dir, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        result["error"] = f"Video file not found: {video_path}"
        logger.error(result["error"])
        return result
    
    try:
        # Use FFmpeg to extract frames
        logger.info(f"Extracting {num_frames} frames from {video_id}")
        
        # Get video duration using FFprobe
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        
        try:
            duration_process = subprocess.run(
                duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, check=True
            )
            duration = float(duration_process.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to get video duration: {str(e)}")
            duration = 0
        
        if duration <= 0:
            logger.warning(f"Invalid duration for {video_id}, assuming 60 seconds")
            duration = 60
        
        # Calculate frame timestamps
        if num_frames == 1:
            # Just the middle frame
            timestamps = [duration / 2]
        else:
            # Evenly spaced frames
            timestamps = [
                duration * i / (num_frames + 1) for i in range(1, num_frames + 1)
            ]
        
        # Extract each frame
        for i, timestamp in enumerate(timestamps):
            frame_path = os.path.join(video_frames_dir, f"frame_{i+1:03d}.jpg")
            
            frame_cmd = [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                "-vframes", "1", "-q:v", "2", frame_path
            ]
            
            subprocess.run(frame_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            if os.path.exists(frame_path):
                result["frame_paths"].append(frame_path)
                logger.debug(f"Extracted frame at {timestamp}s to {frame_path}")
            else:
                logger.warning(f"Failed to extract frame at {timestamp}s")
        
        if result["frame_paths"]:
            result["success"] = True
            logger.info(f"Extracted {len(result['frame_paths'])} frames from {video_id}")
        else:
            result["error"] = "No frames were extracted"
            logger.error(result["error"])
        
        return result
        
    except Exception as e:
        result["error"] = f"Error extracting frames: {str(e)}"
        logger.error(result["error"])
        return result

def extract_frames(video_id: str, num_frames: int = None) -> Dict[str, Any]:
    """Extract frames from video and download thumbnail
    @ Returns: Dictionary with extraction results"""
    
    config = get_config()
    if num_frames is None:
        num_frames = config.get("processing.max_frames_per_video", 3)
    
    result = {
        "success": False,
        "video_id": video_id,
        "thumbnail_path": None,
        "frame_paths": [],
        "error": None
    }
    
    # First try to download the thumbnail
    thumbnail_result = download_thumbnail(video_id)
    result["thumbnail_path"] = thumbnail_result.get("thumbnail_path")
    
    # Then try to extract frames if video file exists
    try:
        raw_dir = config.get_path("raw_data")
        video_path = os.path.join(raw_dir, f"{video_id}.mp4")
        
        if os.path.exists(video_path):
            frames_result = extract_video_frames(video_id, num_frames)
            if frames_result.get("success", False):
                result["frame_paths"] = frames_result.get("frame_paths", [])
                logger.info(f"Extracted {len(result['frame_paths'])} frames from {video_id}")
        
        # Consider success if we have either thumbnail or frames
        if result["thumbnail_path"] or result["frame_paths"]:
            result["success"] = True
        else:
            result["error"] = "Failed to get thumbnail or extract frames"
            logger.error(result["error"])
        
        return result
        
    except Exception as e:
        result["error"] = f"Error in frame extraction: {str(e)}"
        logger.error(result["error"])
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from YouTube videos")
    parser.add_argument("--video_id", type=str, required=True, help="YouTube video ID")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to extract")
    parser.add_argument("--thumbnail_only", action="store_true", help="Only download thumbnail")
    
    args = parser.parse_args()
    
    if args.thumbnail_only:
        result = download_thumbnail(args.video_id)
        if result["success"]:
            print(f"Downloaded thumbnail to {result['thumbnail_path']}")
        else:
            print(f"Failed to download thumbnail: {result['error']}")
            sys.exit(1)
    else:
        result = extract_frames(args.video_id, args.num_frames)
        if result["success"]:
            if result["thumbnail_path"]:
                print(f"Downloaded thumbnail to {result['thumbnail_path']}")
            if result["frame_paths"]:
                print(f"Extracted {len(result['frame_paths'])} frames:")
                for path in result["frame_paths"]:
                    print(f"- {path}")
        else:
            print(f"Failed to extract frames: {result['error']}")
            sys.exit(1)