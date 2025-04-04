# YouTube video information extraction utilities

import re
import logging
from typing import Dict, Any, Optional
import yt_dlp

# Configure logging
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> Optional[str]:
    # Extract video ID from various YouTube URL formats
    
    if not url:
        return None
    
    # Handle already extracted IDs (11 characters)
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
    
    # Match patterns for different YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/watch\?.*v=)([A-Za-z0-9_-]{11})',
        r'(?:youtube\.com\/shorts\/)([A-Za-z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    logger.warning(f"Could not extract video ID from URL: {url}")
    return None

def get_video_info(video_id: str) -> Dict[str, Any]:
    # Get video metadata from YouTube
    
    if not video_id:
        return {"error": "Invalid video ID"}
    
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': True,
            'force_generic_extractor': False
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            
            # Extract relevant fields
            result = {
                "title": info.get("title", "Unknown"),
                "channel": info.get("uploader", info.get("channel", "Unknown")),
                "description": info.get("description", ""),
                "duration": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
                "upload_date": info.get("upload_date", ""),
                "thumbnail": info.get("thumbnail", ""),
                "tags": info.get("tags", [])
            }
            
            return result
    
    except Exception as e:
        logger.error(f"Error fetching video info: {str(e)}")
        return {"error": str(e)} 