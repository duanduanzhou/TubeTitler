# Audio extraction and processing utilities

import os
import logging
import tempfile
from typing import Optional
import yt_dlp

# Configure logging
logger = logging.getLogger(__name__)

def download_audio(video_id: str) -> Optional[str]:
    # Download audio and return path to audio file
    
    if not video_id:
        logger.error("Invalid video ID provided")
        return None
    
    try:
        # Create temp directory for downloads
        temp_dir = os.path.join(tempfile.gettempdir(), "tubetitler_audio")
        os.makedirs(temp_dir, exist_ok=True)
        
        output_file = os.path.join(temp_dir, f"{video_id}.mp3")
        
        # Check if file already exists
        if os.path.exists(output_file):
            logger.info(f"Audio file already exists at: {output_file}")
            return output_file
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(temp_dir, f"{video_id}.%(ext)s"),
            'quiet': True,
            'no_warnings': True
        }
        
        # Download the audio
        logger.info(f"Downloading audio for video {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        
        # Verify file exists
        if os.path.exists(output_file):
            logger.info(f"Audio downloaded successfully: {output_file}")
            return output_file
        else:
            logger.error(f"Failed to download audio: file not found at {output_file}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        return None 