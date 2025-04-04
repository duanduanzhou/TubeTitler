# Visual content extraction utilities

import os
import logging
import tempfile
import requests
from typing import Optional
import yt_dlp

# Configure logging
logger = logging.getLogger(__name__)

def download_thumbnail(video_id: str) -> Optional[str]:
    # Download video thumbnail and return path to image file
    
    if not video_id:
        logger.error("Invalid video ID provided")
        return None
    
    try:
        # Create temp directory for downloads if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), "tubetitler_thumbnails")
        os.makedirs(temp_dir, exist_ok=True)
        
        output_file = os.path.join(temp_dir, f"{video_id}.jpg")
        
        # Check if file already exists
        if os.path.exists(output_file):
            logger.info(f"Thumbnail already exists at: {output_file}")
            return output_file
        
        # Try to get video info to extract thumbnail URL
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
                # Find highest quality thumbnail
                thumbnails = info.get('thumbnails', [])
                if not thumbnails:
                    # Fallback to standard thumbnail URL
                    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"
                else:
                    # Sort by resolution and get highest quality
                    thumbnails = sorted(
                        thumbnails, 
                        key=lambda x: (x.get('width', 0) * x.get('height', 0)), 
                        reverse=True
                    )
                    thumbnail_url = thumbnails[0]['url']
                
                # Download the thumbnail
                logger.info(f"Downloading thumbnail from: {thumbnail_url}")
                response = requests.get(thumbnail_url, timeout=10)
                response.raise_for_status()
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Thumbnail downloaded successfully: {output_file}")
                return output_file
                
        except Exception as e:
            # Fallback to standard YouTube thumbnail URLs
            logger.warning(f"Error getting thumbnail URL from video info: {str(e)}")
            logger.info("Trying standard YouTube thumbnail URLs...")
            
            # Try different resolutions
            resolutions = ["maxresdefault", "sddefault", "hqdefault", "mqdefault", "default"]
            
            for resolution in resolutions:
                try:
                    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/{resolution}.jpg"
                    response = requests.get(thumbnail_url, timeout=10)
                    
                    # If request successful, save the image
                    if response.status_code == 200:
                        with open(output_file, 'wb') as f:
                            f.write(response.content)
                        
                        logger.info(f"Thumbnail downloaded successfully: {output_file}")
                        return output_file
                
                except Exception as inner_e:
                    logger.warning(f"Failed to download {resolution} thumbnail: {str(inner_e)}")
                    continue
            
            logger.error("All thumbnail download attempts failed")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading thumbnail: {str(e)}")
        return None 