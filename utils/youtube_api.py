import os
import yaml
from urllib.parse import urlparse, parse_qs
import requests
import logging
import json
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError):
        logger.warning(f"Config file not found or invalid at {config_path}")
        return {
            "api_keys": {"youtube": None},
            "paths": {},
            "models": {},
            "processing": {}
        }

class YouTubeAPI:
    # API wrapper with fallback for no API key
    
    def __init__(self, api_key=None):
        config = load_config()
        self.api_key = api_key or config.get('api_keys', {}).get('youtube')
        self.has_api_key = bool(self.api_key) and self.api_key != "YOUR_YOUTUBE_API_KEY"
        
        # Initialize official API client if key available
        if self.has_api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
                logger.info("YouTube API initialized with API key")
            except Exception as e:
                logger.warning(f"Failed to initialize YouTube API: {str(e)}")
                self.has_api_key = False
        
        if not self.has_api_key:
            logger.info("YouTube API key N/A - using fallback methods")
    
    def get_video_info(self, video_id):
        if self.has_api_key:
            return self._get_video_info_api(video_id)
        else:
            return self._get_video_info_fallback(video_id)
    
    def _get_video_info_api(self, video_id):
        try:
            response = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            ).execute()
            
            if not response.get('items'):
                logger.warning(f"No video found with ID: {video_id}")
                return {"success": False, "error": "Video not found"}
            
            video_data = response['items'][0]
            snippet = video_data.get('snippet', {})
            statistics = video_data.get('statistics', {})
            
            return {
                "success": True,
                "video_id": video_id,
                "video_info": {
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'channel': snippet.get('channelTitle', ''),
                    'tags': snippet.get('tags', []),
                    'view_count': int(statistics.get('viewCount', 0)),
                    'published_at': snippet.get('publishedAt', '')
                }
            }
            
        except HttpError as e:
            logger.error(f"YouTube API error: {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error with video info API call: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_video_info_fallback(self, video_id):
        # Get video info using yt-dlp as fallback
        try:
            import yt_dlp
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'forcejson': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                
            return {
                "success": True,
                "video_id": video_id,
                "video_info": {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'channel': info.get('channel', ''),
                    'tags': info.get('tags', []),
                    'view_count': info.get('view_count', 0),
                    'published_at': info.get('upload_date', '')
                }
            }
            
        except Exception as e:
            logger.error(f"Error in video info fallback: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def search_videos(self, query, max_results=10):
        # Search for videos with query
        if self.has_api_key:
            return self._search_api(query, max_results)
        else:
            return self._search_fallback(query, max_results)
    
    def _search_api(self, query, max_results=10):
        # Search videos using API
        try:
            # Call the search.list method to search for videos with the query
            response = self.youtube.search().list(
                q=query,
                part="id,snippet",
                maxResults=min(max_results, 50),
                type="video",
                safeSearch="none"
            ).execute()
            
            videos = response.get('items', [])
            video_ids = [video['id']['videoId'] for video in videos if 'videoId' in video['id']]
            
            # If we need more information, we can get video details
            if video_ids:
                video_response = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=','.join(video_ids[:50])  # Max 50 IDs per request
                ).execute()
                detailed_videos = video_response.get('items', [])
            else:
                detailed_videos = []
            
            return {
                "success": True,
                "video_ids": video_ids,
                "videos": detailed_videos,
                "total_results": response.get('pageInfo', {}).get('totalResults', 0)
            }
            
        except HttpError as e:
            logger.error(f"YouTube API search error: {str(e)}")
            return {"success": False, "error": str(e), "video_ids": []}
        except Exception as e:
            logger.error(f"Error in search API: {str(e)}")
            return {"success": False, "error": str(e), "video_ids": []}
    
    def _search_fallback(self, query, max_results=10):
        # Search videos using scraping method
        try:
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            response = requests.get(search_url)
            
            # Simple regex to extract video IDs
            import re
            video_ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", response.text)
            video_ids = list(dict.fromkeys(video_ids))[:max_results]
            
            return {
                "success": True,
                "video_ids": video_ids,
                "videos": [], 
                "total_results": len(video_ids)
            }
            
        except Exception as e:
            logger.error(f"Error in search fallback: {str(e)}")
            return {"success": False, "error": str(e), "video_ids": []}
    
    def get_trending_videos(self, region_code="US", max_results=10):
        if self.has_api_key:
            return self._get_trending_api(region_code, max_results)
        else:
            return self._get_trending_fallback(max_results)
    
    def _get_trending_api(self, region_code="US", max_results=10):
        try:
            response = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=min(max_results, 50)
            ).execute()
            
            videos = response.get('items', [])
            video_ids = [video.get('id') for video in videos]
            
            return {
                "success": True,
                "video_ids": video_ids,
                "videos": videos,
                "total_results": response.get('pageInfo', {}).get('totalResults', 0)
            }
            
        except HttpError as e:
            logger.error(f"YouTube API trending error: {str(e)}")
            return {"success": False, "error": str(e), "video_ids": []}
        except Exception as e:
            logger.error(f"Error in trending API: {str(e)}")
            return {"success": False, "error": str(e), "video_ids": []}
    
    def _get_trending_fallback(self, max_results=10):
        try:
            trending_url = "https://www.youtube.com/feed/trending"
            response = requests.get(trending_url)
            
            import re
            video_ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", response.text)
            
            # Remove duplicates and limit results
            video_ids = list(dict.fromkeys(video_ids))[:max_results]
            
            return {
                "success": True,
                "video_ids": video_ids,
                "videos": [], 
                "total_results": len(video_ids)
            }
            
        except Exception as e:
            logger.error(f"Error in trending fallback: {str(e)}")
            return {"success": False, "error": str(e), "video_ids": []}