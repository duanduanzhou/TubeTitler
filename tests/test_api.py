import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from utils.youtube_api import YouTubeAPI

def test_connection():
    # Test YouTube API connection
    try:
        youtube_api = YouTubeAPI()
        logger.info("Testing fetching trending videos...")
        trending = youtube_api.get_trending_videos(max_results=3)
        
        if trending and trending.get('videos'):
            logger.info(f"Successfully fetched {len(trending['videos'])} trending videos")
            
            # Display video titles
            for i, video in enumerate(trending['videos'][:3], 1):
                snippet = video.get('snippet', {})
                title = snippet.get('title', 'No title')
                channel = snippet.get('channelTitle', 'Unknown channel')
                logger.info(f"  {i}. {title} by {channel}")
        else:
            logger.error("Failed to fetch trending videos")
            return False
        
        # Test search functionality
        logger.info("\nTesting search functionality...")
        search_query = "Python programming tutorial"
        search_results = youtube_api.search_videos(search_query, max_results=3)
        
        if search_results and search_results.get('videos'):
            logger.info(f"Successfully searched for '{search_query}'")
            logger.info(f"Found {search_results.get('total_results', 0)} videos in total")
            
            # show search results
            for i, video in enumerate(search_results['videos'][:3], 1):
                snippet = video.get('snippet', {})
                title = snippet.get('title', 'No title')
                channel = snippet.get('channelTitle', 'Unknown channel')
                logger.info(f"  {i}. {title} by {channel}")
        else:
            logger.error(f"Failed to search for '{search_query}'")
            return False
        
        logger.info("\nYouTube API integration is working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing YouTube API: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1) 