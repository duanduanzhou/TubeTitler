import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config
from utils.youtube_api import YouTubeAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SampleAssembler:
    """Assembles final training samples from processed data components."""
    
    def __init__(self):
        """Initialize sample assembler with configuration."""
        self.config = get_config()
        self.processed_dir = self.config.get_path("processed_data")
        self.transcripts_dir = self.config.get_path("transcripts")
        self.embeddings_dir = self.config.get_path("embeddings")
        self.youtube_api = YouTubeAPI()
    
    def assemble_sample(self, video_id: str, include_raw_embeddings: bool = False) -> Dict[str, Any]:
        """
        Assemble a training sample for a video.
        
        Args:
            video_id: YouTube video ID
            include_raw_embeddings: Whether to include full CLIP embeddings (large)
            
        Returns:
            Dictionary with assembled sample data and success status
        """
        result = {
            "video_id": video_id,
            "success": False,
            "error": None,
            "sample_path": None
        }
        
        try:
            # Get video metadata from YouTube API
            logger.info(f"Fetching metadata for video {video_id}")
            metadata = self.youtube_api.get_video_metadata(video_id)
            
            if not metadata:
                result["error"] = f"Could not fetch metadata for video {video_id}"
                logger.error(result["error"])
                return result
            
            # Get transcript
            transcript_path = os.path.join(self.transcripts_dir, f"{video_id}.json")
            if not os.path.exists(transcript_path):
                result["error"] = f"Transcript not found for video {video_id}"
                logger.error(result["error"])
                return result
                
            with open(transcript_path, 'r') as f:
                transcript_data = json.load(f)
            
            # Get CLIP embeddings
            embeddings_path = os.path.join(self.embeddings_dir, f"{video_id}.json")
            if not os.path.exists(embeddings_path):
                result["error"] = f"Embeddings not found for video {video_id}"
                logger.error(result["error"])
                return result
                
            with open(embeddings_path, 'r') as f:
                embeddings_data = json.load(f)
            
            # Assemble the sample
            sample = {
                "video_id": video_id,
                "title": metadata.get("title", ""),
                "description": metadata.get("description", ""),
                "tags": metadata.get("tags", []),
                "channel": metadata.get("channel_title", ""),
                "transcript": transcript_data.get("text", ""),
                "view_count": metadata.get("view_count", 0),
                "frame_count": len(embeddings_data.get("embeddings", [])),
                "clip_model": embeddings_data.get("model", "ViT-B/32"),
            }
            
            # Calculate average embedding for convenience (for retrieval models)
            embeddings_list = []
            for emb_item in embeddings_data.get("embeddings", []):
                embeddings_list.append(emb_item.get("embedding", []))
                
            if embeddings_list:
                sample["clip_embedding_avg"] = np.mean(embeddings_list, axis=0).tolist()
                
                # Include individual embeddings if requested
                if include_raw_embeddings:
                    sample["clip_embeddings"] = embeddings_list
            
            # Save the assembled sample
            output_path = os.path.join(self.processed_dir, f"{video_id}.json")
            
            with open(output_path, 'w') as f:
                json.dump(sample, f, indent=2)
            
            logger.info(f"Assembled training sample saved to {output_path}")
            
            result["success"] = True
            result["sample_path"] = output_path
            return result
            
        except Exception as e:
            result["error"] = f"Error assembling sample: {str(e)}"
            logger.error(result["error"])
            return result
    
    def batch_assemble(self, video_ids: List[str], include_raw_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Assemble training samples for multiple videos."""
        results = []
        
        for video_id in video_ids:
            result = self.assemble_sample(video_id, include_raw_embeddings)
            results.append(result)
            
        return results
    
    def create_dataset(self, output_file: str, success_only: bool = True):
        """
        Create a consolidated dataset from all assembled samples.
        
        Args:
            output_file: Path to output file (JSON or CSV)
            success_only: Whether to include only successful samples
        """
        # Get all sample files
        sample_files = [f for f in os.listdir(self.processed_dir) 
                       if f.endswith('.json') and f != 'dataset.json']
        
        dataset = []
        
        for filename in sample_files:
            file_path = os.path.join(self.processed_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                    
                dataset.append(sample)
                
            except Exception as e:
                logger.error(f"Error loading sample {filename}: {str(e)}")
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Save the dataset
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(dataset, f)
        elif output_file.endswith('.csv'):
            import pandas as pd
            df = pd.DataFrame(dataset)
            df.to_csv(output_file, index=False)
        else:
            logger.error(f"Unsupported output format: {output_file}")
            return False
            
        logger.info(f"Dataset saved to {output_file}")
        return True

def assemble_training_sample(video_id: str, transcript: Optional[str] = None, 
                             clip_embeddings: Optional[Union[List, np.ndarray]] = None, 
                             title: Optional[str] = None) -> Dict[str, Any]:
    """
    Creates a training sample.
    @ Returns: Dictionary with result status and path
    """
    assembler = SampleAssembler()
    return assembler.assemble_sample(video_id)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Assemble training samples from processed data")
    parser.add_argument("--video_id", type=str, help="YouTube video ID to process")
    parser.add_argument("--batch", action="store_true", help="Process all available videos")
    parser.add_argument("--create_dataset", type=str, help="Create consolidated dataset file")
    parser.add_argument("--include_raw_embeddings", action="store_true", 
                        help="Include raw CLIP embeddings (results in larger files)")
    
    args = parser.parse_args()
    assembler = SampleAssembler()
    
    if args.create_dataset:
        success = assembler.create_dataset(args.create_dataset)
        if not success:
            sys.exit(1)
    elif args.batch:
        # Find all videos with both transcripts and embeddings
        transcript_dir = assembler.transcripts_dir
        embeddings_dir = assembler.embeddings_dir
        
        transcript_ids = {os.path.splitext(f)[0] for f in os.listdir(transcript_dir) 
                         if f.endswith('.json')}
        embedding_ids = {os.path.splitext(f)[0] for f in os.listdir(embeddings_dir) 
                        if f.endswith('.json')}
        
        # Videos with both
        video_ids = transcript_ids.intersection(embedding_ids)
        
        if not video_ids:
            print("No videos found with both transcripts and embeddings")
            sys.exit(1)
            
        print(f"Processing {len(video_ids)} videos")
        results = assembler.batch_assemble(list(video_ids), args.include_raw_embeddings)
        
        success_count = sum(1 for r in results if r["success"])
        print(f"Successfully assembled {success_count}/{len(results)} samples")
        
    elif args.video_id:
        result = assembler.assemble_sample(args.video_id, args.include_raw_embeddings)
        
        if result["success"]:
            print(f"Sample assembled successfully: {result['sample_path']}")
        else:
            print(f"Failed to assemble sample: {result['error']}")
            sys.exit(1)
    else:
        parser.error("Either --video_id, --batch, or --create_dataset is required")