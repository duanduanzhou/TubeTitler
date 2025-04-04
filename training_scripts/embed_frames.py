#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse
import numpy as np
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FrameEmbedder:
    # Extract visual features from video frames using CLIP
    
    def __init__(self, model_name: str = None):
        # Load config
        config = get_config()
        self.model_name = model_name or config.get("models.clip.model_name", "ViT-B/32")
        self.embeddings_dir = config.get_path("embeddings")
        self.thumbnails_dir = config.get_path("thumbnails")
        self.frames_dir = config.get_path("frames")
        
        # Create directories if they don't exist
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        # Initialize model variables
        self.model = None
        self.preprocess = None
        logger.info(f"Initialized FrameEmbedder with model: {self.model_name}")
    
    def load_model(self):
        """Load CLIP model"""
        if self.model is None:
            logger.info(f"Loading CLIP model: {self.model_name}")
            try:
                import torch
                import clip
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load(self.model_name, device=device)
                logger.info(f"CLIP model loaded on {device}")
                return True
            except ImportError:
                logger.error("Failed to import CLIP. Make sure it's installed.")
                logger.info("Try: pip install git+https://github.com/openai/CLIP.git")
                return False
            except Exception as e:
                logger.error(f"Error loading CLIP model: {str(e)}")
                return False
        return True
    
    def embed_frame(self, image_path: str) -> Optional[np.ndarray]:
        # Generate CLIP embedding for a single image
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
        # Load model if not already loaded
        if not self.load_model():
            return None
        
        try:
            import torch
            from PIL import Image
            
            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            processed_image = self.preprocess(image).unsqueeze(0)
            # Move to same device as model
            device = next(self.model.parameters()).device
            processed_image = processed_image.to(device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(processed_image)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to numpy array
            embedding = image_features.cpu().numpy()[0]
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding frame: {str(e)}")
            return None
    
    def embed_frames(self, video_id: str, max_frames: int = None) -> Dict[str, Any]:
        # Generate CLIP embeddings for frames
        if not self.load_model():
            return {"success": False, "error": "Failed to load CLIP model"}
        try:
            # Check for thumbnail first
            thumbnail_path = os.path.join(self.thumbnails_dir, f"{video_id}.jpg")
            if os.path.exists(thumbnail_path):
                logger.info(f"Processing thumbnail for {video_id}")
                thumbnail_embed = self.embed_frame(thumbnail_path)
                if thumbnail_embed is None:
                    return {"success": False, "error": "Failed to embed thumbnail"}
                
                # check for frames directory
                frames_dir = os.path.join(self.frames_dir, video_id)
                frame_embeddings = []
                
                if os.path.exists(frames_dir):
                    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
                    frame_files.sort()  # Sort to ensure consistent ordering
                    
                    # Limit frames if max_frames is set
                    if max_frames and len(frame_files) > max_frames:
                        # Select evenly spaced frames
                        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
                        frame_files = [frame_files[i] for i in indices]
                    
                    logger.info(f"Processing {len(frame_files)} frames for {video_id}")
                    
                    for frame_file in frame_files:
                        frame_path = os.path.join(frames_dir, frame_file)
                        embedding = self.embed_frame(frame_path)
                        if embedding is not None:
                            frame_embeddings.append(embedding.tolist())
                
                # If have frame embeddings, compute average
                if frame_embeddings:
                    avg_embedding = np.mean(frame_embeddings, axis=0).tolist()
                else:
                    # If no frames, just use thumbnail
                    avg_embedding = thumbnail_embed.tolist()
                
                # Save embeddings
                output_path = os.path.join(self.embeddings_dir, f"{video_id}.json")
                result = {
                    "video_id": video_id,
                    "clip_model": self.model_name,
                    "thumbnail_embedding": thumbnail_embed.tolist(),
                    "frame_embeddings": frame_embeddings,
                    "clip_embedding_avg": avg_embedding
                }
                
                with open(output_path, 'w') as f:
                    json.dump(result, f)
                
                logger.info(f"Embeddings saved to {output_path}")
                return {
                    "success": True,
                    "embedding_path": output_path,
                    "frame_count": len(frame_embeddings)
                }
            else:
                return {"success": False, "error": f"Thumbnail not found for {video_id}"}
                
        except Exception as e:
            logger.error(f"Error embedding frames: {str(e)}")
            return {"success": False, "error": str(e)}


def embed_video(video_id: str, max_frames: int = None) -> Dict[str, Any]:
    # generate embeddings for a video
    embedder = FrameEmbedder()
    return embedder.embed_frames(video_id, max_frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for video frames")
    parser.add_argument("--video_id", type=str, required=True, help="YouTube video ID")
    parser.add_argument("--max_frames", type=int, default=None, 
                       help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    result = embed_video(args.video_id, args.max_frames)
    
    if result.get("success", False):
        print(f"Successfully embedded frames for {args.video_id}")
        print(f"Embeddings saved to: {result.get('embedding_path')}")
        print(f"Processed {result.get('frame_count', 0)} frames")
    else:
        print(f"Failed to embed frames: {result.get('error', 'Unknown error')}")
        sys.exit(1)