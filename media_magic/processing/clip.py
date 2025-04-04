# CLIP embedding generation for images

import os
import logging
import torch
import numpy as np
from typing import List, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logger = logging.getLogger(__name__)

# Initialize CLIP model (lazy-loaded)
_clip_model = None
_clip_processor = None

def get_clip_model():
    # Load CLIP model once and reuse
    global _clip_model, _clip_processor
    
    if _clip_model is None:
        logger.info("Loading CLIP model (ViT-B/32)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.to(device)
        logger.info(f"CLIP model loaded successfully on {device}")
    
    return _clip_model, _clip_processor

def get_clip_embedding(image_path: str) -> List[float]:
    # Generate CLIP embedding for an image
    # Returns normalized embedding as Python list
    
    if not image_path or not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return []
    
    try:
        # Get model and preprocessing function
        model, processor = get_clip_model()
        device = next(model.parameters()).device
        
        # Load and preprocess image
        logger.info(f"Processing image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
        except Exception as img_err:
            logger.error(f"Error loading image: {str(img_err)}")
            return []
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list for JSON serialization
        embedding = image_features.cpu().numpy().tolist()[0]
        logger.info(f"CLIP embedding generated successfully: {len(embedding)} dimensions")
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating CLIP embedding: {str(e)}")
        return [] 