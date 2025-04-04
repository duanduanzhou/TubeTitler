# Title generation module using transcripts and visual features

import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Path to trained model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "trained_model")

# Initialize model and tokenizer (lazy-loaded)
_model = None
_tokenizer = None

def load_model():
    # Load model and tokenizer once
    global _model, _tokenizer
    
    if _model is None:
        try:
            logger.info(f"Loading T5 model from {MODEL_PATH}")
            _tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
            _model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)
            logger.info("T5 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    return _model, _tokenizer

def extract_visual_concepts(clip_embedding, top_k=5):
    """Extract visual concepts from CLIP embedding (simplified version)."""
    try:
        # This is a simplified version that doesn't actually match against concepts
        # In a full implementation, you would compare with a set of concept embeddings
        
        # For now, return some generic visual concepts to use as prompts
        concepts = [
            "person", "music", "colorful", "dynamic", "emotional",
            "artistic", "vibrant", "energetic", "professional", "creative"
        ]
        
        # Randomly select some concepts (would normally be based on embedding similarity)
        import random
        selected_concepts = random.sample(concepts, min(top_k, len(concepts)))
        return selected_concepts
    except Exception as e:
        logger.error(f"Error extracting visual concepts: {str(e)}")
        return ["visual", "image", "video"]

def generate_enhanced_title(
    transcript: str, 
    clip_embedding: List[float], 
    channel_name: str, 
    num_titles: int = 3,
    creativity: float = 0.7,
    title_length: float = 0.5
) -> List[str]:
    """Generate engaging YouTube titles using trained multimodal model."""
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load_model()
        
        # Prepare transcript (truncate if too long)
        max_chars = 6000  # Limit transcript length to avoid token limits
        truncated_transcript = transcript[:max_chars] + ("..." if len(transcript) > max_chars else "")
        
        # Adjust temperature based on creativity
        temperature = 0.6 + (creativity * 0.4)  # Scale 0.6-1.0
        
        # Format prompt
        prompt = f"Channel: {channel_name}\nTranscript: {truncated_transcript}"
        
        # Add clip embedding info (our model was trained with this format)
        if clip_embedding and len(clip_embedding) > 0:
            # Just a signal that we have visual info, we don't need to include the actual embeddings
            prompt += "\nHas thumbnail: yes"
        else:
            prompt += "\nHas thumbnail: no"
            
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate titles
        logger.info(f"Generating {num_titles} titles with temperature={temperature}")
        
        # Set decoder start token ID to the pad token (following T5 convention)
        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=100,
            num_return_sequences=num_titles,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_beams=5,
            early_stopping=True,
            decoder_start_token_id=decoder_start_token_id
        )
        
        # Decode the outputs
        titles = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Clean up titles if needed
        clean_titles = []
        for title in titles:
            # Remove any common prefixes like "Title: " if present
            if ":" in title and title.split(":")[0].strip().lower() in ["title", "youtube title"]:
                clean_title = ":".join(title.split(":")[1:]).strip()
            else:
                clean_title = title.strip()
            
            clean_titles.append(clean_title)
        
        # Check if we have valid titles or use fallback
        if not any(clean_titles) or all(len(title) < 5 for title in clean_titles):
            logger.warning("Model generated empty or too short titles, using fallback generation")
            return generate_fallback_titles(transcript, channel_name, num_titles)
            
        return clean_titles
        
    except Exception as e:
        logger.error(f"Error generating titles: {str(e)}")
        # Use fallback in case of errors
        return generate_fallback_titles(transcript, channel_name, num_titles)

def generate_fallback_titles(transcript: str, channel_name: str, num_titles: int = 3) -> List[str]:
    """Generate simple titles from transcript when model generation fails"""
    logger.info("Using fallback title generation")
    
    # Get first few sentences (up to 200 chars)
    first_part = transcript[:200]
    
    # Extract potential key phrases using simple regex
    import re
    sentences = re.split(r'[.!?]', first_part)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [f"{channel_name} - Amazing Video", f"You Won't Believe What {channel_name} Did", f"Watch This {channel_name} Video Now"]
    
    # Generate title formats
    formats = [
        "{} - Must Watch Video",
        "You Won't Believe What Happens in This {} Video",
        "{} - Incredible Moments",
        "The Best of {} - Watch Now",
        "How {} Will Change Your Life",
    ]
    
    titles = []
    
    # First title: use first sentence if it's not too long
    first_sentence = sentences[0]
    if len(first_sentence) > 50:
        first_sentence = first_sentence[:47] + "..."
    titles.append(first_sentence)
    
    # Second title: combine channel name with first sentence fragment
    if len(channel_name) + len(first_sentence) > 45:
        fragment = first_sentence[:40-len(channel_name)] + "..."
        titles.append(f"{channel_name}: {fragment}")
    else:
        titles.append(f"{channel_name}: {first_sentence}")
    
    # Additional titles: use templates
    import random
    while len(titles) < num_titles:
        if sentences:
            sentence = random.choice(sentences)
            title_format = random.choice(formats)
            titles.append(title_format.format(sentence[:30] if len(sentence) > 30 else sentence))
        else:
            titles.append(f"{channel_name} - Amazing Content {len(titles) + 1}")
    
    return titles[:num_titles] 