# Audio transcription utilities

import os
import logging
from typing import Dict, Any, Optional
import whisper

# Configure logging
logger = logging.getLogger(__name__)

# Initialize whisper model (lazy-loaded)
_whisper_model = None

def get_whisper_model():
    # Load the Whisper model once and reuse
    global _whisper_model
    
    if _whisper_model is None:
        logger.info("Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    
    return _whisper_model

def transcribe_audio(audio_file: str) -> str:
    # Transcribe audio file using Whisper
    # Returns full transcript text
    
    if not audio_file or not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return ""
    
    try:
        # Get model and transcribe
        model = get_whisper_model()
        logger.info(f"Transcribing audio file: {audio_file}")
        
        # Run transcription
        result = model.transcribe(audio_file)
        
        # Extract text
        transcript = result.get("text", "")
        
        if not transcript:
            logger.warning("Transcription returned empty text")
        else:
            logger.info(f"Transcription successful: {len(transcript)} characters")
        
        return transcript
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return "" 