import os
import sys
import json
import logging
import tempfile
from typing import Dict, Any, Optional, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size: str = None):
        # Initialize transcriber with configuration
        config = get_config()
        self.model_size = model_size or config.get("models.whisper.model_size", "base")
        self.language = config.get("models.whisper.language", "en")
        self.transcript_dir = config.get_path("transcripts")
        self.model = None
        
        logger.info(f"Initializing Whisper with model size: {self.model_size}")
    
    def load_model(self):
        # Load Whisper model if not already loaded
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Import whisper
            try:
                import whisper
                self.model = whisper.load_model(self.model_size)
                logger.info("Whisper model loaded successfully")
            except ImportError:
                raise ImportError(
                    "Could not import whisper. Make sure you have installed 'openai-whisper'"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load Whisper model: {str(e)}")
    
    def transcribe(self, audio_path: str, save: bool = True) -> Dict[str, Any]:
        """Transcribe audio file using Whisper ASR
        @ Returns: Dictionary with transcription results"""
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            return {"success": False, "error": error_msg}
        
        try:
            self.load_model()
            
            # Extract video_id from filename
            basename = os.path.basename(audio_path)
            video_id = os.path.splitext(basename)[0]
            
            logger.info(f"Transcribing audio for video {video_id}")
            result = self.model.transcribe(
                audio_path, 
                language=self.language,
                verbose=False,
                fp16=False
            )
            
            # Format the result
            transcript_data = {
                "video_id": video_id,
                "text": result["text"],
                "segments": result["segments"],
                "language": result.get("language", self.language),
                "success": True
            }
            
            # Save transcript if requested
            if save:
                self._save_transcript(video_id, transcript_data)
            
            logger.info(f"Transcription completed for {video_id}")
            return transcript_data
            
        except Exception as e:
            return {"success": False, "error": f"Error during transcription: {str(e)}"}
    
    def _save_transcript(self, video_id: str, transcript_data: Dict[str, Any]):
        # Save transcript to files
        os.makedirs(self.transcript_dir, exist_ok=True)
        
        # Save full transcript with segments
        full_path = os.path.join(self.transcript_dir, f"{video_id}.json")
        with open(full_path, 'w') as f:
            json.dump(transcript_data, f, indent=2)
        
        # Save just the text transcript
        text_path = os.path.join(self.transcript_dir, f"{video_id}.txt")
        with open(text_path, 'w') as f:
            f.write(transcript_data["text"])
            
        logger.info(f"Transcript saved to {full_path} and {text_path}")
    
    def batch_transcribe(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files
        @ Returns: List of transcription results"""
        self.load_model()  # Load model once for all files
        results = []
        
        for audio_path in audio_files:
            result = self.transcribe(audio_path)
            results.append(result)
            
        return results
    
    def segment_transcript(self, transcript_data: Dict[str, Any], max_length: int = 500) -> List[str]:
        """Segment a transcript into smaller chunks
        @ Returns: List of transcript segments"""
        segments = []
        current_segment = ""
        
        for seg in transcript_data.get("segments", []):
            text = seg.get("text", "").strip()
            
            if len(current_segment) + len(text) + 1 <= max_length:
                current_segment += " " + text if current_segment else text
            else:
                segments.append(current_segment)
                current_segment = text
                
        if current_segment:
            segments.append(current_segment)
            
        return segments

def transcribe(video_id: str, audio_path: str = None, model_size: str = None) -> Dict[str, Any]:
    """Convenience function to transcribe a single video
    @ Returns: Dictionary with transcription results"""
    config = get_config()
    
    if audio_path is None:
        raw_dir = config.get_path("raw_data")
        audio_path = os.path.join(raw_dir, f"{video_id}.mp3")
    
    transcriber = AudioTranscriber(model_size)
    return transcriber.transcribe(audio_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio files with Whisper")
    parser.add_argument("--video_id", type=str, help="YouTube video ID")
    parser.add_argument("--audio_path", type=str, help="Path to audio file (overrides video_id)")
    parser.add_argument("--model_size", type=str, choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    
    args = parser.parse_args()
    
    if args.audio_path:
        result = transcribe(None, args.audio_path, args.model_size)
    elif args.video_id:
        result = transcribe(args.video_id, model_size=args.model_size)
    else:
        parser.error("Either --video_id or --audio_path is required")
    
    if result.get("success", False):
        print(f"Transcription successful! First 150 characters:")
        print(result["text"][:150] + "...")
    else:
        print(f"Transcription failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)