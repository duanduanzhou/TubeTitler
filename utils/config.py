#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import logging
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    
    _instance = None
    
    def __new__(cls, config_path=None):
        # Create singleton instance
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path=None):
        # Load configuration from YAML
        if config_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config.yaml")
        
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                logger.debug(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "api_keys": {
                "youtube": None
            },
            "paths": {
                "raw_data": "data/raw",
                "processed_data": "data/processed",
                "thumbnails": "data/raw/thumbnails",
                "frames": "data/raw/frames",
                "transcripts": "data/raw/transcripts",
                "embeddings": "data/processed/embeddings",
            },
            "models": {
                "whisper": {
                    "model_size": "base",
                    "language": "en"
                },
                "clip": {
                    "model_name": "ViT-B/32"
                },
                "title_generator": {
                    "model_name": "t5-base",
                    "checkpoint_dir": "models/checkpoints/title_generator"
                }
            },
            "processing": {
                "max_frames_per_video": 3,
                "max_transcript_length": 512,
                "max_title_length": 64
            },
            "training": {
                "batch_size": 8,
                "learning_rate": 3.0e-5,
                "epochs": 5,
                "seed": 42
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        # Get a config value
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_path(self, path_key: str) -> str:
        # Get a filesystem path from configuration
        path = self.get(f"paths.{path_key}")
        
        # If not found, use default paths
        if path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            defaults = {
                "raw_data": os.path.join(root_dir, "data", "raw"),
                "processed_data": os.path.join(root_dir, "data", "processed"),
                "thumbnails": os.path.join(root_dir, "data", "raw", "thumbnails"),
                "frames": os.path.join(root_dir, "data", "raw", "frames"),
                "transcripts": os.path.join(root_dir, "data", "raw", "transcripts"),
                "embeddings": os.path.join(root_dir, "data", "processed", "embeddings")
            }
            path = defaults.get(path_key)
            
            if path is None:
                logger.warning(f"Unknown path key: {path_key}, using as-is")
                path = path_key
        
        # Create directory if it doesn't exist
        if os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        return path
    
    def set(self, key_path: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the innermost dict
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def save(self, config_path: Optional[str] = None):
        """Save configuration to a YAML file"""
        if config_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(root_dir, "config.yaml")
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
                logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {str(e)}")

def get_config(config_path: Optional[str] = None) -> Config:
    # Get configuration singleton instance
    return Config(config_path) 