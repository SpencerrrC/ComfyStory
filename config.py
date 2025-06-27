"""
Configuration settings for ComfyStory
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Keys
    gemini_api_key: str
    
    # ComfyUI Settings
    comfyui_base_url: str = "http://localhost:8188"
    comfyui_timeout: int = 300
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Application Settings
    max_scenes: int = 10
    min_scenes: int = 1
    default_image_width: int = 768
    default_image_height: int = 512
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    
    # Paths
    output_dir: Path = Path("outputs")
    
    # Generation Settings
    default_negative_prompt: str = "low quality, bad quality, blurry, pixelated, distorted"
    default_sampler: str = "euler"
    default_steps: int = 20
    default_cfg_scale: float = 8.0
    
    # Model Settings
    default_model: str = "v1-5-pruned.ckpt"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()

# Ensure output directory exists
settings.output_dir.mkdir(exist_ok=True) 