import base64
import io
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def decode_base64_image(base64_string: str) -> Optional[Image.Image]:
    """
    Decode a base64 string to a PIL Image object.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object or None if decoding fails
    """
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        return image
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL Image to base64 string.
    
    Args:
        image: PIL Image object
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')

def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    Resize image to fit within max_size while maintaining aspect ratio.
    
    Args:
        image: PIL Image object
        max_size: Maximum width and height
        
    Returns:
        Resized PIL Image
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def analyze_image_content(image: Image.Image) -> str:
    """
    Generate a basic description of image content for story generation.
    This is a placeholder - in production, you might use a vision model.
    
    Args:
        image: PIL Image object
        
    Returns:
        Description of image content
    """
    # Get basic image properties
    width, height = image.size
    mode = image.mode
    
    # This is a simplified analysis - in production, you'd use a vision model
    description = f"An image ({width}x{height}, {mode} mode)"
    
    # You could enhance this with:
    # - Color analysis
    # - Edge detection
    # - Object detection using a vision model
    
    return description 