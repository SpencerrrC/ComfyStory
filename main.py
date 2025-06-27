import os
import asyncio
import base64
import json
import aiohttp
import websockets
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ComfyStory API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# ComfyUI Configuration
COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "http://localhost:8188")

# Pydantic models
class StoryboardRequest(BaseModel):
    initial_input: str = Field(..., description="Text prompt or base64 image data")
    randomness_level: float = Field(..., ge=0.0, le=1.0, description="Randomness level between 0.0 and 1.0")
    num_scenes: int = Field(..., ge=1, le=10, description="Number of scenes to generate (1-10)")

class StoryboardScene(BaseModel):
    scene_number: int
    text_description: str
    image_url: str

class StoryboardResponse(BaseModel):
    storyboard_scenes: List[StoryboardScene]

# ComfyUI workflow template for text-to-image generation
COMFYUI_WORKFLOW = {
    "3": {
        "inputs": {
            "seed": 156680208700286,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
    },
    "4": {
        "inputs": {
            "ckpt_name": "v1-5-pruned.ckpt"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "5": {
        "inputs": {
            "width": 512,
            "height": 512,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {
            "text": "placeholder_text",
            "clip": ["4", 1]
        },
        "class_type": "CLIPTextEncode"
    },
    "7": {
        "inputs": {
            "text": "low quality, bad quality, sketches",
            "clip": ["4", 1]
        },
        "class_type": "CLIPTextEncode"
    },
    "8": {
        "inputs": {
            "samples": ["3", 0],
            "vae": ["4", 2]
        },
        "class_type": "VAEDecode"
    },
    "9": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["8", 0]
        },
        "class_type": "SaveImage"
    }
}

class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    async def queue_prompt(self, workflow: dict) -> str:
        """Queue a prompt to ComfyUI and return the prompt ID"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow}
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to queue prompt to ComfyUI")
                
                result = await response.json()
                return result["prompt_id"]
    
    async def get_prompt_status(self, prompt_id: str) -> dict:
        """Get the status of a queued prompt"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                if response.status == 200:
                    return await response.json()
                return None
    
    async def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """Wait for prompt completion and return the result"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.get_prompt_status(prompt_id)
            if result and result.get("outputs"):
                return result
            
            await asyncio.sleep(2)
        
        raise HTTPException(status_code=408, detail="ComfyUI generation timed out")

# Initialize ComfyUI client
comfyui_client = ComfyUIClient(COMFYUI_BASE_URL)

async def generate_scene_descriptions(initial_input: str, randomness_level: float, num_scenes: int) -> List[str]:
    """Generate scene descriptions using Gemini API"""
    try:
        # Adjust creativity based on randomness level
        temperature = 0.7 + (randomness_level * 0.3)  # Range: 0.7 to 1.0
        
        prompt = f"""
        Based on this initial input: "{initial_input}"
        
        Generate {num_scenes} scene descriptions for a storyboard. Each description should be:
        - 1-2 sentences long
        - Visually descriptive
        - Suitable for image generation
        - Connected to form a coherent story
        
        Randomness level: {randomness_level} (0=very predictable, 1=very creative)
        
        Return only the scene descriptions, one per line, without numbering or additional text.
        """
        
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1000
        ))
        
        # Parse the response into individual scene descriptions
        scenes = [scene.strip() for scene in response.text.strip().split('\n') if scene.strip()]
        
        # Ensure we have the requested number of scenes
        if len(scenes) < num_scenes:
            # Pad with additional scenes if needed
            while len(scenes) < num_scenes:
                scenes.append(f"Scene {len(scenes) + 1}: A continuation of the story")
        elif len(scenes) > num_scenes:
            scenes = scenes[:num_scenes]
        
        return scenes
    
    except Exception as e:
        logger.error(f"Error generating scene descriptions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate scene descriptions")

async def generate_image_for_scene(scene_description: str) -> str:
    """Generate an image for a scene description using ComfyUI"""
    try:
        # Create a copy of the workflow and update the prompt
        workflow = json.loads(json.dumps(COMFYUI_WORKFLOW))
        workflow["6"]["inputs"]["text"] = scene_description
        
        # Queue the prompt
        prompt_id = await comfyui_client.queue_prompt(workflow)
        
        # Wait for completion
        result = await comfyui_client.wait_for_completion(prompt_id)
        
        # Extract the image filename
        if "9" in result["outputs"] and "images" in result["outputs"]["9"]:
            image_data = result["outputs"]["9"]["images"][0]
            filename = image_data["filename"]
            
            # Return the URL to access the image
            return f"{COMFYUI_BASE_URL}/view?filename={filename}&subfolder=&type="
        else:
            raise HTTPException(status_code=500, detail="No image generated")
    
    except Exception as e:
        logger.error(f"Error generating image for scene: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate image")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ComfyStory API"}

@app.post("/generate-storyboard", response_model=StoryboardResponse)
async def generate_storyboard(request: StoryboardRequest):
    """Generate a complete storyboard with images"""
    try:
        # Generate scene descriptions
        scene_descriptions = await generate_scene_descriptions(
            request.initial_input,
            request.randomness_level,
            request.num_scenes
        )
        
        # Generate images for each scene
        storyboard_scenes = []
        for i, description in enumerate(scene_descriptions, 1):
            image_url = await generate_image_for_scene(description)
            storyboard_scenes.append(StoryboardScene(
                scene_number=i,
                text_description=description,
                image_url=image_url
            ))
        
        return StoryboardResponse(storyboard_scenes=storyboard_scenes)
    
    except Exception as e:
        logger.error(f"Error generating storyboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/view-image/{filename:path}")
async def view_image(filename: str):
    """Serve images from ComfyUI output directory"""
    try:
        # Construct the path to ComfyUI's output directory
        image_path = f"output/{filename}"
        
        if os.path.exists(image_path):
            return FileResponse(image_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve image")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    ) 