import os
import asyncio
import base64
import json
import aiohttp
import websockets
from typing import List, Optional, Union, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from pathlib import Path
import uuid
from datetime import datetime

# Import custom modules
from utils import decode_base64_image, image_to_base64, resize_image, analyze_image_content
from comfyui_workflows import ComfyUIWorkflows

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

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Pydantic models
class StoryboardRequest(BaseModel):
    initial_input: str = Field(..., description="Text prompt or base64 image data")
    randomness_level: float = Field(..., ge=0.0, le=1.0, description="Randomness level between 0.0 and 1.0")
    num_scenes: int = Field(..., ge=1, le=10, description="Number of scenes to generate (1-10)")
    input_type: str = Field("text", description="Type of input: 'text' or 'image'")

class StoryboardScene(BaseModel):
    scene_number: int
    text_description: str
    image_url: str

class StoryboardResponse(BaseModel):
    storyboard_scenes: List[StoryboardScene]



class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.client_id = str(uuid.uuid4())
    
    async def queue_prompt(self, workflow: dict, client_id: Optional[str] = None) -> str:
        """Queue a prompt to ComfyUI and return the prompt ID"""
        if client_id is None:
            client_id = self.client_id
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow, "client_id": client_id}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ComfyUI error: {error_text}")
                    raise HTTPException(status_code=500, detail="Failed to queue prompt to ComfyUI")
                
                result = await response.json()
                return result["prompt_id"]
    
    async def get_prompt_status(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a queued prompt"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                if response.status == 200:
                    history = await response.json()
                    return history.get(prompt_id)
                return None
    
    async def upload_image(self, image_data: bytes, filename: str) -> str:
        """Upload an image to ComfyUI and return the filename"""
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('image', image_data, filename=filename, content_type='image/png')
            
            async with session.post(f"{self.base_url}/upload/image", data=data) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to upload image to ComfyUI")
                
                result = await response.json()
                return result["name"]
    
    async def get_image(self, filename: str, subfolder: str = "", image_type: str = "output") -> bytes:
        """Get an image from ComfyUI"""
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/view", params=params) as response:
                if response.status != 200:
                    raise HTTPException(status_code=404, detail="Image not found")
                
                return await response.read()
    
    async def wait_for_completion_with_ws(self, prompt_id: str, timeout: int = 300, 
                                         progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Wait for prompt completion using WebSocket for real-time updates"""
        ws_url = f"ws://{self.base_url.replace('http://', '').replace('https://', '')}/ws?clientId={self.client_id}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                start_time = asyncio.get_event_loop().time()
                
                while asyncio.get_event_loop().time() - start_time < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Handle different message types
                        if data.get("type") == "executing" and data.get("data", {}).get("prompt_id") == prompt_id:
                            if data["data"].get("node") is None:
                                # Execution completed
                                result = await self.get_prompt_status(prompt_id)
                                if result:
                                    return result
                        
                        elif data.get("type") == "progress" and progress_callback:
                            await progress_callback(data.get("data", {}))
                        
                        elif data.get("type") == "execution_error":
                            logger.error(f"Execution error: {data}")
                            raise HTTPException(status_code=500, detail="ComfyUI execution error")
                    
                    except asyncio.TimeoutError:
                        # Check status directly if no WebSocket message
                        result = await self.get_prompt_status(prompt_id)
                        if result and result.get("outputs"):
                            return result
                
                raise HTTPException(status_code=408, detail="ComfyUI generation timed out")
        
        except websockets.exceptions.WebSocketException as e:
            logger.warning(f"WebSocket connection failed: {e}. Falling back to polling.")
            return await self.wait_for_completion(prompt_id, timeout)
    
    async def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for prompt completion and return the result (polling fallback)"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.get_prompt_status(prompt_id)
            if result and result.get("outputs"):
                return result
            
            await asyncio.sleep(2)
        
        raise HTTPException(status_code=408, detail="ComfyUI generation timed out")

# Initialize ComfyUI client
comfyui_client = ComfyUIClient(COMFYUI_BASE_URL)

async def generate_scene_descriptions(initial_input: str, randomness_level: float, num_scenes: int, 
                                    input_type: str = "text", image_description: Optional[str] = None) -> List[str]:
    """Generate scene descriptions using Gemini API"""
    try:
        # Adjust creativity based on randomness level
        temperature = 0.7 + (randomness_level * 0.3)  # Range: 0.7 to 1.0
        
        if input_type == "image" and image_description:
            context = f"an image showing: {image_description}"
        else:
            context = f'"{initial_input}"'
        
        prompt = f"""
        Based on this initial input: {context}
        
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

async def generate_image_for_scene(scene_description: str, progress_callback: Optional[callable] = None) -> str:
    """Generate an image for a scene description using ComfyUI"""
    try:
        # Create workflow using the workflow manager
        workflow = ComfyUIWorkflows.text_to_image_workflow(
            prompt=scene_description,
            negative_prompt="low quality, bad quality, blurry, pixelated",
            width=768,
            height=512
        )
        
        # Queue the prompt
        prompt_id = await comfyui_client.queue_prompt(workflow)
        
        # Wait for completion with WebSocket support
        result = await comfyui_client.wait_for_completion_with_ws(prompt_id, progress_callback=progress_callback)
        
        # Extract the image filename
        if "9" in result.get("outputs", {}) and "images" in result["outputs"]["9"]:
            image_data = result["outputs"]["9"]["images"][0]
            filename = image_data["filename"]
            
            # Save a copy locally and return the URL
            image_bytes = await comfyui_client.get_image(filename)
            local_filename = f"{uuid.uuid4()}.png"
            local_path = OUTPUT_DIR / local_filename
            
            with open(local_path, "wb") as f:
                f.write(image_bytes)
            
            # Return the URL to access the image
            return f"/outputs/{local_filename}"
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
        image_description = None
        
        # Handle image input
        if request.input_type == "image":
            # Decode base64 image
            image = decode_base64_image(request.initial_input)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image data")
            
            # Analyze image content (in production, use a vision model)
            image_description = analyze_image_content(image)
            
            # Save the initial image
            initial_image_filename = f"initial_{uuid.uuid4()}.png"
            initial_image_path = OUTPUT_DIR / initial_image_filename
            image.save(initial_image_path)
        
        # Generate scene descriptions
        scene_descriptions = await generate_scene_descriptions(
            request.initial_input,
            request.randomness_level,
            request.num_scenes,
            request.input_type,
            image_description
        )
        
        # Generate images for each scene
        storyboard_scenes = []
        for i, description in enumerate(scene_descriptions, 1):
            # Broadcast progress via WebSocket
            await manager.broadcast(json.dumps({
                "type": "progress",
                "scene": i,
                "total": request.num_scenes,
                "status": "generating",
                "description": description
            }))
            
            image_url = await generate_image_for_scene(description)
            storyboard_scenes.append(StoryboardScene(
                scene_number=i,
                text_description=description,
                image_url=image_url
            ))
            
            # Broadcast completion for this scene
            await manager.broadcast(json.dumps({
                "type": "scene_complete",
                "scene": i,
                "image_url": image_url
            }))
        
        return StoryboardResponse(storyboard_scenes=storyboard_scenes)
    
    except Exception as e:
        logger.error(f"Error generating storyboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await manager.send_personal_message(data, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and return its base64 representation"""
    try:
        # Read image data
        contents = await file.read()
        
        # Validate it's an image
        image = decode_base64_image(base64.b64encode(contents).decode())
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Resize if too large
        image = resize_image(image, max_size=(1024, 1024))
        
        # Convert to base64
        base64_string = image_to_base64(image)
        
        return {
            "base64": f"data:image/png;base64,{base64_string}",
            "filename": file.filename,
            "size": image.size
        }
    
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

@app.get("/storyboard/{storyboard_id}")
async def get_storyboard(storyboard_id: str):
    """Get a saved storyboard by ID"""
    # This is a placeholder - in production, you'd retrieve from a database
    storyboard_file = OUTPUT_DIR / f"storyboard_{storyboard_id}.json"
    
    if storyboard_file.exists():
        with open(storyboard_file, "r") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=404, detail="Storyboard not found")

@app.post("/save-storyboard")
async def save_storyboard(storyboard: StoryboardResponse):
    """Save a storyboard for later retrieval"""
    try:
        storyboard_id = str(uuid.uuid4())
        storyboard_file = OUTPUT_DIR / f"storyboard_{storyboard_id}.json"
        
        # Save storyboard data
        with open(storyboard_file, "w") as f:
            json.dump(storyboard.dict(), f, indent=2)
        
        return {"storyboard_id": storyboard_id}
    
    except Exception as e:
        logger.error(f"Error saving storyboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to save storyboard")

# Mount static files directory for serving generated images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    ) 