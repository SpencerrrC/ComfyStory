"""
Test script for ComfyStory API
"""

import asyncio
import aiohttp
import json
import base64
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

async def test_health_check():
    """Test the health check endpoint"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE_URL}/health") as response:
            if response.status == 200:
                data = await response.json()
                print("✅ Health check passed:", data)
            else:
                print("❌ Health check failed:", response.status)

async def test_text_to_storyboard():
    """Test text-based storyboard generation"""
    payload = {
        "initial_input": "A brave knight discovers a magical sword in an ancient castle",
        "randomness_level": 0.7,
        "num_scenes": 3,
        "input_type": "text"
    }
    
    print("\n🎬 Testing text-to-storyboard generation...")
    print(f"Input: {payload['initial_input']}")
    print(f"Scenes: {payload['num_scenes']}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{API_BASE_URL}/generate-storyboard", json=payload) as response:
            if response.status == 200:
                data = await response.json()
                print("\n✅ Storyboard generated successfully!")
                for scene in data["storyboard_scenes"]:
                    print(f"\nScene {scene['scene_number']}:")
                    print(f"  Description: {scene['text_description']}")
                    print(f"  Image URL: {scene['image_url']}")
            else:
                print(f"❌ Generation failed: {response.status}")
                print(await response.text())

async def test_image_upload():
    """Test image upload endpoint"""
    # Create a simple test image
    from PIL import Image
    import io
    
    # Create a 100x100 red image
    img = Image.new('RGB', (100, 100), color='red')
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    print("\n📤 Testing image upload...")
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', img_buffer, filename='test.png', content_type='image/png')
        
        async with session.post(f"{API_BASE_URL}/upload-image", data=data) as response:
            if response.status == 200:
                result = await response.json()
                print("✅ Image uploaded successfully!")
                print(f"  Filename: {result['filename']}")
                print(f"  Size: {result['size']}")
                print(f"  Base64 length: {len(result['base64'])}")
                return result['base64']
            else:
                print(f"❌ Upload failed: {response.status}")
                return None

async def test_websocket_connection():
    """Test WebSocket connection"""
    print("\n🔌 Testing WebSocket connection...")
    
    try:
        import websockets
        ws_url = f"ws://localhost:8000/ws"
        
        async with websockets.connect(ws_url) as websocket:
            # Send a test message
            await websocket.send("ping")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            
            if response == "ping":
                print("✅ WebSocket connection successful!")
            else:
                print(f"❌ Unexpected response: {response}")
                
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")

async def main():
    """Run all tests"""
    print("🧪 Starting ComfyStory API tests...\n")
    
    # Test health check
    await test_health_check()
    
    # Test image upload
    base64_image = await test_image_upload()
    
    # Test WebSocket
    await test_websocket_connection()
    
    # Test text-to-storyboard
    await test_text_to_storyboard()
    
    # Test image-to-storyboard if upload was successful
    if base64_image:
        print("\n🎬 Testing image-to-storyboard generation...")
        payload = {
            "initial_input": base64_image,
            "randomness_level": 0.5,
            "num_scenes": 2,
            "input_type": "image"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{API_BASE_URL}/generate-storyboard", json=payload) as response:
                if response.status == 200:
                    print("✅ Image-based storyboard generation successful!")
                else:
                    print(f"❌ Generation failed: {response.status}")
    
    print("\n✨ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 