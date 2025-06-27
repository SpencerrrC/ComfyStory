# ComfyStory

A web application that generates storyboards using AI-powered text generation (Google Gemini) and image generation (ComfyUI).

## Features

- **Text-to-Storyboard**: Generate storyboard scenes from initial text input
- **Image-to-Storyboard**: Generate storyboard scenes from an initial image
- **Customizable Randomness**: Control the creativity level of story generation
- **Multiple Scenes**: Generate 1-10 scenes per storyboard
- **Real-time Generation**: Watch as scenes are generated with ComfyUI

## Backend Setup

### Prerequisites

1. **Python 3.8+** installed on your system
2. **ComfyUI** running locally (see [ComfyUI Setup](#comfyui-setup))
3. **Google Gemini API Key** (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ComfyStory
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   COMFYUI_BASE_URL=http://localhost:8188
   ```

### Running the Backend

1. **Start the FastAPI server**:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## ComfyUI Setup

1. **Install ComfyUI** (if not already installed):
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI
   cd ComfyUI
   pip install -r requirements.txt
   ```

2. **Download a model** (e.g., Stable Diffusion v1.5):
   ```bash
   mkdir models/checkpoints
   # Download v1-5-pruned.ckpt to models/checkpoints/
   ```

3. **Start ComfyUI**:
   ```bash
   python main.py --listen 0.0.0.0 --port 8188
   ```

4. **Verify ComfyUI is running**:
   - Web UI: http://localhost:8188
   - API: http://localhost:8188/history

## API Endpoints

### POST /generate-storyboard

Generate a complete storyboard with images.

**Request Body**:
```json
{
  "initial_input": "A young wizard discovers a mysterious book in an ancient library",
  "randomness_level": 0.7,
  "num_scenes": 5
}
```

**Response**:
```json
{
  "storyboard_scenes": [
    {
      "scene_number": 1,
      "text_description": "A young wizard with flowing robes stands in a dimly lit library, reaching for a dusty tome on a high shelf.",
      "image_url": "http://localhost:8188/view?filename=ComfyUI_00001_.png&subfolder=&type="
    }
  ]
}
```

### GET /health

Health check endpoint.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `COMFYUI_BASE_URL` | ComfyUI server URL | `http://localhost:8188` |
| `HOST` | FastAPI host | `0.0.0.0` |
| `PORT` | FastAPI port | `8000` |
| `DEBUG` | Enable debug mode | `False` |

## Troubleshooting

### Common Issues

1. **"GEMINI_API_KEY environment variable is required"**
   - Make sure you've created a `.env` file with your API key

2. **"Failed to queue prompt to ComfyUI"**
   - Ensure ComfyUI is running on the correct port
   - Check that the model file exists in ComfyUI's models directory

3. **"ComfyUI generation timed out"**
   - Increase the timeout in the code or check ComfyUI's performance
   - Ensure you have sufficient GPU memory for image generation

### Logs

The application logs important events. Check the console output for detailed error messages.

## Development

### Project Structure

```
ComfyStory/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create from env.example)
├── env.example          # Example environment file
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

### Adding New Features

1. **New API endpoints**: Add them to `main.py`
2. **New dependencies**: Add to `requirements.txt`
3. **Environment variables**: Add to `env.example` and update the code

## License

[Add your license here]
