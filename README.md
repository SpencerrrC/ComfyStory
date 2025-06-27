# ComfyStory

## Persona
You are an expert full-stack developer with extensive experience in building AI-powered web applications and integrating with complex backend systems like ComfyUI. Your task is to generate the code for a web application that serves as a storyboard generator.

## Project Description

The application will allow users to:
1.  **Upload a first image or input initial text**: This serves as the starting point for the story.
2.  **Generate subsequent scenes**: Based on the initial input, Gemini (via an API) will generate textual descriptions for the next scenes of the story.
3.  **Control randomness**: Users can adjust a slider to control the "randomness" or creative deviation in the story generation by Gemini.
4.  **Specify number of scenes**: Users can specify how many scenes they want generated.
5.  **Visualize scenes with ComfyUI**: The generated text descriptions for each scene will be fed to a ComfyUI backend to generate corresponding images. These images will then be displayed in the web application as a storyboard.

## Optimal Tech Stack

Considering the need for quick development, good performance, and seamless integration with ComfyUI (which is Python-based and offers a robust API), the most optimal tech stack is:

**Backend:**
* **Python with FastAPI**: FastAPI is highly performant, easy to learn, and excellent for building APIs. Its async capabilities are well-suited for handling potentially long-running ComfyUI requests. It also provides automatic interactive API documentation (Swagger UI/ReDoc).
* **ComfyUI**: Running as a separate service, exposed via its API (HTTP and WebSocket). We will interact with it to queue prompts and retrieve generated images.
* **Google Gemini API**: For generating text descriptions of the story scenes.

**Frontend:**
* **React (with Next.js)**: React provides a component-based structure for building a dynamic and responsive UI. Next.js offers server-side rendering (SSR) or static site generation (SSG) for better performance and SEO (though not critical for this internal tool, it's good practice), and simplified API routing. Using `fetch` or `axios` for API calls.
* **Chakra UI (or similar component library)**: For rapid UI development with pre-built, accessible, and customizable components (sliders, image upload, text input, buttons, etc.).

**Deployment Considerations (for future thought, but keep in mind for architecture):**
* **Docker**: Containerize both the FastAPI backend and ComfyUI for easy deployment and scaling.
* **Cloud Platform (e.g., AWS, GCP, Azure, Modal, RunComfy)**: For hosting. ComfyUI can be resource-intensive, so a cloud GPU instance would be ideal for it.

## API Endpoints

The FastAPI backend should expose the following endpoints:

* `POST /generate-storyboard`:
    * **Request Body**:
        ```json
        {
            "initial_input": "string" | "base64_image_data", // Either text prompt or base64 image data
            "randomness_level": "float", // Value between 0.0 and 1.0
            "num_scenes": "integer"
        }
        ```
    * **Functionality**:
        1.  Take `initial_input` as context.
        2.  Call Gemini API with `initial_input`, `randomness_level`, and `num_scenes` to get an array of scene descriptions (text).
        3.  For each scene description, call the ComfyUI API to generate an image. This will involve:
            * Loading a predefined ComfyUI workflow JSON (you can assume a basic text-to-image workflow for now, which can be dynamically updated with the scene text).
            * Queueing the prompt to ComfyUI.
            * Polling the ComfyUI WebSocket or history API to get the generated image.
            * Handling potential delays and errors from ComfyUI.
        4.  Return an array of generated image URLs (or base64 data) and their corresponding text descriptions.
    * **Response Body**:
        ```json
        {
            "storyboard_scenes": [
                {
                    "scene_number": "integer",
                    "text_description": "string",
                    "image_url": "string" // URL to the generated image
                }
            ]
        }
        ```
* `GET /health`: Simple health check endpoint.

## Frontend Requirements

* A main page with:
    * An input field for initial text or an image upload component.
    * A slider for "randomness" (0-100%).
    * A numeric input for "number of scenes".
    * A "Generate Storyboard" button.
    * A loading indicator while generation is in progress.
    * A display area for the generated storyboard (series of images with their corresponding text descriptions).
* Error handling and user feedback.

## Implementation Details & Assumptions

* Assume ComfyUI is running and accessible at a specified URL (e.g., `http://localhost:8188`).
* Assume the Gemini API key is configured securely (e.g., via environment variables in the backend).
* For ComfyUI interaction, use its official API for queuing prompts (`/prompt`) and potentially the WebSocket (`/ws`) for real-time status updates or the `/history/{prompt_id}` endpoint to retrieve results.
* The ComfyUI workflow JSON for text-to-image generation should be provided as a template within the FastAPI application, where the prompt text can be dynamically inserted.
* The generated images from ComfyUI will need to be accessible by the frontend. This might involve the FastAPI server serving them, or ComfyUI itself serving them from its output directory if directly accessible. For simplicity, assume FastAPI will serve the images by fetching them from ComfyUI's output.

## Code Generation Task for Claude Opus 4

Generate the full code for this web application based on the described tech stack and requirements. This includes:

1.  **FastAPI Backend (`main.py` and any helper modules)**:
    * Setup of a FastAPI application.
    * Integration with Google Gemini API for story generation.
    * Functions for interacting with the ComfyUI API (sending prompts, polling for results, fetching images).
    * The `generate-storyboard` endpoint as described above.
    * A basic ComfyUI workflow JSON template (you can use a simple text-to-image workflow).
    * Basic image serving from the ComfyUI output directory.
2.  **React/Next.js Frontend**:
    * Project structure setup.
    * Main page components for user input (text/image upload, slider, number input).
    * Logic to call the FastAPI backend.
    * Components to display the storyboard.
    * Basic styling (using Chakra UI or simple CSS).
3.  **Instructions for running the application**:
    * How to set up and run the FastAPI backend.
    * How to set up and run ComfyUI (briefly, referencing its own documentation).
    * How to set up and run the Next.js frontend.
    * Any necessary environment variables.

Focus on clear, modular, and maintainable code. Provide comments where necessary to explain complex logic. Prioritize functionality over exhaustive error handling for the first pass, but include basic error messages. Assume `pip` and `npm`/`yarn` for package management.