from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import base64
import os

# Initialize Vertex AI
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT", "project-1af0f617-fd7b-4ca6-b43"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

async def generate_anime_image(prompt: str, aspect_ratio: str = "1:1") -> dict:
    """
    Generates an anime-style image from a text prompt using Imagen on Vertex AI.
    
    Args:
        prompt: Text description of the anime image to generate.
        aspect_ratio: Image aspect ratio. Options: "1:1", "9:16", "16:9", "3:4", "4:3".
    
    Returns:
        A dict with status, base64-encoded image, and prompt used.
    """
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

        enhanced_prompt = (
            f"anime style illustration, {prompt}, "
            "vibrant colors, detailed linework, Studio Ghibli quality, "
            "cinematic lighting, high resolution"
        )

        response = model.generate_images(
            prompt=enhanced_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            negative_prompt="low quality, blurry, realistic photography, 3D render, nsfw",
            add_watermark=False,
        )

        image = response.images[0]
        image_bytes = image._image_bytes  # raw bytes
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        return {
            "status": "success",
            "image_base64": image_b64,
            "media_type": "image/png",
            "prompt_used": enhanced_prompt,
            "aspect_ratio": aspect_ratio,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Image generation failed: {str(e)}",
        }


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='An anime image generator that creates anime-style artwork from text descriptions using Imagen on Vertex AI.',
    instruction="""
        You are an Anime Image Generator powered by Google Imagen. Help users create 
        stunning anime-style images from their text descriptions.

        When a user describes a scene, character, or concept:
        1. Identify the ideal aspect_ratio:
           - "1:1"  → profile pics, icons, square art
           - "9:16" → phone wallpapers, portrait characters
           - "16:9" → landscape scenes, cinematic shots
           - "3:4"  → character portraits
           - "4:3"  → wide scenes, group shots
        2. Refine their description into a rich, evocative anime art prompt.
           Add details like: art style (e.g., shonen, shoujo, mecha, isekai),
           mood (dramatic, serene, adventurous), lighting (golden hour, neon night),
           and composition (close-up, wide shot, bird's eye).
        3. Call generate_anime_image with the refined prompt and chosen aspect_ratio.
        4. Report back: confirm success and describe what was generated,
           or explain the error and suggest prompt adjustments if it failed.
    """,
    tools=[FunctionTool(func=generate_anime_image)],
)