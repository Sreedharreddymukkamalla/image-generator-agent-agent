import os
import base64
import asyncio
import vertexai
from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import storage

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT", "project-1af0f617-fd7b-4ca6-b43"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "project-1af0f617-fd7b-4ca6-b43-imagescoring-bucket")  # add this env var to Cloud Run

def _upload_to_gcs(image_bytes: bytes, filename: str) -> str:
    """Upload image bytes to GCS and return a public URL."""
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(f"anime-images/{filename}")
    blob.upload_from_string(image_bytes, content_type="image/png")
    return f"https://storage.googleapis.com/{GCS_BUCKET}/anime-images/{filename}"

async def generate_anime_image(prompt: str, aspect_ratio: str = "1:1") -> dict:
    """
    Generates an anime-style image from a text prompt using Imagen on Vertex AI.
    Returns a GCS URL instead of raw base64 to avoid token limits.

    Args:
        prompt: Text description of the anime image to generate.
        aspect_ratio: "1:1", "9:16", "16:9", "3:4", or "4:3".
    """
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

        enhanced_prompt = (
            f"anime style illustration, {prompt}, "
            "vibrant colors, detailed linework, Studio Ghibli quality, "
            "cinematic lighting, high resolution"
        )

        response = await asyncio.to_thread(
            model.generate_images,
            prompt=enhanced_prompt,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            negative_prompt="low quality, blurry, realistic photography, 3D render, nsfw",
            add_watermark=False,
        )

        image_bytes = response.images[0]._image_bytes
        import uuid
        filename = f"{uuid.uuid4().hex}.png"
        url = await asyncio.to_thread(_upload_to_gcs, image_bytes, filename)

        return {
            "status": "success",
            "image_url": url,           # ← small string, safe for Gemini
            "prompt_used": prompt,
            "aspect_ratio": aspect_ratio,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }


root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='An anime image generator using Imagen on Vertex AI.',
    instruction="""
        You are an Anime Image Generator powered by Google Imagen.

        When a user describes a scene, character, or concept:
        1. Pick the ideal aspect_ratio: "1:1" for square, "9:16" for portrait,
           "16:9" for landscape, "3:4" for character, "4:3" for group/wide.
        2. Enrich the prompt with anime details: art style, mood, lighting, composition.
        3. Call generate_anime_image with the refined prompt and aspect_ratio.
        4. On success, respond with the image_url so the user can view their image.
           On error, explain and suggest prompt changes.
    """,
    tools=[FunctionTool(func=generate_anime_image)],
)