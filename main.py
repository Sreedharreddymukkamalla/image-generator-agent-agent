import os
import sys
import uvicorn
import vertexai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from google.adk.cli.fast_api import get_fast_api_app

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(AGENT_DIR, "my_agent_new"))
from executor import AnimeAgentExecutor

# Initialize Vertex AI
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT", "project-1af0f617-fd7b-4ca6-b43"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

SESSION_SERVICE_URI = "sqlite+aiosqlite:///./sessions.db"
ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "*"]
SERVE_WEB_INTERFACE = True

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=SESSION_SERVICE_URI,
    allow_origins=ALLOWED_ORIGINS,
    web=SERVE_WEB_INTERFACE,
)

AGENT_URL = os.getenv(
    "AGENT_PUBLIC_URL",
    "https://image-generator-agent-475756125529.us-central1.run.app/anime/",
)

agent_card = AgentCard(
    name="Anime Image Generator Agent",
    description="Generates anime-style images from text descriptions using Imagen on Vertex AI.",
    url=AGENT_URL,
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text", "image"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="generate_anime_image",
            name="Generate anime image",
            description="Creates anime-style artwork from a text description or prompt.",
            inputModes=["text"],
            outputModes=["text", "image"],
            tags=["anime", "image generation", "art"],
        )
    ],
)

a2a_app = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=DefaultRequestHandler(
        agent_executor=AnimeAgentExecutor(),
        task_store=None,
    ),
)

app.mount("/anime", a2a_app.build())

@app.get("/.well-known/agent.json", include_in_schema=False)
def redirect_agent_json():
    return RedirectResponse(url="/anime/.well-known/agent.json")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
