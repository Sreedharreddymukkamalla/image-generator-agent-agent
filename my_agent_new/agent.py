from google.adk.agents.llm_agent import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

root_agent = Agent(
    model='gemini-2.5-flash',
    name='image_generator_agent',
    description='Google image generator',
    instruction="""
        Generate a image based on User input
    """,
    tools=[
        McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url="https://aeo-mcp-server.amdal-dev.workers.dev/mcp",
            ),
        )
    ],
)
