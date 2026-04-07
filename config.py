import os
from dotenv import load_dotenv

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Token server URL for frontend to fetch tokens
TOKEN_SERVER_URL = os.getenv("TOKEN_SERVER_URL", "http://127.0.0.1:8000/get-token")
LIVEKIT_ROOM = os.getenv("LIVEKIT_ROOM", "voice-room")
AGENT_IDENTITY = os.getenv("AGENT_IDENTITY", "agent1")

# All AI features use these keys:
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
