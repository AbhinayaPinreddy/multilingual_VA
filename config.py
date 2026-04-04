import os
from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
TOKEN_SERVER_URL = os.getenv("TOKEN_SERVER_URL", "http://127.0.0.1:8000/get-token")
LIVEKIT_ROOM = os.getenv("LIVEKIT_ROOM", "voice-room")
# Agent identity must match token server defaults if you do not pass query params.
AGENT_IDENTITY = os.getenv("AGENT_IDENTITY", "agent1")

# Latency tuning (env overrides)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "64"))

# Qdrant local store (relative to project root)
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(_ROOT, "qdrant_data"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "products_v1")

# If Whisper language_probability is below this, we try langdetect on the transcript.
LANG_CONFIDENCE_MIN = float(os.getenv("LANG_CONFIDENCE_MIN", "0.55"))

# Reconnect backoff after LiveKit disconnect (seconds)
LIVEKIT_RECONNECT_DELAY = float(os.getenv("LIVEKIT_RECONNECT_DELAY", "2.0"))
