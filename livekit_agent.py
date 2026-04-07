import json
import os

from dotenv import load_dotenv

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import groq, silero, cartesia

load_dotenv()

# Load product catalog once at startup
_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_ROOT, "products.json")) as f:
    PRODUCTS = json.load(f)

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and extremely fast multilingual voice shopping assistant. "
    "You help users find clothing and accessories from a product catalog. "
    "\n\nCRITICAL RULES:\n"
    "1. NEVER use your own knowledge to answer questions about products, prices, or availability. "
    "Use `get_catalog_overview` for broad questions like 'what do you have', 'what categories', 'show me products'. "
    "Use `search_products` for specific product, category, or color searches. "
    "ONLY respond based on what the tools return. If nothing is found, say so honestly.\n"
    "2. LANGUAGE: Always detect and respond in the same language the user is speaking. "
    "Naturally mirror their tone and language style — do not force a specific language. "
    "Always keep product names, categories, prices, and technical terms in English regardless of the user's language. "
    "If the user mixes languages (e.g. speaking casually with some English words), match that same casual mixed style. "
    "Never use overly formal or pure native script — keep it conversational and natural.\n"
    "3. KEEP RESPONSES SHORT AND CONVERSATIONAL — you are a voice assistant.\n"
    "4. EVERY response MUST end with ONE relevant follow-up question "
    "(e.g. about color, size, budget, occasion, etc.)."
)


@function_tool
async def search_products(
    context: RunContext,
    query: str,
) -> str:
    """Search for products by name, category, color, or price range.
    Call this whenever the user asks anything about products, prices, or availability.

    Args:
        query: The search keyword — can be a product name, category (ethnic, western, formal,
               casual, sports, winter, kids), or color (red, blue, black, etc.)
    """
    q = query.strip().lower()
    results = []
    for p in PRODUCTS:
        if (
            q in p["name"].lower()
            or q in p["category"].lower()
            or any(q in c.lower() for c in p.get("colors", []))
        ):
            results.append(p)

    if not results:
        return json.dumps({"found": False, "message": f"No products found for '{query}'"})

    top = results[:5]
    return json.dumps({"found": True, "count": len(results), "showing": len(top), "products": top})


@function_tool
async def get_catalog_overview(context: RunContext) -> str:
    """Get a summary of all product categories and types available in the store.
    Call this when the user asks broad questions like 'what do you have', 'what kind of products',
    'show me your catalog', or 'what categories do you sell'.
    """
    categories: dict[str, list[str]] = {}
    for p in PRODUCTS:
        cat = p["category"]
        name = p["name"]
        if cat not in categories:
            categories[cat] = []
        if name not in categories[cat]:
            categories[cat].append(name)

    summary = [
        {"category": cat, "products": names}
        for cat, names in categories.items()
    ]
    return json.dumps({"total_products": len(PRODUCTS), "categories": summary})


async def entrypoint(ctx: JobContext):
    # Auto-subscribe to user's audio only (no video needed)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Create the agent with the new v1.x API
    agent = Agent(
        instructions=SYSTEM_PROMPT,
        stt=groq.STT(model="whisper-large-v3-turbo"),
        llm=groq.LLM(model="openai/gpt-oss-20b"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        tools=[search_products, get_catalog_overview],
    )

    # Create a session and start it
    session = AgentSession()
    await session.start(agent, room=ctx.room)

    # Greet the user once the session is fully running
    await session.say("Hi! Welcome! How can I help you today?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )
