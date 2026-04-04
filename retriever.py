import os
import re

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config

_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        os.makedirs(config.QDRANT_PATH, exist_ok=True)
        _client = QdrantClient(path=config.QDRANT_PATH)
    return _client


def extract_price(query: str):
    nums = re.findall(r"\d+", query)
    return int(nums[0]) if nums else None


def _payload_to_product(p: dict) -> dict:
    return {
        "name": p["name"],
        "price": p["price"],
        "description": p["description"],
        "category": p["category"],
        "colors": p["colors"],
    }


def retrieve(query: str):
    """Multilingual query; cosine search in Qdrant, then optional price filter."""
    client = _get_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        print(
            "WARN: Qdrant collection missing — run `python embedder.py` once. "
            "Returning empty RAG context."
        )
        return []

    top_k = min(config.RAG_TOP_K, 64)
    pool = min(max(12, top_k * 4), 64)

    qv = _model.encode([query], show_progress_bar=False)[0]

    hits = client.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=qv,
        limit=pool,
        with_payload=True,
    )

    ranked = sorted(hits, key=lambda h: h.score or 0.0, reverse=True)
    results = [_payload_to_product(h.payload) for h in ranked if h.payload]

    max_price = extract_price(query)
    if max_price:
        results = [p for p in results if p["price"] <= max_price]

    return results[:top_k]
