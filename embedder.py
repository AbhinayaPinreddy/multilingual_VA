"""
Index products into Qdrant (run once after install or when products.json changes):
  python embedder.py
"""
import json
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

import config

_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    products_path = os.path.join(_ROOT, "products.json")
    with open(products_path, encoding="utf-8") as f:
        products = json.load(f)

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    texts = [
        f"{p['name']} {p['description']} {p['category']}"
        for p in products
    ]
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]

    os.makedirs(config.QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=config.QDRANT_PATH)

    if client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "name": p["name"],
                "price": p["price"],
                "description": p["description"],
                "category": p["category"],
                "colors": p["colors"],
            },
        )
        for i, p in enumerate(products)
    ]

    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    print(f"Indexed {len(points)} products into Qdrant at {config.QDRANT_PATH}")


if __name__ == "__main__":
    main()
