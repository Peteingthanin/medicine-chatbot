"""
02_embed.py — Embed chunks into Qdrant vector store

Reads:  extracted_data/chunks/*.json
Embeds: chunk["text"] using Qwen3-Embedding-8B via llama-cpp-python
Stores: vectors + full payload into local Qdrant collection
"""

import os
import json
import glob
import uuid
from pathlib import Path
from tqdm import tqdm

from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBED_MODEL_PATH, EMBED_N_CTX, EMBED_N_GPU_LAYERS, EMBED_DIMENSIONS,
    QDRANT_PATH, QDRANT_COLLECTION, CHUNKS_DIR
)

LOG_FILE = r".\extracted_data\embed_log.txt"


# ---------------------------------------------------------------------------
# Model + DB init
# ---------------------------------------------------------------------------

def load_embed_model() -> Llama:
    print(f"Loading embedding model: {EMBED_MODEL_PATH}")
    return Llama(
        model_path=EMBED_MODEL_PATH,
        embedding=True,
        n_ctx=EMBED_N_CTX,
        n_gpu_layers=EMBED_N_GPU_LAYERS,
        verbose=False,
    )


def init_qdrant(recreate: bool = False) -> QdrantClient:
    Path(QDRANT_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)

    collections = [c.name for c in client.get_collections().collections]

    if recreate and QDRANT_COLLECTION in collections:
        print(f"Deleting existing collection: {QDRANT_COLLECTION}")
        client.delete_collection(QDRANT_COLLECTION)
        collections = []

    if QDRANT_COLLECTION not in collections:
        print(f"Creating collection: {QDRANT_COLLECTION} (dim={EMBED_DIMENSIONS})")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=EMBED_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
        # Create payload indexes for fast metadata filtering
        client.create_payload_index(QDRANT_COLLECTION, "pill_name",  PayloadSchemaType.KEYWORD)
        client.create_payload_index(QDRANT_COLLECTION, "section",    PayloadSchemaType.KEYWORD)
        client.create_payload_index(QDRANT_COLLECTION, "subsection", PayloadSchemaType.KEYWORD)
    else:
        print(f"Using existing collection: {QDRANT_COLLECTION}")

    return client


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(model: Llama, text: str) -> list:
    """
    Embed a document text (no instruction prefix — raw content).
    For queries, use embed_query() in retriever_vector.py.
    """
    response = model.create_embedding(text)
    return response["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def embed_all(recreate: bool = False):
    chunk_files = sorted(glob.glob(os.path.join(CHUNKS_DIR, "*.json")))
    if not chunk_files:
        print(f"No chunk JSON files found in {CHUNKS_DIR}")
        return

    print(f"Found {len(chunk_files)} chunk files\n")

    # Load model and DB
    embed_model = load_embed_model()
    qdrant      = init_qdrant(recreate=recreate)

    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("status   | chunks | file\n")
        log.write("-" * 60 + "\n")

    total_points = 0
    total_errors = 0

    for chunk_file in tqdm(chunk_files, desc="Embedding files"):
        with open(chunk_file, encoding="utf-8") as f:
            chunks = json.load(f)

        points = []
        errors = 0

        for chunk in chunks:
            text     = chunk.get("text", "")
            data     = chunk.get("data", {})
            metadata = chunk.get("metadata", {})

            if not text.strip():
                continue

            try:
                vector = embed_text(embed_model, text)
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, text))
                point  = PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={
                        # Full chunk stored in payload for retrieval
                        "text":       text,
                        "topic":      data.get("topic"),
                        "subtopic":   data.get("subtopic"),
                        "detail":     data.get("detail", []),
                        # Metadata fields (indexed for filtering)
                        "pill_name":  metadata.get("pill_name"),
                        "section":    metadata.get("section"),
                        "subsection": metadata.get("subsection"),
                        "source_file":metadata.get("source_file"),
                    }
                )
                points.append(point)
            except Exception as e:
                print(f"\n  Error embedding chunk: {e}")
                errors += 1

        # Upsert batch into Qdrant
        if points:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

        total_points += len(points)
        total_errors += errors

        status = "OK" if errors == 0 else f"PARTIAL ({errors} errors)"
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"{status:<8} | {len(points):>6} | {Path(chunk_file).name}\n")

    print(f"\n--- Done ---")
    print(f"Total vectors stored : {total_points}")
    print(f"Total errors         : {total_errors}")
    print(f"Qdrant collection    : {QDRANT_COLLECTION}")
    print(f"Qdrant path          : {QDRANT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the Qdrant collection before embedding")
    args = parser.parse_args()
    embed_all(recreate=args.recreate)
