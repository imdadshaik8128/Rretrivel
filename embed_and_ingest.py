"""
embed_and_ingest.py
===================
Reads chunks from chunker.py output → embeds with nomic-embed-text-v1.5 → upserts into Qdrant.

Setup:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
    pip install sentence-transformers qdrant-client tqdm

Run:
    python embed_and_ingest.py --chunks_dir ./chunks
    python embed_and_ingest.py --chunks_dir ./chunks --recreate   # fresh start
"""

import argparse
import json
import uuid
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "nomic-ai/nomic-embed-text-v1.5"
VECTOR_SIZE = 768
COLLECTION  = "edu_chunks"
QDRANT_URL  = "http://localhost:6333"
BATCH_SIZE  = 64

PAYLOAD_FIELDS = [
    "chunk_id", "source_file", "subject",
    "chapter_number", "chapter_title",
    "section_title", "subsection_title", "heading_path",
    "chunk_index", "start_line", "end_line",
    "token_count", "char_count",
    "has_activity", "has_equation", "has_table", "has_image",
    "keywords", "text",
]

# ── LOAD ──────────────────────────────────────────────────────────────────────
def load_chunks(chunks_dir: str) -> list:
    combined = Path(chunks_dir) / "all_chunks.json"
    if combined.exists():
        with open(combined, encoding="utf-8") as f:
            chunks = json.load(f)
    else:
        chunks = []
        for p in sorted(Path(chunks_dir).glob("*_chunks.json")):
            with open(p, encoding="utf-8") as f:
                chunks.extend(json.load(f))
    print(f"Loaded {len(chunks)} chunks")
    return chunks

# ── EMBED TEXT BUILDER ────────────────────────────────────────────────────────
def build_embed_text(chunk: dict) -> str:
    """
    nomic-embed-text-v1.5 requires task prefixes.
    At ingestion time use 'search_document:'.
    At query time use 'search_query:' (handled in retrieval script).
    We prepend heading_path as context so the model understands structure.
    """
    heading = chunk.get("heading_path", "")
    text    = chunk.get("text", "")
    prefix  = f"[{heading}]\n\n" if heading else ""
    return f"search_document: {prefix}{text}"

# ── QDRANT SETUP ──────────────────────────────────────────────────────────────
def setup_collection(client: QdrantClient, collection: str, recreate: bool):
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        if recreate:
            print(f"Dropping '{collection}' for fresh start...")
            client.delete_collection(collection)
        else:
            print(f"Collection '{collection}' exists — will upsert (idempotent).")
            return

    print(f"Creating collection '{collection}' (cosine, dim={VECTOR_SIZE})")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Indexed payload fields enable O(1) pre-filtering before ANN search
    for field, schema in {
        "subject":        PayloadSchemaType.KEYWORD,
        "chapter_number": PayloadSchemaType.KEYWORD,
        "has_equation":   PayloadSchemaType.BOOL,
        "has_table":      PayloadSchemaType.BOOL,
        "has_image":      PayloadSchemaType.BOOL,
        "has_activity":   PayloadSchemaType.BOOL,
        "token_count":    PayloadSchemaType.INTEGER,
    }.items():
        client.create_payload_index(collection, field, schema)
    print("Payload indexes created.")

def chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID — re-runs won't create duplicates."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

# ── INGEST ────────────────────────────────────────────────────────────────────
def ingest(chunks, model, client, collection, batch_size):
    batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
    print(f"\nEmbedding {len(chunks)} chunks in {len(batches)} batches...\n")

    for batch in tqdm(batches, unit="batch"):
        texts   = [build_embed_text(c) for c in batch]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()

        points = []
        for chunk, vec in zip(batch, vectors):
            payload = {f: chunk.get(f) for f in PAYLOAD_FIELDS}
            kw = payload.get("keywords")
            if isinstance(kw, str):
                payload["keywords"] = [k.strip() for k in kw.split(",")]
            elif kw is None:
                payload["keywords"] = []

            points.append(PointStruct(
                id      = chunk_id_to_uuid(chunk["chunk_id"]),
                vector  = vec,
                payload = payload,
            ))

        client.upsert(collection_name=collection, points=points)

    print(f"\nDone! {len(chunks)} chunks in Qdrant.")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", default="./chunks")
    ap.add_argument("--qdrant_url", default=QDRANT_URL)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--recreate",   action="store_true")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks_dir)
    if not chunks:
        print("No chunks found. Run chunker.py first."); return

    print(f"Loading model: {MODEL_NAME}  (~270MB, cached after first run)")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    print(f"Connecting to Qdrant: {args.qdrant_url}")
    client = QdrantClient(url=args.qdrant_url)

    setup_collection(client, args.collection, args.recreate)
    ingest(chunks, model, client, args.collection, args.batch_size)

    info = client.get_collection(args.collection)
    print(f"\nStats — points: {info.points_count}, status: {info.status}")

if __name__ == "__main__":
    main()