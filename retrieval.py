"""
retrieval.py
============
Retrieves relevant chunks from Qdrant for a given query.
Supports semantic search + optional metadata pre-filters.

Setup:
    pip install -U sentence-transformers qdrant-client rich

Run (single query):
    python retrieval.py --query "What is Newton's third law?"
    python retrieval.py --query "photosynthesis steps" --subject Biology
"""

import argparse
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
COLLECTION = "edu_chunks"
QDRANT_URL = "http://localhost:6333"
TOP_K      = 5

# Global caches
_model  = None
_client = None

def get_model():
    global _model
    if _model is None:
        # trust_remote_code=True is required for Nomic 1.5
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _model

def get_client(url=QDRANT_URL):
    global _client
    if _client is None:
        _client = QdrantClient(url=url)
    return _client

# ── FILTER BUILDER ────────────────────────────────────────────────────────────
def build_filter(
    subject: str       = None,
    chapter_number: str = None,
    has_equation: bool  = None,
    has_table: bool     = None,
    has_image: bool     = None,
    has_activity: bool  = None,
) -> Filter | None:
    must = []

    if subject:
        must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
    
    if chapter_number:
        must.append(FieldCondition(key="chapter_number", match=MatchValue(value=str(chapter_number))))
    
    # In newer qdrant-client, MatchValue handles booleans directly. 
    # MatchBool is no longer needed/available in many versions.
    if has_equation:
        must.append(FieldCondition(key="has_equation", match=MatchValue(value=True)))
    if has_table:
        must.append(FieldCondition(key="has_table", match=MatchValue(value=True)))
    if has_image:
        must.append(FieldCondition(key="has_image", match=MatchValue(value=True)))
    if has_activity:
        must.append(FieldCondition(key="has_activity", match=MatchValue(value=True)))

    return Filter(must=must) if must else None


# ── CORE RETRIEVAL ────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_k: int            = TOP_K,
    subject: str          = None,
    chapter_number: str   = None,
    has_equation: bool    = None,
    has_table: bool       = None,
    has_image: bool       = None,
    has_activity: bool    = None,
    collection: str       = COLLECTION,
    score_threshold: float = 0.3,
) -> list[dict]:
    model  = get_model()
    client = get_client()

    # Nomic 1.5 requirement: prefix search queries
    query_vec = model.encode(
        f"search_query: {query}",
        normalize_embeddings=True
    ).tolist()

    filt = build_filter(
        subject=subject,
        chapter_number=chapter_number,
        has_equation=has_equation,
        has_table=has_table,
        has_image=has_image,
        has_activity=has_activity,
    )

    # Verify search method exists
    if not hasattr(client, "search"):
        print("Error: Your qdrant-client version is outdated or corrupted.")
        print("Please run: pip install --upgrade qdrant-client")
        sys.exit(1)

    try:
        results = client.search(
            collection_name=collection,
            query_vector=query_vec,
            query_filter=filt,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
    except Exception as e:
        print(f"Qdrant search error: {e}")
        return []

    return [
        {
            "score":           round(r.score, 4),
            "chunk_id":        r.payload.get("chunk_id"),
            "subject":         r.payload.get("subject"),
            "heading_path":    r.payload.get("heading_path"),
            "chapter_title":   r.payload.get("chapter_title"),
            "section_title":   r.payload.get("section_title"),
            "has_equation":    r.payload.get("has_equation", False),
            "has_table":       r.payload.get("has_table", False),
            "text":            r.payload.get("text", ""),
        }
        for r in results
    ]


# ── DISPLAY ───────────────────────────────────────────────────────────────────
def display_results(results: list[dict], query: str):
    if not results:
        print("\nNo results found above threshold. Try a broader query.\n")
        return

    if HAS_RICH:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        console.print(f"[bold]Found {len(results)} result(s)[/bold]\n")
        for i, r in enumerate(results, 1):
            flags = []
            if r.get("has_equation"): flags.append("📐 equation")
            if r.get("has_table"):    flags.append("📊 table")
            tag_str = "  " + " · ".join(flags) if flags else ""
            
            title = (f"[{i}] Score: {r['score']:.4f}  |  "
                     f"{r['subject']}  |  {r['heading_path']}{tag_str}")
            
            preview = r["text"][:600].replace("\n", " ") + ("..." if len(r["text"]) > 600 else "")
            console.print(Panel(preview, title=title, border_style="dim"))
    else:
        print(f"\nQuery: {query}")
        print(f"Found {len(results)} result(s)\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] Score: {r['score']:.4f} | {r['subject']} | {r['heading_path']}")
            print(f"     {r['text'][:400]}...")
            print("-" * 40)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Retrieve relevant chunks from Qdrant")
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--subject", type=str, default=None)
    ap.add_argument("--chapter", type=str, default=None)
    ap.add_argument("--has_equation", action="store_true", default=None)
    ap.add_argument("--has_table", action="store_true", default=None)
    ap.add_argument("--has_image", action="store_true", default=None)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--collection", default=COLLECTION)
    args = ap.parse_args()

    # If no query provided via CLI, enter a simple interactive mode
    if not args.query:
        print("No query provided. Entering interactive mode (type 'exit' to quit).")
        while True:
            q = input("\nEnter Query: ").strip()
            if q.lower() in ['exit', 'quit']: break
            if not q: continue
            results = retrieve(query=q, top_k=args.top_k, subject=args.subject, collection=args.collection)
            display_results(results, q)
        return

    print(f"Loading model: {MODEL_NAME}...")
    get_model()
    
    print(f"Searching Qdrant at {QDRANT_URL}...")
    results = retrieve(
        query=args.query,
        top_k=args.top_k,
        subject=args.subject,
        chapter_number=args.chapter,
        has_equation=args.has_equation,
        has_table=args.has_table,
        has_image=args.has_image,
        score_threshold=args.threshold,
        collection=args.collection,
    )
    display_results(results, args.query)

if __name__ == "__main__":
    main()