import requests
import json
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================================
# CONFIG
# ==========================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "edu_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5

# ==========================================================
# LOAD MODELS
# ==========================================================
print(f"Loading embedding model ({EMBEDDING_MODEL_NAME})...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Loading rerank model...")
rerank_model = CrossEncoder(RERANK_MODEL_NAME)
print("Connecting to Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# ==========================================================
# DIAGNOSTIC TOOL: Inspect Collection Schema
# ==========================================================
def inspect_collection():
    print("\n--- SCHEMA INSPECTION ---")
    try:
        points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1)
        if points:
            payload = points[0].payload
            print("Example Payload from Qdrant:")
            for key, value in payload.items():
                print(f"  - {key}: {value} (Type: {type(value).__name__})")
        else:
            print("Collection is empty!")
    except Exception as e:
        print(f"Error inspecting collection: {e}")
    print("------------------------\n")

# ==========================================================
# RERANK FUNCTION
# ==========================================================
def rerank_hits(query: str, hits: List, top_k: int = TOP_K):
    if not hits:
        return []
    pairs = [(query, hit.payload.get("text", "")) for hit in hits]
    scores = rerank_model.predict(pairs)
    for hit, score in zip(hits, scores):
        hit.score = float(score)
    hits.sort(key=lambda x: x.score, reverse=True)
    return hits[:top_k]

# ==========================================================
# RETRIEVAL ROUTER
# ==========================================================
def retrieve_from_parsed(parsed, subject, user_query, top_k=TOP_K):
    """
    Robust retrieval router.
    Handles String/Int chapter conversion and falls back to broader semantic search.
    """
    item_number = parsed.get("activity_number") or parsed.get("exercise_number")
    
    # Standardize Subject casing (Match the "Biology" found in schema)
    # We use .title() but you should match your DB exactly.
    target_subject = str(subject).strip()

    # 1. STRICT SEARCH (Case 1)
    if item_number is not None:
        print(f"DEBUG: Case 1 Triggered (Item {item_number})")
        
        # Build conditions dynamically based on available metadata
        must_conditions = [
            FieldCondition(key="subject", match=MatchValue(value=target_subject))
        ]
        
        # Attempt to match activity_number as a string (per schema inspection findings)
        must_conditions.append(FieldCondition(key="activity_number", match=MatchValue(value=str(item_number))))

        if parsed.get("chapter_number") is not None:
            must_conditions.append(FieldCondition(key="chapter_number", match=MatchValue(value=str(parsed["chapter_number"]))))

        scroll_result, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=must_conditions),
            limit=100
        )
        if scroll_result:
            return scroll_result
        print(f"DEBUG: Case 1 (Strict) for Item {item_number} returned 0. Moving to Semantic Search...")

    # 3. SEMANTIC SEARCH (Case 3)
    print(f"DEBUG: Case 3 Triggered (Semantic). Filtering by Subject: {target_subject}")
    
    def perform_search(conditions):
        # Use 'topic' if available, otherwise raw query
        search_text = parsed.get("topic") or user_query
        query_vec = embedding_model.encode(search_text, normalize_embeddings=True).tolist()
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            query_filter=Filter(must=conditions),
            limit=top_k * 10 # Wider pool for reranker to find the actual needle
        )

    # Attempt A: Subject + Chapter
    current_conditions = [FieldCondition(key="subject", match=MatchValue(value=target_subject))]
    if parsed.get("chapter_number") is not None:
        current_conditions.append(FieldCondition(key="chapter_number", match=MatchValue(value=str(parsed["chapter_number"]))))
    
    hits = perform_search(current_conditions)

    # Attempt B: Subject Only (If specific chapter search failed or gave poor results)
    if not hits:
        print(f"DEBUG: Search with Chapter={parsed.get('chapter_number')} failed. Retrying with Subject only...")
        current_conditions = [FieldCondition(key="subject", match=MatchValue(value=target_subject))]
        hits = perform_search(current_conditions)

    # Rerank and return
    return rerank_hits(user_query, hits, top_k)

if __name__ == "__main__":
    # Diagnostic check
    inspect_collection()

    # Simulation
    parsed_example = {
        "intent": "explain",
        "chapter_number": 2, 
        "chunk_type": "theory", # Changed to theory to see if it helps relevance
        "activity_number": None, # Removed to force semantic search for "digestion"
        "topic": "digestion process in humans", # Added topic to help the embedding model
        "confidence": "high"
    }

    subject = "Biology" 
    query = "explain activity 1 from chapter 2"

    results = retrieve_from_parsed(parsed_example, subject, query)
    print(f"\nFINAL RESULT COUNT: {len(results)}")

    if len(results) > 0:
        for i, r in enumerate(results):
            print(f"\n[{i+1}] Score: {r.score:.4f} | Ch: {r.payload.get('chapter_number')} | Sub: {r.payload.get('subject')}")
            # Show a bit more text to verify if it's the right content
            text_preview = r.payload.get("text", "").replace("\n", " ")[:250]
            print(f"Text: {text_preview}...")
    else:
        print("No results found. Please check if the 'digestion' content is actually in your Qdrant index.")    