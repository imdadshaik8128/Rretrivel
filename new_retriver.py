import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def parse_query_with_slm(query):

    prompt = f"""
You are an intent parser for a school textbook Q&A system.
Output ONLY a JSON object. No explanation. No markdown. No extra text.

Extract structured information from the query.

Return ONLY valid JSON.

Schema:
{{
  "intent": "explain | solve | define | list | unknown",
  "chunk_type": "exercise | activity | theory | unknown",
  "chapter_number": integer or null,
  "chapter_name": string or null,
  "activity_number": integer or null,
  "topic": short phrase,
  "subject" : Biology
}}

Examples:

Query: explain the activity 2 from chapter 1
Output:
{{
  "intent": "explain",
  "chunk_type": "activity",
  "chapter_number": 1,
  "chapter_name": null,
  "activity_number" : 2,
  "subject" : Biology,
  "topic": "Activity 2"
}}

Query: solve exercise 3.1 chapter 4
Output:
{{
  "intent": "solve",
  "chunk_type": "exercise",
  "chapter_number": 4,
  "chapter_name": null,
  "activity_number" : null,
  "subject" : Biology,
  "topic": "Exercise 3.1"
}}

Now extract from:

Query: "{query}"
Output:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "qwen2.5:0.5b-instruct",
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0}
        }
    )

    raw_output = response.json()["response"]

    # 🔥 Convert JSON string → Python dict
    parsed_dict = json.loads(raw_output)

    return parsed_dict

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "edu_chunks"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def build_filter_from_parsed_query(parsed_query):
    """
    Dynamically build metadata filter
    Only uses fields that have real values
    """

    conditions = []

    for key in [
        "chapter_number",
        "chunk_type",
        "activity_number",
        "subject"
    ]:
        value = parsed_query.get(key)

        if value is not None and value != "unknown":
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )

    return Filter(must=conditions) if conditions else None

# =====================================================


# ================== RETRIEVER ==================

def retrieve_edu_content(parsed_query, user_query, top_k=5, debug=True):

    q_filter = build_filter_from_parsed_query(parsed_query)

    is_structured_query = (
        parsed_query.get("chapter_number") is not None and
        parsed_query.get("chunk_type") != "unknown" and
        (
            parsed_query.get("activity_number") is not None or
            parsed_query.get("exercise_number") is not None
        )
    )

    if debug:
        print("\n========= RETRIEVAL DEBUG =========")
        print("Parsed Query:", parsed_query)
        print("Structured Query:", is_structured_query)

        if q_filter:
            print("Filter Conditions:")
            for cond in q_filter.must:
                print(f"  {cond.key} = {cond.match.value}")
        else:
            print("No metadata filter applied")

    # ---------------- FILTER ONLY ----------------

    if is_structured_query and q_filter:

        if debug:
            print("\nUsing FILTER ONLY retrieval")

        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=q_filter,
            limit=top_k
        )

        return results

    # ---------------- HYBRID / VECTOR ----------------

    else:

        if debug:
            print("\nUsing VECTOR / HYBRID retrieval")

        query_embedding = embed_model.encode(user_query).tolist()

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=q_filter,
            limit=top_k
        )

        return results

# =====================================================


# ================== MAIN PIPELINE ==================

def run_query(user_query):

    parsed_query = parse_query_with_slm(user_query)

    results = retrieve_edu_content(parsed_query, user_query)

    print("\n========= RESULTS =========")

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print("Payload:", r.payload)
        print("Text:", r.payload.get("text", "No text field"))

# =====================================================


# ================== TEST ==================

if __name__ == "__main__":
    run_query("explain the activity 2 from chapter 2")