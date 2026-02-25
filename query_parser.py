import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def parse_query_with_slm(query):

    prompt = f"""
You are an intent parser for a school textbook Q&A system.

IMPORTANT:
- Output ONLY a valid JSON object.
- No explanation.
- No markdown.
- No extra text.
- Always follow the schema strictly.

Schema:
{{
  "intent": "explain | solve | define | list | unknown",

  "chapter_number": integer or null,
  "chapter_name": string or null,

  "chunk_type": "theory | activity | exercise | null",

  "activity_number": integer or null,
  "exercise_number": string or null,

  "topic": "short cleaned topic phrase or null",

  "needs_full_chapter": boolean,

  "confidence": "high | medium | low"
}}

Rules:
- "what is" → intent = "define"
- "explain" → intent = "explain"
- "solve" → intent = "solve"
- If "activity <number>" → chunk_type = "activity"
- If "exercise <number>" → chunk_type = "exercise"
- If general concept question → chunk_type = "theory"
- If user asks full chapter → needs_full_chapter = true and topic = null
- Extract topic ONLY for theory queries.
- If unsure about extraction → confidence = "low"
- If clear structured query → confidence = "high"

Examples:

Query: explain activity 2 from chapter 1
Output:
{{
  "intent": "explain",
  "chapter_number": 1,
  "chapter_name": null,
  "chunk_type": "activity",
  "activity_number": 2,
  "exercise_number": null,
  "topic": null,
  "needs_full_chapter": false,
  "confidence": "high"
}}

Query: solve exercise 3.1 chapter 4
Output:
{{
  "intent": "solve",
  "chapter_number": 4,
  "chapter_name": null,
  "chunk_type": "exercise",
  "activity_number": null,
  "exercise_number": "3.1",
  "topic": null,
  "needs_full_chapter": false,
  "confidence": "high"
}}

Query: what is photosynthesis
Output:
{{
  "intent": "define",
  "chapter_number": null,
  "chapter_name": null,
  "chunk_type": "theory",
  "activity_number": null,
  "exercise_number": null,
  "topic": "Photosynthesis",
  "needs_full_chapter": false,
  "confidence": "high"
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
            "options": {
                "temperature": 0
            }
        }
    )

    raw_output = response.json().get("response", "{}")

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "chapter_number": None,
            "chapter_name": None,
            "chunk_type": None,
            "activity_number": None,
            "exercise_number": None,
            "topic": None,
            "needs_full_chapter": False,
            "confidence": "low"
        }

from qdrant_client.models import Filter, FieldCondition, MatchValue


def retrieve_from_parsed(parsed, subject, user_query, top_k=5):
    """
    Retrieval router based on structured parser output.
    Reuses v7 embedding + rerank pipeline.  
    """

    # -----------------------------
    # CASE 1: Activity / Exercise (STRICT LOOKUP)
    # -----------------------------
    item_number = parsed.get("activity_number") or parsed.get("exercise_number")

    if item_number is not None:
        must_conditions = [
            FieldCondition(key="subject", match=MatchValue(value=subject)),
            FieldCondition(key="chunk_type", match=MatchValue(value=parsed["chunk_type"])),
            FieldCondition(key="activity_number", match=MatchValue(value=str(item_number)))
        ]

        if parsed.get("chapter_number") is not None:
            must_conditions.append(
                FieldCondition(
                    key="chapter_number",
                    match=MatchValue(value=parsed["chapter_number"])
                )
            )

        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=50
        )

        # Sort by chunk_index if exists
        scroll_result.sort(
            key=lambda x: x.payload.get("chunk_index", 0)
        )

        return scroll_result


    # -----------------------------
    # CASE 2: Full Chapter Retrieval
    # -----------------------------
    if parsed.get("needs_full_chapter") is True:

        must_conditions = [
            FieldCondition(key="subject", match=MatchValue(value=subject)),
            FieldCondition(key="chapter_number", match=MatchValue(value=parsed["chapter_number"]))
        ]

        scroll_result, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=500
        )

        scroll_result.sort(
            key=lambda x: x.payload.get("chunk_index", 0)
        )

        return scroll_result


    # -----------------------------
    # CASE 3: Semantic Search (Theory / General)
    # -----------------------------

    # If confidence low → reduce strict filtering
    must_conditions = [
        FieldCondition(key="subject", match=MatchValue(value=subject))
    ]

    if parsed.get("confidence") != "low":

        if parsed.get("chapter_number") is not None:
            must_conditions.append(
                FieldCondition(
                    key="chapter_number",
                    match=MatchValue(value=parsed["chapter_number"])
                )
            )

        if parsed.get("chunk_type") is not None:
            must_conditions.append(
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=parsed["chunk_type"])
                )
            )

    search_filter = Filter(must=must_conditions)

    # Use topic if available, else full query
    search_text = parsed.get("topic") or user_query

    query_vector = embedding_model.encode(search_text).tolist()

    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=search_filter,
        limit=top_k * 3
    )

    # Reuse your v7 cross-encoder rerank
    reranked_hits = rerank_hits(user_query, hits)

    return reranked_hits[:top_k]

if __name__ == "__main__":

    subject = "Biology"   # <-- set subject manually here

    test_queries = [
        "explain activity 1 from chapter 2",
      
    ]

    for query in test_queries:
        print("\n==============================")
        print("QUERY:", query)

        parsed = parse_query_with_slm(query)
        print("PARSED:", parsed)

        results = retrieve_from_parsed(parsed, subject, query)

        print("RESULT COUNT:", len(results))

        for r in results:
            print("-" * 40)
            print("Chapter:", r.payload.get("chapter_number"))
            print("Chunk Type:", r.payload.get("chunk_type"))
            print("Activity/Exercise:", r.payload.get("activity_number"))
            print("Text Preview:", r.payload.get("text", "")[:200])