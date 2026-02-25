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
  "exercise_number" : integer or null,
  "topic": short phrase
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
  "exercise_number" : null  ,
  "topic": "Activity 2"
}}

Query: solve exercise 3.1 chapter 4
Output:
{{
  "intent": "solve",
  "chunk_type": "exercise",
  "chapter_number": 4,
  "chapter_name": null,
  "exercise_number" : 3.1,
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
            "format": "json",  # 🔥 Forces JSON output
            "stream": False,
            "options": {
                "temperature": 0
            }
        }
    )

    return response.json()["response"]

print(parse_query_with_slm("explain the activity 2 from chapter 2"))

