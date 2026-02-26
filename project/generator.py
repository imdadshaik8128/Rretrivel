"""
generator.py — Structured Answer Generator
============================================
Takes retrieved chunks + raw query → produces a dual-format answer:
  - spoken_answer      : clean natural language for TTS (no markdown, no symbols)
  - display_answer_markdown : rich markdown for UI display with headings + citations

Two answer types, each with its own prompt template:
  - "reference" : activity / exercise lookup  → present the content directly
  - "concept"   : topic / concept query       → synthesise + explain across chunks

Local LLM via Ollama (offline, no API key needed).
Install : https://ollama.com  →  ollama pull mistral

Architecture:
  Retriever  →  [RetrievedChunk list]
                        │
                  Generator.generate()
                        │
               ┌────────┴────────┐
          spoken_answer     display_answer_markdown
               │
              TTS
"""

from __future__ import annotations

import json
import logging
import re
import math
from dataclasses import dataclass, field
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
OLLAMA_URL         = "http://localhost:11434/api/generate"
OLLAMA_MODEL       = "qwen2.5:0.5b-instruct"          # change to llama3, gemma2, etc. as needed
LLM_TIMEOUT        = 60                 # seconds
CONFIDENCE_FLOOR   = 0.40               # below this → low_confidence_warning fires
REFERENCE_CONFIDENCE = 1.0              # exact metadata match is always 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Output schema
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Citation:
    chunk_id:        str
    chapter_number:  str
    chapter_title:   str
    section_title:   str
    chunk_type:      str
    activity_number: str


@dataclass
class GeneratedAnswer:
    answer_type:              str              # "reference" | "concept"
    spoken_answer:            str              # TTS-safe, plain natural language
    display_answer_markdown:  str              # rich markdown for UI
    citations:                list[Citation]
    confidence:               float            # 0.0 – 1.0
    low_confidence_warning:   Optional[str]    # None if confidence is acceptable
    filter_path:              str              # provenance from retriever

    def to_dict(self) -> dict:
        return {
            "answer_type":             self.answer_type,
            "spoken_answer":           self.spoken_answer,
            "display_answer_markdown": self.display_answer_markdown,
            "citations": [
                {
                    "chunk_id":        c.chunk_id,
                    "chapter_number":  c.chapter_number,
                    "chapter_title":   c.chapter_title,
                    "section_title":   c.section_title,
                    "chunk_type":      c.chunk_type,
                    "activity_number": c.activity_number,
                }
                for c in self.citations
            ],
            "confidence":              round(self.confidence, 4),
            "low_confidence_warning":  self.low_confidence_warning,
            "filter_path":             self.filter_path,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Prompt templates
# ══════════════════════════════════════════════════════════════════════════════

def _build_reference_prompt(query: str, chunks: list) -> str:
    """
    Use Case 1 — activity / exercise lookup.
    The LLM presents the content clearly in two versions.
    It does NOT invent or add information not in the chunks.
    """
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[CHUNK {i} | {c.chunk_type.upper()} | "
            f"Chapter {c.chapter_number} | {c.section_title}]\n{c.text}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are a textbook assistant for school students.
A student asked: "{query}"

The following content was retrieved from the textbook. Present it clearly.
Do NOT add information not present in the chunks below.

---
{context}
---

You MUST respond with ONLY a JSON object. Follow these rules exactly:
- Start your response with {{ and end with }}
- Use double quotes " for all keys and string values
- Use \\n inside strings for line breaks. NEVER use triple quotes.
- Do NOT include markdown fences (no ```)
- Do NOT add any text before or after the JSON

The JSON must have EXACTLY these two keys:

"spoken_answer": A plain natural explanation in 3-5 sentences for text-to-speech.
No bullet points. No markdown symbols. No asterisks. No hashes. Write naturally.

"display_answer_markdown": A markdown response using \\n for newlines. Include:
- A heading with the activity title and chapter
- A short explanation paragraph
- Key steps as a bullet list using - prefix
- A Source line at the bottom

Example of correct format:
{{"spoken_answer": "This activity is about plants. Students observe leaves.", "display_answer_markdown": "## Activity\\n\\nThis activity explains plants.\\n\\n- Step one\\n- Step two\\n\\n**Source:** Chapter 3"}}

Now respond with the JSON for the student query above:"""


def _build_concept_prompt(query: str, chunks: list) -> str:
    """
    Use Case 2 — topic / concept query.
    The LLM synthesises across multiple theory + activity chunks.
    It must cite which chunk each key point came from.
    """
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        context_blocks.append(
            f"[CHUNK {i} | {c.chunk_type.upper()} | "
            f"Chapter {c.chapter_number} – {c.chapter_title} | {c.section_title}]\n{c.text}"
        )
    context = "\n\n".join(context_blocks)

    return f"""You are a textbook assistant for school students.
A student asked: "{query}"

Answer using ONLY the textbook content provided below.
Do NOT use outside knowledge. If the chunks do not contain enough information,
say so clearly instead of guessing.

---
{context}
---

You MUST respond with ONLY a JSON object. Follow these rules exactly:
- Start your response with {{ and end with }}
- Use double quotes " for all keys and string values
- Use \\n inside strings for line breaks. NEVER use triple quotes.
- Do NOT include markdown fences (no ```)
- Do NOT add any text before or after the JSON

The JSON must have EXACTLY these two keys:

"spoken_answer": A plain natural explanation in 4-7 sentences for text-to-speech.
No bullet points. No markdown symbols. No asterisks. No hashes. Write naturally.
Do NOT say "Chunk 1 says" — explain the concept directly.

"display_answer_markdown": A markdown response using \\n for newlines. Include:
- A heading with the topic name
- A clear explanation paragraph
- Key concepts as a bullet list using - prefix
- A Sources section listing chapters referenced

Example of correct format:
{{"spoken_answer": "Osmosis is the movement of water. It happens through a membrane.", "display_answer_markdown": "## Osmosis\\n\\nOsmosis is the movement of water molecules.\\n\\n- Water moves from high to low concentration\\n- A semi-permeable membrane is required\\n\\n**Sources:** Chapter 3"}}

Now respond with the JSON for the student query above:"""


# ══════════════════════════════════════════════════════════════════════════════
# TTS safety cleaner
# ══════════════════════════════════════════════════════════════════════════════

def _clean_for_tts(text: str) -> str:
    """
    Post-process the spoken_answer to guarantee TTS safety.
    The LLM is already prompted to produce clean text, but this is a hard
    safety net in case it leaks any markdown or symbols.
    """
    # Strip markdown headings
    text = re.sub(r"#{1,6}\s+", "", text)
    # Strip bold/italic
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Strip inline code
    text = re.sub(r"`(.+?)`", r"\1", text)
    # Strip bullet points / list markers
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    # Strip numbered list markers like "1." or "2)"
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    # Expand common abbreviations for TTS
    text = text.replace("Fig.", "Figure")
    text = text.replace("fig.", "Figure")
    text = text.replace("e.g.", "for example")
    text = text.replace("i.e.", "that is")
    text = text.replace("vs.", "versus")
    text = text.replace("Ch.", "Chapter")
    # Remove chunk references like "[CHUNK 1]"
    text = re.sub(r"\[CHUNK\s*\d+\]", "", text)
    # Collapse multiple spaces / newlines
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Confidence scoring
# ══════════════════════════════════════════════════════════════════════════════

def _compute_confidence(chunks: list, answer_type: str) -> float:
    """
    Reference queries: always 1.0 (deterministic metadata match).
    Concept queries  : sigmoid-normalise the mean of top chunk scores.
                       Cross-encoder scores are raw logits — sigmoid maps them
                       to [0,1] in a meaningful way.
    """
    if answer_type == "reference":
        return REFERENCE_CONFIDENCE

    scores = [c.score for c in chunks if c.score is not None]
    if not scores:
        return 0.0

    # Use the top score (already sorted best-first by retriever)
    top_score = scores[0]

    # Cross-encoder logits: sigmoid normalisation
    # logit of 3+ → ~0.95, logit of 0 → 0.5, logit of -2 → ~0.12
    confidence = 1.0 / (1.0 + math.exp(-top_score))
    return round(confidence, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Ollama LLM client
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str) -> str:
    """
    Calls local Ollama instance. Returns raw text response.
    Raises ConnectionError if Ollama is not running.
    Raises ValueError if response is malformed.
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,   # low temp = factual, consistent answers
            "top_p": 0.9,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Ollama is not running. Start it with: ollama serve\n"
            f"Then pull a model: ollama pull {OLLAMA_MODEL}"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama did not respond within {LLM_TIMEOUT}s.")

    data = resp.json()
    raw  = data.get("response", "").strip()
    if not raw:
        raise ValueError("Ollama returned an empty response.")
    return raw


def _repair_json(raw: str) -> str:
    """
    Attempt to repair common small-model JSON violations before parsing.
    Handles the specific failures seen with qwen2.5:0.5b and similar models.
    """
    s = raw.strip()

    # 1. Strip markdown fences
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()

    # 2. Replace Python-style triple quotes with a placeholder, then fix
    #    e.g.  """some\ntext"""  →  "some\\ntext"
    def _triple_quote_to_single(m):
        inner = m.group(1)
        # escape any unescaped double quotes inside
        inner = inner.replace('\\"', '\x00DQUOTE\x00')  # protect already-escaped
        inner = inner.replace('"', '\\"')
        inner = inner.replace('\x00DQUOTE\x00', '\\"')
        # escape newlines
        inner = inner.replace('\n', '\\n')
        inner = inner.replace('\r', '')
        return f'"{inner}"'

    s = re.sub(r'"""(.*?)"""', _triple_quote_to_single, s, flags=re.DOTALL)

    # 3. Fix unescaped literal newlines inside JSON string values
    #    Walk character by character to find strings and escape newlines inside them
    result   = []
    in_str   = False
    i        = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and in_str:
            result.append(ch)
            if i + 1 < len(s):
                result.append(s[i + 1])
                i += 2
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
        elif ch == '\n' and in_str:
            result.append('\\n')   # escape the literal newline
        elif ch == '\r' and in_str:
            pass                   # drop carriage returns inside strings
        else:
            result.append(ch)
        i += 1
    s = "".join(result)

    # 4. Remove trailing commas before } or ]  (common small model mistake)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s


def _regex_fallback(raw: str) -> dict:
    """
    Last-resort field extraction using regex when JSON parsing fails entirely.
    Extracts spoken_answer and display_answer_markdown directly from the raw string.
    Returns a dict with whatever it can find, or raises ValueError.
    """
    result = {}

    # Try to extract spoken_answer — look for the key and grab until next key or end
    spoken_match = re.search(
        r'"spoken_answer"\s*:\s*"(.*?)(?<!\\)"(?=\s*[,}])',
        raw,
        re.DOTALL,
    )
    if spoken_match:
        result["spoken_answer"] = spoken_match.group(1).replace('\\n', '\n')

    # Try to extract display_answer_markdown
    display_match = re.search(
        r'"display_answer_markdown"\s*:\s*"(.*?)(?<!\\)"(?=\s*[,}])',
        raw,
        re.DOTALL,
    )
    if display_match:
        result["display_answer_markdown"] = display_match.group(1).replace('\\n', '\n')

    if "spoken_answer" in result and "display_answer_markdown" in result:
        log.warning("JSON parse failed — recovered both fields via regex fallback.")
        return result

    # Could not recover — give up
    raise ValueError(
        f"Could not parse LLM response as JSON and regex fallback also failed.\n"
        f"Raw response (first 400 chars):\n{raw[:400]}"
    )


def _parse_llm_json(raw: str) -> dict:
    """
    Robust JSON parser for small-model output.

    Attempt order:
      1. Direct json.loads              — fast path, works when model behaves
      2. _repair_json + json.loads      — fixes triple quotes, literal newlines
      3. Extract first {...} + repair   — handles leading/trailing junk text
      4. _regex_fallback                — field-level extraction, last resort
    """
    # Attempt 1: direct parse after stripping fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair then parse
    try:
        repaired = _repair_json(raw)
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception):
        pass

    # Attempt 3: extract first {...} block, then repair
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            repaired = _repair_json(match.group())
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            pass

    # Attempt 4: regex field extraction — never loses the answer content
    return _regex_fallback(raw)


# ══════════════════════════════════════════════════════════════════════════════
# Public API: Generator
# ══════════════════════════════════════════════════════════════════════════════

class Generator:
    """
    Single entry point for answer generation.

    Usage:
        generator = Generator()
        answer    = generator.generate(retrieved_chunks, raw_query, answer_type)
        print(json.dumps(answer.to_dict(), indent=2))
    """

    def generate(
        self,
        chunks:      list,          # list[RetrievedChunk] from retriever
        query:       str,           # raw user query text
        answer_type: str = "auto",  # "reference" | "concept" | "auto"
    ) -> GeneratedAnswer:
        """
        Parameters
        ----------
        chunks      : retrieved chunks from Retriever.retrieve()
        query       : original user query string
        answer_type : "reference" for activity/exercise lookups,
                      "concept" for topic/explanation queries,
                      "auto" infers from chunk content (recommended)

        Returns
        -------
        GeneratedAnswer with spoken_answer, display_answer_markdown,
        citations, confidence, and low_confidence_warning.
        """
        if not chunks:
            return self._empty_answer(query)

        # Infer answer type if not explicitly set
        if answer_type == "auto":
            answer_type = self._infer_answer_type(chunks)

        log.info("Generating answer — type=%s, chunks=%d", answer_type, len(chunks))

        # Build citations from all retrieved chunks
        citations = [
            Citation(
                chunk_id        = c.chunk_id,
                chapter_number  = c.chapter_number,
                chapter_title   = c.chapter_title,
                section_title   = c.section_title,
                chunk_type      = c.chunk_type,
                activity_number = c.activity_number,
            )
            for c in chunks
        ]

        # Compute confidence before calling LLM
        confidence = _compute_confidence(chunks, answer_type)
        warning    = None
        if confidence < CONFIDENCE_FLOOR:
            warning = (
                f"The retrieved content may not fully answer this question "
                f"(confidence: {confidence:.0%}). "
                "Consider rephrasing or checking the chapter/subject."
            )
            log.warning("Low confidence %.3f for query: %s", confidence, query)

        # Select prompt template
        if answer_type == "reference":
            prompt = _build_reference_prompt(query, chunks)
        else:
            prompt = _build_concept_prompt(query, chunks)

        # Call LLM
        raw_response = _call_ollama(prompt)
        log.info("LLM responded — parsing JSON …")

        parsed = _parse_llm_json(raw_response)

        # Extract the two required fields
        spoken_raw  = parsed.get("spoken_answer", "")
        display_raw = parsed.get("display_answer_markdown", "")

        if not spoken_raw or not display_raw:
            raise ValueError(
                "LLM response missing 'spoken_answer' or 'display_answer_markdown'. "
                f"Got keys: {list(parsed.keys())}"
            )

        # Hard clean the spoken answer regardless of LLM compliance
        spoken_clean = _clean_for_tts(spoken_raw)

        return GeneratedAnswer(
            answer_type             = answer_type,
            spoken_answer           = spoken_clean,
            display_answer_markdown = display_raw,
            citations               = citations,
            confidence              = confidence,
            low_confidence_warning  = warning,
            filter_path             = chunks[0].filter_path if chunks else "",
        )

    # ------------------------------------------------------------------
    def generate_safe(
        self,
        chunks:      list,
        query:       str,
        answer_type: str = "auto",
    ) -> tuple[Optional[GeneratedAnswer], Optional[str]]:
        """
        Non-raising version. Returns (answer, error_message).
        error_message is None on success.
        """
        try:
            return self.generate(chunks, query, answer_type), None
        except (ConnectionError, TimeoutError, ValueError) as e:
            log.error("Generator error: %s", e)
            return None, str(e)

    # ------------------------------------------------------------------
    @staticmethod
    def _infer_answer_type(chunks: list) -> str:
        """
        Infer answer_type from chunk metadata.
        If any chunk has a non-empty activity_number → reference query.
        Otherwise → concept query.
        """
        for c in chunks:
            if str(c.activity_number).strip() not in ("", "None", "none"):
                return "reference"
        return "concept"

    @staticmethod
    def _empty_answer(query: str) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer_type             = "concept",
            spoken_answer           = (
                "I could not find relevant content in the textbook "
                "to answer your question. Please try rephrasing or "
                "check the chapter and subject."
            ),
            display_answer_markdown = (
                "## No Results Found\n\n"
                "The retrieval system could not find relevant chunks "
                "for your query. Please check:\n"
                "- The subject and chapter are correct\n"
                "- The query is specific enough\n"
            ),
            citations               = [],
            confidence              = 0.0,
            low_confidence_warning  = "No chunks were retrieved.",
            filter_path             = "",
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json as _json
    from query_parser_v2 import parse_query_with_slm
    from retriever import Retriever

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "explain activity 2 from chapter 3 biology"
    print(f"\nQuery: {query!r}\n")

    # 1. Parse
    raw_parse = parse_query_with_slm(query)
    parsed    = _json.loads(raw_parse)
    print("Parsed metadata:", _json.dumps(parsed, indent=2))

    # 2. Retrieve
    retriever = Retriever()
    chunks, ret_err = retriever.retrieve_safe(parsed, query)

    if ret_err:
        print(f"\n⚠  Retrieval error: {ret_err}")
        sys.exit(1)

    print(f"\nRetrieved {len(chunks)} chunks.")

    # 3. Generate
    generator = Generator()
    answer, gen_err = generator.generate_safe(chunks, query)

    if gen_err:
        print(f"\n⚠  Generation error: {gen_err}")
        sys.exit(1)

    # 4. Print result
    print("\n" + "═" * 60)
    print(_json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    print("═" * 60)

    print("\n── SPOKEN (TTS) ──────────────────────────────────────────")
    print(answer.spoken_answer)

    print("\n── DISPLAY (MARKDOWN) ────────────────────────────────────")
    print(answer.display_answer_markdown)