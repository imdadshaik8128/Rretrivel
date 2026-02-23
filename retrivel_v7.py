"""
retrieval_v7.py  — SLM Intent + Content-Type Entity Filtering
=================================================================
NEW in v7:
  - QueryIntent now has a `content_type_filter` field
  - SLM detects when the query asks for a specific content type:
      "theory"           → chunk_type filter: theory
      "activity"         → chunk_type filter: activity
      "exercise"         → chunk_type filter: exercise
      "mcq" / "multiple choice" / "choose correct" → heading_path filter: mcq
      "fill in blank"    → heading_path filter: fill_in_blank
  - semantic_search applies a hard chunk_type / heading_path filter
    BEFORE embedding search when content_type_filter is set
  - reference_lookup unchanged (still works for "activity 3", "example 2.4" etc.)

Setup (one-time):
    pip install ollama sentence-transformers qdrant-client rich

Run:
    python retrieval_v7.py --query "give me MCQs on digestion"         --subject Biology
    python retrieval_v7.py --query "fill in the blanks for chapter 2"  --subject Biology
    python retrieval_v7.py --query "exercises on osmosis"              --subject Biology
    python retrieval_v7.py --query "show me activity 3"                --subject Biology
    python retrieval_v7.py --query "explain photosynthesis"            --subject Biology
    python retrieval_v7.py   # interactive mode
"""

import os
import re
import json
import argparse
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

try:
    from rich.console import Console
    from rich.panel   import Panel
    HAS_RICH = True
    console  = Console()
except ImportError:
    HAS_RICH = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BI_ENCODER_MODEL    = "nomic-ai/nomic-embed-text-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

COLLECTION      = "edu_chunks"
QDRANT_URL      = "http://localhost:6333"

PASS2_TOP_K     = 30   # increased slightly to compensate for content-type filtering
FINAL_TOP_K     = 5
SCORE_THRESHOLD = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT TYPE CONSTANTS
# These map from the detected content_type_filter value to:
#   - chunk_type values stored in Qdrant
#   - heading_path substrings to match (for types not tracked as chunk_type)
# ─────────────────────────────────────────────────────────────────────────────

# chunk_type values that exist in Qdrant metadata
CHUNK_TYPE_VALUES = {"theory", "activity", "exercise"}

# content_type_filter → (qdrant_chunk_type or None, heading_keyword_list)
# If qdrant_chunk_type is set, filter by chunk_type field directly.
# If qdrant_chunk_type is None, filter post-retrieval by heading_path keywords.
CONTENT_TYPE_MAP = {
    "theory":         ("theory",    []),
    "activity":       ("activity",  []),
    "exercise":       ("exercise",  []),
    "mcq":            (None,        ["choose the correct", "multiple choice", "mcq"]),
    "fill_in_blank":  (None,        ["fill in the blank", "fillin the blank", "fill in the"]),
}


# ─────────────────────────────────────────────────────────────────────────────
# QUERY INTENT  (data class — filled by SLM)
# ─────────────────────────────────────────────────────────────────────────────

class QueryIntent:
    """
    mode:                "reference" | "semantic"
    sub_intent:          "DEFINITION" | "COMPARISON" | "GENERAL"
    item_type:           "activity" | "example" | "exercise" | "figure" | None
    item_number:         "2" | "3.1" | None
    chapter_hint:        "1" | "3" | None
    is_conceptual:       False only for calculate/solve/compute queries
    keywords:            ["digestion", "absorption"]
    content_type_filter: "theory" | "activity" | "exercise" | "mcq" | "fill_in_blank" | None
                         Set when user explicitly asks for a content type.
                         This triggers a hard filter before embedding search.
    """
    def __init__(self):
        self.mode                = "semantic"
        self.sub_intent          = "GENERAL"
        self.item_type           = None
        self.item_number         = None
        self.chapter_hint        = None
        self.is_conceptual       = True
        self.keywords            = []
        self.content_type_filter = None   # ← NEW


# ─────────────────────────────────────────────────────────────────────────────
# SLM  —  intent + entity extraction
# ─────────────────────────────────────────────────────────────────────────────

OLLAMA_MODEL = "qwen2.5:0.5b-instruct"

_SLM_SYSTEM_PROMPT = """\
You are an intent parser for a school textbook Q&A system.
Output ONLY a JSON object. No explanation. No markdown. No extra text.

Fields:
  mode                : "reference" if asking for a specific numbered activity/example/exercise/figure, else "semantic"
  item_type           : "activity"|"example"|"exercise"|"figure"|null  (only when mode=reference, numbered item)
  item_number         : digit string like "2" or "3.1", or null        (only when mode=reference)
  chapter_hint        : digit string if a chapter number is mentioned, else null
  sub_intent          : "DEFINITION"|"COMPARISON"|"GENERAL"
  is_conceptual       : false only for calculate/solve/compute queries, else true
  keywords            : list of subject-topic nouns only (NOT: process, type, method, system, step, role, function, form, part)
  content_type_filter : one of ["theory","activity","exercise","mcq","fill_in_blank"] if the user explicitly
                        asks for that type of content, else null.
                        Rules:
                          - "theory"        → user asks for explanation/definition/notes/theory
                          - "activity"      → user asks for activity (without a specific number → not reference mode)
                          - "exercise"      → user asks for exercises/problems/questions
                          - "mcq"           → user asks for MCQ / multiple choice / choose correct answer
                          - "fill_in_blank" → user asks for fill in the blank / fill in the blanks
                        IMPORTANT: set content_type_filter even when mode=semantic.
                        Do NOT set content_type_filter when the query is a plain concept question with no content-type request."""

_FEW_SHOT_MESSAGES = [
    # semantic: plain concept
    {"role": "user",      "content": "Query: explain the process of digestion"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"DEFINITION","is_conceptual":true,"keywords":["digestion"],"content_type_filter":null}'},

    # semantic: what is
    {"role": "user",      "content": "Query: what is osmosis"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"DEFINITION","is_conceptual":true,"keywords":["osmosis"],"content_type_filter":null}'},

    # semantic: comparison
    {"role": "user",      "content": "Query: compare photosynthesis and respiration"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"COMPARISON","is_conceptual":true,"keywords":["photosynthesis","respiration"],"content_type_filter":null}'},

    # MCQ request
    {"role": "user",      "content": "Query: give me MCQs on digestion"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":["digestion"],"content_type_filter":"mcq"}'},

    # MCQ alternate phrasing
    {"role": "user",      "content": "Query: multiple choice questions on osmosis"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":["osmosis"],"content_type_filter":"mcq"}'},

    # fill in the blank
    {"role": "user",      "content": "Query: fill in the blanks for photosynthesis"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":["photosynthesis"],"content_type_filter":"fill_in_blank"}'},

    # fill in blank with chapter
    {"role": "user",      "content": "Query: fill in the blanks chapter 2"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":"2","sub_intent":"GENERAL","is_conceptual":true,"keywords":[],"content_type_filter":"fill_in_blank"}'},

    # exercise request
    {"role": "user",      "content": "Query: exercises on Newton laws"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":["newton","laws"],"content_type_filter":"exercise"}'},

    # activity request without number (semantic, not reference)
    {"role": "user",      "content": "Query: show me activities on respiration"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":["respiration"],"content_type_filter":"activity"}'},

    # reference: specific numbered activity (mode=reference, NOT content_type_filter)
    {"role": "user",      "content": "Query: explain activity 3"},
    {"role": "assistant", "content": '{"mode":"reference","item_type":"activity","item_number":"3","chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":[],"content_type_filter":null}'},

    # reference: activity with chapter
    {"role": "user",      "content": "Query: activity-2 from chapter 1"},
    {"role": "assistant", "content": '{"mode":"reference","item_type":"activity","item_number":"2","chapter_hint":"1","sub_intent":"GENERAL","is_conceptual":true,"keywords":[],"content_type_filter":null}'},

    # reference: ordinal
    {"role": "user",      "content": "Query: show me the second activity"},
    {"role": "assistant", "content": '{"mode":"reference","item_type":"activity","item_number":"2","chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":true,"keywords":[],"content_type_filter":null}'},

    # reference: example with chapter
    {"role": "user",      "content": "Query: show example 2.4 chapter 3"},
    {"role": "assistant", "content": '{"mode":"reference","item_type":"example","item_number":"2.4","chapter_hint":"3","sub_intent":"GENERAL","is_conceptual":true,"keywords":[],"content_type_filter":null}'},

    # theory request
    {"role": "user",      "content": "Query: theory of digestion chapter 1"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":"1","sub_intent":"DEFINITION","is_conceptual":true,"keywords":["digestion"],"content_type_filter":"theory"}'},

    # calculation — no content_type_filter
    {"role": "user",      "content": "Query: calculate the force if mass is 5kg and acceleration is 2"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":null,"sub_intent":"GENERAL","is_conceptual":false,"keywords":["force","mass","acceleration"],"content_type_filter":null}'},

    # chapter mention with MCQ
    {"role": "user",      "content": "Query: MCQ questions chapter 3 biology"},
    {"role": "assistant", "content": '{"mode":"semantic","item_type":null,"item_number":null,"chapter_hint":"3","sub_intent":"GENERAL","is_conceptual":true,"keywords":[],"content_type_filter":"mcq"}'},
]

_ORDINALS = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
}

_VALID_CTF = {"theory", "activity", "exercise", "mcq", "fill_in_blank"}


def detect_intent(query: str) -> QueryIntent:
    """
    Parse query into a QueryIntent using Ollama (qwen2.5:0.5b-instruct).
    Ollama must be running: `ollama serve`
    Model must be pulled:   `ollama pull qwen2.5:0.5b-instruct`
    """
    try:
        import ollama
    except ImportError:
        raise RuntimeError("ollama package not installed. Run: pip install ollama")

    response = ollama.chat(
        model   = OLLAMA_MODEL,
        messages= [
            {"role": "system", "content": _SLM_SYSTEM_PROMPT},
            *_FEW_SHOT_MESSAGES,
            {"role": "user",   "content": f"Query: {query}"},
        ],
        options = {"temperature": 0},
    )

    raw = response["message"]["content"].strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    parsed = json.loads(raw)

    intent = QueryIntent()
    intent.mode          = str(parsed.get("mode", "semantic"))
    intent.item_type     = parsed.get("item_type") or None
    intent.sub_intent    = str(parsed.get("sub_intent", "GENERAL"))
    intent.is_conceptual = bool(parsed.get("is_conceptual", True))
    intent.keywords      = [k.lower().strip() for k in parsed.get("keywords", []) if k]

    ch = parsed.get("chapter_hint")
    intent.chapter_hint = str(ch) if ch else None

    raw_num = parsed.get("item_number")
    if raw_num:
        s = str(raw_num).lower().strip()
        intent.item_number = _ORDINALS.get(s, s)
    else:
        intent.item_number = None

    # Validate content_type_filter
    ctf = parsed.get("content_type_filter")
    if ctf and str(ctf).lower() in _VALID_CTF:
        intent.content_type_filter = str(ctf).lower()
    else:
        intent.content_type_filter = None

    return intent


# ─────────────────────────────────────────────────────────────────────────────
# REGEX FALLBACK — detect content_type_filter from query without SLM
# Used as a safety net if SLM misses it.
# ─────────────────────────────────────────────────────────────────────────────

_CTF_PATTERNS = [
    (re.compile(r'\b(fill\s*in\s*the\s*blank|fill\s*in\s*blank|fillin)', re.I), "fill_in_blank"),
    (re.compile(r'\b(mcq|multiple\s*choice|choose\s*(the\s*)?correct)', re.I),   "mcq"),
    (re.compile(r'\b(exercise|problem|question)s?\b', re.I),                      "exercise"),
    (re.compile(r'\bactivit(y|ies)\b', re.I),                                     "activity"),
    (re.compile(r'\b(theory|explanation|notes|explain)\b', re.I),                 "theory"),
]

def regex_content_type(query: str) -> str | None:
    """Fallback: extract content_type_filter purely from regex if SLM missed it."""
    for pattern, ctype in _CTF_PATTERNS:
        if pattern.search(query):
            return ctype
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL & CLIENT CACHES
# ─────────────────────────────────────────────────────────────────────────────

_bi_encoder    = None
_cross_encoder = None
_qdrant_client = None


def get_bi_encoder():
    global _bi_encoder
    if _bi_encoder is None:
        print(f"[Loading bi-encoder: {BI_ENCODER_MODEL}]")
        _bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, trust_remote_code=True)
    return _bi_encoder


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[Loading cross-encoder: {CROSS_ENCODER_MODEL}]")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def get_client(url: str = QDRANT_URL):
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=url)
    return _qdrant_client


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_filter(
    subject:         str = None,
    chapter_number:  str = None,
    chunk_type:      str = None,
    activity_number: str = None,
    example_number:  str = None,
) -> Filter | None:
    must = []
    if subject:
        must.append(FieldCondition(key="subject",
                                   match=MatchValue(value=subject)))
    if chapter_number:
        must.append(FieldCondition(key="chapter_number",
                                   match=MatchValue(value=str(chapter_number))))
    if chunk_type:
        must.append(FieldCondition(key="chunk_type",
                                   match=MatchValue(value=chunk_type)))
    if activity_number:
        must.append(FieldCondition(key="activity_number",
                                   match=MatchValue(value=str(activity_number))))
    if example_number:
        must.append(FieldCondition(key="example_number",
                                   match=MatchValue(value=str(example_number))))
    return Filter(must=must) if must else None


def payload_to_result(payload: dict, score: float = 0.0) -> dict:
    return {
        "score":           round(score, 4),
        "chunk_id":        payload.get("chunk_id", ""),
        "subject":         payload.get("subject", ""),
        "chapter_number":  payload.get("chapter_number", ""),
        "chapter_title":   payload.get("chapter_title", ""),
        "section_title":   payload.get("section_title", ""),
        "heading_path":    payload.get("heading_path", ""),
        "chunk_index":     payload.get("chunk_index", 0),
        "has_equation":    payload.get("has_equation", False),
        "has_table":       payload.get("has_table", False),
        "chunk_type":      payload.get("chunk_type", "theory"),
        "activity_number": payload.get("activity_number", ""),
        "example_number":  payload.get("example_number", ""),
        "text":            payload.get("text", ""),
    }


def qdrant_search(query_vec: list, filt, top_k: int,
                  collection: str, threshold: float) -> list[dict]:
    client = get_client()
    hits = client.search(
        collection_name = collection,
        query_vector    = query_vec,
        query_filter    = filt,
        limit           = top_k,
        score_threshold = threshold,
        with_payload    = True,
    )
    return [payload_to_result(h.payload, h.score) for h in hits]


def qdrant_scroll_all(filt, collection: str) -> list[dict]:
    """Fetch ALL matching points without vector search."""
    client  = get_client()
    results = []
    offset  = None
    while True:
        batch, next_offset = client.scroll(
            collection_name = collection,
            scroll_filter   = filt,
            limit           = 100,
            offset          = offset,
            with_payload    = True,
            with_vectors    = False,
        )
        results.extend([payload_to_result(p.payload) for p in batch])
        if next_offset is None:
            break
        offset = next_offset
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT TYPE POST-FILTER
# Applied after embedding search for mcq / fill_in_blank (no Qdrant field)
# ─────────────────────────────────────────────────────────────────────────────

def apply_content_type_postfilter(results: list[dict], content_type_filter: str) -> list[dict]:
    """
    For content types that don't have a dedicated Qdrant field (mcq, fill_in_blank),
    filter by checking heading_path or first 300 chars of text.
    """
    _, heading_keywords = CONTENT_TYPE_MAP.get(content_type_filter, (None, []))
    if not heading_keywords:
        return results

    filtered = []
    for r in results:
        haystack = (r.get("heading_path", "") + " " + r.get("section_title", "") +
                    " " + r.get("text", "")[:300]).lower()
        if any(kw in haystack for kw in heading_keywords):
            filtered.append(r)

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — REFERENCE LOOKUP  (hard stop — no fallback to semantic)
# ─────────────────────────────────────────────────────────────────────────────

def reference_lookup(
    intent:     QueryIntent,
    subject:    str,
    collection: str,
    verbose:    bool = False,
) -> list[dict]:
    """
    Pure metadata lookup — no vector search, no semantic fallback.
    Returns all matching chunks sorted by chunk_index, or [] if nothing found.
    """
    item_type   = intent.item_type
    item_number = intent.item_number
    chapter     = intent.chapter_hint

    if verbose:
        print(f"\n[Reference Lookup]  item={item_type} #{item_number}  chapter={chapter}")

    # ── Activities ────────────────────────────────────────────────────────────
    if item_type == "activity":
        matched = qdrant_scroll_all(
            build_filter(subject=subject, chapter_number=chapter,
                         chunk_type="activity", activity_number=item_number),
            collection,
        )
        if not matched:
            candidates = qdrant_scroll_all(
                build_filter(subject=subject, chapter_number=chapter, chunk_type="activity"),
                collection,
            )
            needle = f"activity {item_number}".lower()
            matched = [r for r in candidates if needle in r["heading_path"].lower()]

        if verbose:
            print(f"  activity chunks found: {len(matched)}")
        matched.sort(key=lambda x: x["chunk_index"])
        return matched

    # ── Examples ──────────────────────────────────────────────────────────────
    if item_type == "example":
        matched = qdrant_scroll_all(
            build_filter(subject=subject, chapter_number=chapter, example_number=item_number),
            collection,
        )
        if not matched:
            candidates = qdrant_scroll_all(
                build_filter(subject=subject, chapter_number=chapter),
                collection,
            )
            needle = f"example {item_number}".lower()
            matched = [r for r in candidates
                       if needle in r["heading_path"].lower()
                       or needle in r["text"][:200].lower()]

        if verbose:
            print(f"  example chunks found: {len(matched)}")
        matched.sort(key=lambda x: x["chunk_index"])
        return matched

    # ── Figures / Exercises / Other ───────────────────────────────────────────
    candidates = qdrant_scroll_all(
        build_filter(subject=subject, chapter_number=chapter),
        collection,
    )
    needle = f"{item_type} {item_number}".lower()
    matched = [r for r in candidates
               if needle in r["heading_path"].lower()
               or needle in r["text"][:200].lower()]

    if verbose:
        print(f"  '{needle}' match: {len(matched)} chunks")
    matched.sort(key=lambda x: x["chunk_index"])
    return matched


# ─────────────────────────────────────────────────────────────────────────────
# HEADING-FIRST CHAPTER ANCHORING  (metadata only — no embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def score_heading(heading_text: str, keywords: list[str]) -> tuple[int, int]:
    heading_tokens = set(re.findall(r"[a-zA-Z]{3,}", heading_text.lower()))
    keyword_set    = set(keywords)

    def matches(token: str) -> bool:
        for kw in keyword_set:
            if token == kw:
                return True
            if len(token) >= 5 and len(kw) >= 5:
                stem = min(len(token), len(kw), 5)
                if token[:stem] == kw[:stem]:
                    return True
        return False

    matched_kws   = set()
    unmatched_tok = set()

    for tok in heading_tokens:
        if matches(tok):
            for kw in keyword_set:
                stem = min(len(tok), len(kw), 5)
                if tok == kw or (len(tok) >= 5 and len(kw) >= 5 and tok[:stem] == kw[:stem]):
                    matched_kws.add(kw)
                    break
        else:
            unmatched_tok.add(tok)

    return len(matched_kws), len(unmatched_tok)


def heading_anchor_search(
    intent:     QueryIntent,
    subject:    str,
    collection: str,
    verbose:    bool = False,
) -> list[dict]:
    """
    Scan chapter_title / section_title / heading_path metadata for all chunks.
    Score each chapter by best heading match against intent.keywords.

    When content_type_filter is set (e.g. mcq, fill_in_blank, exercise), we
    look for keyword matches across ALL chunks (not just theory headings),
    because the topic keyword may only appear in the content, not in the heading.
    """
    keywords = intent.keywords
    if not keywords:
        # If no keywords but we have a chapter_hint, we skip anchor search
        # (chapter_hint will be used directly in semantic_search)
        return []

    if verbose:
        print(f"\n[Heading Anchor]  keywords={keywords}  sub_intent={intent.sub_intent}")

    all_chunks = qdrant_scroll_all(build_filter(subject=subject), collection)
    if not all_chunks:
        return []

    chapter_best: dict[str, dict] = {}

    for chunk in all_chunks:
        ch_num = chunk["chapter_number"]
        if not ch_num:
            continue

        heading_text = " ".join(filter(None, [
            chunk.get("chapter_title", ""),
            chunk.get("section_title", ""),
            chunk.get("heading_path",  ""),
        ]))

        exact, extra = score_heading(heading_text, keywords)
        if exact == 0:
            continue

        score = exact - extra
        prev  = chapter_best.get(ch_num)
        if prev is None or score > prev["score"]:
            chapter_best[ch_num] = {
                "chapter_number": ch_num,
                "chapter_title":  chunk.get("chapter_title", ""),
                "score":          score,
                "exact":          exact,
                "extra":          extra,
                "heading_path":   heading_text[:80],
            }

    if not chapter_best:
        return []

    candidates = sorted(
        chapter_best.values(),
        key=lambda x: (
            -x["score"],
            int(x["chapter_number"]) if x["chapter_number"].isdigit() else 999,
        ),
    )

    if verbose:
        for c in candidates[:5]:
            print(f"    ch={c['chapter_number']}  score={c['score']}  "
                  f"exact={c['exact']}  extra={c['extra']}  '{c['heading_path'][:60]}'")

    best_score = candidates[0]["score"]

    if intent.sub_intent == "DEFINITION":
        zero_extra = [c for c in candidates if c["score"] == best_score and c["extra"] == 0]
        candidates = zero_extra if zero_extra else [c for c in candidates if c["score"] == best_score]
    else:
        candidates = [c for c in candidates if c["score"] == best_score]

    for i, c in enumerate(candidates):
        c["label"] = "primary" if i == 0 else "secondary"
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK TYPE PRIORITY  (used only when content_type_filter is None)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_PRIORITY = {
    "DEFINITION": ["theory", "activity", "exercise"],
    "COMPARISON": ["theory", "activity"],
    "GENERAL":    ["theory", "activity", "exercise"],
}


def filter_by_chunk_type(results: list[dict], intent: QueryIntent) -> list[dict]:
    """
    Order results by chunk type priority for the given intent.
    Only applied when content_type_filter is None (no explicit type requested).
    """
    if not intent.is_conceptual or intent.content_type_filter:
        return results

    priority = _CHUNK_PRIORITY.get(intent.sub_intent, ["theory", "activity", "exercise"])
    buckets  = defaultdict(list)
    for r in results:
        buckets[r["chunk_type"]].append(r)

    ordered = []
    for ctype in priority:
        ordered.extend(buckets.pop(ctype, []))
    for items in buckets.values():
        ordered.extend(items)
    return ordered


# ─────────────────────────────────────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────────────────────────────────────

def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    if not results:
        return []
    ce     = get_cross_encoder()
    pairs  = [(query, r["text"]) for r in results]
    scores = ce.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = round(float(s), 4)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — SEMANTIC SEARCH  (heading-first anchored + content-type filtered)
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(
    query:      str,
    intent:     QueryIntent,
    subject:    str,
    top_k:      int,
    collection: str,
    threshold:  float,
    verbose:    bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Step 1 — heading-first chapter anchor (metadata only)
    Step 2 — embedding search restricted to anchored chapter
              + content_type_filter applied as Qdrant filter (if applicable)
    Step 3 — post-filter for heading-keyword types (mcq, fill_in_blank)
    Step 4 — chunk type priority ordering (skipped if content_type_filter is set)
    Step 5 — cross-encoder rerank
    Returns (results, anchor_chapters)
    """
    bi  = get_bi_encoder()
    vec = bi.encode(f"search_query: {query}", normalize_embeddings=True).tolist()

    ctf = intent.content_type_filter

    # ── Determine Qdrant chunk_type filter ───────────────────────────────────
    # For "theory", "activity", "exercise" → use Qdrant field filter directly
    # For "mcq", "fill_in_blank"           → no Qdrant field; post-filter later
    qdrant_chunk_type = None
    if ctf and ctf in CONTENT_TYPE_MAP:
        qdrant_chunk_type, _ = CONTENT_TYPE_MAP[ctf]

    if verbose:
        print(f"\n[ContentType] filter={ctf}  qdrant_chunk_type={qdrant_chunk_type}")

    # ── Step 1: anchor chapter ────────────────────────────────────────────────
    if intent.chapter_hint:
        anchor_chapters = [{
            "chapter_number": intent.chapter_hint,
            "chapter_title":  "",
            "label":          "primary",
            "score":          999,
            "exact":          0,
            "extra":          0,
            "heading_path":   "",
        }]
        if verbose:
            print(f"\n[Anchor] Explicit chapter from query/CLI: {intent.chapter_hint}")
    else:
        anchor_chapters = heading_anchor_search(intent, subject, collection, verbose)
        if verbose:
            if anchor_chapters:
                print(f"[Anchor] {[(c['chapter_number'], c['label']) for c in anchor_chapters]}")
            else:
                print("[Anchor] No chapter matched — running unanchored search")

    # ── Step 2: embedding search ──────────────────────────────────────────────
    if not anchor_chapters:
        if verbose:
            print("[Semantic] Unanchored (no heading match)")
        # Apply chunk_type filter globally if we have one
        filt = build_filter(subject=subject, chunk_type=qdrant_chunk_type)
        results = qdrant_search(vec, filt, PASS2_TOP_K, collection, threshold)
    else:
        primary   = next((c for c in anchor_chapters if c["label"] == "primary"), anchor_chapters[0])
        anchor_ch = primary["chapter_number"]
        if verbose:
            print(f"[Semantic] Anchored to chapter {anchor_ch}  chunk_type_filter={qdrant_chunk_type}")

        filt = build_filter(
            subject        = subject,
            chapter_number = anchor_ch,
            chunk_type     = qdrant_chunk_type,  # None for mcq/fill_in_blank
        )
        results = qdrant_search(vec, filt, PASS2_TOP_K, collection, threshold)

        # If anchored search returned nothing, try without chapter constraint
        if not results and ctf:
            if verbose:
                print(f"[Semantic] No results in chapter {anchor_ch} with type={ctf} — expanding to full subject")
            filt = build_filter(subject=subject, chunk_type=qdrant_chunk_type)
            results = qdrant_search(vec, filt, PASS2_TOP_K, collection, threshold)

        if not results and verbose:
            print(f"[Semantic] No results — returning empty")

    # ── Step 3: post-filter for mcq / fill_in_blank ───────────────────────────
    if ctf in ("mcq", "fill_in_blank") and results:
        filtered = apply_content_type_postfilter(results, ctf)
        if filtered:
            if verbose:
                print(f"[PostFilter] {ctf}: {len(results)} → {len(filtered)} chunks")
            results = filtered
        else:
            # Soft fallback: expand search without chapter anchor, post-filter again
            if verbose:
                print(f"[PostFilter] No {ctf} chunks in anchored results — expanding globally")
            filt = build_filter(subject=subject)
            expanded = qdrant_search(vec, filt, PASS2_TOP_K * 2, collection, threshold)
            filtered = apply_content_type_postfilter(expanded, ctf)
            if filtered:
                results = filtered
                if verbose:
                    print(f"[PostFilter] Found {len(results)} {ctf} chunks globally")
            # else: fall back to original results (better than nothing)

    # ── Step 4: chunk type priority (only when no explicit content_type_filter) ──
    ordered = filter_by_chunk_type(results, intent)
    if verbose:
        print(f"[Types]  {dict(Counter(r['chunk_type'] for r in ordered))}  "
              f"sub_intent={intent.sub_intent}  ctf={ctf}")

    # ── Step 5: rerank ────────────────────────────────────────────────────────
    return rerank(query, ordered, top_k), anchor_chapters


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:      str,
    subject:    str   = None,
    chapter:    str   = None,
    top_k:      int   = FINAL_TOP_K,
    collection: str   = COLLECTION,
    threshold:  float = SCORE_THRESHOLD,
    verbose:    bool  = False,
) -> tuple[list[dict], str, list[dict]]:
    """
    Returns (results, mode, anchor_chapters).
      mode:            "reference" | "semantic"
      anchor_chapters: chapter candidates used for anchoring
    """
    intent = detect_intent(query)

    # Safety net: if SLM missed content_type_filter, try regex
    if intent.content_type_filter is None and intent.mode == "semantic":
        regex_ctf = regex_content_type(query)
        if regex_ctf:
            if verbose:
                print(f"[Regex Fallback] content_type_filter={regex_ctf} (SLM missed it)")
            intent.content_type_filter = regex_ctf

    # CLI --chapter overrides chapter parsed from query text
    if chapter:
        intent.chapter_hint = str(chapter)

    if verbose:
        print(f"\n[Intent]  mode={intent.mode}  sub_intent={intent.sub_intent}  "
              f"item={intent.item_type} #{intent.item_number}  "
              f"chapter={intent.chapter_hint}  "
              f"conceptual={intent.is_conceptual}  "
              f"keywords={intent.keywords}  "
              f"content_type_filter={intent.content_type_filter}")

    # ── REFERENCE — hard stop, no fallback ───────────────────────────────────
    if intent.mode == "reference":
        results = reference_lookup(intent, subject, collection, verbose)
        if not results and verbose:
            print("[Reference] Nothing found — returning empty (no semantic fallback)")
        return results, "reference", []

    # ── SEMANTIC ──────────────────────────────────────────────────────────────
    results, anchor_chapters = semantic_search(
        query, intent, subject, top_k, collection, threshold, verbose,
    )
    return results, "semantic", anchor_chapters


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

TYPE_ICON = {"theory": "📖", "activity": "🔬", "exercise": "📝"}
CTF_LABEL = {
    "theory":        "📖 Theory",
    "activity":      "🔬 Activity",
    "exercise":      "📝 Exercise",
    "mcq":           "🔘 MCQ",
    "fill_in_blank": "✏️  Fill-in-Blank",
}


def display_results(results: list[dict], query: str, mode: str,
                    anchor_chapters: list[dict] = None,
                    content_type_filter: str = None):
    if not results:
        msg = ("Reference not found. No fallback applied."
               if mode == "reference"
               else "No results found. Try rephrasing your query.")
        (console.print(f"\n[yellow]{msg}[/yellow]\n") if HAS_RICH else print(f"\n{msg}\n"))
        return

    # Ambiguity banner
    if anchor_chapters and len(anchor_chapters) > 1:
        if HAS_RICH:
            lines = "\n".join(
                f"  [{c['label'].upper()}] Chapter {c['chapter_number']}: {c.get('chapter_title','')}"
                for c in anchor_chapters
            )
            console.print(
                f"\n[bold yellow]⚠ Ambiguous — multiple chapters matched:[/bold yellow]\n"
                f"{lines}\n"
                f"[dim]Results shown from PRIMARY chapter. Add --chapter N to pin.[/dim]\n"
            )
        else:
            print("\n⚠ Ambiguous — multiple chapters matched:")
            for c in anchor_chapters:
                print(f"  [{c['label'].upper()}] Chapter {c['chapter_number']}: {c.get('chapter_title','')}")
            print("Results from PRIMARY chapter. Add --chapter N to pin.\n")

    if HAS_RICH:
        ctf_str = f"  [bold magenta]{CTF_LABEL.get(content_type_filter, '')}[/bold magenta]" if content_type_filter else ""
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}  [dim][{mode}][/dim]{ctf_str}")

        if mode == "reference":
            console.print(
                f"\n[bold green]📋 Complete content — {len(results)} chunk(s)[/bold green]\n"
            )
            meta  = results[0]
            title = (f"ch={meta['chapter_number']}  "
                     f"{TYPE_ICON.get(meta['chunk_type'],'?')} {meta['chunk_type']}  "
                     f"|  {meta['heading_path']}")
            console.print(Panel("\n\n".join(r["text"] for r in results),
                                 title=title, border_style="green"))
        else:
            console.print(f"[bold]Top {len(results)} result(s)[/bold]\n")
            for i, r in enumerate(results, 1):
                ctype = r.get("chunk_type", "theory")
                flags = [f"{TYPE_ICON.get(ctype,'?')} {ctype}"]
                if r.get("has_equation"): flags.append("📐 eq")
                if r.get("has_table"):    flags.append("📊 tbl")
                title = (
                    f"[{i}] ch={r.get('chapter_number','?')}  "
                    f"rerank={r.get('rerank_score',0):.3f}  "
                    f"cosine={r.get('score',0):.3f}  "
                    f"{'  '.join(flags)}  |  {r.get('heading_path','')}"
                )
                preview = r["text"][:600].replace("\n", " ")
                if len(r["text"]) > 600:
                    preview += "…"
                console.print(Panel(preview, title=title, border_style="dim"))
    else:
        ctf_str = f" [{CTF_LABEL.get(content_type_filter, '')}]" if content_type_filter else ""
        print(f"\nQuery: {query}  [{mode}]{ctf_str}")
        if mode == "reference":
            print(f"Complete content ({len(results)} chunks):\n")
            for r in results:
                print(r["text"]); print()
        else:
            for i, r in enumerate(results, 1):
                print(f"[{i}] ch={r.get('chapter_number','?')} | "
                      f"rerank={r.get('rerank_score',0):.3f} | "
                      f"type={r.get('chunk_type','?')} | {r.get('heading_path','')}")
                print(f"     {r['text'][:400]}...")
                print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(args):
    print("\n=== EDU-RAG v7 — SLM Intent · Content-Type Entity Filtering ===")
    print("Type your query, 'verbose' to toggle debug, 'exit' to quit.\n")
    print("Examples:")
    print("  give me MCQs on digestion")
    print("  fill in the blanks chapter 2")
    print("  exercises on Newton laws")
    print("  explain photosynthesis")
    print("  activity 3  (reference lookup)\n")
    verbose = False

    while True:
        try:
            raw = input("Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            break
        if raw.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose: {'ON' if verbose else 'OFF'}")
            continue

        results, mode, anchor_chapters = retrieve(
            query      = raw,
            subject    = args.subject or None,
            chapter    = args.chapter or None,
            top_k      = args.top_k,
            collection = args.collection,
            threshold  = args.threshold,
            verbose    = verbose,
        )
        # Pass content_type_filter for display label
        # (detect again or parse from intent — simplest: re-detect)
        ctf = regex_content_type(raw)
        display_results(results, raw, mode, anchor_chapters, content_type_filter=ctf)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EDU-RAG v7 — SLM Intent + Content-Type Entity Filtering")
    ap.add_argument("--query",      type=str,   default=None)
    ap.add_argument("--subject",    type=str,   default=None,
                    help="Subject filter: Physics, Biology, Maths_sem_1 ...")
    ap.add_argument("--chapter",    type=str,   default=None,
                    help="Pin to a specific chapter number (e.g. 1, 2)")
    ap.add_argument("--top_k",      type=int,   default=FINAL_TOP_K)
    ap.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD)
    ap.add_argument("--collection", type=str,   default=COLLECTION)
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    get_bi_encoder()
    get_cross_encoder()
    get_client()

    if args.query:
        results, mode, anchor_chapters = retrieve(
            query      = args.query,
            subject    = args.subject,
            chapter    = args.chapter,
            top_k      = args.top_k,
            collection = args.collection,
            threshold  = args.threshold,
            verbose    = args.verbose,
        )
        # detect ctf for display
        ctf = regex_content_type(args.query)
        display_results(results, args.query, mode, anchor_chapters, content_type_filter=ctf)
    else:
        interactive_loop(args)


if __name__ == "__main__":
    main()