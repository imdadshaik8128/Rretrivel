"""
retrieval_v2.py  (v3 — Intent-Aware)
======================================
Two retrieval MODES:

  MODE 1 — Reference Lookup  (direct metadata-driven)
    Triggered when query names a specific item:
      "explain activity 3"  /  "show example 2.4"  /  "exercise 5 question 2"
    → Bypass semantic search entirely
    → Filter Qdrant by chapter + chunk_type/activity_number (no vector search)
    → Fetch ALL chunks belonging to that item sorted by chunk_index
    → Returns COMPLETE content, not just a fragment

  MODE 2 — Semantic Search   (two-pass + chapter anchor + rerank)
    Triggered for all conceptual / open queries.
    → Pass 1: broad subject search → chapter majority vote
    → Pass 2: narrow chapter search → chunk type filter → cross-encoder rerank

chunk_type values (ground-truth from chunker.py, NOT computed at runtime):
    "theory"   — explanations, definitions, concept text
    "activity" — Activity 1, Activity 3.2 (hands-on, embedded in theory flow)
    "exercise" — question banks, fill in blanks, MCQs, improve your learning

Install:
    pip install sentence-transformers qdrant-client rich

Run:
    python retrieval_v2.py --query "explain photosynthesis"   --subject Biology
    python retrieval_v2.py --query "explain activity 3"       --subject Biology --chapter 2
    python retrieval_v2.py --query "what is newton third law" --subject Physics
    python retrieval_v2.py   # interactive mode
"""

import re
import argparse
from collections import Counter

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

PASS1_TOP_K     = 10
PASS2_TOP_K     = 20
FINAL_TOP_K     = 5
VOTE_WINDOW     = 3
SCORE_THRESHOLD = 0.25


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Patterns: (regex, item_type)
# Each regex must capture the item number in group 1
REFERENCE_PATTERNS = [
    (r'\bactivity\s*([\d.]+)',          "activity"),
    (r'\bexample\s*([\d.]+)',           "example"),
    (r'\bexercise\s+(\d[\d.]*)',        "exercise"),   # "exercise 2", not just "exercise"
    (r'\b(?:figure|fig)\s*([\d.]+)',    "figure"),
]
_REF_COMPILED = [(re.compile(p, re.IGNORECASE), t) for p, t in REFERENCE_PATTERNS]

CHAPTER_IN_QUERY_RE = re.compile(r'\bchapter\s*(\d+)', re.IGNORECASE)

_CALC_RE = re.compile(
    r'\bcalculate\b|\bsolve\b|\bfind\s+the\b|\bcompute\b'
    r'|\bhow\s+much\b|\bhow\s+many\b'
    r'|\bwhat\s+is\s+the\s+(value|answer|result)\b',
    re.IGNORECASE
)


class QueryIntent:
    def __init__(self):
        self.mode          = "semantic"   # "reference" | "semantic"
        self.item_type     = None         # "activity" | "example" | "exercise" | "figure"
        self.item_number   = None         # "3" or "3.2"
        self.chapter_hint  = None         # explicitly stated chapter number
        self.is_conceptual = True         # False → don't suppress exercise chunks


def detect_intent(query: str) -> QueryIntent:
    intent = QueryIntent()

    for pattern, item_type in _REF_COMPILED:
        m = pattern.search(query)
        if m:
            intent.mode        = "reference"
            intent.item_type   = item_type
            intent.item_number = m.group(1)
            break

    m = CHAPTER_IN_QUERY_RE.search(query)
    if m:
        intent.chapter_hint = m.group(1)

    if _CALC_RE.search(query):
        intent.is_conceptual = False

    return intent


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
    subject:        str = None,
    chapter_number: str = None,
    chunk_type:     str = None,
    section_type:   str = None,
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
    if section_type:
        must.append(FieldCondition(key="section_type",
                                   match=MatchValue(value=section_type)))
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
        "section_type":    payload.get("section_type", ""),
        "activity_number": payload.get("activity_number", ""),
        "text":            payload.get("text", ""),
    }


def qdrant_search(query_vec: list, filt, top_k: int,
                  collection: str, threshold: float) -> list[dict]:
    client = get_client()
    hits   = client.search(
        collection_name=collection,
        query_vector=query_vec,
        query_filter=filt,
        limit=top_k,
        score_threshold=threshold,
        with_payload=True,
    )
    return [payload_to_result(h.payload, h.score) for h in hits]


def qdrant_scroll_all(filt, collection: str) -> list[dict]:
    """
    Fetch ALL points matching a filter without vector search.
    Used by reference lookup to retrieve complete content.
    """
    client  = get_client()
    results = []
    offset  = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=filt,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        results.extend([payload_to_result(p.payload) for p in batch])
        if next_offset is None:
            break
        offset = next_offset

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — REFERENCE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────

def reference_lookup(
    intent:     QueryIntent,
    subject:    str,
    collection: str,
    verbose:    bool = False,
) -> list[dict]:
    """
    Direct metadata lookup — no vector search.

    For activities: uses activity_number payload field (set by chunker.py).
    For examples/figures: scans heading_path string for the reference number.

    Returns all matching chunks sorted by chunk_index (complete, in order).
    Falls back to [] if nothing found — caller will fall back to semantic search.
    """
    item_type   = intent.item_type
    item_number = intent.item_number
    chapter     = intent.chapter_hint

    if verbose:
        print(f"\n[Mode: Reference Lookup]")
        print(f"  item={item_type} #{item_number}  chapter_hint={chapter}")

    # ── Activities: use dedicated activity_number index ───────────────────────
    if item_type == "activity":
        filt = build_filter(
            subject=subject,
            chapter_number=chapter,
            chunk_type="activity",
        )
        candidates = qdrant_scroll_all(filt, collection)

        # Try exact activity_number match first
        matched = [r for r in candidates if r["activity_number"] == item_number]

        # Fallback: activity number appears anywhere in heading_path
        if not matched and item_number:
            search = f"activity {item_number}".lower()
            matched = [r for r in candidates
                       if search in r["heading_path"].lower()]

        if verbose:
            print(f"  Activity chunks found: {len(matched)}")

        if matched:
            matched.sort(key=lambda x: x["chunk_index"])
            return matched

    # ── Examples / figures / named exercises: scan heading_path ──────────────
    filt = build_filter(subject=subject, chapter_number=chapter)
    all_chunks = qdrant_scroll_all(filt, collection)

    search_term = f"{item_type} {item_number}".lower()
    matched = [
        r for r in all_chunks
        if search_term in r["heading_path"].lower()
        or search_term in r["text"][:200].lower()
    ]

    if verbose:
        print(f"  Heading/text match '{search_term}': {len(matched)} chunks")

    if matched:
        matched.sort(key=lambda x: x["chunk_index"])
        return matched

    # Nothing found — caller handles fallback
    return []


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — SEMANTIC SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def detect_anchor_chapter(pass1: list[dict]) -> str | None:
    """
    Majority vote across top VOTE_WINDOW results.
    Returns chapter_number string if majority agree, else None.

    [3, 5, 3]  → "3"   (2/3 agree)
    [3, 5, 7]  → None  (all different — cross-chapter query)
    """
    if not pass1:
        return None
    chapters = [r["chapter_number"] for r in pass1[:VOTE_WINDOW] if r["chapter_number"]]
    if not chapters:
        return None
    vote_counts            = Counter(chapters)
    top_chapter, top_count = vote_counts.most_common(1)[0]
    # If every result is a different chapter → ambiguous
    if len(vote_counts) >= VOTE_WINDOW and top_count == 1:
        return None
    return top_chapter


def filter_by_chunk_type(results: list[dict], is_conceptual: bool) -> list[dict]:
    """
    Conceptual queries  → theory + activity first, exercises pushed to back.
    Calculation queries → show everything (exercises contain relevant problems).
    Activities are NEVER suppressed for conceptual queries — they are content.
    """
    if not is_conceptual:
        return results
    priority   = [r for r in results if r["chunk_type"] in ("theory", "activity")]
    last_resort= [r for r in results if r["chunk_type"] == "exercise"]
    return priority + last_resort


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """Cross-encoder rescoring for precise relevance ordering."""
    if not results:
        return []
    ce     = get_cross_encoder()
    pairs  = [(query, r["text"]) for r in results]
    scores = ce.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = round(float(s), 4)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def semantic_search(
    query:      str,
    intent:     QueryIntent,
    subject:    str,
    top_k:      int,
    collection: str,
    threshold:  float,
    verbose:    bool,
) -> list[dict]:
    bi  = get_bi_encoder()
    vec = bi.encode(f"search_query: {query}", normalize_embeddings=True).tolist()

    # If user explicitly stated a chapter, use it; otherwise vote from pass-1
    anchor = intent.chapter_hint

    if not anchor:
        pass1 = qdrant_search(vec, build_filter(subject=subject),
                              PASS1_TOP_K, collection, threshold)
        if verbose:
            print(f"\n[Pass-1] {len(pass1)} results")
            for r in pass1[:VOTE_WINDOW]:
                print(f"  ch={r['chapter_number']:>3}  "
                      f"score={r['score']}  {r['heading_path'][:60]}")
        anchor = detect_anchor_chapter(pass1)
        if verbose:
            print(f"[Vote]   anchor chapter → {anchor or 'None (ambiguous)'}")
    else:
        if verbose:
            print(f"\n[Chapter] Explicit chapter from query: {anchor}")

    # Pass 2: anchored search
    pass2 = qdrant_search(vec, build_filter(subject=subject, chapter_number=anchor),
                          PASS2_TOP_K, collection, threshold)

    if verbose:
        print(f"[Pass-2] {len(pass2)} candidates  chapter={anchor}")

    if not pass2:
        pass2 = qdrant_search(vec, build_filter(subject=subject),
                              PASS2_TOP_K, collection, threshold)

    typed = filter_by_chunk_type(pass2, intent.is_conceptual)

    if verbose:
        by_type = Counter(r["chunk_type"] for r in typed)
        print(f"[Types]  {dict(by_type)}  conceptual={intent.is_conceptual}")

    return rerank(query, typed, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:      str,
    subject:    str   = None,
    top_k:      int   = FINAL_TOP_K,
    collection: str   = COLLECTION,
    threshold:  float = SCORE_THRESHOLD,
    verbose:    bool  = False,
) -> tuple[list[dict], str]:
    """
    Returns (results, mode) where mode is "reference" or "semantic".
    """
    intent = detect_intent(query)

    if verbose:
        print(f"\n[Intent]  mode={intent.mode}  "
              f"item={intent.item_type} #{intent.item_number}  "
              f"chapter_hint={intent.chapter_hint}  "
              f"conceptual={intent.is_conceptual}")

    if intent.mode == "reference":
        results = reference_lookup(intent, subject, collection, verbose)
        if results:
            return results, "reference"
        if verbose:
            print("[Fallback] Reference lookup empty → semantic search")

    results = semantic_search(query, intent, subject, top_k, collection,
                              threshold, verbose)
    return results, "semantic"


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

TYPE_ICON = {"theory": "📖", "activity": "🔬", "exercise": "📝"}


def display_results(results: list[dict], query: str, mode: str):
    if not results:
        msg = "No results found. Try rephrasing your query."
        (console.print(f"\n[yellow]{msg}[/yellow]\n") if HAS_RICH
         else print(f"\n{msg}\n"))
        return

    if HAS_RICH:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}  "
                      f"[dim][mode={mode}][/dim]")

        if mode == "reference":
            # Show all chunks joined in reading order as one complete answer
            console.print(
                f"\n[bold green]📋 Complete content — "
                f"{len(results)} chunk(s) in reading order[/bold green]\n"
            )
            full_text = "\n\n".join(r["text"] for r in results)
            meta  = results[0]
            title = (f"ch={meta['chapter_number']}  "
                     f"{TYPE_ICON.get(meta['chunk_type'],'?')} {meta['chunk_type']}  "
                     f"|  {meta['heading_path']}")
            console.print(Panel(full_text, title=title, border_style="green"))

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
        print(f"\nQuery: {query}  [mode={mode}]")
        if mode == "reference":
            print(f"Complete content ({len(results)} chunks):\n")
            for r in results:
                print(r["text"])
                print()
        else:
            for i, r in enumerate(results, 1):
                print(f"[{i}] ch={r.get('chapter_number','?')} | "
                      f"rerank={r.get('rerank_score',0):.3f} | "
                      f"type={r.get('chunk_type','?')} | "
                      f"{r.get('heading_path','')}")
                print(f"     {r['text'][:400]}...")
                print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(args):
    print("\n=== EDU-RAG v3 — Intent-Aware Retrieval ===")
    print("Reference : 'explain activity 3'  |  'show example 2.4 chapter 1'")
    print("Semantic  : 'explain photosynthesis'  |  'what is newton third law'")
    print("Commands  : 'verbose' to toggle debug  |  'exit' to quit\n")
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

        results, mode = retrieve(
            query=raw,
            subject=args.subject or None,
            top_k=args.top_k,
            collection=args.collection,
            threshold=args.threshold,
            verbose=verbose,
        )
        display_results(results, raw, mode)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EDU-RAG v3 — Intent-Aware Retrieval")
    ap.add_argument("--query",      type=str,   default=None)
    ap.add_argument("--subject",    type=str,   default=None,
                    help="Subject filter: Physics, Biology, Maths_sem_1 ...")
    ap.add_argument("--top_k",      type=int,   default=FINAL_TOP_K)
    ap.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD)
    ap.add_argument("--collection", type=str,   default=COLLECTION)
    ap.add_argument("--verbose",    action="store_true")
    args = ap.parse_args()

    get_bi_encoder()
    get_cross_encoder()
    get_client()

    if args.query:
        results, mode = retrieve(
            query=args.query,
            subject=args.subject,
            top_k=args.top_k,
            collection=args.collection,
            threshold=args.threshold,
            verbose=args.verbose,
        )
        display_results(results, args.query, mode)
    else:
        interactive_loop(args)


if __name__ == "__main__":
    main()
