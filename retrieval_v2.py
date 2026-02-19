"""
retrieval.py  (v2 — High Precision)
=====================================
Two-pass retrieval with:
  1. Chunk-type classification  → exercise chunks suppressed for theory queries
  2. Chapter anchoring          → majority-vote on top-3 to lock chapter
  3. Cross-encoder re-ranking   → precise ordering within the anchored chapter

Install:
    pip install sentence-transformers qdrant-client rich

Run:
    python retrieval.py --query "explain photosynthesis" --subject Biology
    python retrieval.py --query "newton third law"       --subject Physics
    python retrieval.py   # interactive mode
"""

import re
import sys
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

COLLECTION  = "edu_chunks"
QDRANT_URL  = "http://localhost:6333"

# Pass-1: broad search to detect chapter (cast wide net)
PASS1_TOP_K = 10

# Pass-2: narrowed to anchored chapter, then re-ranked
PASS2_TOP_K = 20   # retrieve more, reranker will trim to FINAL_TOP_K
FINAL_TOP_K = 5    # what we show the user

# Chapter vote: look at this many top results for majority vote
VOTE_WINDOW = 3

# If spread across this many distinct chapters in VOTE_WINDOW → skip anchoring
ANCHOR_SKIP_THRESHOLD = VOTE_WINDOW   # all different → ambiguous query

# Score threshold for pass-1 (lower = broader)
SCORE_THRESHOLD = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# CHUNK TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that strongly indicate an exercise/question chunk
EXERCISE_PATTERNS = [
    r"^\s*\d+[\.\)]\s",                      # numbered list  "1. ..."
    r"answer\s+the\s+following",
    r"fill\s+in\s+the\s+blank",
    r"choose\s+the\s+(correct|best)",
    r"match\s+the\s+following",
    r"true\s+or\s+false",
    r"short\s+answer",
    r"long\s+answer",
    r"give\s+reason",
    r"define\s+the\s+following",
    r"explain\s+with\s+example",
    r"what\s+do\s+you\s+mean\s+by",
    r"^\s*[a-e][\.\)]\s",                    # option list  "a) ..."
    r"exercise\s+\d",
    r"questions\s+and\s+answers",
    r"^\s*Q\d+[\.\)]",                        # Q1. Q2.
    r"let['']?s\s+(recall|practice|check)",
    r"solve\s+the\s+following",
    r"calculate\s+the",
    r"find\s+the\s+(value|area|volume|mass|speed|force|current)",
]

EXERCISE_RE = re.compile("|".join(EXERCISE_PATTERNS), re.IGNORECASE | re.MULTILINE)

def classify_chunk(text: str, has_activity: bool = False) -> str:
    """
    Returns 'exercise' or 'theory'.
    Simple heuristic — no model needed.
    """
    if has_activity:
        return "exercise"

    # Count exercise-pattern matches
    matches = len(EXERCISE_RE.findall(text))

    # Ratio of question marks to total sentences
    sentences    = max(text.count(".") + text.count("?"), 1)
    q_mark_ratio = text.count("?") / sentences

    # Heuristic: if many exercise patterns OR heavy on question marks → exercise
    if matches >= 3 or q_mark_ratio > 0.4:
        return "exercise"

    return "theory"


def is_query_conceptual(query: str) -> bool:
    """
    Decide if the user is asking a conceptual/theory question
    vs requesting problem solving / calculation.
    Used to decide whether to suppress exercise chunks.
    """
    CALCULATION_SIGNALS = [
        r"\bcalculate\b", r"\bsolve\b", r"\bfind\s+the\b",
        r"\bcompute\b",   r"\bhow\s+much\b", r"\bhow\s+many\b",
        r"\bwhat\s+is\s+the\s+(value|answer|result)\b",
    ]
    for pattern in CALCULATION_SIGNALS:
        if re.search(pattern, query, re.IGNORECASE):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHES
# ─────────────────────────────────────────────────────────────────────────────

_bi_encoder    = None
_cross_encoder = None
_client        = None


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


def get_client(url=QDRANT_URL):
    global _client
    if _client is None:
        _client = QdrantClient(url=url)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT FILTER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_filter(subject: str = None, chapter_number: str = None) -> Filter | None:
    must = []
    if subject:
        must.append(FieldCondition(key="subject",
                                   match=MatchValue(value=subject)))
    if chapter_number:
        must.append(FieldCondition(key="chapter_number",
                                   match=MatchValue(value=str(chapter_number))))
    return Filter(must=must) if must else None


# ─────────────────────────────────────────────────────────────────────────────
# QDRANT SEARCH (raw)
# ─────────────────────────────────────────────────────────────────────────────

def qdrant_search(query_vec: list, filt, top_k: int,
                  collection: str, threshold: float) -> list[dict]:
    client  = get_client()
    results = client.search(
        collection_name=collection,
        query_vector=query_vec,
        query_filter=filt,
        limit=top_k,
        score_threshold=threshold,
        with_payload=True,
    )
    return [
        {
            "score":          round(r.score, 4),
            "chunk_id":       r.payload.get("chunk_id", ""),
            "subject":        r.payload.get("subject", ""),
            "chapter_number": r.payload.get("chapter_number", ""),
            "chapter_title":  r.payload.get("chapter_title", ""),
            "section_title":  r.payload.get("section_title", ""),
            "heading_path":   r.payload.get("heading_path", ""),
            "has_equation":   r.payload.get("has_equation", False),
            "has_table":      r.payload.get("has_table", False),
            "has_activity":   r.payload.get("has_activity", False),
            "text":           r.payload.get("text", ""),
        }
        for r in results
    ]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CHAPTER ANCHOR via MAJORITY VOTE
# ─────────────────────────────────────────────────────────────────────────────

def detect_anchor_chapter(pass1_results: list[dict]) -> str | None:
    """
    Look at the top VOTE_WINDOW results.
    If a majority (or plurality) agree on a chapter_number → return it.
    If all are different (ambiguous query) → return None (skip anchoring).

    Example:
        top-3 chapters: [3, 5, 3]  → anchor = "3"  (appears 2/3 times)
        top-3 chapters: [3, 5, 7]  → anchor = None  (all different)
        top-3 chapters: [3, 3, 3]  → anchor = "3"  (unanimous)
    """
    if not pass1_results:
        return None

    window  = pass1_results[:VOTE_WINDOW]
    chapters = [r["chapter_number"] for r in window if r["chapter_number"]]

    if not chapters:
        return None

    vote_counts  = Counter(chapters)
    top_chapter, top_count = vote_counts.most_common(1)[0]

    # Skip anchoring if every result is from a different chapter
    distinct_chapters = len(vote_counts)
    if distinct_chapters >= ANCHOR_SKIP_THRESHOLD and top_count == 1:
        return None   # genuinely cross-chapter query

    return top_chapter


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CHUNK TYPE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def filter_by_chunk_type(results: list[dict], suppress_exercises: bool) -> list[dict]:
    """
    Tag each result with its chunk_type.
    If suppress_exercises=True, move exercise chunks to the back
    (don't discard — user may want them if nothing else matches).
    """
    theory   = []
    exercise = []

    for r in results:
        ctype = classify_chunk(r["text"], r.get("has_activity", False))
        r["chunk_type"] = ctype
        if ctype == "exercise":
            exercise.append(r)
        else:
            theory.append(r)

    if suppress_exercises:
        # Theory first, exercises as fallback
        return theory + exercise
    else:
        return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CROSS-ENCODER RE-RANKING
# ─────────────────────────────────────────────────────────────────────────────

def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """
    Cross-encoder reads query+chunk_text together and produces
    a more accurate relevance score than cosine similarity.
    """
    if not results:
        return []

    cross_encoder = get_cross_encoder()
    pairs  = [(query, r["text"]) for r in results]
    scores = cross_encoder.predict(pairs)

    for r, score in zip(results, scores):
        r["rerank_score"] = round(float(score), 4)

    ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RETRIEVAL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query:      str,
    subject:    str  = None,
    top_k:      int  = FINAL_TOP_K,
    collection: str  = COLLECTION,
    threshold:  float = SCORE_THRESHOLD,
    verbose:    bool  = False,
) -> list[dict]:
    """
    Full two-pass retrieval with chapter anchoring and re-ranking.

    Pass 1:  broad search (subject only)   → detect anchor chapter
    Pass 2:  narrow search (subject+chapter) → retrieve candidates
    Filter:  suppress exercise chunks for conceptual queries
    Rerank:  cross-encoder scores candidates precisely
    Return:  top_k results
    """

    bi_encoder = get_bi_encoder()
    query_vec  = bi_encoder.encode(
        f"search_query: {query}",
        normalize_embeddings=True
    ).tolist()

    conceptual = is_query_conceptual(query)

    # ── PASS 1: Broad search to detect chapter ────────────────────────────────
    pass1_filter  = build_filter(subject=subject)
    pass1_results = qdrant_search(
        query_vec, pass1_filter, PASS1_TOP_K, collection, threshold
    )

    if verbose:
        print(f"\n[Pass-1] Got {len(pass1_results)} results")
        for r in pass1_results[:VOTE_WINDOW]:
            print(f"  ch={r['chapter_number']:>3}  score={r['score']}  {r['heading_path'][:60]}")

    # ── CHAPTER VOTE ──────────────────────────────────────────────────────────
    anchor_chapter = detect_anchor_chapter(pass1_results)

    if verbose:
        if anchor_chapter:
            print(f"\n[Chapter Vote] Anchored to chapter: {anchor_chapter}")
        else:
            print(f"\n[Chapter Vote] Ambiguous — skipping chapter anchor")

    # ── PASS 2: Narrow search within anchored chapter ─────────────────────────
    pass2_filter  = build_filter(subject=subject, chapter_number=anchor_chapter)
    pass2_results = qdrant_search(
        query_vec, pass2_filter, PASS2_TOP_K, collection, threshold
    )

    if verbose:
        print(f"\n[Pass-2] Got {len(pass2_results)} results from chapter {anchor_chapter}")

    if not pass2_results:
        # Fallback: use pass-1 results if pass-2 is empty
        pass2_results = pass1_results

    # ── CHUNK TYPE FILTER ─────────────────────────────────────────────────────
    typed_results = filter_by_chunk_type(pass2_results, suppress_exercises=conceptual)

    if verbose:
        theory_count   = sum(1 for r in typed_results if r.get("chunk_type") == "theory")
        exercise_count = sum(1 for r in typed_results if r.get("chunk_type") == "exercise")
        print(f"\n[Chunk Types] theory={theory_count}  exercise={exercise_count}"
              f"  suppress_exercises={conceptual}")

    # ── CROSS-ENCODER RE-RANK ─────────────────────────────────────────────────
    final_results = rerank(query, typed_results, top_k)

    return final_results


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def display_results(results: list[dict], query: str):
    if not results:
        if HAS_RICH:
            console.print("\n[yellow]No results found. Try a broader query.[/yellow]\n")
        else:
            print("\nNo results found.\n")
        return

    if HAS_RICH:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        console.print(f"[bold]Top {len(results)} result(s)[/bold]\n")

        for i, r in enumerate(results, 1):
            flags = []
            if r.get("chunk_type") == "exercise": flags.append("📝 exercise")
            else:                                  flags.append("📖 theory")
            if r.get("has_equation"):              flags.append("📐 eq")
            if r.get("has_table"):                 flags.append("📊 table")

            bi_score     = r.get("score", 0)
            rerank_score = r.get("rerank_score", 0)
            chapter      = r.get("chapter_number", "?")

            title = (
                f"[{i}] ch={chapter}  "
                f"rerank={rerank_score:.3f}  cosine={bi_score:.3f}  "
                f"{'  '.join(flags)}  |  {r['heading_path']}"
            )
            preview = r["text"][:600].replace("\n", " ")
            if len(r["text"]) > 600:
                preview += "…"
            console.print(Panel(preview, title=title, border_style="dim"))
    else:
        print(f"\nQuery: {query}")
        print(f"Top {len(results)} result(s)\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] ch={r.get('chapter_number','?')} | "
                  f"rerank={r.get('rerank_score',0):.3f} | "
                  f"cosine={r.get('score',0):.3f} | "
                  f"type={r.get('chunk_type','?')} | "
                  f"{r.get('heading_path','')}")
            print(f"     {r['text'][:400]}...")
            print("-" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def interactive_loop(args):
    print("\n=== EDU-RAG v2 (interactive) ===")
    print("Commands: 'exit' to quit, 'verbose' to toggle debug output\n")

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
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        results = retrieve(
            query=raw,
            subject=args.subject or None,
            top_k=args.top_k,
            collection=args.collection,
            threshold=args.threshold,
            verbose=verbose,
        )
        display_results(results, raw)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="High-precision EDU-RAG retrieval")
    ap.add_argument("--query",      type=str,   default=None,
                    help="Query string (omit for interactive mode)")
    ap.add_argument("--subject",    type=str,   default=None,
                    help="Subject filter: Physics, Biology, Maths_sem_1, ...")
    ap.add_argument("--top_k",      type=int,   default=FINAL_TOP_K)
    ap.add_argument("--threshold",  type=float, default=SCORE_THRESHOLD)
    ap.add_argument("--collection", type=str,   default=COLLECTION)
    ap.add_argument("--verbose",    action="store_true",
                    help="Print pass-1, vote, and chunk-type debug info")
    args = ap.parse_args()

    # Pre-load both models once
    get_bi_encoder()
    get_cross_encoder()
    get_client()

    if args.query:
        results = retrieve(
            query=args.query,
            subject=args.subject,
            top_k=args.top_k,
            collection=args.collection,
            threshold=args.threshold,
            verbose=args.verbose,
        )
        display_results(results, args.query)
    else:
        interactive_loop(args)


if __name__ == "__main__":
    main()
