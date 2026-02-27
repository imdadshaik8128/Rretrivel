"""
api.py — FastAPI Backend for RAG Pipeline
==========================================
Wraps: query_parser_v2 → parse_sanitizer → Retriever → Generator

Run:
    pip install fastapi uvicorn
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /query          — full pipeline: parse → retrieve → generate
    GET  /health         — liveness check
    GET  /subjects       — list available subjects

CORS is open for localhost:3000 / 5173 (React dev servers).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Pipeline imports ────────────────────────────────────────────────────────
from query_parser_v2 import parse_query_with_slm
from parse_sanitizer import sanitize
from retriever import Retriever, AmbiguityError
from generator import Generator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Textbook RAG API",
    description="Deterministic textbook Q&A pipeline: parse → retrieve → generate",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline once at startup (heavy — models load here) ─────────────────
log.info("Loading Retriever (bi-encoder + cross-encoder)…")
retriever = Retriever()
log.info("Loading Generator…")
generator = Generator()
log.info("✓ Pipeline ready.")

# ── Subjects (must match all_chunks.json subject field exactly) ───────────────
AVAILABLE_SUBJECTS = [
    "Biology",
    "Economics",
    "Geography",
    "History",
    "Maths_sem_1",
    "Maths_sem_2",
    "Physics",
    "Social_political",
]


# ══════════════════════════════════════════════════════════════════════════════
# Request / Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str                        # raw user query text
    subject: str                      # session-locked subject (UI always sends this)
    session_id: Optional[str] = None  # for logging/tracing (not used in retrieval)


class CitationOut(BaseModel):
    chunk_id: str
    chapter_number: str
    chapter_title: str
    section_title: str
    chunk_type: str
    activity_number: str


class QueryResponse(BaseModel):
    # Answer fields
    answer_type: str                          # "reference" | "concept"
    spoken_answer: str                        # TTS-safe plain text
    display_answer_markdown: str              # rich markdown for UI
    citations: list[CitationOut]
    confidence: float                         # 0.0 – 1.0
    confidence_pct: int                       # integer 0–100 for UI bars
    low_confidence_warning: Optional[str]
    filter_path: str                          # provenance string

    # Debug / timing
    parsed_query: dict                        # what the SLM extracted
    retrieval_ms: int
    generation_ms: int
    total_ms: int


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "pipeline": "ready"}


@app.get("/subjects")
def subjects():
    return {"subjects": AVAILABLE_SUBJECTS}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Full RAG pipeline:
      1. SLM parse  (query_parser_v2)
      2. Sanitize   (parse_sanitizer — strips hallucinated fields)
      3. Inject subject from UI session (hard override)
      4. Retrieve   (Retriever — deterministic metadata filter + semantic rerank)
      5. Generate   (Generator → Ollama LLM)
    """
    t_start = time.perf_counter()
    log.info("Query: %r  Subject: %s  Session: %s", req.query, req.subject, req.session_id)

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    try:
        raw_parse = parse_query_with_slm(req.query)
        parsed_dict: dict = json.loads(raw_parse)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Query parser error: {e}")

    # ── Step 2: Sanitize (strip hallucinated metadata) ───────────────────────
    parsed_dict = sanitize(parsed_dict, req.query)

    # ── Step 3: Inject subject — UI session always wins ──────────────────────
    parsed_dict["subject"] = req.subject
    log.info("Parsed (sanitized + injected): %s", json.dumps(parsed_dict))

    # ── Step 4: Retrieve ─────────────────────────────────────────────────────
    t_ret = time.perf_counter()
    chunks, ret_err = retriever.retrieve_safe(parsed_dict, req.query)
    retrieval_ms = int((time.perf_counter() - t_ret) * 1000)

    if ret_err:
        # Surface AmbiguityError or other retrieval failures to the UI
        raise HTTPException(
            status_code=404,
            detail=f"Retrieval error: {ret_err}"
        )

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. Try rephrasing your question."
        )

    log.info("Retrieved %d chunks in %dms", len(chunks), retrieval_ms)

    # ── Step 5: Generate ─────────────────────────────────────────────────────
    t_gen = time.perf_counter()
    answer, gen_err = generator.generate_safe(chunks, req.query)
    generation_ms = int((time.perf_counter() - t_gen) * 1000)

    if gen_err:
        raise HTTPException(
            status_code=503,
            detail=f"Generation error: {gen_err}. Make sure Ollama is running: ollama serve"
        )

    total_ms = int((time.perf_counter() - t_start) * 1000)
    log.info(
        "Done — type=%s  confidence=%.2f  ret=%dms  gen=%dms  total=%dms",
        answer.answer_type, answer.confidence, retrieval_ms, generation_ms, total_ms,
    )

    # ── Build response ────────────────────────────────────────────────────────
    return QueryResponse(
        answer_type              = answer.answer_type,
        spoken_answer            = answer.spoken_answer,
        display_answer_markdown  = answer.display_answer_markdown,
        citations                = [
            CitationOut(
                chunk_id        = c.chunk_id,
                chapter_number  = c.chapter_number,
                chapter_title   = c.chapter_title,
                section_title   = c.section_title,
                chunk_type      = c.chunk_type,
                activity_number = c.activity_number,
            )
            for c in answer.citations
        ],
        confidence               = answer.confidence,
        confidence_pct           = int(answer.confidence * 100),
        low_confidence_warning   = answer.low_confidence_warning,
        filter_path              = answer.filter_path,
        parsed_query             = parsed_dict,
        retrieval_ms             = retrieval_ms,
        generation_ms            = generation_ms,
        total_ms                 = total_ms,
    )
