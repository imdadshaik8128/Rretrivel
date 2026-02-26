"""
retriever.py — Production RAG Retriever
========================================
Architecture:
  1. DETERMINISTIC METADATA FILTER  (hard rules, no guessing)
  2. SEMANTIC RERANK on the filtered candidate pool
  3. Returns top-K chunks with full provenance

Rules enforced:
  - Chapter selection is ALWAYS by chapter_number from parsed metadata (never embedding)
  - Subject selection is ALWAYS by subject field   (never embedding)
  - Reference queries (activity / exercise) NEVER fall back to semantic search alone
  - SLM never performs retrieval
  - Ambiguity is surfaced, not guessed
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Embedding backend (local, offline) ────────────────────────────────────────
# Uses sentence-transformers with a small, fast model.
# Install: pip install sentence-transformers
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNKS_PATH      = Path("all_chunks.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # ~80 MB, fully offline after first download
TOP_K            = 5
MIN_CANDIDATES   = 20   # minimum pool size before semantic ranking
SCORE_THRESHOLD  = 0.20 # cosine similarity floor — below this we warn


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedQuery:
    """Output contract of query_parser_v2.py (SLM)."""
    intent:           str            = "unknown"
    chunk_type:       str            = "unknown"
    chapter_number:   Optional[int]  = None
    chapter_name:     Optional[str]  = None
    activity_number:  Optional[int]  = None
    exercise_number:  Optional[float]= None
    topic:            str            = ""
    subject:          Optional[str]  = None   # optional; parser may not emit this

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedQuery":
        return cls(
            intent          = d.get("intent", "unknown"),
            chunk_type      = d.get("chunk_type", "unknown"),
            chapter_number  = _safe_int(d.get("chapter_number")),
            chapter_name    = d.get("chapter_name"),
            activity_number = _safe_int(d.get("activity_number")),
            exercise_number = _safe_float(d.get("exercise_number")),
            topic           = d.get("topic", ""),
            subject         = d.get("subject"),
        )


@dataclass
class RetrievedChunk:
    chunk_id:       str
    subject:        str
    chapter_number: str
    chapter_title:  str
    section_title:  str
    chunk_type:     str
    activity_number:str
    text:           str
    score:          float
    filter_path:    str   # human-readable explanation of why this chunk was selected


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def _normalise_chunk_type(ct: str) -> str:
    ct = ct.lower().strip()
    if ct in {"exercise", "exercises"}:
        return "exercise"
    if ct in {"activity", "activities"}:
        return "activity"
    if ct in {"theory", "content", "text"}:
        return "theory"
    return ct


# ══════════════════════════════════════════════════════════════════════════════
# ChunkStore — loads and indexes all chunks once at startup
# ══════════════════════════════════════════════════════════════════════════════

class ChunkStore:
    def __init__(self, chunks_path: Path, embed_model_name: str):
        log.info("Loading chunks from %s …", chunks_path)
        with open(chunks_path, encoding="utf-8") as f:
            raw: list[dict] = json.load(f)
        self.chunks: list[dict] = raw
        log.info("Loaded %d chunks.", len(self.chunks))

        log.info("Loading embedding model '%s' …", embed_model_name)
        self._model = SentenceTransformer(embed_model_name)

        log.info("Encoding all chunks (this runs once) …")
        texts = [c.get("text", "") for c in self.chunks]
        self._embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        log.info("Embeddings ready — shape %s", self._embeddings.shape)

        # Pre-build lookup indexes for O(1) metadata access
        self._by_chapter: dict[tuple[str, str], list[int]] = {}  # (subject, chapter) → indices
        self._by_subject: dict[str, list[int]] = {}

        for idx, c in enumerate(self.chunks):
            subj = c.get("subject", "").strip().lower()
            chap = str(c.get("chapter_number", "")).strip()

            self._by_subject.setdefault(subj, []).append(idx)
            self._by_chapter.setdefault((subj, chap), []).append(idx)

    # ------------------------------------------------------------------
    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )

    def subjects(self) -> list[str]:
        return list(self._by_subject.keys())

    def chapters_for_subject(self, subject: str) -> list[str]:
        subj = subject.strip().lower()
        return [k[1] for k in self._by_chapter if k[0] == subj]


# ══════════════════════════════════════════════════════════════════════════════
# Filter pipeline  — deterministic, rule-based
# ══════════════════════════════════════════════════════════════════════════════

class MetadataFilter:
    """
    Applies hard metadata constraints in order.
    Returns (candidate_indices, filter_description).
    Never guesses — raises AmbiguityError when intent is unresolvable.
    """

    def __init__(self, store: ChunkStore):
        self.store = store

    def filter(
        self,
        pq: ParsedQuery,
        query_text: str,
    ) -> tuple[list[int], str]:

        steps: list[str] = []
        candidates: list[int] = list(range(len(self.store.chunks)))

        # ── Step 1: Subject filter ────────────────────────────────────────────
        if pq.subject:
            subj_key = pq.subject.strip().lower()
            if subj_key not in self.store._by_subject:
                raise AmbiguityError(
                    f"Subject '{pq.subject}' not found. "
                    f"Available: {self.store.subjects()}"
                )
            candidates = self.store._by_subject[subj_key]
            steps.append(f"subject={pq.subject}")
        else:
            # No subject in parse → do not filter by subject
            # (subject leakage guard: we still won't cross-inject later)
            steps.append("subject=ALL (not specified)")

        # ── Step 2: Chapter filter ────────────────────────────────────────────
        # RULE: chapter_number from parser is authoritative. Embeddings NEVER pick chapter.
        if pq.chapter_number is not None:
            chap_str = str(pq.chapter_number)

            if pq.subject:
                subj_key = pq.subject.strip().lower()
                key = (subj_key, chap_str)
                if key not in self.store._by_chapter:
                    raise AmbiguityError(
                        f"Chapter {pq.chapter_number} not found in subject '{pq.subject}'. "
                        f"Available chapters: {self.store.chapters_for_subject(pq.subject)}"
                    )
                chapter_idxs = set(self.store._by_chapter[key])
            else:
                # No subject given — gather matching chapter across ALL subjects
                chapter_idxs = set()
                for (_, chap), idxs in self.store._by_chapter.items():
                    if chap == chap_str:
                        chapter_idxs.update(idxs)

            candidates = [i for i in candidates if i in chapter_idxs]
            steps.append(f"chapter={pq.chapter_number}")

            if not candidates:
                raise AmbiguityError(
                    f"No chunks found for chapter={pq.chapter_number}. "
                    "Check chapter number or subject."
                )

        # ── Step 3: chunk_type filter ─────────────────────────────────────────
        ct = _normalise_chunk_type(pq.chunk_type)
        if ct not in ("unknown", ""):
            candidates = [
                i for i in candidates
                if _normalise_chunk_type(
                    self.store.chunks[i].get("chunk_type", "")
                ) == ct
            ]
            steps.append(f"chunk_type={ct}")

        # ── Step 4: Activity / Exercise exact match ───────────────────────────
        # RULE: reference queries (activity/exercise number) NEVER fall back to semantic
        if pq.activity_number is not None:
            act_str = str(pq.activity_number)
            exact = [
                i for i in candidates
                if str(self.store.chunks[i].get("activity_number", "")).strip() == act_str
            ]
            if exact:
                candidates = exact
                steps.append(f"activity_number={pq.activity_number} [EXACT]")
            else:
                # Exact match failed — surface the ambiguity, don't silently fall back
                raise AmbiguityError(
                    f"Activity {pq.activity_number} not found "
                    f"in chapter={pq.chapter_number}. "
                    "Verify the activity number or chapter."
                )

        if pq.exercise_number is not None:
            ex_str = str(pq.exercise_number)
            exact = [
                i for i in candidates
                if str(self.store.chunks[i].get("activity_number", "")).strip() == ex_str
            ]
            if exact:
                candidates = exact
                steps.append(f"exercise_number={pq.exercise_number} [EXACT]")
            else:
                raise AmbiguityError(
                    f"Exercise {pq.exercise_number} not found "
                    f"in chapter={pq.chapter_number}. "
                    "Verify the exercise number or chapter."
                )

        filter_path = " → ".join(steps) if steps else "no-filter (open query)"
        return candidates, filter_path


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Ranker — runs ONLY within the filtered candidate pool
# ══════════════════════════════════════════════════════════════════════════════

class SemanticRanker:
    def __init__(self, store: ChunkStore):
        self.store = store

    def rank(
        self,
        query_vec: np.ndarray,
        candidate_indices: list[int],
        top_k: int = TOP_K,
    ) -> list[tuple[int, float]]:
        if not candidate_indices:
            return []

        scores = [
            (idx, _cosine(query_vec, self.store._embeddings[idx]))
            for idx in candidate_indices
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Public API: Retriever
# ══════════════════════════════════════════════════════════════════════════════

class AmbiguityError(Exception):
    """Raised when the query cannot be resolved deterministically."""

class Retriever:
    """
    Single entry point for the RAG pipeline.

    Usage:
        retriever = Retriever()                        # load once at startup
        results   = retriever.retrieve(parsed_query, raw_query_text)
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embed_model: str  = EMBED_MODEL_NAME,
        top_k: int        = TOP_K,
    ):
        self.top_k  = top_k
        self.store  = ChunkStore(chunks_path, embed_model)
        self.filter = MetadataFilter(self.store)
        self.ranker = SemanticRanker(self.store)

    # ------------------------------------------------------------------
    def retrieve(
        self,
        parsed_query: dict | ParsedQuery,
        raw_query_text: str,
    ) -> list[RetrievedChunk]:
        """
        Parameters
        ----------
        parsed_query   : dict output of query_parser_v2 (or ParsedQuery instance)
        raw_query_text : original user query string (used for embedding)

        Returns
        -------
        List of RetrievedChunk, ordered by relevance (best first).

        Raises
        ------
        AmbiguityError : if metadata constraints cannot be resolved deterministically.
        """
        if isinstance(parsed_query, dict):
            pq = ParsedQuery.from_dict(parsed_query)
        else:
            pq = parsed_query

        log.info("Parsed query: %s", pq)

        # 1. Deterministic metadata filter
        try:
            candidates, filter_path = self.filter.filter(pq, raw_query_text)
        except AmbiguityError:
            raise  # let caller handle / surface to user

        log.info(
            "Filter path: [%s] → %d candidates", filter_path, len(candidates)
        )

        # Warn if candidate pool is suspiciously small before semantic ranking
        if len(candidates) == 0:
            raise AmbiguityError(
                "Metadata filters produced 0 candidates. "
                "Relax constraints or check the query."
            )

        # 2. Semantic ranking within filtered pool
        query_vec = self.store.embed_query(raw_query_text)
        ranked    = self.ranker.rank(query_vec, candidates, top_k=self.top_k)

        # 3. Warn on low-confidence results
        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            if score < SCORE_THRESHOLD:
                log.warning(
                    "Low similarity score %.3f for chunk '%s'. "
                    "Consider rephrasing the query.",
                    score,
                    self.store.chunks[idx]["chunk_id"],
                )
            c = self.store.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id       = c.get("chunk_id", ""),
                    subject        = c.get("subject", ""),
                    chapter_number = str(c.get("chapter_number", "")),
                    chapter_title  = c.get("chapter_title", ""),
                    section_title  = c.get("section_title", ""),
                    chunk_type     = c.get("chunk_type", ""),
                    activity_number= str(c.get("activity_number", "")),
                    text           = c.get("text", ""),
                    score          = round(score, 4),
                    filter_path    = filter_path,
                )
            )

        return results

    # ------------------------------------------------------------------
    def retrieve_safe(
        self,
        parsed_query: dict | ParsedQuery,
        raw_query_text: str,
    ) -> tuple[list[RetrievedChunk], Optional[str]]:
        """
        Non-raising version. Returns (results, error_message).
        error_message is None on success.
        """
        try:
            return self.retrieve(parsed_query, raw_query_text), None
        except AmbiguityError as e:
            return [], str(e)


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json as _json
    from query_parser_v2 import parse_query_with_slm

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "explain activity 2 from chapter 1"
    print(f"\nQuery: {query!r}\n")

    # 1. Parse
    raw_parse = parse_query_with_slm(query)
    parsed    = _json.loads(raw_parse)
    print("Parsed metadata:", _json.dumps(parsed, indent=2))

    # 2. Retrieve
    retriever = Retriever()
    results, err = retriever.retrieve_safe(parsed, query)

    if err:
        print(f"\n⚠  Ambiguity / Error: {err}")
        sys.exit(1)

    print(f"\nTop {len(results)} chunks:\n")
    for i, r in enumerate(results, 1):
        print(f"{'─'*60}")
        print(f"[{i}] {r.chunk_id}  score={r.score}")
        print(f"    Subject  : {r.subject}")
        print(f"    Chapter  : {r.chapter_number} — {r.chapter_title}")
        print(f"    Section  : {r.section_title}")
        print(f"    Type     : {r.chunk_type}  activity={r.activity_number}")
        print(f"    Filter   : {r.filter_path}")
        print(f"    Text     : {r.text[:200].strip()} …")
    print(f"{'─'*60}")
