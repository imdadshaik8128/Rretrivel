# 📚 Textbook RAG — Offline Q&A System

A fully offline, production-grade Retrieval-Augmented Generation (RAG) system for school textbook question answering. Built with deterministic metadata filtering, semantic reranking, a local LLM via Ollama, and a React chat UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Structure](#3-file-structure)
4. [How the Pipeline Works](#4-how-the-pipeline-works)
5. [Installation](#5-installation)
6. [Running the System](#6-running-the-system)
7. [File-by-File Reference](#7-file-by-file-reference)
8. [The UI — Feature Guide](#8-the-ui--feature-guide)
9. [API Reference](#9-api-reference)
10. [Configuration & Tuning](#10-configuration--tuning)
11. [Non-Negotiable Design Rules](#11-non-negotiable-design-rules)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

This system lets students ask natural language questions about their textbooks and get structured, cited answers — completely offline. No OpenAI, no cloud APIs.

**Two query types are supported:**

| Type | Example | Behaviour |
|---|---|---|
| **Reference** | "Explain Activity 2 from Chapter 1" | Exact metadata match → deterministic lookup, confidence = 1.0 |
| **Concept** | "What is photosynthesis?" | Semantic search → bi-encoder recall + cross-encoder rerank |

**Subjects supported:** Biology, Economics, Geography, History, Maths Sem 1, Maths Sem 2, Physics, Social & Political

---

## 2. Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  query_parser_v2  (SLM — Qwen 0.5B via Ollama)      │
│  Extracts: intent, chunk_type, chapter_number,       │
│            activity_number, exercise_number, topic   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  parse_sanitizer                                     │
│  Strips hallucinated fields the user never said      │
│  (chapter/activity/exercise must appear in raw text) │
└──────────────────────┬──────────────────────────────┘
                       │
              inject session subject
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Retriever                                           │
│                                                      │
│  ① MetadataFilter  (deterministic, hard rules)       │
│     subject → chapter → chunk_type → ref number     │
│                                                      │
│  ② USE CASE 1 — Reference query                      │
│     Exact match on activity_number / exercise_number │
│     Bi-encoder scores for ordering only              │
│     NEVER falls back to semantic search              │
│                                                      │
│  ③ USE CASE 2 — Concept query                        │
│     BiEncoder recall (top-5) → CrossEncoder rerank   │
│     Returns top-K chunks                             │
└──────────────────────┬──────────────────────────────┘
                       │
                  RetrievedChunk list
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Generator                                           │
│  Prompt template (reference or concept)              │
│  → Ollama LLM (qwen2.5:0.5b-instruct)               │
│  → Parse JSON response                               │
│  → spoken_answer (TTS-safe)                          │
│  → display_answer_markdown (rich UI)                 │
│  → confidence score (sigmoid of cross-encoder logit) │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              GeneratedAnswer
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    FastAPI (api.py)           CLI (Main.py)
          │
          ▼
    React UI (rag_chat_ui.jsx)
    display_answer_markdown + citations + confidence
```

---

## 3. File Structure

```
project/
│
├── all_chunks.json          ← Textbook chunks (YOU must provide this)
├── metadata.csv             ← Chunk metadata index
│
├── retriever.py             ← Core RAG retriever (bi-encoder + cross-encoder)
├── generator.py             ← LLM answer generator via Ollama
├── parse_sanitizer.py       ← Guards against SLM hallucinated fields
├── query_parser_v2.py       ← SLM query parser (NOT in repo — you provide)
│
├── Main.py                  ← Full CLI with TTS + Rich display
├── chat.py                  ← Lightweight CLI (retrieval only, no generator)
│
├── api.py                   ← FastAPI backend (NEW — connects backend to UI)
└── rag_chat_ui.jsx          ← React chat UI (NEW — full frontend)
```

> **Note:** `query_parser_v2.py` and `all_chunks.json` are not included in this repo — you supply them. `query_parser_v2.py` must export a `parse_query_with_slm(query: str) -> str` function that returns a JSON string.

---

## 4. How the Pipeline Works

### Step 1 — Query Parsing (`query_parser_v2.py`)
A small local LLM (run via Ollama) reads the raw query and extracts structured metadata:
```json
{
  "intent": "explain",
  "chunk_type": "activity",
  "chapter_number": 1,
  "activity_number": 2,
  "exercise_number": null,
  "topic": "cell structure",
  "subject": null
}
```

### Step 2 — Sanitization (`parse_sanitizer.py`)
Strips any field the user never actually mentioned. For example if the query says "what is osmosis?" and the SLM hallucinates `chapter_number: 3`, the sanitizer sets it back to `null`. Rules:
- `chapter_number` only kept if query contains `chapter N` or `ch. N`
- `activity_number` only kept if query contains `activity N`
- `exercise_number` only kept if query contains `exercise N.N`
- `topic` is always kept

### Step 3 — Subject Injection
The session subject selected in the UI is **hard-injected** into the parsed dict, overriding whatever the SLM emitted. This is the **subject isolation guarantee** — cross-subject leakage is impossible.

### Step 4 — Retrieval (`retriever.py`)

**USE CASE 1 — Reference queries** (activity/exercise number present):
```
subject filter → chapter filter → chunk_type filter → exact activity/exercise match
```
If any step finds zero results, an `AmbiguityError` is raised immediately. There is no fallback to semantic search. Confidence is always `1.0`.

**USE CASE 2 — Concept queries** (no reference number):
```
subject filter → optional chapter filter → theory+activity chunks only
→ bi-encoder recall (top-5)
→ cross-encoder rerank (top-2)
```
Confidence is computed as `sigmoid(cross_encoder_logit)`.

### Step 5 — Generation (`generator.py`)
Two prompt templates (reference / concept) feed the retrieved chunks into the local LLM. Output is a strict JSON object with:
- `spoken_answer` — plain text, TTS-safe (no markdown, no symbols)
- `display_answer_markdown` — rich markdown for the UI

A 4-stage JSON repair pipeline handles small-model output failures gracefully.

---

## 5. Installation

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.9+ | python.org |
| Node.js | 18+ | nodejs.org |
| Ollama | latest | ollama.com |

### Python dependencies

```bash
pip install fastapi uvicorn
pip install sentence-transformers
pip install numpy requests
pip install rich pyttsx3        # optional — for CLI only
```

### Pull the LLM model

```bash
ollama pull qwen2.5:0.5b-instruct
```

You can swap this for any model by editing `OLLAMA_MODEL` in `generator.py`. Larger models give better answers:
```bash
ollama pull llama3              # better quality
ollama pull mistral             # good balance
ollama pull qwen2.5:0.5b-instruct  # fastest, smallest (default)
```

### React frontend

```bash
# In your React project folder
npm install
```

Place `rag_chat_ui.jsx` as a component in your React app (e.g. `src/App.jsx` or `src/components/RAGChat.jsx`).

---

## 6. Running the System

You need **3 terminals** running simultaneously.

### Terminal 1 — Ollama (LLM server)
```bash
ollama serve
```
Runs on `http://localhost:11434`. Must be running before the backend starts.

### Terminal 2 — FastAPI backend
```bash
# From the folder containing api.py, retriever.py, generator.py, etc.
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

On first start, this will:
1. Load `all_chunks.json`
2. Download and cache `all-MiniLM-L6-v2` (~80 MB) if not already cached
3. Download and cache `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80 MB) if not already cached
4. Encode all chunks into embeddings (runs once, cached in memory)

**Wait for:** `INFO | ✓ Pipeline ready.` before sending queries.

### Terminal 3 — React frontend
```bash
npm run dev
```
Open `http://localhost:5173` in your browser.

### Alternatively — CLI only (no UI, no API)

Full pipeline with TTS and Rich display:
```bash
python Main.py
```

Retrieval only (no LLM generation):
```bash
python chat.py
```

Smoke test the retriever directly:
```bash
python retriever.py "explain activity 2 from chapter 1"
```

Smoke test the full pipeline:
```bash
python generator.py "what is photosynthesis chapter 3 biology"
```

---

## 7. File-by-File Reference

### `retriever.py`
The core of the system. Never guesses — raises `AmbiguityError` when a query cannot be resolved deterministically.

Key classes:
- `ChunkStore` — loads `all_chunks.json`, encodes all chunks on startup, builds O(1) lookup indexes by subject and chapter
- `MetadataFilter` — applies hard deterministic rules (subject → chapter → chunk_type → reference number)
- `SemanticRanker` — cosine similarity bi-encoder ranking within the filtered candidate pool
- `CrossEncoderReranker` — precision reranker for concept queries only
- `Retriever` — public API; call `retriever.retrieve(parsed_query, raw_text)` or `retriever.retrieve_safe(...)` for the non-raising version

Key constants (edit at top of file):
```python
TOP_K = 2                    # number of final chunks returned
BI_ENCODER_RECALL_K = 5      # bi-encoder recall set size
SCORE_THRESHOLD = 0.20       # cosine similarity floor for warnings
```

### `generator.py`
Wraps Ollama with structured output. Handles JSON repair for small models.

Key constants:
```python
OLLAMA_MODEL = "qwen2.5:0.5b-instruct"   # change to llama3, mistral, etc.
LLM_TIMEOUT = 60                          # seconds
CONFIDENCE_FLOOR = 0.40                   # below this → warning shown in UI
```

Key classes:
- `Generator` — call `generator.generate(chunks, query)` or `generator.generate_safe(...)`
- `GeneratedAnswer` — output dataclass with `spoken_answer`, `display_answer_markdown`, `citations`, `confidence`, `filter_path`

JSON repair pipeline (handles small model failures):
1. Direct `json.loads` after stripping fences
2. `_repair_json()` — fixes triple quotes, literal newlines, trailing commas
3. Extract first `{...}` block + repair
4. `_regex_fallback()` — field-level regex extraction (last resort, never loses content)

### `parse_sanitizer.py`
Stateless utility. Call `sanitize(parsed_dict, raw_query)` after every SLM parse call. Returns a cleaned copy — never mutates the input.

### `api.py` *(new)*
FastAPI server exposing the pipeline over HTTP. Handles CORS for localhost React dev servers (ports 3000 and 5173).

Endpoints:
- `GET /health` — liveness check
- `GET /subjects` — list available subjects
- `POST /query` — full pipeline endpoint

### `Main.py`
Full-featured CLI. Loads retriever + generator + TTS engine at startup. Supports Rich terminal rendering (install `pip install rich`). TTS via `pyttsx3` + `espeak`.

### `chat.py`
Lightweight CLI. Runs retrieval only — no LLM generation, no TTS. Useful for debugging the retrieval pipeline without needing Ollama running.

### `rag_chat_ui.jsx` *(new)*
React component. Zero external dependencies beyond React itself. All state is in-memory (no localStorage). Drop into any React + Vite or Create React App project.

---

## 8. The UI — Feature Guide

### Sidebar

| Element | Behaviour |
|---|---|
| **＋ New Chat** | Creates a new isolated session with its own message memory, locked to current subject |
| **Subject list** | Click any subject to switch. If a conversation is in progress, triggers a new session automatically |
| **Recent sessions** | All sessions in the current browser session, with message count and subject label |
| **API status dot** | 🟢 Green = backend connected · 🔴 Red = backend offline · 🟡 Yellow = checking |
| **‹ / › toggle** | Collapses sidebar to icon-only mode |

### Subject switching mid-conversation
When you click a different subject while messages exist in the current session:
1. A yellow system notice appears: `⚡ SUBJECT CHANGE DETECTED: A new session has been initialized…`
2. After 500ms, a new session is created and becomes active
3. The old session is preserved in the Recent list — you can switch back to it

This ensures **subject isolation** — the retriever never mixes chunks from different subjects.

### Message bubbles

Each AI response shows:
- **● REFERENCE** or **● CONCEPT** badge (from `answer_type`)
- **Confidence bar** — green ≥75%, yellow ≥50%, red <50%
- **Answer text** — rendered markdown (headings, bullets, bold)
- **⚠ Low confidence warning** — shown when confidence < 40%
- **📚 Sources** — citations with chapter number, chapter title, chunk type, activity number
- **🔎 Filter path** — exact retrieval provenance (e.g. `subject=Biology → chapter=3 → activity_number=2 [EXACT]`)
- **Timings** — `ret:42ms gen:1203ms` for debugging
- **debug ▾** — toggle to see raw parsed query JSON from the SLM

### Loading stages
While a query is processing, the typing bubble shows which pipeline stage is running:
- `Parsing query…` — SLM extracting metadata
- `Retrieving chunks…` — MetadataFilter + semantic ranking
- `Generating answer…` — Ollama LLM producing the answer

---

## 9. API Reference

### `POST /query`

**Request:**
```json
{
  "query": "explain activity 2 from chapter 1",
  "subject": "Biology",
  "session_id": "optional-string-for-logging"
}
```

**Response:**
```json
{
  "answer_type": "reference",
  "spoken_answer": "Activity 2 in Chapter 1 asks students to observe...",
  "display_answer_markdown": "## Activity 2 — Cell Observation\n\nThis activity...",
  "citations": [
    {
      "chunk_id": "bio_ch1_act2",
      "chapter_number": "1",
      "chapter_title": "The Living World",
      "section_title": "Observing Cells",
      "chunk_type": "activity",
      "activity_number": "2"
    }
  ],
  "confidence": 1.0,
  "confidence_pct": 100,
  "low_confidence_warning": null,
  "filter_path": "subject=Biology → chapter=1 → chunk_type=activity → activity_number=2 [EXACT]",
  "parsed_query": {
    "intent": "explain",
    "chunk_type": "activity",
    "chapter_number": 1,
    "activity_number": 2,
    "subject": "Biology"
  },
  "retrieval_ms": 38,
  "generation_ms": 1842,
  "total_ms": 1891
}
```

**Error responses:**
- `422` — Query parser failed
- `404` — No chunks found / AmbiguityError (e.g. wrong chapter number)
- `503` — Ollama not running

### `GET /health`
```json
{ "status": "ok", "pipeline": "ready" }
```

### `GET /subjects`
```json
{
  "subjects": ["Biology", "Economics", "Geography", "History",
               "Maths_sem_1", "Maths_sem_2", "Physics", "Social_political"]
}
```

---

## 10. Configuration & Tuning

### Change the LLM model
In `generator.py`:
```python
OLLAMA_MODEL = "llama3"           # better quality answers
OLLAMA_MODEL = "mistral"          # good balance of speed + quality
OLLAMA_MODEL = "qwen2.5:0.5b-instruct"  # fastest, default
```
Then pull it: `ollama pull llama3`

### Increase results returned
In `retriever.py`:
```python
TOP_K = 4              # return top 4 chunks instead of 2
BI_ENCODER_RECALL_K = 10   # wider recall set for cross-encoder
```

### Adjust confidence threshold
In `generator.py`:
```python
CONFIDENCE_FLOOR = 0.40   # lower = fewer warnings, higher = stricter
```

### Change the API port
```bash
uvicorn api:app --port 9000 --reload
```
Then update `API_BASE` in `rag_chat_ui.jsx`:
```js
const API_BASE = "http://localhost:9000";
```

### Production deployment (CORS)
In `api.py`, replace the `allow_origins` list with your actual domain:
```python
allow_origins=["https://yourdomain.com"]
```

---

## 11. Non-Negotiable Design Rules

These rules are enforced throughout the codebase and must never be violated:

| Rule | Enforcement |
|---|---|
| **Structure beats similarity** — metadata filters always run before embeddings | `MetadataFilter` runs before `SemanticRanker` in every path |
| **Headings decide chapters** — chapter number comes from parsed metadata, never from embedding similarity | `chapter_number` is only set via `ParsedQuery`, never inferred |
| **SLM never performs retrieval** — the query parser only extracts metadata | `query_parser_v2` returns JSON metadata; `Retriever` does all lookups |
| **Reference queries never fall back to semantic search** — if activity/exercise not found, raise error | `AmbiguityError` raised immediately; no fallback path exists |
| **Ambiguity must be exposed, not guessed** — wrong chapter/activity = error, not best-guess result | `AmbiguityError` surfaces to UI as a clear error message |
| **Subject leakage is a critical bug** — Biology chunks must never appear in a Physics session | Session subject is hard-injected and overrides SLM output every time |

---

## 12. Troubleshooting

### 🔴 API status dot is red
The FastAPI backend is not running. Start it:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### ⚠ "Generation error: Ollama is not running"
Ollama must be running separately:
```bash
ollama serve
```
Then verify: `curl http://localhost:11434`

### ⚠ "Activity X not found in chapter Y"
The activity or chapter number doesn't exist in `all_chunks.json` for that subject. Check:
- The chapter number is correct
- The activity exists in that chapter
- The subject is set correctly in the UI

### ⚠ Low confidence warnings appearing often
The cross-encoder score is low — the retrieved chunks may not be closely related to the query. Try:
- Being more specific in the query
- Including the chapter number if known
- Lowering `CONFIDENCE_FLOOR` in `generator.py` if warnings are too aggressive

### 🐌 First query is slow
The models are loaded from disk on the first request and embeddings are computed for all chunks. This is a one-time cost per server start. Subsequent queries are fast.

### ❌ JSON parse errors in generation
The LLM produced malformed JSON. The system has a 4-stage repair pipeline so this is usually recovered automatically. If it fails entirely, try a larger/better model:
```bash
ollama pull mistral
# then set OLLAMA_MODEL = "mistral" in generator.py
```

### React UI shows blank / errors
Make sure `rag_chat_ui.jsx` is used as a default export in your React app and that you have React 18+ with hooks support. The component has no external npm dependencies beyond React itself.

---

## Quick Reference Card

```
Start Ollama:    ollama serve
Start backend:   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Start frontend:  npm run dev
Open browser:    http://localhost:5173

CLI (full):      python Main.py
CLI (retrieval): python chat.py
Smoke test:      python retriever.py "your query here"
```

---

*Built for offline educational use. All processing — embedding, reranking, and generation — runs locally on your machine. No data leaves your device.*
