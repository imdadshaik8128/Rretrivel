# 10th Class Subject — Chunking Pipeline

## Files
| File | Purpose |
|------|---------|
| `chunker.py` | Splits each `.md` into chunks, extracts metadata, writes JSON |
| `metadata_extractor.py` | Analyses chunks, prints stats table, exports CSV |

---

## Step 1 – Install dependencies
```bash
pip install tiktoken tabulate
# tiktoken  → accurate token counting (OpenAI cl100k_base)
# tabulate  → pretty table in terminal
```

---

## Step 2 – Put your markdown files in one folder
```
subjects/
  Physics.md
  Biology.md
  Chemistry.md
  Maths_sem_1.md
  Maths_sem_2.md
  History.md
  Geography.md
  Economics.md
  Social_political.md
```

---

## Step 3 – Run the chunker
```bash
python chunker.py \
  --input_dir  ./subjects \
  --output_dir ./chunks \
  --min_tokens 80 \
  --max_tokens 512
```

**What it produces:**
```
chunks/
  Physics_chunks.json        ← chunks for Physics only
  Biology_chunks.json
  …
  all_chunks.json            ← every chunk combined (use this for vector DB)
  chunking_report.json       ← summary: chunk counts per file
```

---

## Step 4 – Analyse metadata
```bash
python metadata_extractor.py \
  --chunks_dir ./chunks \
  --export_csv ./chunks/metadata.csv
```

Prints a subject-level table and exports a CSV for inspection in Excel/Sheets.

---

## Chunking Strategy Explained

```
H1  (#)   → Chapter boundary          → ALWAYS split
H2  (##)  → Section boundary          → ALWAYS split
H3  (###) → Sub-section boundary      → Split only if current chunk ≥ min_tokens
blank line → Paragraph boundary       → Split only if chunk > max_tokens
```

**Why this strategy?**
- Respects the natural hierarchy of the textbooks (Chapter → Section → Activity)
- Keeps related content (e.g. an Activity + its explanation) together
- Prevents oversized chunks that confuse retrieval models
- min_tokens guard stops tiny H3 chunks (e.g. single-line headings)

---

## Metadata Fields per Chunk

| Field | Description |
|-------|-------------|
| `chunk_id` | Unique ID: `physics_ch01_0042` |
| `subject` | `Physics`, `Biology`, `Maths_sem_1` … |
| `chapter_number` | `"1"`, `"2"` … or `null` |
| `chapter_title` | Full H1 heading |
| `section_title` | H2 heading (empty if none) |
| `subsection_title` | H3 heading (empty if none) |
| `heading_path` | `"Ch1 > 1.1 Chemical Equations > Activity 1.2"` |
| `text` | Raw Markdown content of the chunk |
| `token_count` | Approx. tokens |
| `has_activity` | `true` if Activity reference found |
| `has_equation` | `true` if math formula (`$…$`) found |
| `has_table` | `true` if Markdown table found |
| `has_image` | `true` if image link found |
| `keywords` | Top-10 domain keywords |

---

## Next Steps (after chunking)

```
all_chunks.json
      │
      ▼
Embed each chunk["text"]     ← e.g. OpenAI text-embedding-3-small
      │                              or sentence-transformers
      ▼
Store in vector DB           ← Chroma / Pinecone / Qdrant / FAISS
      │    + store all other fields as metadata filters
      ▼
RAG pipeline                 ← query → retrieve top-k chunks → LLM answer
```

**Filter examples you can do with the metadata:**
- "Only search Physics chapters 1–3" → filter `subject == "Physics" AND chapter_number IN [1,2,3]`
- "Find chunks with equations" → filter `has_equation == true`
- "Search only Activities" → filter `has_activity == true`
