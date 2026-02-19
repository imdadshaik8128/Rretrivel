"""
chunker.py
==========
Hierarchical Markdown Chunker for 10th Class Subject Files
Splits content by headings, extracts rich metadata, and saves chunks as JSON.

Usage:
    python chunker.py --input_dir ./subjects --output_dir ./chunks

Dependencies:
    pip install tiktoken langchain-text-splitters
"""

import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Optional: token counter (falls back to word count) ──────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def token_count(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def token_count(text: str) -> int:
        return len(text.split())          # rough fallback


# ────────────────────────────────────────────────────────────────────────────
# DATA MODEL
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    # --- Identity ---
    chunk_id:        str          # e.g. "physics_ch01_sec1.1_001"
    source_file:     str          # original .md filename
    subject:         str          # Physics / Biology / Maths_sem_1 …

    # --- Hierarchical position ---
    chapter_number:  Optional[str]
    chapter_title:   str
    section_title:   str          # H2 heading (empty if none)
    subsection_title:str          # H3 heading (empty if none)
    heading_path:    str          # full breadcrumb: "Ch1 > 1.1 > Activity 1.2"

    # --- Content ---
    text:            str
    token_count:     int
    char_count:      int

    # --- Document position ---
    chunk_index:     int          # 0-based index within the file
    start_line:      int
    end_line:        int

    # --- Content flags ---
    has_activity:    bool
    has_equation:    bool
    has_table:       bool
    has_image:       bool
    keywords:        list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
# HEADING DETECTION
# ────────────────────────────────────────────────────────────────────────────

H1 = re.compile(r'^#\s+(.+)')
H2 = re.compile(r'^##\s+(.+)')
H3 = re.compile(r'^###\s+(.+)')

CHAPTER_NUM = re.compile(
    r'chapter\s*(\d+[\w.]*)',
    re.IGNORECASE
)

IMG_RE      = re.compile(r'!\[.*?\]\(.*?\)')
TABLE_RE    = re.compile(r'^\|.+\|', re.MULTILINE)
EQ_RE       = re.compile(r'\$[^$]+\$|\\\[.+?\\\]')
ACTIVITY_RE = re.compile(r'activity\s*\d+', re.IGNORECASE)

# Very rough keyword extractor (top nouns / domain terms)
STOP = set("the a an is are was were be been being have has had do does did "
           "will would could should may might shall not and or but in on at to "
           "for of with by from this that these those it its we our you your "
           "they their he she his her as if so no yes".split())

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    words = re.findall(r'\b[A-Za-z]{4,}\b', text)
    freq: dict[str, int] = {}
    for w in words:
        w = w.lower()
        if w not in STOP:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


# ────────────────────────────────────────────────────────────────────────────
# CHUNKING ENGINE
# ────────────────────────────────────────────────────────────────────────────

def subject_from_filename(filename: str) -> str:
    """Physics.md -> Physics, Maths_sem_1.md -> Maths_sem_1"""
    return Path(filename).stem


def parse_chapter_number(heading_text: str) -> Optional[str]:
    m = CHAPTER_NUM.search(heading_text)
    return m.group(1) if m else None


def chunk_markdown(filepath: str, 
                   min_tokens: int = 80,
                   max_tokens: int = 512) -> list[Chunk]:
    """
    Strategy: Heading-based hierarchical splitting.

    1. Walk lines; each time we hit a heading we start a new segment.
    2. H1 = Chapter boundary (always split)
       H2 = Section boundary (always split)
       H3 = Sub-section (split if segment > min_tokens, else append)
    3. If a segment still exceeds max_tokens after splitting by H3, 
       apply a sliding-window paragraph split.
    """

    source   = Path(filepath).name
    subject  = subject_from_filename(source)

    with open(filepath, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # ── Pass 1: collect raw segments separated by any heading ────────────────
    segments: list[dict] = []          # {level, heading, lines[], start_line}
    current:  dict | None = None

    chapter_title    = ""
    section_title    = ""
    subsection_title = ""

    for i, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        if m := H1.match(line):
            if current: segments.append(current)
            chapter_title    = m.group(1).strip()
            section_title    = ""
            subsection_title = ""
            current = dict(level=1, heading=chapter_title,
                           chapter=chapter_title,
                           section="", subsection="",
                           lines=[], start=i)

        elif m := H2.match(line):
            if current: segments.append(current)
            section_title    = m.group(1).strip()
            subsection_title = ""
            current = dict(level=2, heading=section_title,
                           chapter=chapter_title,
                           section=section_title, subsection="",
                           lines=[], start=i)

        elif m := H3.match(line):
            sub = m.group(1).strip()
            # only split if current segment is big enough
            if current and token_count("\n".join(current["lines"])) >= min_tokens:
                segments.append(current)
                subsection_title = sub
                current = dict(level=3, heading=sub,
                               chapter=chapter_title,
                               section=section_title, subsection=sub,
                               lines=[], start=i)
            else:
                # small segment: merge H3 heading into current
                if current:
                    current["lines"].append(line)
                else:
                    subsection_title = sub
                    current = dict(level=3, heading=sub,
                                   chapter=chapter_title,
                                   section=section_title, subsection=sub,
                                   lines=[], start=i)
        else:
            if current is None:
                current = dict(level=0, heading="",
                               chapter="", section="", subsection="",
                               lines=[], start=i)
            current["lines"].append(line)

    if current:
        segments.append(current)

    # ── Pass 2: large segments → paragraph split ─────────────────────────────
    final_segments: list[dict] = []
    for seg in segments:
        body = "\n".join(seg["lines"]).strip()
        if token_count(body) <= max_tokens:
            seg["body"] = body
            final_segments.append(seg)
        else:
            # split by blank lines (paragraph boundaries)
            paras  = re.split(r'\n{2,}', body)
            bucket_lines: list[str] = []
            bucket_start = seg["start"]
            for para in paras:
                combined = "\n\n".join(bucket_lines + [para]) if bucket_lines else para
                if token_count(combined) > max_tokens and bucket_lines:
                    s = dict(seg)
                    s["body"] = "\n\n".join(bucket_lines).strip()
                    s["start"] = bucket_start
                    final_segments.append(s)
                    bucket_lines = [para]
                    bucket_start = seg["start"]   # approximate
                else:
                    bucket_lines.append(para)
            if bucket_lines:
                s = dict(seg)
                s["body"] = "\n\n".join(bucket_lines).strip()
                s["start"] = bucket_start
                final_segments.append(s)

    # ── Pass 3: build Chunk objects ───────────────────────────────────────────
    chunks: list[Chunk] = []
    for idx, seg in enumerate(final_segments):
        body = seg.get("body", "\n".join(seg.get("lines", [])).strip())
        if not body:
            continue

        chapter     = seg.get("chapter", "")
        section     = seg.get("section", "")
        subsection  = seg.get("subsection", "")
        ch_num      = parse_chapter_number(chapter)

        breadcrumb_parts = [p for p in [chapter, section, subsection] if p]
        breadcrumb = " > ".join(breadcrumb_parts) if breadcrumb_parts else "intro"

        safe_subject = re.sub(r'\W+', '_', subject.lower())
        safe_ch      = f"ch{ch_num.zfill(2)}" if ch_num else "ch00"
        chunk_id     = f"{safe_subject}_{safe_ch}_{idx:04d}"

        toks = token_count(body)
        chunk = Chunk(
            chunk_id         = chunk_id,
            source_file      = source,
            subject          = subject,
            chapter_number   = ch_num,
            chapter_title    = chapter,
            section_title    = section,
            subsection_title = subsection,
            heading_path     = breadcrumb,
            text             = body,
            token_count      = toks,
            char_count       = len(body),
            chunk_index      = idx,
            start_line       = seg.get("start", 0),
            end_line         = seg.get("start", 0) + body.count("\n"),
            has_activity     = bool(ACTIVITY_RE.search(body)),
            has_equation     = bool(EQ_RE.search(body)),
            has_table        = bool(TABLE_RE.search(body)),
            has_image        = bool(IMG_RE.search(body)),
            keywords         = extract_keywords(body),
        )
        chunks.append(chunk)

    return chunks


# ────────────────────────────────────────────────────────────────────────────
# FILE RUNNER
# ────────────────────────────────────────────────────────────────────────────

def process_directory(input_dir: str, output_dir: str,
                      min_tokens: int = 80, max_tokens: int = 512):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    summary = []

    md_files = sorted(input_path.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {input_dir}")
        return

    for md_file in md_files:
        print(f"\n📄 Processing: {md_file.name}")
        chunks = chunk_markdown(str(md_file), min_tokens, max_tokens)

        # per-subject JSON
        out_file = output_path / f"{md_file.stem}_chunks.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)

        print(f"   ✅ {len(chunks)} chunks → {out_file.name}")
        summary.append({"file": md_file.name, "chunks": len(chunks),
                        "output": str(out_file)})
        all_chunks.extend([asdict(c) for c in chunks])

    # combined JSON (for vector store ingestion)
    combined_path = output_path / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # summary report
    report_path = output_path / "chunking_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks": len(all_chunks),
            "files": summary,
            "config": {"min_tokens": min_tokens, "max_tokens": max_tokens}
        }, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Total chunks across all subjects: {len(all_chunks)}")
    print(f"📦 Combined output  : {combined_path}")
    print(f"📊 Report           : {report_path}")


# ────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 10th-class subject MD files")
    parser.add_argument("--input_dir",  default="./subjects",
                        help="Directory containing .md files")
    parser.add_argument("--output_dir", default="./chunks",
                        help="Directory to write chunk JSONs")
    parser.add_argument("--min_tokens", type=int, default=80,
                        help="Min tokens before an H3 triggers a split (default 80)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per chunk before paragraph-splitting (default 512)")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir,
                      args.min_tokens, args.max_tokens)
