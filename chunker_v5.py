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

    # --- Chunk classification ---
    chunk_type:      str = "theory"   # "theory" | "exercise" | "activity"
    section_type:    str = ""         # normalised section label, e.g. "fill_in_the_blanks"
    activity_number: str = ""         # e.g. "3" or "3.2" — only set when chunk_type="activity"


# ────────────────────────────────────────────────────────────────────────────
# HEADING DETECTION
# ────────────────────────────────────────────────────────────────────────────

H1 = re.compile(r'^#\s+(.+)')
H2 = re.compile(r'^##\s+(.+)')
H3 = re.compile(r'^###\s+(.+)')

CHAPTER_NUM = re.compile(
    r'^chapter\s*(\d+[\w.]*|[IVXivx]+)\s*$',   # "Chapter N" or "Chapter IV" alone on a line
    re.IGNORECASE
)

CHAPTER_NUM_INLINE = re.compile(
    r'chapter\s*(\d+[\w.]*|[IVXivx]+)',  # arabic or Roman numeral after "chapter"
    re.IGNORECASE
)

ROMAN = {"I":1,"II":2,"III":3,"IV":4,"V":5,"VI":6,"VII":7,"VIII":8,"IX":9,"X":10}

def roman_to_int(s: str) -> Optional[str]:
    return str(ROMAN.get(s.upper())) if s.upper() in ROMAN else None

IMG_RE      = re.compile(r'!\[.*?\]\(.*?\)')
TABLE_RE    = re.compile(r'^\|.+\|', re.MULTILINE)
EQ_RE       = re.compile(r'\$[^$]+\$|\\\[.+?\\\]')
ACTIVITY_RE     = re.compile(r'\bactivity\s*[\-]?\s*[\d.]+', re.IGNORECASE)
ACTIVITY_NUM_RE = re.compile(r'\bactivity\s*[\-]?\s*([\d.]+)', re.IGNORECASE)


# ────────────────────────────────────────────────────────────────────────────
# CHUNK TYPE & SECTION TYPE DETECTION
# ────────────────────────────────────────────────────────────────────────────

# Activity heading patterns → chunk_type="activity"  (NOT exercise)
# Covers all Biology/Physics variants found in textbooks:
#   "Activity 3"   "Activity 3.2"   "Activity-1"   "Activity-2T"
#   "Lab Activity"  ")LabActivity"   "activity"  (standalone)
ACTIVITY_HEADING_RE = re.compile(
    r'^\s*[)\s]*'       # optional leading ) or space (OCR artefact in Biology)
    r'(?:lab\s*)?'      # optional "Lab " prefix
    r'activity'         # the word activity
    r'[\s\-]*'          # space or hyphen (Activity-1 style)
    r'[\d.]*'           # optional number (no number = standalone "activity" heading)
    r'[a-zA-Z]?\s*$',   # optional trailing letter e.g. "2T", then end of string
    re.IGNORECASE
)

# Exercise section headings → chunk_type="exercise"
# Built from the exact heading variants found across all 8 subject files.
# "activity" / "lab activity" are intentionally absent — handled by ACTIVITY_HEADING_RE above.
EXERCISE_SECTION_MAP: list[tuple[str, list[str]]] = [
    ("exercise",              ["exercise", "exercises"]),
    ("improve_your_learning", ["improve your learning"]),
    ("fill_in_the_blanks",    ["fill in the blanks", "fill in the blank"]),
    ("choose_correct_answer", ["choose the correct answer", "choose the correct",
                               "multiple choice", "mcq"]),
    ("write_in_brief",        ["write in brief", "answer in brief",
                               "short answer", "short questions"]),
    ("discuss",               ["discuss", "discussion questions",
                               "think and discuss", "think about it"]),
    ("project",               ["project", "projects", "project work"]),
    ("questions",             ["questions", "questions and answers",
                               "comprehension questions", "review questions",
                               "key questions", "intext questions",
                               "let us recall", "check your progress",
                               "test yourself",
                               "let's work these out", "let's work this out",
                               "let\u2019s work these out", "let\u2019s work this out",
                               "let's do it", "let\u2019s do it",
                               "work sheet", "worksheet"]),
    ("what_you_have_learnt",  ["what you have learnt", "what have you learnt",
                               "what we have learnt", "summary", "key points",
                               "summing up"]),
    ("group_activity",        ["group activity", "group work", "pair work",
                               "class activity", "do yourself"]),
]

# Theory-only section labels — explicitly classified so metadata_extractor
# can report them separately. These are NOT exercise, NOT activity.
# (chunk_type stays "theory" — these just populate section_type for reference)
THEORY_SECTION_MAP: list[tuple[str, list[str]]] = [
    ("info_box",      ["do you know", "interesting fact", "more to know",
                       "fun fact", "did you know"]),
    ("glossary",      ["new words", "key words", "glossary", "glo ssary",
                       "word bank"]),
    ("worked_example",["solution", "solution :", "example", "solved example",
                       "proof", "proof :"]),
    ("teacher_note",  ["notes for the teacher", "note to the teacher",
                       "a note to the reader"]),
    ("overview",      ["overview", "introduction", "in this chapter"]),
    ("primary_source",["source a", "source b", "source c", "source d",
                       "source e", "source f", "sources for information"]),
    ("sidebar",       ["box 1", "box 2", "box 3", "box 4",
                       "remarks", "remarks :"]),
]

# Combined flat regex lists
_EXERCISE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (label, re.compile(
        r'\b(' + '|'.join(re.escape(p) for p in phrases) + r')\b',
        re.IGNORECASE
    ))
    for label, phrases in EXERCISE_SECTION_MAP
]

_THEORY_PATTERNS: list[tuple[str, re.Pattern]] = [
    (label, re.compile(
        r'\b(' + '|'.join(re.escape(p) for p in phrases) + r')\b',
        re.IGNORECASE
    ))
    for label, phrases in THEORY_SECTION_MAP
]


def classify_section_type(heading: str) -> tuple[str, str]:
    """
    Given a heading string returns (chunk_type, section_type_label).

    chunk_type   : "activity" | "exercise" | "theory"
    section_type : specific label e.g. "fill_in_the_blanks", "info_box", "activity", or ""

    Priority:
      1. Activity headings  → chunk_type="activity"
      2. Exercise headings  → chunk_type="exercise"
      3. Theory sub-types   → chunk_type="theory" with labelled section_type
      4. Everything else    → chunk_type="theory", section_type=""
    """
    # 1. Activity — takes priority over everything
    if ACTIVITY_HEADING_RE.match(heading):
        return "activity", "activity"

    h = heading.lower().strip()

    # 2. Exercise sections
    for label, pattern in _EXERCISE_PATTERNS:
        if pattern.search(h):
            return "exercise", label

    # 3. Named theory sections (info boxes, glossaries, worked examples etc.)
    for label, pattern in _THEORY_PATTERNS:
        if pattern.search(h):
            return "theory", label

    # 4. Default
    return "theory", ""


def extract_activity_number(text: str) -> str:
    """
    Scan the chunk text/heading for the activity number.
    Returns the number as a string e.g. "3" or "3.2", or "" if not found.
    """
    m = ACTIVITY_NUM_RE.search(text)
    return m.group(1) if m else ""


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
    # Try arabic number first
    m = CHAPTER_NUM_INLINE.search(heading_text)
    if m:
        val = m.group(1)
        # Convert Roman numeral if needed
        if not val.isdigit():
            return roman_to_int(val) or val
        return val
    return None


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
    chapter_number   = None           # track current chapter number
    section_title    = ""
    subsection_title = ""

    # Pre-scan: merge consecutive H1 lines where the first is "# Chapter N"
    # Pattern in these textbooks:
    #   # Chapter 1          ← chapter number line
    #   # Chemical Reactions ← chapter title line
    # We collapse them into one heading: "Chapter 1 - Chemical Reactions"
    merged_lines: list[tuple[int, str]] = []  # (1-based line num, text)
    raw_lines = [l.rstrip("\n") for l in lines]
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if H1.match(line):
            h1_text = H1.match(line).group(1).strip()
            # Is this line purely a chapter-number declaration? (arabic OR roman)
            if CHAPTER_NUM.match(h1_text) and i + 1 < len(raw_lines) and H1.match(raw_lines[i + 1]):
                next_text = H1.match(raw_lines[i + 1]).group(1).strip()
                # Merge: "Chapter 1" + "Chemical Reactions" → "Chapter 1 - Chemical Reactions"
                merged = f"# {h1_text} - {next_text}"
                merged_lines.append((i + 1, merged))
                i += 2          # skip both lines
                continue
        merged_lines.append((i + 1, line))
        i += 1

    for line_num, line in merged_lines:

        if m := H1.match(line):
            if current: segments.append(current)
            new_chapter_title = m.group(1).strip()
            new_chapter_number = parse_chapter_number(new_chapter_title)
            # Only update chapter tracking when we find an actual chapter heading
            # Non-chapter H1s (like "NOTES FOR THE TEACHER") inherit the current chapter
            if new_chapter_number is not None:
                chapter_title  = new_chapter_title
                chapter_number = new_chapter_number
            else:
                chapter_title  = new_chapter_title
                # chapter_number stays as-is (inherited from previous chapter)
            section_title    = ""
            subsection_title = ""
            h1_ctype, h1_stype = classify_section_type(new_chapter_title)
            current = dict(level=1, heading=new_chapter_title,
                           chapter=chapter_title,
                           chapter_number=chapter_number,
                           section="", subsection="",
                           chunk_type=h1_ctype, section_type=h1_stype,
                           lines=[], start=line_num)

        elif m := H2.match(line):
            if current: segments.append(current)
            section_title    = m.group(1).strip()
            subsection_title = ""
            h2_ctype, h2_stype = classify_section_type(section_title)
            current = dict(level=2, heading=section_title,
                           chapter=chapter_title,
                           chapter_number=chapter_number,
                           section=section_title, subsection="",
                           chunk_type=h2_ctype, section_type=h2_stype,
                           lines=[], start=line_num)

        elif m := H3.match(line):
            sub = m.group(1).strip()
            # only split if current segment is big enough
            if current and token_count("\n".join(current["lines"])) >= min_tokens:
                segments.append(current)
                subsection_title = sub
                h3_ctype, h3_stype = classify_section_type(sub)
                current = dict(level=3, heading=sub,
                               chapter=chapter_title,
                               chapter_number=chapter_number,
                               section=section_title, subsection=sub,
                               chunk_type=h3_ctype, section_type=h3_stype,
                               lines=[], start=line_num)
            else:
                # small segment: merge H3 heading into current
                if current:
                    current["lines"].append(line)
                else:
                    subsection_title = sub
                    h3_ctype, h3_stype = classify_section_type(sub)
                    current = dict(level=3, heading=sub,
                                   chapter=chapter_title,
                                   chapter_number=chapter_number,
                                   section=section_title, subsection=sub,
                                   chunk_type=h3_ctype, section_type=h3_stype,
                                   lines=[], start=line_num)
        else:
            if current is None:
                current = dict(level=0, heading="",
                               chapter="", chapter_number=None,
                               section="", subsection="",
                               lines=[], start=line_num)
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
        ch_num      = seg.get("chapter_number") or parse_chapter_number(chapter)

        breadcrumb_parts = [p for p in [chapter, section, subsection] if p]
        breadcrumb = " > ".join(breadcrumb_parts) if breadcrumb_parts else "intro"

        safe_subject = re.sub(r'\W+', '_', subject.lower())
        safe_ch      = f"ch{ch_num.zfill(2)}" if ch_num else "ch00"
        chunk_id     = f"{safe_subject}_{safe_ch}_{idx:04d}"

        toks = token_count(body)

        # ── Determine chunk_type ───────────────────────────────────────────
        # Inherit from heading classification first.
        # Then handle edge case: heading was "theory" but body IS an activity
        # (e.g. an activity embedded mid-section without its own heading).
        seg_ctype    = seg.get("chunk_type", "theory")
        seg_stype    = seg.get("section_type", "")

        if seg_ctype == "theory" and ACTIVITY_HEADING_RE.search(body.split("\n")[0]):
            # First line of body looks like an activity heading
            seg_ctype = "activity"
            seg_stype = "activity"

        # Extract activity number for fast lookup later
        act_number = ""
        if seg_ctype == "activity":
            # Try heading path first, then body text
            act_number = (extract_activity_number(breadcrumb) or
                          extract_activity_number(body))

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
            chunk_type       = seg_ctype,
            section_type     = seg_stype,
            activity_number  = act_number,
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

        # quick type breakdown per file
        theory_n   = sum(1 for c in chunks if c.chunk_type == "theory")
        exercise_n = sum(1 for c in chunks if c.chunk_type == "exercise")
        activity_n = sum(1 for c in chunks if c.chunk_type == "activity")
        print(f"      theory={theory_n}  exercise={exercise_n}  activity={activity_n}")

        summary.append({"file": md_file.name, "chunks": len(chunks),
                        "theory": theory_n, "exercise": exercise_n,
                        "activity": activity_n,
                        "output": str(out_file)})
        all_chunks.extend([asdict(c) for c in chunks])

    # combined JSON (for vector store ingestion)
    combined_path = output_path / "all_chunks.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    total_theory   = sum(1 for c in all_chunks if c.get("chunk_type") == "theory")
    total_exercise = sum(1 for c in all_chunks if c.get("chunk_type") == "exercise")
    total_activity = sum(1 for c in all_chunks if c.get("chunk_type") == "activity")

    # summary report
    report_path = output_path / "chunking_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_chunks":    len(all_chunks),
            "total_theory":    total_theory,
            "total_exercise":  total_exercise,
            "total_activity":  total_activity,
            "files":           summary,
            "config": {"min_tokens": min_tokens, "max_tokens": max_tokens}
        }, f, indent=2)

    print(f"\n{'='*55}")
    print(f"✅ Total chunks : {len(all_chunks)}")
    print(f"   theory      : {total_theory}")
    print(f"   exercise    : {total_exercise}")
    print(f"   activity    : {total_activity}")
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
