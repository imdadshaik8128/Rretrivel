"""
metadata_extractor.py
=====================
Reads the chunk JSON files produced by chunker.py and:
  1. Validates / enriches metadata
  2. Prints a per-subject and per-chapter breakdown
  3. Exports a flat CSV for quick inspection in Excel / Sheets

Usage:
    python metadata_extractor.py --chunks_dir ./chunks

Dependencies:
    pip install pandas tabulate
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ────────────────────────────────────────────────────────────────────────────
# LOAD
# ────────────────────────────────────────────────────────────────────────────

def load_all_chunks(chunks_dir: str) -> list[dict]:
    combined = Path(chunks_dir) / "all_chunks.json"
    if combined.exists():
        with open(combined, encoding="utf-8") as f:
            return json.load(f)

    # fallback: merge individual files
    chunks = []
    for p in sorted(Path(chunks_dir).glob("*_chunks.json")):
        with open(p, encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ────────────────────────────────────────────────────────────────────────────

def analyse(chunks: list[dict]):
    stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "tokens": 0,
        "with_activity": 0, "with_equation": 0,
        "with_table": 0, "with_image": 0,
        "chapters": set()
    })

    for c in chunks:
        sub = c["subject"]
        s   = stats[sub]
        s["total"]         += 1
        s["tokens"]        += c["token_count"]
        s["with_activity"] += int(c.get("has_activity", False))
        s["with_equation"] += int(c.get("has_equation", False))
        s["with_table"]    += int(c.get("has_table", False))
        s["with_image"]    += int(c.get("has_image", False))
        if c.get("chapter_title"):
            s["chapters"].add(c["chapter_title"])

    rows = []
    for sub, s in sorted(stats.items()):
        rows.append([
            sub,
            s["total"],
            s["tokens"],
            round(s["tokens"] / max(s["total"], 1)),
            len(s["chapters"]),
            s["with_activity"],
            s["with_equation"],
            s["with_table"],
            s["with_image"],
        ])

    headers = ["Subject", "Chunks", "TotalTokens", "AvgTokens",
               "Chapters", "HasActivity", "HasEquation", "HasTable", "HasImage"]

    if HAS_TABULATE:
        print("\n" + tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("\n" + "\t".join(headers))
        for r in rows:
            print("\t".join(str(x) for x in r))

    return rows, headers


# ────────────────────────────────────────────────────────────────────────────
# EXPORT CSV
# ────────────────────────────────────────────────────────────────────────────

EXPORT_FIELDS = [
    "chunk_id", "subject", "source_file",
    "chapter_number", "chapter_title",
    "section_title", "subsection_title",
    "heading_path",
    "chunk_index", "start_line", "end_line",
    "token_count", "char_count",
    "has_activity", "has_equation", "has_table", "has_image",
    "keywords",
    # NOTE: 'text' is intentionally omitted from CSV to keep it readable
]

def export_csv(chunks: list[dict], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for c in chunks:
            row = {k: c.get(k, "") for k in EXPORT_FIELDS}
            # flatten list fields
            if isinstance(row.get("keywords"), list):
                row["keywords"] = ", ".join(row["keywords"])
            writer.writerow(row)
    print(f"\n📋 Metadata CSV exported → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# METADATA SCHEMA PRINTER
# ────────────────────────────────────────────────────────────────────────────

SCHEMA = {
    "chunk_id":          "Unique ID: <subject>_ch<nn>_<index>",
    "source_file":       "Original .md file name",
    "subject":           "Subject name (Physics, Biology, Maths_sem_1 …)",
    "chapter_number":    "Chapter number as a string ('1', '2' …) or null",
    "chapter_title":     "Full H1 heading text",
    "section_title":     "H2 heading (empty string if none)",
    "subsection_title":  "H3 heading (empty string if none)",
    "heading_path":      "Breadcrumb: Chapter > Section > Subsection",
    "text":              "Raw chunk content (Markdown)",
    "token_count":       "Approximate tokens (cl100k_base / word count fallback)",
    "char_count":        "Character length of text",
    "chunk_index":       "0-based position within the source file",
    "start_line":        "Approx. start line in source file",
    "end_line":          "Approx. end line in source file",
    "has_activity":      "True if chunk contains an 'Activity N.N' heading/reference",
    "has_equation":      "True if chunk contains $…$ or \\[…\\] math",
    "has_table":         "True if chunk contains a Markdown table (|col|)",
    "has_image":         "True if chunk contains an ![…](…) image reference",
    "keywords":          "Top-10 domain keywords by frequency (stop-words removed)",
}

def print_schema():
    print("\n━━━━ CHUNK METADATA SCHEMA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for field, desc in SCHEMA.items():
        print(f"  {field:<22}  {desc}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


# ────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse chunk metadata")
    parser.add_argument("--chunks_dir", default="./chunks",
                        help="Directory produced by chunker.py")
    parser.add_argument("--export_csv", default="",
                        help="Path to export metadata CSV (optional)")
    args = parser.parse_args()

    print_schema()

    chunks = load_all_chunks(args.chunks_dir)
    if not chunks:
        print("No chunks found. Run chunker.py first.")
        raise SystemExit(1)

    print(f"Loaded {len(chunks)} chunks from {args.chunks_dir}\n")
    analyse(chunks)

    csv_path = args.export_csv or str(Path(args.chunks_dir) / "metadata.csv")
    export_csv(chunks, csv_path)

    # print one example chunk (without full text)
    ex = {k: v for k, v in chunks[0].items() if k != "text"}
    ex["text_preview"] = chunks[0]["text"][:300] + "…"
    print("\n━━━━ EXAMPLE CHUNK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(json.dumps(ex, indent=2, ensure_ascii=False))
