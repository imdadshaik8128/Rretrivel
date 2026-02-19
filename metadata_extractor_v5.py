"""
metadata_extractor.py
=====================
Reads chunk JSON files produced by chunker.py and:
  1. Prints updated schema (includes chunk_type, section_type, activity_number)
  2. Prints per-subject breakdown with theory / exercise / activity counts
  3. Prints section_type distribution (what kinds of sections exist per subject)
  4. Exports a flat CSV for inspection in Excel / Sheets

Usage:
    python metadata_extractor.py --chunks_dir ./chunks
    python metadata_extractor.py --chunks_dir ./chunks --export_csv ./chunks/metadata.csv

Dependencies:
    pip install tabulate
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter

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

    chunks = []
    for p in sorted(Path(chunks_dir).glob("*_chunks.json")):
        with open(p, encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


# ────────────────────────────────────────────────────────────────────────────
# ANALYSIS — per-subject summary
# ────────────────────────────────────────────────────────────────────────────

def analyse(chunks: list[dict]):
    stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "tokens": 0,
        "theory": 0, "exercise": 0, "activity": 0,
        "has_equation": 0, "has_table": 0, "has_image": 0,
        "chapters": set(),
    })

    for c in chunks:
        sub = c["subject"]
        s   = stats[sub]
        s["total"]        += 1
        s["tokens"]       += c.get("token_count", 0)
        s["has_equation"] += int(c.get("has_equation", False))
        s["has_table"]    += int(c.get("has_table", False))
        s["has_image"]    += int(c.get("has_image", False))
        ctype = c.get("chunk_type", "theory")
        s[ctype]          += 1
        if c.get("chapter_title"):
            s["chapters"].add(c["chapter_title"])

    rows = []
    for sub, s in sorted(stats.items()):
        rows.append([
            sub,
            s["total"],
            s["theory"],
            s["exercise"],
            s["activity"],
            round(s["tokens"] / max(s["total"], 1)),
            len(s["chapters"]),
            s["has_equation"],
            s["has_table"],
            s["has_image"],
        ])

    headers = ["Subject", "Total", "Theory", "Exercise", "Activity",
               "AvgTokens", "Chapters", "Equations", "Tables", "Images"]

    print("\n━━━━ PER-SUBJECT BREAKDOWN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(str(x) for x in r))

    return rows, headers


# ────────────────────────────────────────────────────────────────────────────
# ANALYSIS — section_type distribution
# ────────────────────────────────────────────────────────────────────────────

def analyse_section_types(chunks: list[dict]):
    """
    Show how many chunks of each section_type exist per subject.
    Helps verify the classification is working correctly.
    """
    # subject → Counter of section_types
    by_subject: dict[str, Counter] = defaultdict(Counter)

    for c in chunks:
        stype = c.get("section_type", "") or "(theory — no label)"
        by_subject[c["subject"]][stype] += 1

    print("\n━━━━ SECTION_TYPE DISTRIBUTION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for sub in sorted(by_subject):
        print(f"\n  {sub}")
        for stype, count in by_subject[sub].most_common():
            bar = "█" * min(count // 2, 40)
            print(f"    {stype:<30}  {count:>4}  {bar}")


# ────────────────────────────────────────────────────────────────────────────
# VALIDATION — flag potential misclassifications
# ────────────────────────────────────────────────────────────────────────────

def validate(chunks: list[dict]):
    """
    Quick sanity checks — prints warnings for suspicious chunks.
    """
    warnings = []

    for c in chunks:
        ctype  = c.get("chunk_type", "theory")
        stype  = c.get("section_type", "")
        hpath  = c.get("heading_path", "")
        text   = c.get("text", "")

        # Activity chunks with no activity_number — may need manual check
        if ctype == "activity" and not c.get("activity_number"):
            warnings.append(
                f"  [no activity_number]  {c['chunk_id']}  |  {hpath[:60]}"
            )

        # Very short chunks — likely OCR noise or mis-split
        if c.get("token_count", 0) < 20 and ctype == "theory":
            warnings.append(
                f"  [tiny theory chunk]   {c['chunk_id']}  tokens={c['token_count']}"
                f"  |  {text[:60]!r}"
            )

    if warnings:
        print(f"\n━━━━ VALIDATION WARNINGS ({len(warnings)}) ━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for w in warnings[:30]:   # cap at 30 to avoid flooding
            print(w)
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more")
    else:
        print("\n✅  No validation warnings.")


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
    # Classification (new fields)
    "chunk_type", "section_type", "activity_number",
    # Content flags
    "has_activity", "has_equation", "has_table", "has_image",
    "keywords",
    # NOTE: 'text' intentionally omitted — CSV would be unreadable
]

def export_csv(chunks: list[dict], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for c in chunks:
            row = {k: c.get(k, "") for k in EXPORT_FIELDS}
            if isinstance(row.get("keywords"), list):
                row["keywords"] = ", ".join(row["keywords"])
            writer.writerow(row)
    print(f"\n📋 Metadata CSV exported → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# SCHEMA
# ────────────────────────────────────────────────────────────────────────────

SCHEMA = {
    "chunk_id":          "Unique ID: <subject>_ch<nn>_<index>",
    "source_file":       "Original .md file name",
    "subject":           "Subject name (Physics, Biology, Maths_sem_1 …)",
    "chapter_number":    "Chapter number as string ('1', '2' …) or null",
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
    # Classification
    "chunk_type":        "'theory' | 'exercise' | 'activity'",
    "section_type":      "Specific label: 'fill_in_the_blanks', 'info_box', 'activity' …",
    "activity_number":   "Activity number string e.g. '3' or '3.2' (activity chunks only)",
    # Flags
    "has_activity":      "True if body text references an Activity N.N",
    "has_equation":      "True if chunk contains $…$ or \\[…\\] math",
    "has_table":         "True if chunk contains a Markdown table (|col|)",
    "has_image":         "True if chunk contains an ![…](…) image reference",
    "keywords":          "Top-10 domain keywords by frequency (stop-words removed)",
}

def print_schema():
    print("\n━━━━ CHUNK METADATA SCHEMA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    for field, desc in SCHEMA.items():
        print(f"  {field:<22}  {desc}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


# ────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse chunk metadata")
    parser.add_argument("--chunks_dir",  default="./chunks",
                        help="Directory produced by chunker.py")
    parser.add_argument("--export_csv",  default="",
                        help="CSV output path (default: chunks_dir/metadata.csv)")
    parser.add_argument("--no_validate", action="store_true",
                        help="Skip validation warnings")
    args = parser.parse_args()

    print_schema()

    chunks = load_all_chunks(args.chunks_dir)
    if not chunks:
        print("No chunks found. Run chunker.py first.")
        raise SystemExit(1)

    print(f"Loaded {len(chunks)} chunks from {args.chunks_dir}")

    # Overall totals
    total_theory   = sum(1 for c in chunks if c.get("chunk_type") == "theory")
    total_exercise = sum(1 for c in chunks if c.get("chunk_type") == "exercise")
    total_activity = sum(1 for c in chunks if c.get("chunk_type") == "activity")
    print(f"\nOverall → theory={total_theory}  exercise={total_exercise}  activity={total_activity}")

    analyse(chunks)
    analyse_section_types(chunks)

    if not args.no_validate:
        validate(chunks)

    csv_path = args.export_csv or str(Path(args.chunks_dir) / "metadata.csv")
    export_csv(chunks, csv_path)

    # Print one example chunk of each type
    for ctype in ("theory", "exercise", "activity"):
        example = next((c for c in chunks if c.get("chunk_type") == ctype), None)
        if example:
            ex = {k: v for k, v in example.items() if k != "text"}
            ex["text_preview"] = example["text"][:200] + "…"
            print(f"\n━━━━ EXAMPLE {ctype.upper()} CHUNK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(json.dumps(ex, indent=2, ensure_ascii=False))
