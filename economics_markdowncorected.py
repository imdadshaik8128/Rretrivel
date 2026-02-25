import re

def fix_economics_headings(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    corrected_lines = []
    keep_next_as_h1 = False

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("#"):
            corrected_lines.append(line)
            continue

        heading_text = stripped.lstrip("#").strip()

        # ----------------------------
        # RULE 1: Keep Chapter as H1
        # ----------------------------
        if re.match(r"^CHAPTER\s+\d+", heading_text, re.IGNORECASE):
            corrected_lines.append(f"# {heading_text}\n")
            keep_next_as_h1 = True
            continue

        # ----------------------------
        # RULE 2: Keep immediate title after Chapter as H1
        # ----------------------------
        if keep_next_as_h1:
            corrected_lines.append(f"# {heading_text}\n")
            keep_next_as_h1 = False
            continue

        # ----------------------------
        # RULE 3: Convert other single # to ##
        # ----------------------------
        if stripped.startswith("# ") and not stripped.startswith("##"):
            corrected_lines.append(f"## {heading_text}\n")
            continue

        corrected_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    print("✅ Chapter and Title kept as H1. Others normalized.")

fix_economics_headings("Economics.md","new_Economics.md")