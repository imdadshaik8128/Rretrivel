import re

def standardize_textbook_hierarchy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    corrected_lines = []
    last_section_depth = 0

    for line in lines:
        stripped = line.strip()

        if not stripped.startswith("#"):
            corrected_lines.append(line)
            continue

        heading_text = stripped.lstrip("#").strip()

        # ----------------------------------
        # KEEP ALL LEVEL 1 HEADINGS AS IS
        # ----------------------------------
        if stripped.startswith("# ") and not stripped.startswith("##"):
            corrected_lines.append(line)
            last_section_depth = 1
            continue

        # ----------------------------------
        # DETECT NUMBERED SECTIONS
        # ----------------------------------
        match = re.match(r"^(\d+(\.\d+)*)", heading_text)

        if match:
            section_number = match.group(1)
            dot_count = section_number.count(".")

            if dot_count == 0:
                corrected_lines.append(f"## {heading_text}\n")
                last_section_depth = 2

            elif dot_count == 1:
                corrected_lines.append(f"## {heading_text}\n")
                last_section_depth = 2

            else:
                corrected_lines.append(f"### {heading_text}\n")
                last_section_depth = 3

            continue

        # ----------------------------------
        # ACTIVITY RULE (SUBSECTION AWARE)
        # ----------------------------------
        if "activity" in heading_text.lower():

            if last_section_depth == 3:
                # If under 1.1.1 type subsection
                corrected_lines.append(f"#### {heading_text}\n")
            else:
                corrected_lines.append(f"## {heading_text}\n")

            continue

        # ----------------------------------
        # DEFAULT
        # ----------------------------------
        corrected_lines.append(line)

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    print("✅ Hierarchy standardized correctly.")



standardize_textbook_hierarchy("Physics.md","new_Physics.md")