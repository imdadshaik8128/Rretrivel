def fix_activity_hierarchy(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    corrected_lines = []
    i = 0
    total_lines = len(lines)

    while i < total_lines:
        line = lines[i].strip()

        # Check for ### Activity
        if line.startswith("### ") and "activity" in line.lower():

            activity_line = line.replace("### ", "").strip()

            # Look ahead skipping blank lines
            j = i + 1
            while j < total_lines and lines[j].strip() == "":
                j += 1

            # If next non-empty line is ## heading
            if j < total_lines and lines[j].strip().startswith("## "):

                heading_line = lines[j].strip().replace("## ", "").strip()

                # Write corrected order
                corrected_lines.append(f"## {activity_line}\n\n")
                corrected_lines.append(f"### {heading_line}\n\n")

                i = j + 1
                continue

        # Default case
        corrected_lines.append(lines[i])
        i += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    print("✅ Activity hierarchy corrected successfully.")


fix_activity_hierarchy("Biology.md","new_biology.md")