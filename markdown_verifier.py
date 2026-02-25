import os
import re
ROOT_FOLDER = "Subjects_v2"  # your root folder


def extract_hierarchy(md_path):
    hierarchy_lines = []

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")

            # Ignore lines that are not headings
            if not stripped.lstrip().startswith("#"):
                continue

            # Remove leading spaces safely
            clean_line = stripped.lstrip()

            # Count hashes manually
            hash_count = 0
            for ch in clean_line:
                if ch == "#":
                    hash_count += 1
                else:
                    break

            # Extract title after hashes
            title = clean_line[hash_count:].strip()

            # Create indentation
            indent = "  " * (hash_count - 1)

            # Preserve original hash level
            hierarchy_lines.append(f"{indent}{'#' * hash_count} {title}")

    return hierarchy_lines


def process_root_folder(root_folder):
    for subject in os.listdir(root_folder):
        subject_path = os.path.join(root_folder, subject)

        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):
            if not file.lower().endswith(".md"):
                continue

            md_path = os.path.join(subject_path, file)

            hierarchy = extract_hierarchy(md_path)

            if not hierarchy:
                continue

            output_filename = f"{subject}__{os.path.splitext(file)[0]}_hierarchy.txt"
            output_path = os.path.join(root_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(hierarchy))

            print(f"✔ Saved hierarchy: {output_path}")


def main():
    process_root_folder(ROOT_FOLDER)
    print("\n✅ ALL HIERARCHIES EXTRACTED SUCCESSFULLY")


if __name__ == "__main__":
    main()
