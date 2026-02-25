import json
import random
from pathlib import Path

def generate_slm_dataset(chunks_path, output_path):
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    dataset = []

    for chunk in chunks:
        ctype = chunk.get("chunk_type", "theory")
        text = chunk.get("text", "")
        hpath = chunk.get("heading_path", "General Information")
        
        # 1. Handle GLOSSARY / NEW WORDS (High value for SLMs)
        if chunk.get("section_type") == "glossary":
            # Split by dash or colon to find definitions
            lines = text.split('\n')
            for line in lines:
                if '–' in line or '-' in line:
                    parts = line.split('–') if '–' in line else line.split('-')
                    term = parts[0].strip()
                    defn = parts[1].strip()
                    dataset.append({
                        "instruction": f"Define the term '{term}' in the context of {hpath}.",
                        "context": text,
                        "response": defn
                    })

        # 2. Handle ACTIVITIES
        elif ctype == "activity":
            dataset.append({
                "instruction": f"Based on the activity described in {hpath}, explain the objective or the steps required.",
                "context": text,
                "response": f"This activity involves: {text[:200]}..." # Or full text for SLM training
            })

        # 3. Handle THEORY (General Knowledge)
        elif ctype == "theory":
            # Generate a "What is this section about" pair
            dataset.append({
                "instruction": f"Summarize the key information regarding {hpath.split(' > ')[-1]}.",
                "context": text,
                "response": f"In {hpath}, the text discusses: {text[:150]}..."
            })

    # Save as JSONL (Standard format for training)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"✅ Generated {len(dataset)} training pairs -> {output_path}")

if __name__ == "__main__":
    generate_slm_dataset("./chunks/all_chunks.json", "slm_training_set.jsonl")