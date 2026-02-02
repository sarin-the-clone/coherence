import os
import json
import argparse
import sys

# Allow imports from coherence root
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from tokenizer import Tokenizer
from metrics import CoherenceMetrics


def load_metadata_for_book(book_path):
    meta_path = book_path.replace(".txt", "_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def compute_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = Tokenizer(kind="gpt4", add_space_punct=False, lowercase=False)
    tokens = tok.encode(text)

    cm = CoherenceMetrics(tokens)
    return cm.compute_all(), len(tokens)


def process_folder(folder, output_json):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
    files.sort()

    results = []

    for path in files:
        print(f"\n[Processing] {path}")

        meta = load_metadata_for_book(path)
        metrics, n_tokens = compute_metrics(path)

        entry = {
            **(meta or {}),
            "file": os.path.basename(path),
            "n_tokens": n_tokens,
            **metrics
        }

        results.append(entry)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", output_json)
    return results


def process_single(path, out_json):
    meta = load_metadata_for_book(path)
    metrics, n_tokens = compute_metrics(path)

    entry = {
        **(meta or {}),
        "file": os.path.basename(path),
        "n_tokens": n_tokens,
        **metrics
    }

    out_json = out_json or path.replace(".txt", "_metrics.json")
    with open(out_json, "w") as f:
        json.dump([entry], f, indent=2)

    print("\nSaved:", out_json)
    return [entry]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=None, help="Path to the folder containing generated texts. Default is generated_texts_og/Texts")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to a single file to process")
    parser.add_argument("--out", type=str, default=None, help="Path to the output JSON file. Default is generated_texts_og/Output/gmetrics_all.json")
    args = parser.parse_args()

    if args.file:
        return process_single(args.file, args.out)

    # Default to generated_texts_og if no folder is specified
    folder = args.folder
    if folder is None:
        # Try to find generated_texts_og relative to the script or root
        possible_paths = [
            os.path.join(ROOT_DIR, "generated_texts_og", "Texts"),
            os.path.join(os.path.dirname(__file__), "..", "generated_texts_og", "Texts")
        ]
        for p in possible_paths:
            if os.path.exists(p):
                folder = p
                break
        
        if folder is None:
            print("Error: Could not find default generated_texts_og/Texts folder. Please specify --folder.")
            return

    out_json = args.out
    if out_json is None:
        # Default output location relative to the input folder's parent
        parent_dir = os.path.dirname(folder)
        out_json = os.path.join(parent_dir, "Output", "gmetrics_all.json")

    return process_folder(folder, out_json)


if __name__ == "__main__":
    main()
