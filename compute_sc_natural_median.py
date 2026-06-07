"""
Compute SC scalars for the 5 natural median books (one per size bin).

Reads texts from the Gutenberg benchmark cache zip and runs
SemanticCoherence (retrospective, stride=1.0) for all window scales.

Output:
    /tmp/natural_books_sc_median.json
    {book_id: {title, bin, n_tokens, W1k: {...}, W2k: {...}, ...}}

Usage:
    python compute_sc_natural_median.py
"""

import json
import os
import sys
import time
import zipfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherence, WINDOW_NAMES, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ZIP_PATH   = "natural_texts/Output/gutenberg_benchmark_cache/books.zip"
OUTPUT     = "/tmp/natural_books_sc_median.json"
WINDOWS    = [1, 2, 3, 4]
BATCH_SIZE = 64
MAX_LENGTH = 512
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SCALAR_KEYS = ["off_diag_mean", "novelty_mean", "sim_lag1_mean", "mean_sim_to_past_mean"]

# Three books per bin: median representative + 2 genre-diverse additions
MEDIAN_BOOKS = [
    # Bin 0 (~121k tokens)
    {"book_id": 62081, "bin": 0, "title": "The Timber Pirate",
     "member": "books/62081.txt"},
    {"book_id":  1448, "bin": 0, "title": "Heidi",
     "member": "books/1448.txt"},
    {"book_id": 37948, "bin": 0, "title": "The Beautiful White Devil",
     "member": "books/37948.txt"},
    # Bin 1 (~194k tokens)
    {"book_id": 63866, "bin": 1, "title": "Hero-Tales of Ireland",
     "member": "books/63866.txt"},
    {"book_id": 24000, "bin": 1, "title": "Miss Mackenzie",
     "member": "books/24000.txt"},
    {"book_id": 28791, "bin": 1, "title": "A Son of the Middle Border",
     "member": "books/28791.txt"},
    # Bin 2 (~354k tokens)
    {"book_id": 70731, "bin": 2, "title": "A History of English Literature",
     "member": "books/70731.txt"},
    {"book_id":  1257, "bin": 2, "title": "The Three Musketeers",
     "member": "books/1257.txt"},
    {"book_id":  6400, "bin": 2, "title": "Lives of the Twelve Caesars",
     "member": "books/6400.txt"},
    # Bin 3 (~580k tokens)
    {"book_id":   996, "bin": 3, "title": "Don Quixote",
     "member": "books/996.txt"},
    {"book_id": 38194, "bin": 3, "title": "Wealth of Nations",
     "member": "books/38194.txt"},
    {"book_id": 14300, "bin": 3, "title": "Life of Napoleon I (Complete)",
     "member": "books/14300.txt"},
    # Bin 4 (~1M tokens)
    {"book_id": 10706, "bin": 4, "title": "The History of Rome (Vols 1-5)",
     "member": "books/10706.txt"},
    {"book_id": 44700, "bin": 4, "title": "Holinshed Chronicles",
     "member": "books/44700.txt"},
    {"book_id": 15474, "bin": 4, "title": "The Mahabharata Vol. 1",
     "member": "books/15474.txt"},
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device : {DEVICE}")
    print(f"Zip    : {ZIP_PATH}\n")

    print(f"Loading {DEFAULT_MODEL} ...")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

    results = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            results = json.load(f)

    with zipfile.ZipFile(ZIP_PATH) as zf:
        for binfo in MEDIAN_BOOKS:
            bid = str(binfo["book_id"])
            if bid in results:
                print(f"Bin {binfo['bin']}  {binfo['title'][:40]}  (skip)")
                continue

            print(f"\nBin {binfo['bin']}  {binfo['title'][:40]}")
            t_book = time.perf_counter()

            text = zf.read(binfo["member"]).decode("utf-8", errors="replace")
            n_chars = len(text)
            print(f"  {n_chars:,} chars  ≈ {n_chars/4e6:.2f}M tokens")

            sc = SemanticCoherence(
                text=text,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
                tokenizer=tokenizer,
                model=model,
            )

            entry = {
                "book_id":  binfo["book_id"],
                "title":    binfo["title"],
                "bin":      binfo["bin"],
                "n_chars":  n_chars,
                "n_tokens": int(n_chars / 4.0),
            }

            for w in WINDOWS:
                t_w = time.perf_counter()
                sc.compute_window(w)
                wname = WINDOW_NAMES[w]
                prefix_key = f"sc_{wname}_"
                scalars = {k: sc.results.get(f"{prefix_key}{k}") for k in SCALAR_KEYS}
                n_win = sc.results.get(f"{prefix_key}n_windows", 0)
                entry[wname] = scalars
                print(f"  {wname}: {n_win} windows  "
                      f"odm={scalars['off_diag_mean']:.3f}  "
                      f"nov={scalars['novelty_mean']:.3f}  "
                      f"sl1={scalars['sim_lag1_mean']:.3f}  "
                      f"({time.perf_counter()-t_w:.0f}s)")

            results[bid] = entry
            with open(OUTPUT, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  saved ({time.perf_counter()-t_book:.0f}s total)")

    print(f"\nDone. {OUTPUT}")


if __name__ == "__main__":
    main()
