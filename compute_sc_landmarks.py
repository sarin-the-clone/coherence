"""
Compute SC scalars (W2k/W8k/W40k) for the 12 landmark books, so they can be
projected into the combined PLS-DA score space alongside the natural + generated
books. Landmarks are not in the SC subcorpus and have no SC computed, so we
compute them here from their Selected_Books full texts (same pipeline as the
natural SC in compute_sc_natural_extra.py).

Output: /tmp/landmark_sc.json  keyed by gutenberg id ->
        {title, n_tokens, W2k:{...}, W8k:{...}, W40k:{...}}
"""

import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherence, WINDOW_NAMES, DEFAULT_MODEL

LANDMARK_CSV = "natural_texts/Output/pca/generated_prefixes/landmark_prefix_projection_scores.csv"
SELECTED_DIR = "natural_texts/Selected_Books"
OUTPUT_PATH  = "/tmp/landmark_sc.json"

WINDOWS     = [2, 3, 4]          # W2k / W8k / W40k (W1k dropped from the model)
SCALAR_KEYS = ["off_diag_mean", "novelty_mean", "sim_lag1_mean"]
BATCH_SIZE  = 64
MAX_LENGTH  = 512
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def landmark_endpoints() -> pd.DataFrame:
    lm = pd.read_csv(LANDMARK_CSV)
    ep = lm.loc[lm.groupby("title")["n_tokens"].idxmax(),
                ["title", "file", "source_text_file", "n_tokens"]].reset_index(drop=True)
    ep["gid"] = ep["file"].astype(str).str.extract(r"^(\d+)_")
    return ep


def main():
    books = landmark_endpoints()
    print(f"Landmarks: {len(books)}   Device: {DEVICE}   Output: {OUTPUT_PATH}\n")

    results = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            results = json.load(f)

    pending = [r for _, r in books.iterrows() if str(r["gid"]) not in results]
    print(f"{len(results)} already done, {len(pending)} to compute\n")
    if not pending:
        print("nothing to do."); return

    print(f"Loading {DEFAULT_MODEL} ...")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s\n")

    for i, r in enumerate(pending):
        gid = str(r["gid"])
        path = os.path.join(SELECTED_DIR, str(r["source_text_file"]))
        print(f"[{i+1}/{len(pending)}] {r['title'][:44]}  (gid {gid})")
        t_book = time.perf_counter()
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        print(f"  {len(text):,} chars")

        sc = SemanticCoherence(text=text, device=DEVICE, batch_size=BATCH_SIZE,
                               max_length=MAX_LENGTH, tokenizer=tokenizer, model=model)
        entry = {"title": r["title"], "n_tokens": int(r["n_tokens"])}
        for w in WINDOWS:
            t_w = time.perf_counter()
            sc.compute_window(w)
            wname = WINDOW_NAMES[w]
            pk = f"sc_{wname}_"
            entry[wname] = {k: sc.results.get(f"{pk}{k}") for k in SCALAR_KEYS}
            print(f"  {wname}: odm={entry[wname]['off_diag_mean']:.3f} "
                  f"nov={entry[wname]['novelty_mean']:.3f} "
                  f"sl1={entry[wname]['sim_lag1_mean']:.3f}  ({time.perf_counter()-t_w:.0f}s)")

        results[gid] = entry
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved ({time.perf_counter()-t_book:.0f}s)\n")

    print(f"Done -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
