"""
Compute SC scalars on generated text prefixes truncated to natural bin lengths.

For each generated run and each bin, truncates the full text to the bin's
representative char length and runs SemanticCoherence offline (one pass).

Natural bin reference lengths (from median natural books):
  Bin 0: 121,140 tok  (Timber Pirate)
  Bin 1: 193,795 tok  (Hero-Tales of Ireland)
  Bin 2: 353,917 tok  (History of English Lit.)
  Bin 3: 593,869 tok  (Don Quixote)

Output:
  generated_texts_og/Output/sc_prefix_bins.json
  {run_id: {label, conditioning_mode, bins: {0: {W1k: {...}, ...}, ...}}}

Usage:
    python compute_sc_prefix_bins.py
"""

import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherence, WINDOW_NAMES, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPTS_DIR  = "generated_texts_og/prompts"
OUTPUT_FILE  = "generated_texts_og/Output/sc_prefix_bins.json"

SKIP_TEMPS   = {str([0.6, 0.9])}

BATCH_SIZE   = 64
MAX_LENGTH   = 512
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
WINDOWS      = [1, 2, 3, 4]

CHARS_PER_TOKEN = 4.0

# Bin 4 is the open-ended top size bin (>800k tok): generated texts are taken at
# their full endpoint (no truncation), gated on the run exceeding this many tokens.
BIN4_ENDPOINT = True
BIN4_MIN_TOKENS = 800_000

# Natural median book lengths in tokens → converted to chars for truncation
# (bins 0-3). Bin 4 is handled as an endpoint, see BIN4_ENDPOINT above.
BINS = {
    0: {"n_tokens":   121_140, "label": "Timber Pirate"},
    1: {"n_tokens":   193_795, "label": "Hero-Tales"},
    2: {"n_tokens":   353_917, "label": "Hist. Eng. Lit."},
    3: {"n_tokens":   593_869, "label": "Don Quixote"},
    4: {"n_tokens": 1_072_000, "label": "endpoint (full text, >800k tok)"},
}

NATURAL_SCALARS = {
    0: {"W1k":  {"off_diag_mean": 0.240, "novelty_mean": 0.442, "sim_lag1_mean": 0.288},
        "W2k":  {"off_diag_mean": 0.373, "novelty_mean": 0.417, "sim_lag1_mean": 0.431},
        "W8k":  {"off_diag_mean": 0.654, "novelty_mean": 0.245, "sim_lag1_mean": 0.693},
        "W40k": {"off_diag_mean": 0.891, "novelty_mean": 0.087, "sim_lag1_mean": 0.889}},
    1: {"W1k":  {"off_diag_mean": 0.296, "novelty_mean": 0.415, "sim_lag1_mean": 0.345},
        "W2k":  {"off_diag_mean": 0.432, "novelty_mean": 0.378, "sim_lag1_mean": 0.491},
        "W8k":  {"off_diag_mean": 0.705, "novelty_mean": 0.207, "sim_lag1_mean": 0.757},
        "W40k": {"off_diag_mean": 0.877, "novelty_mean": 0.086, "sim_lag1_mean": 0.907}},
    2: {"W1k":  {"off_diag_mean": 0.204, "novelty_mean": 0.418, "sim_lag1_mean": 0.277},
        "W2k":  {"off_diag_mean": 0.314, "novelty_mean": 0.402, "sim_lag1_mean": 0.402},
        "W8k":  {"off_diag_mean": 0.591, "novelty_mean": 0.229, "sim_lag1_mean": 0.688},
        "W40k": {"off_diag_mean": 0.800, "novelty_mean": 0.102, "sim_lag1_mean": 0.884}},
    3: {"W1k":  {"off_diag_mean": 0.289, "novelty_mean": 0.375, "sim_lag1_mean": 0.325},
        "W2k":  {"off_diag_mean": 0.432, "novelty_mean": 0.357, "sim_lag1_mean": 0.474},
        "W8k":  {"off_diag_mean": 0.717, "novelty_mean": 0.192, "sim_lag1_mean": 0.751},
        "W40k": {"off_diag_mean": 0.904, "novelty_mean": 0.068, "sim_lag1_mean": 0.922}},
}

SCALAR_KEYS = ["off_diag_mean", "novelty_mean", "sim_lag1_mean", "mean_sim_to_past_mean"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_full_text(run_dir):
    all_chunks = {}
    for fname in os.listdir(run_dir):
        m = re.match(r"chunk_(\d+)_attempt_(\d+)\.txt$", fname)
        if m:
            cid, att = int(m.group(1)), int(m.group(2))
            all_chunks.setdefault(cid, []).append(att)

    # Build accepted-attempt map: sidecar preferred, marker files as fallback
    accepted = {}
    sidecar_path = os.path.join(run_dir, "token_naturalness_by_chunk.jsonl")
    if os.path.exists(sidecar_path):
        with open(sidecar_path) as f:
            for line in f:
                r = json.loads(line)
                accepted[r["chunk_id"]] = r["accepted_attempt"]
    else:
        for cid in all_chunks:
            marker = os.path.join(run_dir, f"chunk_{cid:03d}_accepted.txt")
            if os.path.exists(marker):
                try:
                    data = json.loads(open(marker).read())
                    accepted[cid] = data.get("attempt", min(all_chunks[cid]))
                except Exception:
                    pass

    text = ""
    for cid in sorted(all_chunks.keys()):
        att = accepted.get(cid, 1 if 1 in all_chunks[cid] else min(all_chunks[cid]))
        fname = os.path.join(run_dir, f"chunk_{cid:03d}_attempt_{att}.txt")
        with open(fname) as cf:
            text += json.load(cf).get("new_text", "")
    return text


def collect_runs():
    out = []
    for name in sorted(os.listdir(PROMPTS_DIR)):
        run_dir = os.path.join(PROMPTS_DIR, name)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.isdir(run_dir) or not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if str(meta.get("temperature_range", "")) in SKIP_TEMPS:
            continue
        out.append((name, run_dir, meta))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    runs = collect_runs()
    # pending = runs missing at least one bin
    pending = [(r, d, m) for r, d, m in runs
               if any(str(b) not in all_results.get(r, {}).get("bins", {}) for b in BINS)]
    print(f"Device  : {DEVICE}")
    print(f"Runs    : {len(runs)} total, {len(pending)} need work")
    print(f"Bins    : {list(BINS.keys())} ({[BINS[b]['n_tokens'] for b in BINS]} tokens)\n")

    print(f"Loading {DEFAULT_MODEL} ...")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

    # gpt4 tokenizer only used to count tokens for the bin-4 endpoint gate
    from tokenizer import Tokenizer
    gpt4_tok = Tokenizer(kind="gpt4")

    for i, (run_id, run_dir, meta) in enumerate(runs):
        existing_bins = all_results.get(run_id, {}).get("bins", {})
        missing_bins = [str(b) for b in BINS if str(b) not in existing_bins]
        if not missing_bins:
            print(f"[{i+1}/{len(runs)}] skip  {run_id[:60]}")
            continue

        print(f"\n[{i+1}/{len(runs)}] {run_id[:60]}")
        t_run = time.perf_counter()

        try:
            full_text = load_full_text(run_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        n_chars = len(full_text)
        print(f"  Full text: {n_chars:,} chars  ≈ {n_chars/CHARS_PER_TOKEN/1e6:.2f}M tokens")
        n_tokens_full = None  # lazily computed for the bin-4 endpoint gate

        if run_id in all_results:
            entry = all_results[run_id]
        else:
            entry = {
                "run_id":            run_id,
                "conditioning_mode": meta.get("conditioning_mode", "?"),
                "broader_genre":     meta.get("broader_genre", "?"),
                "n_chars":           n_chars,
                "bins":              {},
            }

        for bin_idx, binfo in BINS.items():
            if str(bin_idx) not in missing_bins:
                continue

            if bin_idx == 4 and BIN4_ENDPOINT:
                # open-ended top bin: full generated text (endpoint), gated on >800k tok
                if n_tokens_full is None:
                    n_tokens_full = len(gpt4_tok.encode(full_text))
                if n_tokens_full <= BIN4_MIN_TOKENS:
                    print(f"  Bin 4 (endpoint): {n_tokens_full:,} tok ≤ "
                          f"{BIN4_MIN_TOKENS:,}, not bin-4 eligible, skip")
                    continue
                prefix = full_text
                print(f"  Bin 4 (endpoint): full text, {n_tokens_full:,} tok")
            else:
                target_chars = int(binfo["n_tokens"] * CHARS_PER_TOKEN)
                if target_chars > n_chars:
                    print(f"  Bin {bin_idx} ({binfo['n_tokens']}k tok): text too short, skip")
                    continue
                prefix = full_text[:target_chars]

            t_bin = time.perf_counter()

            sc = SemanticCoherence(
                text=prefix,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                max_length=MAX_LENGTH,
                tokenizer=tokenizer,
                model=model,
            )

            bin_entry = {}
            for w in WINDOWS:
                sc.compute_window(w)
                wname = WINDOW_NAMES[w]
                prefix_key = f"sc_{wname}_"
                scalars = {k: sc.results.get(f"{prefix_key}{k}") for k in SCALAR_KEYS}
                bin_entry[wname] = scalars

            entry["bins"][str(bin_idx)] = bin_entry
            elapsed = time.perf_counter() - t_bin

            w1 = bin_entry.get("W1k", {})
            print(f"  Bin {bin_idx} ({binfo['n_tokens']}k tok): "
                  f"odm={w1.get('off_diag_mean', float('nan')):.3f}  "
                  f"nov={w1.get('novelty_mean', float('nan')):.3f}  "
                  f"sl1={w1.get('sim_lag1_mean', float('nan')):.3f}  "
                  f"({elapsed:.0f}s)")

        all_results[run_id] = entry
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  saved ({time.perf_counter()-t_run:.0f}s total)")

    print(f"\nDone. Results in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
