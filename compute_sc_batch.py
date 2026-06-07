"""
Batch retrospective SemanticCoherence computation across all generated runs.

Loads the referee model once, processes every run with SemanticCoherence
(batched NxN retrospective), and appends results to a single JSON file.

Skips:
  - runs with conditioning_mode in SKIP_MODES
  - runs with temperature_range matching SKIP_TEMPS
  - runs that already have an entry in the output JSON

Output:
  generated_texts_og/Output/sc_batch_results.json
  {run_id: {conditioning_mode, genre, total_tokens, W1k: {...}, W2k: {...}, ...}}

Usage:
    python compute_sc_batch.py [--windows 1 2 3 4]
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherence, WINDOW_NAMES, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPTS_DIR  = "generated_texts_og/prompts"
OUTPUT_FILE  = "generated_texts_og/Output/sc_batch_results.json"

SKIP_MODES   = set()                              # none skipped by mode
SKIP_TEMPS   = {str([0.6, 0.9])}                 # skip low-temperature summaries runs

STRIDE_FACTOR = 1.0
BATCH_SIZE    = 64
MAX_LENGTH    = 512
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

SCALAR_KEYS = [
    "n_windows", "off_diag_mean", "off_diag_std",
    "novelty_mean", "novelty_std", "novelty_final",
    "mean_sim_to_past_mean", "mean_sim_to_past_final",
    "sim_lag1_mean", "sim_lag1_std",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_full_text(run_dir):
    path = os.path.join(run_dir, "token_naturalness_by_chunk.jsonl")
    text = ""
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            fname = os.path.join(run_dir,
                f"chunk_{r['chunk_id']:03d}_attempt_{r['accepted_attempt']}.txt")
            with open(fname) as cf:
                text += json.load(cf).get("new_text", "")
    return text


def load_metadata(run_dir):
    path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def collect_runs(windows):
    """Return list of (run_id, run_dir) to process, respecting skip rules."""
    out = []
    for name in sorted(os.listdir(PROMPTS_DIR)):
        run_dir = os.path.join(PROMPTS_DIR, name)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.isdir(run_dir) or not os.path.exists(meta_path):
            continue
        m = load_metadata(run_dir)
        mode = m.get("conditioning_mode", "")
        temp = str(m.get("temperature_range", ""))
        if mode in SKIP_MODES:
            continue
        if temp in SKIP_TEMPS:
            continue
        out.append((name, run_dir, m))
    return out


def extract_scalars(results, window):
    name = WINDOW_NAMES[window]
    prefix = f"sc_{name}_"
    return {k: results.get(f"{prefix}{k}") for k in SCALAR_KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(windows):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load existing results
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    runs = collect_runs(windows)
    print(f"Device  : {DEVICE}")
    print(f"Windows : {', '.join(WINDOW_NAMES[w] for w in windows)}")
    print(f"Runs    : {len(runs)} total, "
          f"{sum(1 for r,_,_ in runs if r not in all_results)} to compute")
    print(f"Output  : {OUTPUT_FILE}\n")

    # Load model once
    print(f"Loading {DEFAULT_MODEL} ...")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

    for i, (run_id, run_dir, meta) in enumerate(runs):
        if run_id in all_results:
            print(f"[{i+1}/{len(runs)}] {run_id[:55]}  (skip — already done)")
            continue

        print(f"[{i+1}/{len(runs)}] {run_id[:55]}")
        t_run = time.perf_counter()

        try:
            text = load_full_text(run_dir)
        except Exception as e:
            print(f"  ERROR loading text: {e}")
            continue

        sc = SemanticCoherence(
            text=text,
            stride_factor=STRIDE_FACTOR,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            tokenizer=tokenizer,
            model=model,
        )

        entry = {
            "run_id":             run_id,
            "conditioning_mode":  meta.get("conditioning_mode", "?"),
            "broader_genre":      meta.get("broader_genre", "?"),
            "temperature_range":  meta.get("temperature_range"),
            "total_tokens":       meta.get("total_tokens"),
            "n_chars":            len(text),
        }

        for w in windows:
            t_w = time.perf_counter()
            sc.compute_window(w)
            scalars = extract_scalars(sc.results, w)
            entry[WINDOW_NAMES[w]] = scalars
            print(f"  {WINDOW_NAMES[w]:5s}: {scalars['n_windows']:5d} windows  "
                  f"odm={scalars['off_diag_mean']:.3f}  "
                  f"nov={scalars['novelty_mean']:.3f}  "
                  f"sl1={scalars['sim_lag1_mean']:.3f}  "
                  f"({time.perf_counter()-t_w:.0f}s)")

        all_results[run_id] = entry

        # Save after each run so progress is not lost
        with open(OUTPUT_FILE, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  → saved  ({time.perf_counter()-t_run:.0f}s total)\n")

    print(f"Done. {len(all_results)} runs in {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int,
                        default=[1, 2, 3, 4],
                        help="Window scales to compute (1=W1k 2=W2k 3=W8k 4=W40k)")
    args = parser.parse_args()
    main(args.windows)
