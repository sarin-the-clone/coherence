"""
Compute SC scalars for bin 4 (800k+ token prefix) of all generated runs.

Runs in parallel across 3 groups. Each group writes to a temp file;
a final merge step adds results back into sc_prefix_bins.json.

Usage:
    python compute_sc_prefix_bin4.py --group 0   # → /tmp/sc_bin4_g0.json
    python compute_sc_prefix_bin4.py --group 1   # → /tmp/sc_bin4_g1.json
    python compute_sc_prefix_bin4.py --group 2   # → /tmp/sc_bin4_g2.json
    python compute_sc_prefix_bin4.py --merge      # merge all 3 into sc_prefix_bins.json
"""

import argparse
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
PREFIX_FILE  = "generated_texts_og/Output/sc_prefix_bins.json"

BATCH_SIZE   = 64
MAX_LENGTH   = 512
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
WINDOWS      = [1, 2, 3, 4]

CHARS_PER_TOKEN = 4.0
BIN4_N_TOKENS   = 1_072_000          # median natural bin-4 book length
BIN4_N_CHARS    = int(BIN4_N_TOKENS * CHARS_PER_TOKEN)

SCALAR_KEYS = ["off_diag_mean", "novelty_mean", "sim_lag1_mean", "mean_sim_to_past_mean"]

OUTPUT_BY_GROUP = {0: "/tmp/sc_bin4_g0.json",
                   1: "/tmp/sc_bin4_g1.json",
                   2: "/tmp/sc_bin4_g2.json"}

# ---------------------------------------------------------------------------
# Helpers (copied from compute_sc_prefix_bins.py)
# ---------------------------------------------------------------------------

def load_full_text(run_dir):
    path = os.path.join(run_dir, "token_naturalness_by_chunk.jsonl")
    accepted = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            accepted[r["chunk_id"]] = r["accepted_attempt"]

    all_chunks = {}
    for fname in os.listdir(run_dir):
        m = re.match(r"chunk_(\d+)_attempt_(\d+)\.txt$", fname)
        if m:
            cid, att = int(m.group(1)), int(m.group(2))
            all_chunks.setdefault(cid, []).append(att)

    text = ""
    for cid in sorted(all_chunks.keys()):
        att = accepted.get(cid, 1 if 1 in all_chunks[cid] else min(all_chunks[cid]))
        fname = os.path.join(run_dir, f"chunk_{cid:03d}_attempt_{att}.txt")
        with open(fname) as cf:
            text += json.load(cf).get("new_text", "")
    return text


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge():
    with open(PREFIX_FILE) as f:
        all_results = json.load(f)

    added = 0
    for g in range(3):
        path = OUTPUT_BY_GROUP[g]
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            g_results = json.load(f)
        for run_id, bin4_entry in g_results.items():
            if run_id in all_results:
                all_results[run_id]["bins"]["4"] = bin4_entry
                added += 1
            else:
                print(f"  WARNING: {run_id} not in {PREFIX_FILE}")

    with open(PREFIX_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Merged bin 4 for {added} runs into {PREFIX_FILE}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, choices=[0, 1, 2],
                        help="Which shard to process (0, 1, or 2)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge all group outputs into sc_prefix_bins.json")
    args = parser.parse_args()

    if args.merge:
        merge()
        return

    if args.group is None:
        parser.error("--group or --merge required")

    output_path = OUTPUT_BY_GROUP[args.group]

    with open(PREFIX_FILE) as f:
        prefix_data = json.load(f)

    all_run_ids = [r for r, v in prefix_data.items() if "4" not in v.get("bins", {})]
    group_run_ids = all_run_ids[args.group::3]

    results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)

    pending = [r for r in group_run_ids if r not in results]
    print(f"Group   : {args.group}")
    print(f"Device  : {DEVICE}")
    print(f"Runs    : {len(group_run_ids)} total, {len(pending)} to compute")
    print(f"Bin 4   : {BIN4_N_TOKENS:,} tokens = {BIN4_N_CHARS:,} chars\n")

    print(f"Loading {DEFAULT_MODEL} ...")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s\n")

    for i, run_id in enumerate(group_run_ids):
        if run_id in results:
            print(f"[{i+1}/{len(group_run_ids)}] skip  {run_id[:60]}")
            continue

        print(f"\n[{i+1}/{len(group_run_ids)}] {run_id[:60]}")
        t_run = time.perf_counter()

        run_dir = os.path.join(PROMPTS_DIR, run_id)
        try:
            full_text = load_full_text(run_dir)
        except Exception as e:
            print(f"  ERROR loading text: {e}")
            continue

        n_chars = len(full_text)
        print(f"  Full text: {n_chars:,} chars  ≈ {n_chars/CHARS_PER_TOKEN/1e6:.2f}M tokens")

        if n_chars < BIN4_N_CHARS:
            print(f"  Text too short for bin 4 ({n_chars:,} < {BIN4_N_CHARS:,} chars), skip")
            continue

        prefix = full_text[:BIN4_N_CHARS]

        sc = SemanticCoherence(
            text=prefix,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            tokenizer=tokenizer,
            model=model,
        )

        bin4_entry = {}
        for w in WINDOWS:
            t_w = time.perf_counter()
            sc.compute_window(w)
            wname = WINDOW_NAMES[w]
            prefix_key = f"sc_{wname}_"
            scalars = {k: sc.results.get(f"{prefix_key}{k}") for k in SCALAR_KEYS}
            bin4_entry[wname] = scalars
            print(f"  {wname}: odm={scalars['off_diag_mean']:.3f}  "
                  f"nov={scalars['novelty_mean']:.3f}  "
                  f"sl1={scalars['sim_lag1_mean']:.3f}  "
                  f"({time.perf_counter()-t_w:.0f}s)")

        results[run_id] = bin4_entry
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  saved ({time.perf_counter()-t_run:.0f}s total)")

    print(f"\nDone. {output_path}")


if __name__ == "__main__":
    main()
