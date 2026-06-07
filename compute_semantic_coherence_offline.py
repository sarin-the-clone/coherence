"""
Compute SemanticCoherence trajectories offline on the full text of a run.

Simpler and faster than the chunk-by-chunk online version when the full text
is already available. No resume overhead: reads all accepted chunks, stitches
them into one string, runs all window scales in a single pass.

Outputs per run folder:
  semantic_coherence_trajectories.json — per-window novelty/sim trajectories

Usage:
    python compute_semantic_coherence_offline.py <run_folder> [<run_folder> ...]

    e.g.
    python compute_semantic_coherence_offline.py \\
        generated_texts_og/prompts/20260518_102401_dune_...
"""

import sys
import os
import json
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherence, WINDOW_CHARS, WINDOW_NAMES, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOWS = [1, 2, 3, 4]   # W1k, W2k, W8k, W40k
STRIDE_FACTOR = 0.5       # overridden by --stride CLI arg
BATCH_SIZE = 64
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHARS_PER_GPT4_TOKEN = 4.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_full_text(run_dir):
    """Stitch all accepted chunks into a single string. Returns (text, n_tokens)."""
    import re
    path = os.path.join(run_dir, "token_naturalness_by_chunk.jsonl")
    total_tokens = 0

    # Build accepted-attempt map from JSONL
    accepted = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            accepted[r["chunk_id"]] = (r["accepted_attempt"], r.get("n_tokens", 0))

    # Discover all chunk IDs from filesystem
    all_chunks = {}
    for fname in os.listdir(run_dir):
        m = re.match(r"chunk_(\d+)_attempt_(\d+)\.txt$", fname)
        if m:
            cid, att = int(m.group(1)), int(m.group(2))
            all_chunks.setdefault(cid, []).append(att)

    text = ""
    for cid in sorted(all_chunks.keys()):
        if cid in accepted:
            att, n_tok = accepted[cid]
            total_tokens += n_tok
        else:
            att = 1 if 1 in all_chunks[cid] else min(all_chunks[cid])
        fname = os.path.join(run_dir, f"chunk_{cid:03d}_attempt_{att}.txt")
        with open(fname) as cf:
            d = json.load(cf)
        text += d.get("new_text", "")
    return text, total_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_run(run_dir, tokenizer, model, stride_factor):
    import numpy as np
    run_dir = run_dir.rstrip("/")
    out_traj = os.path.join(run_dir, "semantic_coherence_trajectories.json")
    folder_name = os.path.basename(run_dir)

    print(f"\n{'='*60}")
    print(f"Run: {folder_name}")

    t0 = time.perf_counter()
    text, n_tokens = load_full_text(run_dir)
    n_chars = len(text)
    approx_tokens = n_chars / CHARS_PER_GPT4_TOKEN
    print(f"Text: {n_chars:,} chars  ≈ {approx_tokens/1e6:.2f}M GPT-4 tokens  ({n_tokens:,} recorded tokens)")

    sc = SemanticCoherence(
        text=text,
        stride_factor=stride_factor,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        tokenizer=tokenizer,
        model=model,
    )

    for w in WINDOWS:
        t_w = time.perf_counter()
        sc.compute_window(w)
        name = WINDOW_NAMES[w]
        n_win = sc.results.get(f"sc_{name}_n_windows", 0)
        print(f"  {name:5s}: {n_win:5d} windows  ({time.perf_counter()-t_w:.0f}s)")

    # Save trajectories
    trajectories = {"folder_name": folder_name, "n_chars": n_chars,
                    "approx_gpt4_tokens": approx_tokens, "stride_factor": stride_factor}
    for w in WINDOWS:
        name = WINDOW_NAMES[w]
        prefix = f"sc_{name}_"
        nov  = sc.results.get(f"{prefix}novelty_trajectory")
        msp  = sc.results.get(f"{prefix}mean_sim_trajectory")
        sim  = sc.results.get(f"{prefix}sim_matrix")
        n    = sim.shape[0] if sim is not None else 0
        lag1 = np.array([sim[i, i-1] for i in range(1, n)]) if n > 1 else np.array([])

        window_chars = WINDOW_CHARS[w]
        stride_chars = max(1, int(window_chars * stride_factor))
        # positions aligned to trajectory (start at window 1)
        positions = [stride_chars * i + window_chars // 2 for i in range(1, n)]

        trajectories[f"{name}_novelty_trajectory"]    = nov.tolist()  if nov  is not None else []
        trajectories[f"{name}_mean_sim_trajectory"]   = msp.tolist()  if msp  is not None else []
        trajectories[f"{name}_sim_lag1_trajectory"]   = lag1.tolist()
        trajectories[f"{name}_window_char_positions"] = positions

    with open(out_traj, "w") as f:
        json.dump(trajectories, f)

    print(f"Saved: {out_traj}  ({time.perf_counter()-t0:.0f}s total)")


def main(run_dirs, stride_factor):
    print(f"Device        : {DEVICE}")
    print(f"Windows       : {', '.join(WINDOW_NAMES[w] for w in WINDOWS)}")
    print(f"Stride factor : {stride_factor}")
    print(f"\nLoading referee model: {DEFAULT_MODEL}")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    for run_dir in run_dirs:
        process_run(run_dir, tokenizer, model, stride_factor)

    print("\nAll runs done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--stride", type=float, default=STRIDE_FACTOR,
                        help="Stride factor (default 0.5; use 10 for ~20x speedup)")
    args = parser.parse_args()
    main(args.run_dirs, args.stride)
