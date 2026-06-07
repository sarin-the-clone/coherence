"""
Compute SemanticCoherence metrics incrementally as accepted chunks accumulate.

Reads accepted chunks from token_naturalness_by_chunk.jsonl (which records
which attempt was accepted for each chunk), accumulates the text, and runs
SemanticCoherenceMonitor at each level.

Outputs per run folder:
  semantic_coherence_by_chunk.jsonl   — one record per chunk (like token_naturalness_by_chunk.jsonl)
  semantic_coherence_trajectories.json — full novelty/sim trajectories per level for plotting

Usage:
    python compute_semantic_coherence_by_chunk.py <run_folder>

    e.g.
    python compute_semantic_coherence_by_chunk.py \\
        generated_texts_og/prompts/20260518_102401_dune_karikon_s_hidden_secrets_the_aria_chronicles_adventure_extended_conditioning_large_retrieval
"""

import sys
import os
import json
import math
import time
import torch

sys.path.insert(0, os.path.dirname(__file__))
from semantic_coherence import SemanticCoherenceMonitor, WINDOW_CHARS, WINDOW_NAMES, DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WINDOWS = [1, 2, 3, 4]   # W1k, W2k, W8k, W40k
STRIDE_FACTOR = 0.5
BATCH_SIZE = 32
MAX_LENGTH = 512          # per sub-window; hierarchical pooling handles large windows
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Approximate loop lag in windows for the ~900-token (~3600 char) looping signal.
# W1k stride=500 → lag≈7; W2k stride=1024 → lag≈4; W8k/W40k: arc-level lag
LOOP_LAG = {1: 7, 2: 4, 3: 2, 4: 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_accepted_chunks(run_dir):
    """Return list of (chunk_id, accepted_attempt, n_tokens) from token_naturalness_by_chunk.jsonl."""
    path = os.path.join(run_dir, "token_naturalness_by_chunk.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records.append((r["chunk_id"], r["accepted_attempt"], r["n_tokens"]))
    return records


def read_new_text(run_dir, chunk_id, attempt):
    fname = os.path.join(run_dir, f"chunk_{chunk_id:03d}_attempt_{attempt}.txt")
    with open(fname) as f:
        d = json.load(f)
    return d.get("new_text", "")


def _window_trailing_metrics(monitor, window):
    """Return the latest rolling metrics from a monitor, prefixed with {WINDOW_NAMES[window]}_."""
    traj = monitor.trajectory
    if not traj:
        return {}
    last = traj[-1]
    name = WINDOW_NAMES[window]
    out = {f"{name}_n_windows": last["sc_n_windows"]}
    for key in ["sc_novelty", "sc_max_sim_to_past", "sc_mean_sim_to_past",
                "sc_sim_lag1", "sc_sim_lag2", "sc_sim_lag3", "sc_sim_at_loop_lag"]:
        if key in last:
            metric_name = key.replace("sc_", f"{name}_")
            out[metric_name] = last[key]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir):
    run_dir = run_dir.rstrip("/")
    out_jsonl = os.path.join(run_dir, "semantic_coherence_by_chunk.jsonl")
    out_traj  = os.path.join(run_dir, "semantic_coherence_trajectories.json")
    folder_name = os.path.basename(run_dir)

    print(f"Run folder : {folder_name}")
    print(f"Device     : {DEVICE}")
    print(f"Windows    : {', '.join(WINDOW_NAMES[w] for w in WINDOWS)}")

    # ---- load accepted chunk list ----
    accepted = load_accepted_chunks(run_dir)
    print(f"Accepted chunks: {len(accepted)}")

    # ---- check existing output (skip already computed chunks) ----
    done_chunks = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                done_chunks.add(r["chunk_id"])
        print(f"Resuming: {len(done_chunks)} chunks already done")

    # ---- load model once ----
    print(f"\nLoading referee model: {DEFAULT_MODEL}")
    t0 = time.perf_counter()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DEFAULT_MODEL, trust_remote_code=True, dtype=torch.float16,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    # ---- init one monitor per window scale ----
    monitors = {
        w: SemanticCoherenceMonitor(
            level=w,
            loop_lag=LOOP_LAG[w],
            stride_factor=STRIDE_FACTOR,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            tokenizer=tokenizer,
            model=model,
        )
        for w in WINDOWS
    }

    # ---- if resuming, replay text to restore monitor state ----
    accumulated_text = ""
    if done_chunks:
        print("Replaying text to restore monitor state...")
        for chunk_id, attempt, _ in accepted:
            if chunk_id > max(done_chunks):
                break
            accumulated_text += read_new_text(run_dir, chunk_id, attempt)
        t0 = time.perf_counter()
        for w, monitor in monitors.items():
            monitor.update_from_text(accumulated_text)
        print(f"Monitor state restored in {time.perf_counter()-t0:.1f}s  "
              f"(W1 windows: {monitors[1].trajectory[-1]['sc_n_windows'] if monitors[1].trajectory else 0})")

    # ---- process chunks ----
    out_f = open(out_jsonl, "a")
    t_start = time.perf_counter()

    for i, (chunk_id, attempt, n_tokens) in enumerate(accepted):
        if chunk_id in done_chunks:
            continue

        new_text = read_new_text(run_dir, chunk_id, attempt)
        accumulated_text += new_text
        n_chars = len(accumulated_text)

        record = {
            "folder_name": folder_name,
            "chunk_id": chunk_id,
            "accepted_attempt": attempt,
            "n_tokens": n_tokens,
            "n_chars": n_chars,
        }

        new_windows_total = 0
        for w, monitor in monitors.items():
            new = monitor.update_from_text(accumulated_text)
            new_windows_total += len(new)
            record.update(_window_trailing_metrics(monitor, w))

        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

        elapsed = time.perf_counter() - t_start
        chunks_done = i + 1 - len(done_chunks)
        remaining = len(accepted) - len(done_chunks) - chunks_done
        rate = chunks_done / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else float("nan")
        eta_str = f"{eta/60:.1f}min" if math.isfinite(eta) else "?"
        print(
            f"  chunk {chunk_id:4d}/{len(accepted)}  "
            f"n_chars={n_chars:>9,}  "
            f"W1k={monitors[1].trajectory[-1]['sc_n_windows'] if monitors[1].trajectory else 0:5d}wins  "
            f"W2k={monitors[2].trajectory[-1]['sc_n_windows'] if monitors[2].trajectory else 0:4d}wins  "
            f"+{new_windows_total} new  "
            f"ETA {eta_str}",
            flush=True,
        )

    out_f.close()

    # ---- save full trajectories for plotting ----
    trajectories = {}
    for w, monitor in monitors.items():
        name = WINDOW_NAMES[w]
        trajectories[f"{name}_novelty_trajectory"]      = monitor.novelty_trajectory().tolist()
        trajectories[f"{name}_mean_sim_trajectory"]     = monitor.mean_sim_trajectory().tolist()
        window_chars = WINDOW_CHARS[w]
        stride_chars = max(1, int(window_chars * STRIDE_FACTOR))
        n_windows = len(monitor.trajectory)
        trajectories[f"{name}_window_char_positions"] = [
            stride_chars * i + window_chars // 2 for i in range(n_windows)
        ]

    with open(out_traj, "w") as f:
        json.dump(trajectories, f)

    total_time = time.perf_counter() - t_start
    print(f"\nDone in {total_time/60:.1f} min")
    print(f"Saved: {out_jsonl}")
    print(f"Saved: {out_traj}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_semantic_coherence_by_chunk.py <run_folder>")
        sys.exit(1)
    main(sys.argv[1])
