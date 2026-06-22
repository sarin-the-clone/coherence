#!/usr/bin/env python3
"""
Compute full td_/tr_/tp_ token metrics at each bin prefix length for all
generated runs that have SC computed.

Results cached to:
  /tmp/sc_generated_prefix_token_metrics.json
  {run_id: {bin_id: {td_*: ..., tr_*: ..., tp_*: ...}}}

Skips (run, bin) pairs already in the cache.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import token_distribution_scaling_metrics, token_recurrence_metrics, token_predictability_metrics
from tokenizer import Tokenizer

GENERATED_SC_FILE = ROOT / "generated_texts_og" / "Output" / "sc_prefix_bins.json"
PROMPTS_DIR       = ROOT / "generated_texts_og" / "prompts"
CACHE_FILE        = Path("/tmp/sc_generated_prefix_token_metrics.json")

BIN_TOKEN_TARGETS = {
    "0":  121_140,
    "1":  193_795,
    "2":  353_917,
    "3":  593_869,
    "4": 1_072_000,  # unused: bin 4 is the endpoint, see BIN4_MIN_TOKENS
}

# Bin 4 is the open-ended top size bin (>800k tok): use the full text (endpoint),
# gated on the run exceeding this many tokens, matching compute_sc_prefix_bins.py.
BIN4_MIN_TOKENS = 800_000


def load_accepted_text(folder: Path) -> str:
    """Reconstruct full accepted text from chunk attempt files.

    Prefers the token_naturalness_by_chunk.jsonl sidecar; falls back to the
    chunk_NNN_accepted.txt marker files when no sidecar is present.
    """
    all_chunks: dict[int, list[int]] = {}
    for fname in folder.iterdir():
        m = re.match(r"chunk_(\d+)_attempt_(\d+)\.txt$", fname.name)
        if m:
            cid, att = int(m.group(1)), int(m.group(2))
            all_chunks.setdefault(cid, []).append(att)

    accepted = {}
    nat_path = folder / "token_naturalness_by_chunk.jsonl"
    if nat_path.exists():
        with open(nat_path) as f:
            for line in f:
                r = json.loads(line)
                accepted[r["chunk_id"]] = r["accepted_attempt"]
    else:
        for cid in all_chunks:
            marker = folder / f"chunk_{cid:03d}_accepted.txt"
            if marker.exists():
                try:
                    accepted[cid] = json.loads(marker.read_text()).get(
                        "attempt", min(all_chunks[cid]))
                except Exception:
                    pass

    text = ""
    for cid in sorted(all_chunks):
        att = accepted.get(cid, 1 if 1 in all_chunks[cid] else min(all_chunks[cid]))
        p = folder / f"chunk_{cid:03d}_attempt_{att}.txt"
        with open(p) as f:
            text += json.load(f).get("new_text", "")
    return text


def main() -> None:
    with open(GENERATED_SC_FILE) as f:
        sc_data = json.load(f)

    cache: dict = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            cache = json.load(f)

    tokenizer = Tokenizer(kind="gpt4")
    total_runs = len(sc_data)

    for run_idx, (run_id, run_rec) in enumerate(sc_data.items()):
        folder = PROMPTS_DIR / run_id
        has_sidecar = (folder / "token_naturalness_by_chunk.jsonl").exists()
        has_markers = any(folder.glob("chunk_*_accepted.txt"))
        if not has_sidecar and not has_markers:
            print(f"[{run_idx+1}/{total_runs}] skip (no sidecar/markers): {run_id[:60]}")
            continue

        bins_needed = [
            b for b in run_rec.get("bins", {})
            if b not in cache.get(run_id, {})
        ]
        if not bins_needed:
            print(f"[{run_idx+1}/{total_runs}] skip (cached): {run_id[:60]}")
            continue

        print(f"\n[{run_idx+1}/{total_runs}] {run_id[:60]}")
        t0 = time.perf_counter()

        try:
            text = load_accepted_text(folder)
        except Exception as e:
            print(f"  ERROR loading text: {e}")
            continue

        tokens_full = tokenizer.encode(text)
        print(f"  Full tokens: {len(tokens_full):,}")

        run_cache = cache.setdefault(run_id, {})

        for bin_id in sorted(bins_needed, key=int):
            if bin_id == "4":
                # open-ended top bin: full text (endpoint), gated on >800k tok
                if len(tokens_full) <= BIN4_MIN_TOKENS:
                    print(f"  Bin 4 (endpoint): {len(tokens_full):,} tok ≤ "
                          f"{BIN4_MIN_TOKENS:,}, not bin-4 eligible, skip")
                    continue
                prefix_tokens = tokens_full
                target = len(tokens_full)
                print(f"  Bin 4 (endpoint): full text, {len(tokens_full):,} tok")
            else:
                target = BIN_TOKEN_TARGETS[bin_id]
                if len(tokens_full) < target:
                    print(f"  Bin {bin_id}: text too short ({len(tokens_full):,} < {target:,}), skip")
                    continue
                prefix_tokens = tokens_full[:target]

            t_bin = time.perf_counter()

            td = token_distribution_scaling_metrics(prefix_tokens)
            tr = token_recurrence_metrics(prefix_tokens)
            tp = token_predictability_metrics(prefix_tokens, max_order=4, device="cpu")

            result = {}
            result.update(td)
            result.update(tr)
            result.update(tp)
            # remove non-finite
            result = {k: v for k, v in result.items()
                      if isinstance(v, (int, float)) and v == v}

            run_cache[bin_id] = result
            print(f"  Bin {bin_id} ({target//1000}k tok): {len(result)} metrics  ({time.perf_counter()-t_bin:.1f}s)")

        # save after each run
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        print(f"  saved ({time.perf_counter()-t0:.1f}s total)")

    print(f"\nDone. Cache: {CACHE_FILE}")
    n_runs  = len([r for r in cache if cache[r]])
    n_pairs = sum(len(v) for v in cache.values())
    print(f"  {n_runs} runs, {n_pairs} (run, bin) pairs cached")


if __name__ == "__main__":
    main()
