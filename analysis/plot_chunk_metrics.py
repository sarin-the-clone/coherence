#!/usr/bin/env python3
"""
Plot per-chunk metric evolution from a prompt/debug folder.

Usage:
  python coherence/analysis/plot_chunk_metrics.py --folder /path/to/prompts/<book_folder> [--out metrics_evolution.jpg]

What it does:
  - Reads all chunk_*_attempt_*.txt JSON files (skips *_prompt.txt).
  - For each chunk, treats the highest attempt index as the accepted attempt.
  - Plots the accepted attempts as a line.
  - Overlays all failed attempts as faint scatter points.
"""

import argparse
import json
import re
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl

# Ensure repo root on path for coherence_decision import
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tokenizer import Tokenizer
from metrics import CoherenceMetrics

# Coherence thresholds (soft gate) pulled from coherence_decision.py
try:
    from generated_texts_og.utils.coherence_decision import _DEFAULT_CFG as COH_LIMITS
except Exception:
    COH_LIMITS = {}


# Size-binned natural text stats for overlays
SIZE_STATS_PATH = Path(__file__).resolve().parents[1] / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"

def load_size_stats():
    if SIZE_STATS_PATH.exists():
        try:
            with open(SIZE_STATS_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Could not load size-binned stats: {e}")
    return None

SIZE_STATS = load_size_stats()

def load_attempts(folder: Path) -> Dict[int, Dict[int, Dict]]:
    """Return {chunk: {attempt: record}}."""
    pattern = re.compile(r"chunk_(\d+)_attempt_(\d+)\.txt$")

    attempts_by_chunk: Dict[int, Dict[int, Dict]] = {}

    for file in folder.glob("chunk_*_attempt_*.txt"):
        if "prompt" in file.name:
            continue
        m = pattern.search(file.name)
        if not m:
            continue
        chunk = int(m.group(1))
        attempt = int(m.group(2))
        try:
            with open(file, "r") as fh:
                data = json.load(fh)
            
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
            continue
        # Store file path for later usage (e.g. finding prompt files)
        data["_file_path"] = str(file)
        attempts_by_chunk.setdefault(chunk, {})[attempt] = data

    return attempts_by_chunk



# Helper to find which attempt was actually accepted based on the next chunk's prompt
def find_accepted_attempt(chunk_idx: int, attempts_by_chunk: Dict[int, Dict[int, Dict]]) -> int:
    attempts_dict = attempts_by_chunk[chunk_idx]
    
    # If this is the last chunk effectively (no next chunk found), fallback to max
    next_chunk = chunk_idx + 1
    if next_chunk not in attempts_by_chunk:
        return max(attempts_dict)
    
    # If we represent the next chunk, we can check its prompt file matching
    # We look for any prompt file for next_chunk (e.g. chunk_32_attempt_1_prompt.txt)
    # Since we only have the folder path available via the attempt filenames, let's derive it
    sample_file = list(attempts_dict.values())[0].get("_file_path") 
    if not sample_file:
        # Fallback if we didn't store file path (we need to modify load_attempts to store it)
        return max(attempts_dict)
        
    folder_path = Path(sample_file).parent
    
    # Find any prompt file for the next chunk
    next_prompts = list(folder_path.glob(f"chunk_{next_chunk:03d}_attempt_*_prompt.txt"))
    if not next_prompts:
        # Try with non-padded if 3-digit failed (though usually it is 3 padded)
        next_prompts = list(folder_path.glob(f"chunk_{next_chunk}_attempt_*_prompt.txt"))
    
    if not next_prompts:
        return max(attempts_dict)
        
    # Read one prompt file (they should all share context if they are attempts for same chunk)
    try:
        with open(next_prompts[0], "r") as f:
            next_prompt_content = f.read()
    except Exception:
        return max(attempts_dict)
        
    # Check attempts
    for att_id, att_data in sorted(attempts_dict.items(), key=lambda x: x[0], reverse=True):
        # We check reverse order (highest first) as it's most likely, but check VALIDITY
        text = att_data.get("new_text", "").strip()
        if not text:
            continue
        
        # Check if the *start* of the text is in the prompt (Context/Summary/Previous Text)
        # We interpret "linkage" as: the next chunk's prompt contains a snippet of this chunk's text.
        # Usually it's in the "PREVIOUS TEXT" section or "Relevant Earlier Scenes".
        # A robust check is the first 64 chars of the text.
        snippet = text[:64]
        if len(snippet) < 10: 
            snippet = text # Very short text
            
        if snippet in next_prompt_content:
            return att_id
    
    # If no match found, fallback
    return max(attempts_dict)


def estimate_cumulative_tokens(attempts_by_chunk: Dict[int, Dict[int, Dict]]) -> Dict[int, int]:
    """
    Get cumulative token counts per chunk.
    Priority 1: Use 'accumulated_metrics.n_tokens' if available (Global count).
    Priority 2: Estimate by summing 'new_text' lengths (Local count).
    """
    cumulative = {}
    
    # Check if we have explicit global tokens
    has_global = False
    global_counts = {}
    
    # First pass: collect global counts for accepted chunks
    sorted_chunks = sorted(attempts_by_chunk.keys())
    for chunk in sorted_chunks:
        # USE CORRECT ACCEPTED LOGIC HERE
        best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
        rec = attempts_by_chunk[chunk][best_attempt]
        # Check deep key
        if rec.get("accumulated_metrics") and "n_tokens" in rec["accumulated_metrics"]:
            val = rec["accumulated_metrics"]["n_tokens"]
            global_counts[chunk] = val
            has_global = True

    if has_global:
        # Check integrity/monotonicity
        is_monotonic = True
        prev_val = -1
        for chunk in sorted_chunks:
            val = global_counts.get(chunk, prev_val) # handle gaps if any
            if val < prev_val:
                print(f"⚠️ Non-monotonic global tokens detected at chunk {chunk}: {val} < {prev_val}. Switching to estimation.")
                is_monotonic = False
                break
            prev_val = val
        
        if is_monotonic:
            print(f"DEBUG: Using explicit 'n_tokens' from accumulated_metrics (Monotonicity Check Passed).")
            # Fill gaps if any simplisticly
            return global_counts
        else:
            print("DEBUG: Global tokens valid but non-monotonic. Discarding and recalculating.")

    # Fallback: Local Summation
    lengths = {}
    for chunk, attempts in attempts_by_chunk.items():
        best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
        rec = attempts[best_attempt]
        text = rec.get("new_text") or rec.get("full_output") or ""
        tokens = len(Tokenizer(kind="gpt4").encode(text))
        lengths[chunk] = tokens

    avg_calc = int(sum(lengths.values()) / len(lengths)) if lengths else 1500
    print(f"DEBUG: Calculated average tokens per chunk: {avg_calc}")
    print(f"DEBUG: Total estimated tokens: {sum(lengths.values())}")
    
    avg_len = avg_calc
    cum = 0
    for chunk in sorted(attempts_by_chunk.keys()):
        chunk_tokens = lengths.get(chunk, avg_len)
        chunk_tokens = max(chunk_tokens, int(0.5 * avg_len))
        cum += chunk_tokens
        cumulative[chunk] = cum
    return cumulative


def build_size_segments(chunks_sorted, cumulative_tokens):
    """
    Collapse consecutive chunks into size-bucket segments based on cumulative tokens.
    """
    if not SIZE_STATS or not cumulative_tokens:
        return []

    bins = SIZE_STATS.get("bins", [])

    def bin_for(tok):
        if tok is None:
            return None
        for b in bins:
            lo, hi = b.get("range_tokens", [None, None])
            if lo is None or hi is None:
                continue
            if lo <= tok < hi:
                return b
        return None

    segments = []
    prev_bin = None
    start = None
    end = None
    for chunk in chunks_sorted:
        b = bin_for(cumulative_tokens.get(chunk))
        if b is None:
            if prev_bin is not None:
                segments.append((start, end, prev_bin))
                prev_bin = None
            start = end = None
            continue
        if prev_bin is None:
            prev_bin = b
            start = end = chunk
            continue
        if b.get("name") == prev_bin.get("name"):
            end = chunk
        else:
            segments.append((start, end, prev_bin))
            prev_bin = b
            start = end = chunk
    if prev_bin is not None:
        segments.append((start, end, prev_bin))
    return segments


def plot_metrics(
    attempts_by_chunk: Dict[int, Dict[int, Dict]], out_path: Path, 
    structural_metrics: List[str], informational_metrics: List[str]
) -> None:
    """Create the metrics plot with two columns: structural and informational."""
    chunks_sorted = sorted(attempts_by_chunk.keys())
    if not chunks_sorted:
        raise SystemExit("No chunk attempts found.")

    cumulative_tokens = estimate_cumulative_tokens(attempts_by_chunk)
    size_segments = build_size_segments(chunks_sorted, cumulative_tokens)
    palette = ["#5B8FF9", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]
    bin_colors = {}
    for idx, (_, _, b) in enumerate(size_segments):
        bin_colors[b["name"]] = palette[idx % len(palette)]
    band_labels_used = set()
    mean_labels_used = set()  # Track mean labels globally

    rows = max(len(structural_metrics), len(informational_metrics))
    fig, axes = plt.subplots(rows, 2, figsize=(20, 3 * rows), sharex=True)
    
    # Flatten axes if needed but here we have 2D array if rows > 1
    if rows == 1:
        axes = axes.reshape(1, 2)

    def get_metric(rec: Dict, metric: str):
        """Prefer accumulated_metrics, then chunk/metrics."""
        if rec.get("accumulated_metrics") and metric in rec["accumulated_metrics"]:
            return rec["accumulated_metrics"][metric]
        if rec.get("chunk_metrics") and metric in rec["chunk_metrics"]:
            return rec["chunk_metrics"][metric]
        return rec.get("metrics", {}).get(metric)

    # Helper to plot on a specific axis
    def plot_on_ax(ax, metric):
        if not metric:
            ax.axis('off')
            return

        is_structural = metric in structural_metrics
        # is_informational = metric in informational_metrics (Implicit if not structural)

        xs_line = []
        ys_line = []

        for chunk in chunks_sorted:
            attempts = attempts_by_chunk[chunk]
            best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
            
            # Plot failed attempts as faint scatter
            for att, rec in attempts.items():
                y = get_metric(rec, metric)
                if y is None:
                    continue  # missing metric
                x = cumulative_tokens.get(chunk, 0) / 1000.0  # Convert to k tokens
                if att == best_attempt:
                    continue
                ax.scatter(x, y, color="gray", alpha=0.35, s=12, marker="x", label="rejected chunk")

            # Record accepted attempt for line
            best_rec = attempts[best_attempt]
            y_best = get_metric(best_rec, metric)
            if y_best is not None:
                xs_line.append(cumulative_tokens.get(chunk, 0) / 1000.0)
                ys_line.append(y_best)

        ax.plot(xs_line, ys_line, marker="o", markersize=4, linewidth=1.5, label="accepted")
        ax.lines[-1].set_label("accepted chunk")
        
        # Overlay size-binned natural statistics as local bands
        for start, end, b in size_segments:
            mstats = b.get("metrics", {}).get(metric)
            if not mstats:
                continue
            # Use p2.5 and p97.5 for the wide gate visualization
            p25 = mstats.get("p2_5") or mstats.get("p5") or mstats.get("p25")
            p75 = mstats.get("p97_5") or mstats.get("p95") or mstats.get("p75")
            
            mean = mstats.get("mean")
            
            # New token-based logic
            tok_start = cumulative_tokens.get(start, 0) / 1000.0
            tok_end = cumulative_tokens.get(end, 0) / 1000.0
            
            x0, x1 = tok_start, tok_end
            lo_tok, hi_tok = b.get("range_tokens", [None, None])
            
            # Special case: Check if this is the 100k-150k bin
            is_100k_bin = (lo_tok is not None and abs(lo_tok - 100000) < 1000)
            
            label_suffix = ""
            if lo_tok is not None and hi_tok is not None:
                lo_k = int(lo_tok // 1000)
                hi_k = "∞" if hi_tok > 900_000 else int(hi_tok // 1000)
                label_suffix = f" for natural books, {lo_k}k-{hi_k}k tokens long"
            
            color = bin_colors.get(b["name"], "skyblue")
            
            # Shading Logic
            if p25 is not None and p75 is not None:
                label_key = f"{metric}-{b['name']}"
                label = None if label_key in band_labels_used else f"p2.5-p97.5{label_suffix}"
                
                # Default draw range: within the bin (likely >100k for the first bin in new logic)
                draw_x0 = x0
                
                if is_100k_bin:
                    # If it's the 100k bin, handle <100k visualization
                    if is_structural:
                        # Structural: Do NOT draw < 100k.
                        # Clamp start to 100k
                        draw_x0 = max(x0, 100.0)
                        
                        if draw_x0 < x1:
                            ax.fill_between(
                                [draw_x0, x1], [p25, p25], [p75, p75],
                                color=color, alpha=0.08, zorder=1.5,
                                label=label
                            )
                            band_labels_used.add(label_key)
                    else:
                        # Informational: Still treat as normal (start > 100k) or just suppress special logic
                        # Actually we want normal plotting for informational too now, but clamped at 100k?
                        # "remove the gray colored area <100k (proxy)"
                        # This just means we plot the main bin like structural.
                        draw_x0 = max(x0, 100.0)
                        if draw_x0 < x1:
                            ax.fill_between(
                                [draw_x0, x1], [p25, p25], [p75, p75],
                                color=color, alpha=0.08, zorder=1.5,
                                label=label
                            )
                            band_labels_used.add(label_key)

                else:
                    # Normal bin including 100k+
                    ax.fill_between(
                        [draw_x0, x1], [p25, p25], [p75, p75],
                        color=color, alpha=0.08, zorder=1.5,
                        label=label
                    )


                band_labels_used.add(label_key)
                
            if mean is not None:
                label_key = f"mean-{metric}-{b['name']}"
                label = None if label_key in mean_labels_used else f"mean{label_suffix}"
                
                draw_x0 = x0
                if is_100k_bin:
                    # Do not plot mean for <100k proxy region
                    draw_x0 = max(x0, 100.0)
                
                if draw_x0 < x1:
                    ax.plot(
                        [draw_x0, x1], [mean, mean],
                        color=color, linestyle="--", linewidth=1.2, zorder=1.6,
                        label=label
                    )
                mean_labels_used.add(label_key)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    legend_handles = []

    # Plot Structural Column (Left)
    for i in range(rows):
        metric = structural_metrics[i] if i < len(structural_metrics) else None
        plot_on_ax(axes[i, 0], metric)
        if i == rows - 1:
            axes[i, 0].set_xlabel("Tokens (k)")

    # Plot Informational Column (Right)
    for i in range(rows):
        metric = informational_metrics[i] if i < len(informational_metrics) else None
        plot_on_ax(axes[i, 1], metric)
        if i == rows - 1:
            axes[i, 1].set_xlabel("Tokens (k)")

    # Collect legend handles once from the first axis and remove per-plot legends
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
        ax.legend_.remove() if ax.get_legend() else None
    # Deduplicate while preserving order
    seen = set()
    dedup = []
    for h, l in zip(handles, labels):
        if l in seen or not l:
            continue
        dedup.append((h, l))
        seen.add(l)
    legend_handles = dedup

    # Add an extra invisible subplot for the legend
    # Use fig.legend instead for absolute figure positioning
    if legend_handles:
        fig.legend(
            [h for h, _ in legend_handles],
            [l for _, l in legend_handles],
            loc='lower center',
            ncol=3,
            bbox_to_anchor=(0.5, 0.02), # Absolute bottom of figure in reserved space
            fontsize=9,
            frameon=False
        )

    fig.suptitle("Metric Evolution per accumulated generated chunks: Structural (left column) vs Informational (right column)", fontsize=14)
    # Increase bottom margin significantly for legend (12%)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-chunk metrics from prompt folder.")
    parser.add_argument("--folder", "-f", required=True, help="Path to prompt/debug folder for a generated book.")
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output image path (default: <folder>/_metrics_evolution.jpg)",
    )
    # No longer taking generic --metrics arg, but we can keep it if we want to override both lists?
    # For simplicity, let's hardcode the split as requested or allow passing separated lists?
    # Separate them in code. Let's just use the defaults requested.
    
    args = parser.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    attempts_by_chunk = load_attempts(folder)
    out_path = Path(args.out) if args.out else folder / "_metrics_evolution.jpg"

    structural_metrics = ["beta", "alpha", "H_DMA", "H_WT_db4"]
    informational_metrics = ["entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"]

    plot_metrics(attempts_by_chunk, out_path, structural_metrics, informational_metrics)


if __name__ == "__main__":
    main()
