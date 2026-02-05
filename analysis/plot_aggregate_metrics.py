#!/usr/bin/env python3
"""
Plot aggregated metric evolution (Mean +/- Std) across ALL generated books.

Usage:
  python coherence/analysis/plot_aggregate_metrics.py [--out aggregate_evolution.jpg]
"""

import argparse
import json
import re
import glob
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Ensure analysis dir is on path for sibling import
ANALYSIS_DIR = Path(__file__).parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))

import plot_chunk_metrics as pcm

# Alias functions for convenience
load_attempts = pcm.load_attempts
find_accepted_attempt = pcm.find_accepted_attempt
estimate_cumulative_tokens = pcm.estimate_cumulative_tokens
load_size_stats = pcm.load_size_stats
build_size_segments = pcm.build_size_segments

def get_book_folders(base_path: Path) -> List[Path]:
    """Find all run folders in generated_texts_og/prompts/."""
    # We assume any folder starting with a year or 'run_' or just generic foldernames
    # Filter out code files or non-run directories if any.
    folders = []
    for p in base_path.glob("*"):
        if p.is_dir() and not p.name.startswith("_"):
            # Check if it has chunk files to confirm it's a run folder
            if list(p.glob("chunk_*.txt")):
                folders.append(p)
    return folders

def extract_time_series(attempts_by_chunk: Dict[int, Dict]) -> Dict[str, Dict[int, float]]:
    """
    Extract accepted values for each metric for this book.
    Returns: {metric_name: {chunk_idx: value}}
    Also returns estimated tokens mapping: {chunk_idx: tokens}
    """
    cumulative_tokens = estimate_cumulative_tokens(attempts_by_chunk)
    
    series_data = {}
    
    chunk_indices = sorted(attempts_by_chunk.keys())
    for chunk in chunk_indices:
        best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
        rec = attempts_by_chunk[chunk][best_attempt]
        
        # Extract all metrics
        # Standardize extraction: accumulated > chunk_metrics > metrics
        sources = [
            rec.get("accumulated_metrics", {}),
            rec.get("chunk_metrics", {}),
            rec.get("metrics", {})
        ]
        
        # Merge them (priority first)
        combined = {}
        for s in reversed(sources):
            if s:
                combined.update(s)
                
        for m, val in combined.items():
            if isinstance(val, (int, float)):
                if m not in series_data:
                    series_data[m] = {}
                series_data[m][chunk] = val
                
    return series_data, cumulative_tokens

def aggregate_data_by_tokens(folders: List[Path], bin_size: int = 20000, max_tokens: int = 1000000):
    """
    Load all books and aggregate metrics into token bins.
    
    Returns:
       binned_stats: {metric: {bin_center: {'mean': x, 'std': y, 'count': n}}}
    """
    # 1. Collect all (token_count, value) pairs for each metric
    raw_data = {}  # {metric: [(tokens, val), ...]}
    
    print(f"Found {len(folders)} run folders.")
    
    for folder in folders:
        print(f"Loading {folder.name}...")
        attempts = load_attempts(folder)
        if not attempts:
            continue
            
        series, tokens_map = extract_time_series(attempts)
        
        # Align series by tokens
        for metric, chunk_map in series.items():
            if metric not in raw_data:
                raw_data[metric] = []
                
            for chunk, val in chunk_map.items():
                t = tokens_map.get(chunk)
                if t is not None and t <= max_tokens:
                    raw_data[metric].append((t, val))

    # 2. Bin the data
    # Create bins: 0, bin_size, 2*bin_size, ...
    bins = np.arange(0, max_tokens + bin_size, bin_size)
    binned_stats = {}
    
    print(f"Aggregating into {len(bins)-1} bins of size {bin_size}...")
    
    for metric, points in raw_data.items():
        if not points:
            continue
            
        points_arr = np.array(points) # Shape (N, 2) -> col 0: tokens, col 1: values
        token_vals = points_arr[:, 0]
        metric_vals = points_arr[:, 1]
        
        # Digitizing tells us which bin index each point belongs to
        # indices 1..len(bins)-1
        bin_indices = np.digitize(token_vals, bins)
        
        metric_stats = {}
        for i in range(1, len(bins)):
            # Get values falling into this bin
            mask = bin_indices == i
            vals_in_bin = metric_vals[mask]
            
            if len(vals_in_bin) > 0:
                bin_center = (bins[i-1] + bins[i]) / 2.0
                metric_stats[bin_center] = {
                    "mean": np.mean(vals_in_bin),
                    "std": np.std(vals_in_bin),
                    "count": len(vals_in_bin)
                }
        
        binned_stats[metric] = metric_stats
        
    return binned_stats

def plot_aggregate(binned_stats, out_path, structural_metrics, informational_metrics):
    """
    Plot aggregated mean +/- std from token-binned data.
    """
    # Load natural stats for overlays
    SIZE_STATS = load_size_stats()
    
    # We create pseudo-chunks for the segments logic using bin centers
    # Just need a range covering 0 to 1M to generate segments
    dummy_chunks = list(range(0, 10)) # minimal needed
    dummy_tokens = {0: 0, 1: 1000000} # full range
    
    # Actually, we need to map X-axis (tokens) to bins
    # For the background bands, we can just iterate tokens directly or use the helper
    # The helper expects chunk-indexed maps. Let's BYPASS build_size_segments and iterate specific ranges
    # since we are on a pure token axis now. Much easier!
    
    bins_Natural = SIZE_STATS.get("bins", []) if SIZE_STATS else []
    
    palette = ["#5B8FF9", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]
    bin_colors = {}
    for idx, b in enumerate(bins_Natural):
        bin_colors[b["name"]] = palette[idx % len(palette)]
        
    band_labels_used = set()
    mean_labels_used = set()
    rows = max(len(structural_metrics), len(informational_metrics))
    fig, axes = plt.subplots(rows, 2, figsize=(20, 3 * rows), sharex=True)
    if rows == 1:
        axes = axes.reshape(1, 2)

    def plot_on_ax(ax, metric):
        if not metric or metric not in binned_stats:
            ax.axis('off')
            return
            
        stats_map = binned_stats[metric]
        # Sort by bin center
        sorted_centers = sorted(stats_map.keys())
        if not sorted_centers:
            return
            
        xs = [c / 1000.0 for c in sorted_centers] # k tokens
        means = [stats_map[c]["mean"] for c in sorted_centers]
        stds = [stats_map[c]["std"] for c in sorted_centers]
        
        upper = np.array(means) + np.array(stds)
        lower = np.array(means) - np.array(stds)
        
        # Plot Mean
        ax.plot(xs, means, color="#333333", linestyle="-", linewidth=2, label="Mean Generated")
        # Plot Std Dev Shade
        ax.fill_between(xs, lower, upper, color="#333333", alpha=0.2, label="±1 Std Dev")
        
        is_structural = metric in structural_metrics
        
        # Overlay Natural Text Bands (Pure Token Range Logic)
        for b in bins_Natural:
            mstats = b.get("metrics", {}).get(metric)
            if not mstats:
                continue
            
            p25 = mstats.get("p2_5") or mstats.get("p5") or mstats.get("p25")
            p75 = mstats.get("p97_5") or mstats.get("p95") or mstats.get("p75")
            mean_nat = mstats.get("mean")
            
            lo_tok, hi_tok = b.get("range_tokens", [None, None])
            if lo_tok is None: continue
            if hi_tok is None: hi_tok = 10000000 # Infinity
            
            # Constrain to plot window (0 to 1M)
            start_k = max(0, lo_tok / 1000.0)
            end_k = min(1000.0, hi_tok / 1000.0)
            
            if start_k >= end_k: continue

            is_100k_bin = (abs(lo_tok - 100000) < 1000)
            
            label_suffix = ""
            lo_label_k = int(lo_tok // 1000)
            hi_label_k = "∞" if hi_tok > 9000000 else int(hi_tok // 1000)
            label_suffix = f" (Natural {lo_label_k}k-{hi_label_k})"
            
            color = bin_colors.get(b["name"], "skyblue")
            
            if p25 is not None and p75 is not None:
                label = f"Natural p2.5-p97.5{label_suffix}"
                
                # Logic for 100k bin special handling
                if is_100k_bin:
                    # Draw 100k+ normally for both structural and informational (no proxy)
                    sub_start = max(start_k, 100.0)
                    if sub_start < end_k:
                        ax.fill_between([sub_start, end_k], [p25, p25], [p75, p75], color=color, alpha=0.08, zorder=1.5, label=label if label not in band_labels_used else None)
                        band_labels_used.add(label)
                else:
                    # Normal bin
                    ax.fill_between([start_k, end_k], [p25, p25], [p75, p75], color=color, alpha=0.08, zorder=1.5, label=label if label not in band_labels_used else None)
                    band_labels_used.add(label)
                    # Normal bin
                    ax.fill_between([start_k, end_k], [p25, p25], [p75, p75], color=color, alpha=0.08, zorder=1.5, label=label if label not in band_labels_used else None)
                    band_labels_used.add(label)
            
            if mean_nat is not None:
                label = f"Natural Mean{label_suffix}"
                draw_start = start_k
                if is_100k_bin: draw_start = max(start_k, 100.0)
                
                if draw_start < end_k:
                    ax.plot([draw_start, end_k], [mean_nat, mean_nat], color=color, linestyle="--", linewidth=1.2, zorder=1.6, label=label if label not in mean_labels_used else None)
                    mean_labels_used.add(label)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_title(metric)

    # Plot Structural
    for i in range(rows):
        metric = structural_metrics[i] if i < len(structural_metrics) else None
        plot_on_ax(axes[i, 0], metric)
        if i == rows - 1: axes[i, 0].set_xlabel("Tokens (k)")
        
    # Plot Informational
    for i in range(rows):
        metric = informational_metrics[i] if i < len(informational_metrics) else None
        plot_on_ax(axes[i, 1], metric)
        if i == rows - 1: axes[i, 1].set_xlabel("Tokens (k)")

    # Legend
    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    
    seen = set()
    dedup = []
    for h, l in zip(handles, labels):
        if l in seen or not l: continue
        dedup.append((h, l))
        seen.add(l)
    
    if dedup:
        fig.legend(
            [h for h, _ in dedup],
            [l for _, l in dedup],
            loc='lower center',
            ncol=4,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=9,
            frameon=False
        )

    fig.suptitle("Aggregated Metric Evolution (Mean ± Std Dev) vs Natural Texts\n(Binned by Token Count)", fontsize=16)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    print(f"Saved aggregated plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot text metrics aggregated across all books.")
    parser.add_argument("--out", "-o", default="generated_texts_og/prompts/_aggregated_metrics_evolution.jpg")
    parser.add_argument("--bin-size", type=int, default=20000, help="Bin size in tokens for aggregation")
    args = parser.parse_args()

    prompts_dir = ROOT / "generated_texts_og" / "prompts"
    if not prompts_dir.exists():
        print(f"Error: {prompts_dir} does not exist")
        return

    folders = get_book_folders(prompts_dir)
    binned_stats = aggregate_data_by_tokens(folders, bin_size=args.bin_size, max_tokens=1000000)
    
    structural_metrics = ["beta", "alpha", "H_DMA", "H_WT_db4"]
    informational_metrics = ["entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"]
    
    plot_aggregate(binned_stats, args.out, structural_metrics, informational_metrics)


if __name__ == "__main__":
    main()
