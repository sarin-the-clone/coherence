#!/usr/bin/env python3
"""
Plot aggregated metric evolution across generated books.

Default behavior now produces two figures:
  1. legacy low/high temperature runs on a linear 0-1M token axis
  2. chaotic run vs full self-conditioning high-temp runs on a nonlinear axis
"""

import argparse
import json
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

def load_single_book(folder_path: Path):
    """Load time series for a single specific book folder."""
    attempts = load_attempts(folder_path)
    if not attempts:
        return None
    series, tokens_map = extract_time_series(attempts)
    
    # Convert {chunk: val} to {tokens: val}
    tokens_series = {}
    for metric, chunk_map in series.items():
        tokens_series[metric] = {tokens_map[c]: v for c, v in chunk_map.items() if tokens_map.get(c) is not None}
    
    return tokens_series

def plot_aggregate_generic(
    cohort_specs,
    out_path,
    structural_metrics,
    informational_metrics,
    title,
    chaotic_data=None,
    xscale="linear",
    xmax_k=1000.0,
    xlabel="token count",
):
    """Plot aggregated mean +/- std from token-binned data."""
    # Load natural stats for overlays
    SIZE_STATS = load_size_stats()
    bins_Natural = SIZE_STATS.get("bins", []) if SIZE_STATS else []

    palette = ["#FF0000", "#FF7F00", "#FFD700", "#FF1493", "#8A2BE2", "#FF00FF"]
    bin_colors = {}
    for idx, b in enumerate(bins_Natural):
        bin_colors[b["name"]] = palette[idx % len(palette)]

    band_labels_used = set()
    mean_labels_used = set()
    rows = max(len(structural_metrics), len(informational_metrics))
    fig, axes = plt.subplots(rows, 2, figsize=(20, 5 * rows), sharex=True)
    if rows == 1:
        axes = axes.reshape(1, 2)

    def plot_on_ax(ax, metric):
        metric_present = any(metric in stats_map for stats_map, _, _ in cohort_specs)
        if chaotic_data and metric in chaotic_data:
            metric_present = True
        if not metric or not metric_present:
            ax.axis('off')
            return

        def plot_stats(stats_map, color, label_prefix):
            if not stats_map:
                return
            sorted_centers = sorted(stats_map.keys())
            if not sorted_centers:
                return
            xs = [c / 1000.0 for c in sorted_centers]
            means = [stats_map[c]["mean"] for c in sorted_centers]
            stds = [stats_map[c]["std"] for c in sorted_centers]
            upper = np.array(means) + np.array(stds)
            lower = np.array(means) - np.array(stds)

            ax.plot(xs, means, color=color, linestyle="-", linewidth=2.0, label=f"Mean for {label_prefix}", zorder=10)
            ax.fill_between(xs, lower, upper, color=color, alpha=0.25, label=f"±1 Std Dev ({label_prefix})", zorder=9)

        for stats_map, color, label in cohort_specs:
            if metric in stats_map:
                plot_stats(stats_map[metric], color, label)

        if chaotic_data and metric in chaotic_data:
            c_map = chaotic_data[metric]
            c_tokens = sorted(c_map.keys())
            c_xs = []
            c_vals = []
            for t in c_tokens:
                v = c_map[t]
                if v is not None:
                    c_xs.append(t / 1000.0)
                    c_vals.append(v)
                    
            if c_xs and c_vals:
                vals_arr = np.array(c_vals, dtype=np.float64)
                nan_mask = np.isnan(vals_arr)
                if np.any(nan_mask):
                    not_nan_mask = ~nan_mask
                    if np.any(not_nan_mask):
                        vals_arr[nan_mask] = np.interp(np.array(c_xs)[nan_mask], np.array(c_xs)[not_nan_mask], vals_arr[not_nan_mask])
                ax.plot(c_xs, vals_arr, color="gray", linestyle=":", linewidth=2.0, alpha=0.9, label="Chaotic Run (4M tokens)", zorder=8)

        for b in bins_Natural:
            mstats = b.get("metrics", {}).get(metric)
            if not mstats:
                continue

            p25 = mstats.get("p2_5") or mstats.get("p5") or mstats.get("p25")
            p75 = mstats.get("p97_5") or mstats.get("p95") or mstats.get("p75")
            mean_nat = mstats.get("mean")
            lo_tok, hi_tok = b.get("range_tokens", [None, None])
            if lo_tok is None:
                continue
            if hi_tok is None:
                hi_tok = 10000000

            start_k = max(0, lo_tok / 1000.0)
            end_k = min(xmax_k, hi_tok / 1000.0)
            if start_k >= end_k:
                continue

            is_100k_bin = (abs(lo_tok - 100000) < 1000)
            label_suffix = ""
            lo_label_k = int(lo_tok // 1000)
            if hi_tok > 800000:
                label_suffix = f" ( >{lo_label_k}k)"
            else:
                hi_label_k = int(hi_tok // 1000)
                label_suffix = f" ( {lo_label_k}-{hi_label_k}k)"
            
            color = bin_colors.get(b["name"], "#FF00FF")
            if hi_tok > 800000 and xmax_k > 1000:
                color = "#8B0000"

            if p25 is not None and p75 is not None:
                label = f"Natural p2.5-p97.5{label_suffix}"
                if is_100k_bin:
                    sub_start = max(start_k, 100.0)
                    if sub_start < end_k:
                        ax.fill_between([sub_start, end_k], [p25, p25], [p75, p75], color=color, alpha=0.05, zorder=1.5, label=label if label not in band_labels_used else None)
                        band_labels_used.add(label)
                else:
                    # Normal bin
                    ax.fill_between([start_k, end_k], [p25, p25], [p75, p75], color=color, alpha=0.05, zorder=1.5, label=label if label not in band_labels_used else None)
                    band_labels_used.add(label)
            
            if mean_nat is not None:
                label = f"Natural Mean{label_suffix}"
                draw_start = start_k
                if is_100k_bin:
                    draw_start = max(start_k, 100.0)
                if draw_start < end_k:
                    ax.plot([draw_start, end_k], [mean_nat, mean_nat], color=color, linestyle="--", linewidth=2.0, zorder=15, label=label if label not in mean_labels_used else None)
                    mean_labels_used.add(label)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_title(metric)

        if xscale == "linear":
            ax.set_xscale("linear")
            ax.set_xlim(0, xmax_k)
            major_ticks = [0, 200, 400, 600, 800, 1000]
            ax.set_xticks(major_ticks)
            ax.set_xticklabels(['0', '200k', '400k', '600k', '800k', '1M'])
        else:
            ax.set_xscale('symlog', linthresh=1000.0, linscale=5.0)
            ax.set_xlim(0, xmax_k + 200.0)
            major_ticks = [0, 200, 400, 600, 800, 1000, 2000, 4000]
            ax.set_xticks(major_ticks)
            ax.set_xticklabels(['0', '200k', '400k', '600k', '800k', '1M', '2M', '4M'])
        ax.tick_params(axis='x', labelsize=9)

    for i in range(rows):
        metric = structural_metrics[i] if i < len(structural_metrics) else None
        plot_on_ax(axes[i, 0], metric)
        if i == rows - 1:
            axes[i, 0].set_xlabel(xlabel)

    for i in range(rows):
        metric = informational_metrics[i] if i < len(informational_metrics) else None
        plot_on_ax(axes[i, 1], metric)
        if i == rows - 1:
            axes[i, 1].set_xlabel(xlabel)

    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = set()
    dedup = []
    for h, l in zip(handles, labels):
        if l in seen or not l:
            continue
        dedup.append((h, l))
        seen.add(l)

    chaotic_items = []
    final_dedup = []
    for h, l in dedup:
        if "Chaotic Run" in l:
            chaotic_items.append((h, l))
        else:
            final_dedup.append((h, l))
    final_dedup.extend(chaotic_items)
    
    if final_dedup:
        fig.legend(
            [h for h, _ in final_dedup],
            [l for _, l in final_dedup],
            loc='lower center',
            ncol=3 if xscale == "linear" else 4,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=9,
            frameon=False
        )

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    print(f"Saved aggregated plot → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot text metrics aggregated across all books.")
    parser.add_argument("--in-data", "-i", default="generated_texts_og/Output/aggregated_metrics.json", help="Path to aggregated metrics JSON")
    parser.add_argument(
        "--out-linear",
        default="generated_texts_og/Output/_aggregated_metrics_evolution_linear_1m.jpg",
        help="Output path for the low/high linear 0-1M figure",
    )
    parser.add_argument(
        "--out-chaotic-selfcond",
        default="generated_texts_og/Output/_aggregated_metrics_evolution_chaotic_selfcond.jpg",
        help="Output path for the chaotic vs self-conditioned nonlinear figure",
    )
    args = parser.parse_args()

    in_path = ROOT / args.in_data
    if not in_path.exists():
        print(f"Error: Aggregated metrics file not found: {in_path}")
        print("Please run coherence/analysis/extract_aggregate_metrics.py first.")
        return
        
    print(f"Loading aggregated metrics from {in_path.name}...")
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    binned_stats_high_raw = data.get("high_temp", {})
    binned_stats_low_raw = data.get("low_temp", {})
    binned_stats_self_conditioned_high_raw = data.get("self_conditioned_high_temp", {})
    
    # JSON dict keys are strings, convert the numeric token keys back to float
    def cast_keys(binned_stats):
        result = {}
        for metric, bins in binned_stats.items():
            result[metric] = {float(k): v for k, v in bins.items()}
        return result
        
    binned_stats_high = cast_keys(binned_stats_high_raw)
    binned_stats_low = cast_keys(binned_stats_low_raw)
    binned_stats_self_conditioned_high = cast_keys(binned_stats_self_conditioned_high_raw)
    
    # Load Chaotic Book separately from its dedicated json
    chaotic_file = ROOT / "generated_texts_og" / "Output" / "chaotic_run_metrics.json"
    chaotic_data = None
    if chaotic_file.exists():
        print(f"Loading chaotic run: {chaotic_file.name}")
        with open(chaotic_file, "r") as f:
            raw_chaotic = json.load(f)
            # JSON keys are currently strings, convert token keys to int
            chaotic_data = {}
            for metric, data_dict in raw_chaotic.items():
                chaotic_data[metric] = {int(k): v for k, v in data_dict.items()}
    else:
        print(f"Warning: Chaotic run file not found: {chaotic_file}")

    structural_metrics = ["beta", "alpha", "H_DMA", "H_WT_db4"]
    informational_metrics = ["entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"]

    linear_cohorts = [
        (binned_stats_low, "steelblue", "Low Temperature runs"),
        (binned_stats_high, "blue", "High Temperature runs"),
    ]
    nonlinear_cohorts = [
        (
            binned_stats_self_conditioned_high,
            "darkorange",
            "High Temperature + Full Self-Conditioning runs",
        ),
    ]

    plot_aggregate_generic(
        linear_cohorts,
        args.out_linear,
        structural_metrics,
        informational_metrics,
        title="Mean metric evolution vs Natural Texts: Low vs High Temperature runs",
        chaotic_data=None,
        xscale="linear",
        xmax_k=1000.0,
        xlabel="token count",
    )

    plot_aggregate_generic(
        nonlinear_cohorts,
        args.out_chaotic_selfcond,
        structural_metrics,
        informational_metrics,
        title="Mean metric evolution vs Natural Texts: Chaotic run vs Full Self-Conditioning runs",
        chaotic_data=chaotic_data,
        xscale="symlog",
        xmax_k=4000.0,
        xlabel="nonlinear token count",
    )


if __name__ == "__main__":
    main()
