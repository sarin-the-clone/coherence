#!/usr/bin/env python3
"""
Plot per-chunk metric evolution from a prompt/debug folder.

Usage:
  python coherence/analysis/plot_chunk_metrics.py --folder /path/to/prompts/<book_folder> [--out metrics_evolution.jpg]

What it does:
  - Reads all chunk_*_attempt_*.txt JSON files (skips *_prompt.txt).
  - Reads optional token_naturalness_by_chunk.jsonl sidecar if present.
  - Infers accepted attempts from accepted markers / next prompts.
  - Plots the accepted attempts as a line.
  - Overlays all failed attempts as faint scatter points.
  - Does not compute metrics or tokenize text.
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

METRIC_PRESETS = {
    "legacy-vs-additional": (
        ["beta", "alpha", "H_DFA1", "entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"],
        [
            "tp_entropy_slope",
            "tr_repeat_gap_burstiness",
            "tr_repeat_gap_cv",
            "tr_repeat_gap_logbin_entropy",
            "td_hapax_token_fraction",
            "td_zipf_exponent",
            "td_heaps_exponent",
        ],
    ),
    "statistical-descriptors": (
        ["beta", "alpha", "H_DFA1"],
    ),
    "informational-descriptors": (
        ["entropy_bits", "perplexity"],
        ["lz_complexity", "mi_decay_gamma"],
    ),
    "additional-descriptors": (
        ["tp_entropy_slope", "tr_repeat_gap_burstiness", "tr_repeat_gap_cv", "tr_repeat_gap_logbin_entropy"],
        ["td_zipf_exponent", "td_zipf_r2", "td_heaps_exponent", "td_heaps_r2"],
    ),
    "default": (
        ["beta", "alpha", "H_DFA1"],
        ["entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"],
    ),
    "token-fast": (
        ["td_hapax_token_fraction", "td_zipf_exponent", "td_zipf_r2", "td_heaps_exponent", "td_heaps_r2"],
        ["tr_repeat_gap_burstiness", "tr_repeat_gap_cv", "tr_repeat_gap_logbin_entropy"],
    ),
    "token-predictability": (
        ["tp_entropy_0", "tp_redundancy_1", "tp_entropy_slope", "tp_context_efficiency_4"],
    ),
    "token-curated": (
        ["tp_entropy_0", "tp_redundancy_1", "tp_entropy_slope", "tp_context_efficiency_4"],
        ["td_hapax_token_fraction", "td_zipf_exponent", "td_zipf_r2", "td_heaps_exponent", "td_heaps_r2"],
        ["tr_repeat_gap_burstiness", "tr_repeat_gap_cv", "tr_repeat_gap_logbin_entropy"],
    ),
    "default-plus-token-fast": (
        ["beta", "alpha", "H_DMA", "H_WT_db4", "td_zipf_exponent", "td_heaps_exponent"],
        ["entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma", "tr_repeat_gap_burstiness", "tr_repeat_gap_cv", "tr_repeat_gap_logbin_entropy"],
    ),
}

PRESET_TITLES = {
    "legacy-vs-additional": "Legacy vs Additional Descriptor Evolution",
    "statistical-descriptors": "Statistical Descriptor Evolution",
    "informational-descriptors": "Informational Descriptor Evolution",
    "additional-descriptors": "Additional Token Descriptor Evolution",
    "default": "Statistical and Informational Descriptor Evolution",
    "token-fast": "Token Distribution and Recurrence Descriptor Evolution",
    "token-predictability": "Token Predictability Descriptor Evolution",
    "token-curated": "Curated Token Descriptor Evolution",
    "default-plus-token-fast": "Descriptor Evolution",
}

SECONDARY_METRIC_OVERLAYS = {
    "td_zipf_exponent": "td_zipf_r2",
    "td_heaps_exponent": "td_heaps_r2",
}


def parse_metric_list(value: str | None) -> List[str] | None:
    if value is None:
        return None
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    return metrics or None


def normalize_preset_columns(groups: Tuple[List[str], ...]) -> Tuple[List[str], List[str]]:
    """
    Convert a preset with one or more metric groups into two plot columns.

    Some token presets are naturally three groups (tp, td, tr), while the
    plotting layout has two columns. Keep the first two groups as anchors, then
    append any remaining groups to the currently shorter column.
    """
    if not groups:
        return [], []
    if len(groups) == 1:
        return list(groups[0]), []

    left = list(groups[0])
    right = list(groups[1])
    for group in groups[2:]:
        if len(left) <= len(right):
            left.extend(group)
        else:
            right.extend(group)
    return left, right


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


def load_token_naturalness_by_chunk(folder: Path) -> Dict[int, Dict]:
    """Read retrospective accumulated token-naturalness sidecar, if present."""
    path = folder / "token_naturalness_by_chunk.jsonl"
    records: Dict[int, Dict] = {}
    if not path.exists():
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = rec.get("chunk_id")
            if chunk_id is None:
                continue
            records[int(chunk_id)] = rec
    return records


# Helper to find which attempt was actually accepted based on the next chunk's prompt
def find_accepted_attempt(chunk_idx: int, attempts_by_chunk: Dict[int, Dict[int, Dict]]) -> int:
    attempts_dict = attempts_by_chunk[chunk_idx]
    sample_file = list(attempts_dict.values())[0].get("_file_path")
    if sample_file:
        folder_path = Path(sample_file).parent
        accepted_path = folder_path / f"chunk_{chunk_idx:03d}_accepted.txt"
        if accepted_path.exists():
            try:
                accepted_data = json.loads(accepted_path.read_text())
                accepted_attempt = accepted_data.get("accepted_attempt")
                if accepted_attempt is not None and int(accepted_attempt) in attempts_dict:
                    return int(accepted_attempt)
            except Exception:
                pass
    
    # If this is the last chunk effectively (no next chunk found), fallback to max
    next_chunk = chunk_idx + 1
    if next_chunk not in attempts_by_chunk:
        return max(attempts_dict)
    
    # If we represent the next chunk, we can check its prompt file matching
    # We look for any prompt file for next_chunk (e.g. chunk_32_attempt_1_prompt.txt)
    # Since we only have the folder path available via the attempt filenames, let's derive it
    if not sample_file:
        # Fallback if we didn't store file path (we need to modify load_attempts to store it)
        return max(attempts_dict)
        
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
    
    # If no match found, fallback to a read-only inference.
    return max(attempts_dict)


def load_cumulative_tokens(
    attempts_by_chunk: Dict[int, Dict[int, Dict]],
    sidecar_by_chunk: Dict[int, Dict] | None = None,
) -> Dict[int, int]:
    """
    Read cumulative token counts per chunk.

    This intentionally does not estimate or tokenize. If no monotonic token
    counts are available, callers should plot against chunk index.
    """
    sidecar_by_chunk = sidecar_by_chunk or {}
    
    # Check if we have explicit global tokens
    has_global = False
    global_counts = {}
    
    sorted_chunks = sorted(attempts_by_chunk.keys())
    for chunk in sorted_chunks:
        if chunk in sidecar_by_chunk and sidecar_by_chunk[chunk].get("n_tokens") is not None:
            global_counts[chunk] = int(sidecar_by_chunk[chunk]["n_tokens"])
            has_global = True
            continue

        best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
        rec = attempts_by_chunk[chunk][best_attempt]
        if rec.get("accumulated_metrics") and "n_tokens" in rec["accumulated_metrics"]:
            global_counts[chunk] = int(rec["accumulated_metrics"]["n_tokens"])
            has_global = True

    if not has_global:
        return {}

    prev_val = -1
    monotonic_counts = {}
    for chunk in sorted_chunks:
        if chunk not in global_counts:
            continue
        val = global_counts[chunk]
        if val < prev_val:
            print(
                f"⚠️ Non-monotonic stored n_tokens at chunk {chunk}: "
                f"{val} < {prev_val}. Token x-axis disabled."
            )
            return {}
        prev_val = val
        monotonic_counts[chunk] = val

    return monotonic_counts


def compute_gating_frequency(
    attempts_by_chunk: Dict[int, Dict[int, Dict]],
    cumulative_tokens: Dict[int, int],
    start_tokens_threshold: int = 100000,
) -> Tuple[int, int, float]:
    """
    Chunk-level gating proxy based on retries.

    A chunk counts as a single gating event if more than one attempt file exists
    for that chunk, regardless of whether it took 2 or 5 attempts.
    Chunks before the full-gating threshold are excluded.
    """
    eligible_chunks = [
        chunk for chunk in attempts_by_chunk
        if cumulative_tokens.get(chunk, 0) >= start_tokens_threshold
    ]
    total_chunks = len(eligible_chunks)
    if total_chunks == 0:
        return 0, 0, 0.0

    gated_chunks = sum(
        1
        for chunk in eligible_chunks
        if len(attempts_by_chunk[chunk]) > 1
    )
    gating_rate = gated_chunks / total_chunks
    return gated_chunks, total_chunks, gating_rate


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


def finite_float(value) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def apply_y_limits_from_values(ax, values: List[float]) -> None:
    values = [value for value in values if finite_float(value) is not None]
    if not values:
        return

    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1e-6)
    else:
        pad = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_metrics(
    attempts_by_chunk: Dict[int, Dict[int, Dict]], out_path: Path, 
    structural_metrics: List[str], informational_metrics: List[str],
    sidecar_by_chunk: Dict[int, Dict] | None = None,
    y_axis_ignore_first_n: int = 50,
    title: str | None = None,
) -> None:
    """Create the metrics plot with two columns: structural and informational."""
    sidecar_by_chunk = sidecar_by_chunk or {}
    chunks_sorted = sorted(attempts_by_chunk.keys())
    if not chunks_sorted:
        raise SystemExit("No chunk attempts found.")
    y_axis_ignore_first_n = max(0, int(y_axis_ignore_first_n))
    chunk_rank = {chunk: idx for idx, chunk in enumerate(chunks_sorted)}

    cumulative_tokens = load_cumulative_tokens(attempts_by_chunk, sidecar_by_chunk)
    use_token_x_axis = bool(cumulative_tokens)
    gated_chunks, total_chunks, gating_rate = compute_gating_frequency(
        attempts_by_chunk,
        cumulative_tokens,
    )
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

    def get_x(chunk: int) -> float:
        if use_token_x_axis:
            return cumulative_tokens.get(chunk, 0) / 1000.0
        return float(chunk)

    def get_metric(chunk: int, rec: Dict, metric: str, include_sidecar: bool = False):
        """Read metric values only; sidecar values are accepted-history metrics."""
        if include_sidecar and chunk in sidecar_by_chunk and metric in sidecar_by_chunk[chunk]:
            return sidecar_by_chunk[chunk][metric]
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
        y_limit_values: List[float] = []

        rejected_label = f"rejected chunk (gating frequency: {gating_rate * 100:.1f}%)"
        def contributes_to_ylim(chunk: int) -> bool:
            return chunk_rank.get(chunk, 0) >= y_axis_ignore_first_n

        def add_ylim_value(value) -> None:
            value = finite_float(value)
            if value is not None:
                y_limit_values.append(value)

        for chunk in chunks_sorted:
            attempts = attempts_by_chunk[chunk]
            best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
            
            # Plot failed attempts as faint scatter
            for att, rec in attempts.items():
                y = get_metric(chunk, rec, metric, include_sidecar=False)
                if y is None:
                    continue  # missing metric
                x = get_x(chunk)
                if att == best_attempt:
                    continue
                if contributes_to_ylim(chunk):
                    add_ylim_value(y)
                ax.scatter(
                    x, y,
                    color="gray",
                    alpha=0.35,
                    s=12,
                    marker="x",
                    label=rejected_label,
                )

            # Record accepted attempt for line
            best_rec = attempts[best_attempt]
            y_best = get_metric(chunk, best_rec, metric, include_sidecar=True)
            if y_best is not None:
                xs_line.append(get_x(chunk))
                ys_line.append(y_best)
                if contributes_to_ylim(chunk):
                    add_ylim_value(y_best)

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
                segment_contributes_to_ylim = chunk_rank.get(end, 0) >= y_axis_ignore_first_n
                
                if is_100k_bin:
                    # If it's the 100k bin, handle <100k visualization
                    if is_structural:
                        # Structural: Do NOT draw < 100k.
                        # Clamp start to 100k
                        draw_x0 = max(x0, 100.0)
                        
                        if draw_x0 < x1:
                            if segment_contributes_to_ylim:
                                add_ylim_value(p25)
                                add_ylim_value(p75)
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
                            if segment_contributes_to_ylim:
                                add_ylim_value(p25)
                                add_ylim_value(p75)
                            ax.fill_between(
                                [draw_x0, x1], [p25, p25], [p75, p75],
                                color=color, alpha=0.08, zorder=1.5,
                                label=label
                            )
                            band_labels_used.add(label_key)

                else:
                    # Normal bin including 100k+
                    if segment_contributes_to_ylim:
                        add_ylim_value(p25)
                        add_ylim_value(p75)
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
                    if chunk_rank.get(end, 0) >= y_axis_ignore_first_n:
                        add_ylim_value(mean)
                    ax.plot(
                        [draw_x0, x1], [mean, mean],
                        color=color, linestyle="--", linewidth=1.2, zorder=1.6,
                        label=label
                    )
                mean_labels_used.add(label_key)

        secondary_metric = SECONDARY_METRIC_OVERLAYS.get(metric)
        if secondary_metric:
            secondary_color = "#4b5563"
            secondary_ys = []
            for chunk in chunks_sorted:
                attempts = attempts_by_chunk[chunk]
                best_attempt = find_accepted_attempt(chunk, attempts_by_chunk)
                best_rec = attempts[best_attempt]
                y_secondary = get_metric(chunk, best_rec, secondary_metric, include_sidecar=True)
                y_secondary = finite_float(y_secondary)
                if y_secondary is None:
                    continue
                secondary_ys.append(y_secondary)

            if secondary_ys:
                mean_r2 = sum(secondary_ys) / len(secondary_ys)
                ax.text(
                    0.02,
                    0.92,
                    f"mean(R^2) = {mean_r2:.3f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color=secondary_color,
                    bbox=dict(facecolor="white", edgecolor=secondary_color, alpha=0.82, boxstyle="round,pad=0.2"),
                )

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        apply_y_limits_from_values(ax, y_limit_values)

    legend_handles = []

    # Plot Structural Column (Left)
    for i in range(rows):
        metric = structural_metrics[i] if i < len(structural_metrics) else None
        plot_on_ax(axes[i, 0], metric)
        if i == rows - 1:
            axes[i, 0].set_xlabel("Tokens (k)" if use_token_x_axis else "Chunk")

    # Plot Informational Column (Right)
    for i in range(rows):
        metric = informational_metrics[i] if i < len(informational_metrics) else None
        plot_on_ax(axes[i, 1], metric)
        if i == rows - 1:
            axes[i, 1].set_xlabel("Tokens (k)" if use_token_x_axis else "Chunk")

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

    title = title or "Metric Evolution per accumulated generated chunks"
    if y_axis_ignore_first_n:
        title += f"\nY-axis range ignores the first {y_axis_ignore_first_n} chunks"
    fig.suptitle(title, fontsize=14)
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
    parser.add_argument(
        "--preset",
        choices=sorted(METRIC_PRESETS),
        default="default",
        help="Metric preset. Sidecar metrics require token_naturalness_by_chunk.jsonl to exist.",
    )
    parser.add_argument(
        "--structural-metrics",
        default=None,
        help="Comma-separated metrics for left column. Overrides --preset left column.",
    )
    parser.add_argument(
        "--informational-metrics",
        default=None,
        help="Comma-separated metrics for right column. Overrides --preset right column.",
    )
    parser.add_argument(
        "--y-axis-ignore-first-n",
        type=int,
        default=50,
        help="Draw early chunks but exclude the first N chunks from y-axis autoscaling. Use 0 to disable.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title. Defaults to a title derived from --preset.",
    )
    
    args = parser.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    attempts_by_chunk = load_attempts(folder)
    sidecar_by_chunk = load_token_naturalness_by_chunk(folder)
    out_path = Path(args.out) if args.out else folder / "_metrics_evolution.jpg"

    structural_metrics, informational_metrics = normalize_preset_columns(METRIC_PRESETS[args.preset])
    structural_metrics = parse_metric_list(args.structural_metrics) or structural_metrics
    informational_metrics = parse_metric_list(args.informational_metrics) or informational_metrics

    plot_metrics(
        attempts_by_chunk,
        out_path,
        structural_metrics,
        informational_metrics,
        sidecar_by_chunk=sidecar_by_chunk,
        y_axis_ignore_first_n=args.y_axis_ignore_first_n,
        title=args.title or PRESET_TITLES.get(args.preset),
    )


if __name__ == "__main__":
    main()
