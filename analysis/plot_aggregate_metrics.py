#!/usr/bin/env python3
"""
Plot aggregate descriptor evolution by conditioning mode.

The script reads generated-book prompt folders directly:
  - legacy descriptors from accepted chunk attempt JSON
  - additional token descriptors from token_naturalness_by_chunk.jsonl

It bins each run on accumulated token count, averages within each run/bin, then
computes group mean/std across runs. This keeps dense stride-1 sidecars from
overweighting a group relative to sparse sidecars.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ANALYSIS_DIR = Path(__file__).parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))

import plot_chunk_metrics as pcm


DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
DEFAULT_OUT_LEGACY = ROOT / "generated_texts_og" / "Output" / "_aggregate_legacy_descriptors_by_group.jpg"
DEFAULT_OUT_ADDITIONAL = ROOT / "generated_texts_og" / "Output" / "_aggregate_additional_descriptors_by_group.jpg"
DEFAULT_OUT_FIG1 = ROOT / "generated_texts_og" / "Output" / "_fig1_legacy_descriptors_all_modes.jpg"
DEFAULT_CACHE = ROOT / "generated_texts_og" / "Output" / "_aggregate_descriptor_group_stats.json"
CACHE_VERSION = 1

# canonical 8 statistical-scaling + informational descriptors (matches the PCA
# and discriminant scripts: H_DFA1 + H_WT_db4 are the retained Hurst estimators)
LEGACY_METRICS = [
    "beta",
    "alpha",
    "H_DFA1",
    "H_WT_db4",
    "entropy_bits",
    "perplexity",
    "lz_complexity",
    "mi_decay_gamma",
]

ADDITIONAL_METRICS = [
    "tp_entropy_slope",
    "tr_repeat_gap_burstiness",
    "tr_repeat_gap_cv",
    "tr_repeat_gap_logbin_entropy",
    "td_hapax_token_fraction",
    "td_zipf_exponent",
    "td_heaps_exponent",
]

SUMMARIES_GROUPS = [
    "summaries_high_temperature",
    "summaries_low_temperature",
]

CONDITIONING_GROUPS = [
    "extended_conditioning",
    "extended_no_fixed_conditioning",
    "basic_conditioning",
    "minimal_fixed_conditioning",
    "markov_rolling_conditioning",
]

GROUP_ORDER = SUMMARIES_GROUPS + CONDITIONING_GROUPS

# Main-figure mode set: the six conditioning modes used in the discriminant
# figures (low-temperature summaries are an ablation, kept for Extended Data).
FIG1_GROUPS = [
    "summaries_high_temperature",
    "extended_conditioning",
    "extended_no_fixed_conditioning",
    "basic_conditioning",
    "minimal_fixed_conditioning",
    "markov_rolling_conditioning",
]

GROUP_LABELS = {
    "summaries_high_temperature": "Summaries high-temp",
    "summaries_low_temperature": "Summaries low-temp",
    "extended_conditioning": "Extended/full",
    "extended_no_fixed_conditioning": "Extended no fixed",
    "basic_conditioning": "Basic",
    "minimal_fixed_conditioning": "Minimal fixed",
    "markov_rolling_conditioning": "Markov rolling",
}

GROUP_COLORS = {
    "summaries_high_temperature": "#1f77b4",
    "summaries_low_temperature": "#17becf",
    "extended_conditioning": "#d62728",
    "extended_no_fixed_conditioning": "#00897B",
    "basic_conditioning": "#2ca02c",
    "minimal_fixed_conditioning": "#9467bd",
    "markov_rolling_conditioning": "#8c564b",
}

CHAOTIC_METRICS_PATH = ROOT / "generated_texts_og" / "Output" / "chaotic_run_metrics.json"


def finite_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def has_attempt_files(folder: Path) -> bool:
    return any("prompt" not in path.name for path in folder.glob("chunk_*_attempt_*.txt"))


def read_metadata(folder: Path) -> dict:
    path = folder / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_conditioning_group(folder: Path) -> str | None:
    name = folder.name
    meta = read_metadata(folder)
    mode = meta.get("conditioning_mode")

    if not mode:
        for candidate in (
            "summaries_conditioning",
            "extended_conditioning",
            "extended_no_fixed_conditioning",
            "basic_conditioning",
            "minimal_fixed_conditioning",
            "markov_rolling_conditioning",
            "minimal_rolling_conditioning",
        ):
            if candidate in name:
                mode = candidate
                break

    if mode == "summaries_conditioning":
        if "low_temperature" in name:
            return "summaries_low_temperature"
        if "high_temperature" in name:
            return "summaries_high_temperature"
        temp_range = meta.get("temperature_range")
        if isinstance(temp_range, list) and temp_range:
            return "summaries_low_temperature" if max(temp_range) <= 0.9 else "summaries_high_temperature"
        return "summaries_high_temperature"

    if mode == "extended_conditioning":
        return "extended_conditioning"
    if mode == "extended_no_fixed_conditioning":
        return "extended_no_fixed_conditioning"
    if mode == "basic_conditioning":
        return "basic_conditioning"
    if mode == "minimal_fixed_conditioning":
        return "minimal_fixed_conditioning"
    if mode in {"markov_rolling_conditioning", "minimal_rolling_conditioning"}:
        return "markov_rolling_conditioning"
    return None


def iter_run_folders(prompts_dir: Path, pattern: str | None = None) -> List[Path]:
    folders = []
    for folder in sorted(prompts_dir.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        if pattern and pattern not in folder.name:
            continue
        if not has_attempt_files(folder):
            continue
        if infer_conditioning_group(folder) is None:
            continue
        folders.append(folder)
    return folders


def accepted_attempt_metrics(folder: Path) -> Dict[str, List[Tuple[float, float, int]]]:
    attempts_by_chunk = pcm.load_attempts(folder)
    if not attempts_by_chunk:
        return {}
    sidecar_by_chunk = pcm.load_token_naturalness_by_chunk(folder)
    token_map = pcm.load_cumulative_tokens(attempts_by_chunk, sidecar_by_chunk)

    series: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for chunk_id in sorted(attempts_by_chunk):
        attempts = attempts_by_chunk[chunk_id]
        accepted = pcm.find_accepted_attempt(chunk_id, attempts_by_chunk)
        rec = attempts[accepted]
        tokens = finite_float(token_map.get(chunk_id))
        if tokens is None:
            continue

        combined = {}
        for source in (
            rec.get("metrics", {}),
            rec.get("chunk_metrics", {}),
            rec.get("accumulated_metrics", {}),
        ):
            if source:
                combined.update(source)

        for metric in LEGACY_METRICS:
            val = finite_float(combined.get(metric))
            if val is not None:
                series[metric].append((tokens, val, chunk_id))
    return series


def sidecar_metrics(folder: Path) -> Dict[str, List[Tuple[float, float, int]]]:
    records = pcm.load_token_naturalness_by_chunk(folder)
    series: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    metrics = set(ADDITIONAL_METRICS)
    for rec in records.values():
        tokens = finite_float(rec.get("n_tokens"))
        if tokens is None:
            continue
        chunk_id = rec.get("chunk_id")
        if chunk_id is None:
            continue
        chunk_id = int(chunk_id)
        for metric in metrics:
            val = finite_float(rec.get(metric))
            if val is not None:
                series[metric].append((tokens, val, chunk_id))
    return series


def folder_metric_series(folder: Path) -> Dict[str, List[Tuple[float, float, int]]]:
    series = accepted_attempt_metrics(folder)
    for metric, points in sidecar_metrics(folder).items():
        series[metric].extend(points)
    return dict(series)


def bin_folder_series(
    series: Dict[str, List[Tuple[float, float, int]]],
    bin_edges: np.ndarray,
    max_tokens: int,
    y_axis_ignore_first_n: int,
) -> Dict[str, Dict[float, dict]]:
    binned: Dict[str, Dict[float, dict]] = {}
    for metric, points in series.items():
        if not points:
            continue
        values_by_bin: Dict[float, List[float]] = defaultdict(list)
        ylim_values_by_bin: Dict[float, List[float]] = defaultdict(list)
        for tokens, value, chunk_id in points:
            if tokens < 0 or tokens > max_tokens:
                continue
            idx = np.searchsorted(bin_edges, tokens, side="right") - 1
            if idx < 0 or idx >= len(bin_edges) - 1:
                continue
            center = float((bin_edges[idx] + bin_edges[idx + 1]) / 2.0)
            values_by_bin[center].append(value)
            if metric in LEGACY_METRICS or chunk_id > y_axis_ignore_first_n:
                ylim_values_by_bin[center].append(value)
        if values_by_bin:
            binned[metric] = {
                center: {
                    "value": float(np.mean(vals)),
                    "ylim_value": float(np.mean(ylim_values_by_bin[center]))
                    if ylim_values_by_bin.get(center)
                    else None,
                }
                for center, vals in values_by_bin.items()
                if vals
            }
    return binned


def aggregate_groups(
    folders: Iterable[Path],
    bin_size: int,
    max_tokens: int,
    y_axis_ignore_first_n: int,
) -> tuple[dict, Counter]:
    bin_edges = np.arange(0, max_tokens + bin_size, bin_size, dtype=float)
    values: Dict[str, Dict[str, Dict[float, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ylim_values: Dict[str, Dict[str, Dict[float, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    counts: Counter = Counter()

    for folder in folders:
        group = infer_conditioning_group(folder)
        if group is None:
            continue
        counts[group] += 1
        series = folder_metric_series(folder)
        binned = bin_folder_series(series, bin_edges, max_tokens, y_axis_ignore_first_n)
        for metric, bin_map in binned.items():
            for center, rec in bin_map.items():
                values[group][metric][center].append(rec["value"])
                if rec["ylim_value"] is not None:
                    ylim_values[group][metric][center].append(rec["ylim_value"])

    stats = {}
    for group, metric_map in values.items():
        stats[group] = {}
        for metric, bin_map in metric_map.items():
            stats[group][metric] = {}
            for center, vals in sorted(bin_map.items()):
                if not vals:
                    continue
                arr = np.array(vals, dtype=float)
                stats[group][metric][center] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "count": int(len(arr)),
                }
                ylim_arr = np.array(ylim_values[group][metric].get(center, []), dtype=float)
                if ylim_arr.size:
                    stats[group][metric][center].update(
                        {
                            "ylim_mean": float(np.mean(ylim_arr)),
                            "ylim_std": float(np.std(ylim_arr)),
                            "ylim_count": int(len(ylim_arr)),
                        }
                    )
    return stats, counts


def cache_settings(
    prompts_dir: Path,
    pattern: str | None,
    bin_size: int,
    max_tokens: int,
    y_axis_ignore_first_n: int,
) -> dict:
    return {
        "version": CACHE_VERSION,
        "prompts_dir": str(prompts_dir.resolve()),
        "pattern": pattern,
        "bin_size": int(bin_size),
        "max_tokens": int(max_tokens),
        "y_axis_ignore_first_n": int(y_axis_ignore_first_n),
        "legacy_metrics": LEGACY_METRICS,
        "additional_metrics": ADDITIONAL_METRICS,
        "groups": GROUP_ORDER,
    }


def serialize_stats(stats: dict) -> dict:
    return {
        group: {
            metric: {str(center): rec for center, rec in bin_map.items()}
            for metric, bin_map in metric_map.items()
        }
        for group, metric_map in stats.items()
    }


def deserialize_stats(raw_stats: dict) -> dict:
    stats = {}
    for group, metric_map in raw_stats.items():
        stats[group] = {}
        for metric, bin_map in metric_map.items():
            stats[group][metric] = {}
            for center, rec in bin_map.items():
                try:
                    stats[group][metric][float(center)] = rec
                except (TypeError, ValueError):
                    continue
    return stats


def load_cached_aggregate(cache_path: Path, settings: dict) -> tuple[dict, Counter] | None:
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Cache read failed ({exc}); recalculating.")
        return None
    if raw.get("settings") != settings:
        print(f"Cache settings differ from current options; recalculating.")
        return None
    stats = deserialize_stats(raw.get("stats", {}))
    counts = Counter({group: int(count) for group, count in raw.get("counts", {}).items()})
    if not stats:
        return None
    print(f"Loaded aggregate stats cache -> {cache_path}")
    return stats, counts


def write_cached_aggregate(
    cache_path: Path,
    settings: dict,
    folders: Iterable[Path],
    stats: dict,
    counts: Counter,
) -> None:
    payload = {
        "settings": settings,
        "folder_count": int(sum(counts.values())),
        "folders": [folder.name for folder in folders],
        "counts": dict(counts),
        "stats": serialize_stats(stats),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote aggregate stats cache -> {cache_path}")


NATURAL_BENCHMARK_PALETTE = ["#5B8FF9", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B"]
NATURAL_INNER_BAND_ALPHA = 0.18
NATURAL_OUTER_BAND_ALPHA = 0.10
DEFAULT_Y_PAD_FRACTION = 0.18
METRIC_Y_PAD_FRACTION = {
    "beta": 0.25,
    "alpha": 0.25,
}
METRIC_Y_FLOOR = {
    "beta": 0.0,
    "alpha": 0.0,
}


def natural_segments_for_metric(metric: str, xmax_k: float):
    size_stats = pcm.load_size_stats()
    if not size_stats:
        return []

    segments = []
    for idx, b in enumerate(size_stats.get("bins", [])):
        mstats = b.get("metrics", {}).get(metric)
        if not mstats:
            continue
        lo_tok, hi_tok = b.get("range_tokens", [None, None])
        if lo_tok is None:
            continue
        if hi_tok is None:
            hi_tok = 10_000_000
        x0 = max(0.0, lo_tok / 1000.0)
        x1 = min(xmax_k, hi_tok / 1000.0)
        if x0 >= x1:
            continue
        outer_low = mstats.get("p2_5") or mstats.get("p5") or mstats.get("p25")
        outer_high = mstats.get("p97_5") or mstats.get("p95") or mstats.get("p75")
        inner_low = mstats.get("p25")
        inner_high = mstats.get("p75")
        mean = mstats.get("mean")
        color = NATURAL_BENCHMARK_PALETTE[idx % len(NATURAL_BENCHMARK_PALETTE)]
        segments.append((x0, x1, outer_low, outer_high, inner_low, inner_high, mean, color))
    return segments


def plot_natural_overlay(
    ax,
    metric: str,
    xmax_k: float,
    y_values: list[float],
    *,
    contribute_to_ylim: bool = False,
) -> None:
    for idx, (
        x0,
        x1,
        outer_low,
        outer_high,
        inner_low,
        inner_high,
        mean,
        color,
    ) in enumerate(natural_segments_for_metric(metric, xmax_k)):
        if outer_low is not None and outer_high is not None:
            ax.fill_between(
                [x0, x1],
                [outer_low, outer_low],
                [outer_high, outer_high],
                color=color,
                alpha=NATURAL_OUTER_BAND_ALPHA,
                zorder=1,
                label="natural p2.5-p97.5" if idx == 0 else None,
            )
            if contribute_to_ylim:
                y_values.extend([outer_low, outer_high])
        if inner_low is not None and inner_high is not None:
            ax.fill_between(
                [x0, x1],
                [inner_low, inner_low],
                [inner_high, inner_high],
                color=color,
                alpha=NATURAL_INNER_BAND_ALPHA,
                zorder=2,
                label="natural p25-p75" if idx == 0 else None,
            )
            if contribute_to_ylim:
                y_values.extend([inner_low, inner_high])
        if mean is not None:
            ax.plot(
                [x0, x1],
                [mean, mean],
                color=color,
                linestyle="--",
                linewidth=1.15,
                zorder=3,
                label="natural mean" if idx == 0 else None,
            )
            if contribute_to_ylim:
                y_values.append(mean)


def y_limits_from_values(values: list[float], pad_fraction: float = 0.18) -> tuple[float, float] | None:
    finite = np.array([v for v in values if math.isfinite(v)], dtype=float)
    if finite.size < 2:
        return None
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        return None
    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def y_limits_for_metric(metric: str, values: list[float]) -> tuple[float, float] | None:
    limits = y_limits_from_values(
        values,
        pad_fraction=METRIC_Y_PAD_FRACTION.get(metric, DEFAULT_Y_PAD_FRACTION),
    )
    if limits is None:
        return None
    lo, hi = limits
    floor = METRIC_Y_FLOOR.get(metric)
    if floor is not None:
        lo = max(floor, lo)
    return lo, hi


def load_chaotic_data(max_tokens: int) -> Dict[str, Dict[float, float]]:
    if not CHAOTIC_METRICS_PATH.exists():
        return {}
    try:
        raw = json.loads(CHAOTIC_METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    chaotic: Dict[str, Dict[float, float]] = {}
    for metric, mapping in raw.items():
        if not isinstance(mapping, dict):
            continue
        metric_map = {}
        for tok, value in mapping.items():
            try:
                tokens = float(tok)
            except Exception:
                continue
            val = finite_float(value)
            if val is None or tokens > max_tokens:
                continue
            metric_map[tokens] = val
        if metric_map:
            chaotic[metric] = metric_map
    return chaotic


def plot_metric_panel(
    ax,
    stats: dict,
    counts: Counter,
    metric: str,
    max_tokens: int,
    groups: List[str],
    chaotic_data: Dict[str, Dict[float, float]] | None = None,
    y_axis_ignore_first_n: int = 50,
) -> list[float]:
    xmax_k = max_tokens / 1000.0
    y_values: list[float] = []
    plot_natural_overlay(ax, metric, xmax_k, y_values, contribute_to_ylim=True)

    for group in groups:
        metric_bins = stats.get(group, {}).get(metric, {})
        if not metric_bins:
            continue
        centers = sorted(metric_bins)
        xs = np.array(centers, dtype=float) / 1000.0
        means = np.array([metric_bins[c]["mean"] for c in centers], dtype=float)
        stds = np.array([metric_bins[c]["std"] for c in centers], dtype=float)
        color = GROUP_COLORS[group]
        label = f"{GROUP_LABELS[group]} (n={counts[group]})"
        ax.plot(xs, means, color=color, linewidth=1.8, label=label, zorder=5)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.13, zorder=4)

        ylim_means = np.array([metric_bins[c].get("ylim_mean", np.nan) for c in centers], dtype=float)
        ylim_stds = np.array([metric_bins[c].get("ylim_std", np.nan) for c in centers], dtype=float)
        y_values.extend(ylim_means[np.isfinite(ylim_means)].tolist())
        y_values.extend((ylim_means - ylim_stds)[np.isfinite(ylim_means - ylim_stds)].tolist())
        y_values.extend((ylim_means + ylim_stds)[np.isfinite(ylim_means + ylim_stds)].tolist())

    if chaotic_data and metric in chaotic_data:
        token_map = chaotic_data[metric]
        xs = np.array(sorted(token_map), dtype=float) / 1000.0
        ys = np.array([token_map[t] for t in sorted(token_map)], dtype=float)
        ax.plot(xs, ys, color="#111827", linestyle=":", linewidth=1.7, alpha=0.9, label="Chaotic run", zorder=6)
        ymask = np.ones_like(xs, dtype=bool) if metric in LEGACY_METRICS else xs >= float(y_axis_ignore_first_n)
        y_values.extend(ys[ymask & np.isfinite(ys)].tolist())

    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, xmax_k)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticklabels(["0", "200k", "400k", "600k", "800k", "1M"])
    return y_values


def plot_mode_grid(
    stats: dict,
    counts: Counter,
    out_path: Path,
    max_tokens: int,
    metrics: List[str],
    title: str,
    groups: List[str],
    ncols: int = 4,
    chaotic_data: Dict[str, Dict[float, float]] | None = None,
    y_axis_ignore_first_n: int = 50,
) -> None:
    """Main-figure layout: one panel per descriptor, every conditioning mode
    overlaid in the same panel, against the natural reference bands."""
    rows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(rows, ncols, figsize=(5.4 * ncols, 3.6 * rows),
                             sharex=True, squeeze=False)
    flat = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = flat[i]
        values = plot_metric_panel(
            ax, stats, counts, metric, max_tokens,
            groups=groups, chaotic_data=chaotic_data,
            y_axis_ignore_first_n=y_axis_ignore_first_n,
        )
        limits = y_limits_for_metric(metric, values)
        if limits:
            ax.set_ylim(*limits)
    for ax in flat[len(metrics):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Tokens")

    handles, labels, seen, hmap = [], [], set(), {}
    for ax in flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l); hmap[l] = h

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # grey swatch standing in for the per-mode shaded ±1σ bands
    sd_handle = Patch(facecolor="0.45", alpha=0.30, edgecolor="none")

    # ── two-part bottom legend, split by a vertical divider ──
    # Left block: the natural reference bands + the chaotic (no-conditioning)
    # baseline. Ordered column-major (ncol=2) so "Chaotic run" lands directly
    # beneath "natural mean".
    ref_order = ["natural p2.5-p97.5", "natural p25-p75", "natural mean", "Chaotic run"]
    ref_pairs = [(hmap[l], l) for l in ref_order if l in hmap]
    # Right block: the conditioning modes and their ±1σ bands.
    mode_bases = ["Summaries high-temp", "Extended/full", "Extended no fixed",
                  "Basic", "Minimal fixed", "Markov rolling"]
    # Fig 1 uses the manuscript's mode names (GROUP_LABELS stays as-is elsewhere,
    # e.g. the two-column comparison that also plots low-temp summaries)
    mode_display = {"Summaries high-temp": "Hierarchical summaries",
                    "Extended/full": "Extended", "Extended no fixed": "Extended variant",
                    "Minimal fixed": "Fixed contract"}
    mode_pairs = [(sd_handle, "±1σ within conditioning mode")]
    for base in mode_bases:
        match = next((l for l in hmap if l.startswith(base + " (")), None)
        if match:
            mode_pairs.append((hmap[match], match.replace(base, mode_display.get(base, base))))

    if ref_pairs:
        leg_ref = fig.legend([h for h, _ in ref_pairs], [l for _, l in ref_pairs],
                             loc="lower right", bbox_to_anchor=(0.36, 0.005),
                             ncol=2, fontsize=9, frameon=False)
        fig.add_artist(leg_ref)
    if mode_pairs:
        fig.legend([h for h, _ in mode_pairs], [l for _, l in mode_pairs],
                   loc="lower left", bbox_to_anchor=(0.39, 0.005),
                   ncol=4, fontsize=9, frameon=False)
    # divider between the natural/reference block and the conditioning modes
    fig.add_artist(Line2D([0.375, 0.375], [0.012, 0.075], transform=fig.transFigure,
                          color="0.6", linewidth=1.0))

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"figure  → {out_path}")


def plot_two_column_comparison(
    stats: dict,
    counts: Counter,
    out_path: Path,
    max_tokens: int,
    metrics: List[str],
    title: str,
    chaotic_data: Dict[str, Dict[float, float]] | None = None,
    y_axis_ignore_first_n: int = 50,
) -> None:
    rows = len(metrics)
    fig, axes = plt.subplots(rows, 2, figsize=(22, 3.7 * rows), sharex=True)
    if rows == 1:
        axes = axes.reshape(1, 2)

    for row, metric in enumerate(metrics):
        left_values = plot_metric_panel(
            axes[row, 0],
            stats,
            counts,
            metric,
            max_tokens,
            groups=SUMMARIES_GROUPS,
            chaotic_data=chaotic_data,
            y_axis_ignore_first_n=y_axis_ignore_first_n,
        )
        right_values = plot_metric_panel(
            axes[row, 1],
            stats,
            counts,
            metric,
            max_tokens,
            groups=CONDITIONING_GROUPS,
            chaotic_data=None,
            y_axis_ignore_first_n=y_axis_ignore_first_n,
        )
        limits = y_limits_for_metric(metric, left_values + right_values)
        if limits:
            axes[row, 0].set_ylim(*limits)
            axes[row, 1].set_ylim(*limits)

    axes[-1, 0].set_xlabel("Tokens")
    axes[-1, 1].set_xlabel("Tokens")

    handles, labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = set()
    dedup = []
    for h, l in zip(handles, labels):
        if not l or l in seen:
            continue
        seen.add(l)
        dedup.append((h, l))

    if dedup:
        fig.legend(
            [h for h, _ in dedup],
            [l for _, l in dedup],
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=8,
            frameon=False,
        )

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.08, 1, 0.965])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved aggregate plot -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--out-legacy", type=Path, default=DEFAULT_OUT_LEGACY)
    parser.add_argument("--out-additional", type=Path, default=DEFAULT_OUT_ADDITIONAL)
    parser.add_argument("--out-fig1", type=Path, default=DEFAULT_OUT_FIG1)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute grouped aggregate stats and overwrite cache.")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write the aggregate stats cache.")
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--bin-size", type=int, default=20_000)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument(
        "--y-axis-ignore-first-n",
        type=int,
        default=50,
        help="Draw early iterations but exclude first N chunks from y-axis autoscaling.",
    )
    args = parser.parse_args()

    settings = cache_settings(
        args.prompts_dir,
        args.pattern,
        args.bin_size,
        args.max_tokens,
        args.y_axis_ignore_first_n,
    )
    cached = None if args.no_cache or args.refresh_cache else load_cached_aggregate(args.cache, settings)
    if cached is not None:
        stats, counts = cached
    else:
        folders = iter_run_folders(args.prompts_dir, args.pattern)
        print(f"Found {len(folders)} reconstructable grouped folders.")
        group_counts = Counter(infer_conditioning_group(folder) for folder in folders)
        for group in GROUP_ORDER:
            print(f"  {GROUP_LABELS[group]}: {group_counts[group]}")

        stats, counts = aggregate_groups(
            folders,
            bin_size=args.bin_size,
            max_tokens=args.max_tokens,
            y_axis_ignore_first_n=args.y_axis_ignore_first_n,
        )
        if not args.no_cache:
            write_cached_aggregate(args.cache, settings, folders, stats, counts)

    if cached is not None:
        for group in GROUP_ORDER:
            print(f"  {GROUP_LABELS[group]}: {counts[group]}")

    chaotic_data = load_chaotic_data(args.max_tokens)

    # Main figure: one panel per descriptor, all conditioning modes overlaid,
    # no chaotic run / low-temperature ablation (those stay in Extended Data).
    plot_mode_grid(
        stats,
        counts,
        args.out_fig1,
        max_tokens=args.max_tokens,
        metrics=LEGACY_METRICS,
        title="Statistical-scaling and informational descriptors vs accumulated tokens,\n"
              "by conditioning mode, against natural reference bands",
        groups=FIG1_GROUPS,
        ncols=4,
        chaotic_data=chaotic_data,
        y_axis_ignore_first_n=args.y_axis_ignore_first_n,
    )

    plot_two_column_comparison(
        stats,
        counts,
        args.out_legacy,
        max_tokens=args.max_tokens,
        metrics=LEGACY_METRICS,
        title="Legacy Descriptor Evolution by Conditioning Group",
        chaotic_data=chaotic_data,
        y_axis_ignore_first_n=args.y_axis_ignore_first_n,
    )
    plot_two_column_comparison(
        stats,
        counts,
        args.out_additional,
        max_tokens=args.max_tokens,
        metrics=ADDITIONAL_METRICS,
        title="Additional Descriptor Evolution by Conditioning Group",
        chaotic_data=chaotic_data,
        y_axis_ignore_first_n=args.y_axis_ignore_first_n,
    )


if __name__ == "__main__":
    main()
