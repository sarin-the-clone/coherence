#!/usr/bin/env python3
"""
PCA of legacy descriptors on the natural-book corpus, colour-coded by token
size bin, with generated-book prefixes projected onto the natural score plane.

The PCA basis is fit once on the natural books (batch_results). Generated
prefixes are projected with the same scaler/loadings and drawn with the colour
of their matching token size bin, so visual overlap with the same-coloured
natural cloud directly shows that generated prefixes occupy the same
size-binned region of descriptor space as natural books of equal length.

Generated prefix metrics come from local aggregates (the per-run sidecars are
not synced to this machine):
  - generated_texts_og/experimental_metrics.json : 36 main hierarchical-summary
    runs x 5 prefixes (100k, 200k, 400k, 600k, 1M tokens)
  - natural_texts/Output/pca/generated_prefixes/generated_prefix_projection_scores.csv :
    3 ablation runs (minimal fixed, basic, markov rolling) x 5 prefixes

Outputs:
  - legacy_pca_size_bins_combined.png : overview panel (corpus density +
    loadings + size-bin means) with one score panel per size bin (poster,
    2-column span)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BATCH_RESULTS = ROOT / "natural_texts" / "Output" / "batch_results"
DEFAULT_GENERATED_MAIN = ROOT / "generated_texts_og" / "experimental_metrics.json"
DEFAULT_GENERATED_ABLATION = (
    ROOT / "natural_texts" / "Output" / "pca" / "generated_prefixes"
    / "generated_prefix_projection_scores.csv"
)
DEFAULT_SIZE_STATS = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"
DEFAULT_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "legacy_size_bins"
DEFAULT_POSTER_DIR = ROOT / "ppt"

# Canonical 8 stat/info descriptors, consistent with pca_all_metrics.py and
# pca_sc_natural_combined.py (H_DFA1 + H_WT_db4 retained as the Hurst estimators).
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

METRIC_TEX = {
    "beta": r"$\beta$",
    "alpha": r"$\alpha$",
    "H_DFA1": r"$H$",
    "H_WT_db4": r"$H_{db4}$",
    "entropy_bits": r"$\mathcal{S}$",
    "perplexity": r"$PP$",
    "lz_complexity": r"$LZC$",
    "mi_decay_gamma": r"$\gamma$",
}

NAVY = "#0D2545"


def size_bin_palette(n: int) -> list[str]:
    """Sequential yellow→green→blue palette matching ppt/sc_pca.png:
    light yellow for the densest short-book bin, dark blue for rare long books."""
    from matplotlib import cm
    from matplotlib.colors import to_hex
    return [to_hex(cm.YlGnBu(x)) for x in np.linspace(0.08, 0.92, n)]


def format_token_label(n_tokens: int) -> str:
    if n_tokens >= 1_000_000 and n_tokens % 1_000_000 == 0:
        return f"{n_tokens // 1_000_000}M"
    if n_tokens >= 1_000:
        return f"{n_tokens / 1_000:.0f}k"
    return str(n_tokens)


def load_size_bins(path: Path) -> list[dict]:
    fallback = [
        {"label": "100-150k", "range_tokens": [100_000, 150_000]},
        {"label": "150-300k", "range_tokens": [150_000, 300_000]},
        {"label": "300-500k", "range_tokens": [300_000, 500_000]},
        {"label": "500-800k", "range_tokens": [500_000, 800_000]},
        {"label": ">800k", "range_tokens": [800_000, float("inf")]},
    ]
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    bins = []
    for item in data.get("bins", []):
        lo, hi = item.get("range_tokens", [None, None])
        if lo is None or hi is None:
            continue
        if hi > 5_000_000:
            label = f">{format_token_label(int(lo))}"
            hi = float("inf")
        else:
            label = f"{format_token_label(int(lo))}-{format_token_label(int(hi))}"
        bins.append({"label": label, "range_tokens": [int(lo), float(hi)]})
    return bins or fallback


def darken(color: str, factor: float = 0.62) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (r * factor, g * factor, b * factor)


REGION_LEVELS = [
    # central mass, line alpha, line width — 2D analogues of the 2.5–97.5%
    # and 25–75% one-dimensional intervals
    (0.95, 0.55, 2.2),
    (0.50, 1.00, 3.6),
]
KDE_MAX_POINTS = 2500
KDE_GRID = 240


def add_bin_density_contours(ax, xy: np.ndarray, color: str,
                             rng: np.random.Generator | None = None,
                             lw_scale: float = 1.0) -> None:
    """Draw highest-density regions of the 2D scores as KDE contours.

    The density threshold for each level is the empirical quantile of the KDE
    evaluated at the bin's own points, so each contour encloses the stated
    fraction of the bin's books regardless of the cloud's shape.
    """
    rng = rng or np.random.default_rng(0)
    sample = xy
    if len(sample) > KDE_MAX_POINTS:
        sample = sample[rng.choice(len(sample), KDE_MAX_POINTS, replace=False)]
    kde = gaussian_kde(sample.T)
    dens_at_points = kde(sample.T)

    span_x = sample[:, 0].max() - sample[:, 0].min()
    span_y = sample[:, 1].max() - sample[:, 1].min()
    gx = np.linspace(sample[:, 0].min() - 0.15 * span_x,
                     sample[:, 0].max() + 0.15 * span_x, KDE_GRID)
    gy = np.linspace(sample[:, 1].min() - 0.15 * span_y,
                     sample[:, 1].max() + 0.15 * span_y, KDE_GRID)
    gxx, gyy = np.meshgrid(gx, gy)
    dens = kde(np.vstack([gxx.ravel(), gyy.ravel()])).reshape(gxx.shape)

    for level, alpha, lw in REGION_LEVELS:
        thr = np.quantile(dens_at_points, 1 - level)
        # dark halo keeps pale bin colours readable on the white background
        ax.contour(gxx, gyy, dens, levels=[thr], colors=["#37474F"],
                   linewidths=(lw + 1.8) * lw_scale, alpha=alpha, zorder=4)
        ax.contour(gxx, gyy, dens, levels=[thr], colors=[color],
                   linewidths=lw * lw_scale, alpha=1.0, zorder=5)


def assign_bin(n_tokens: float, size_bins: list[dict],
               lower_edge_tolerance: float = 0.0) -> str | None:
    """Map a token count to its size bin.

    `lower_edge_tolerance` admits counts slightly below the first bin edge:
    generated prefixes target 100k tokens but chunks end at learned EOS
    boundaries, so the accumulated prefix can stop just short of the edge.
    """
    for item in size_bins:
        lo, hi = item["range_tokens"]
        if lo <= n_tokens < hi:
            return item["label"]
    if size_bins and n_tokens >= size_bins[0]["range_tokens"][0] * (1 - lower_edge_tolerance):
        return size_bins[0]["label"]
    return None


def load_natural(batch_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=LEGACY_METRICS + ["n_tokens"])
    return df


def load_generated_main(path: Path) -> pd.DataFrame:
    # low-temperature runs are excluded from the projection by design
    records = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    df = pd.DataFrame([r for r in records
                       if r.get("model") == "Mistral"
                       and r.get("temperature_level") == "High"])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=LEGACY_METRICS + ["n_tokens"])
    df["group"] = "main"
    df["mode_label"] = "hierarchical summaries (high T)"
    return df


def load_generated_ablation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=LEGACY_METRICS + ["n_tokens"])
    df["group"] = "ablation"
    df["mode_label"] = df["condition_label"]
    return df


def plot_scores(natural: pd.DataFrame, generated: pd.DataFrame,
                size_bins: list[dict], bin_color: dict[str, str],
                evr: np.ndarray, figsize: tuple[float, float],
                legend_cols: int = 1) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    for item in size_bins:
        label = item["label"]
        sub = natural[natural["size_bin"] == label]
        if not len(sub):
            continue
        ax.scatter(sub["PC1"], sub["PC2"], s=14, c=bin_color[label],
                   alpha=0.35, linewidths=0, zorder=2)
        if len(sub) >= 20:
            add_bin_density_contours(ax, sub[["PC1", "PC2"]].to_numpy(), bin_color[label])

    # connected per-bin natural means (length trajectory of the corpus mean)
    means = (natural.groupby("size_bin")[["PC1", "PC2"]].mean()
             .reindex([b["label"] for b in size_bins]).dropna())
    ax.plot(means["PC1"], means["PC2"], "-", color=NAVY, lw=2.0, zorder=5)
    for label, row in means.iterrows():
        ax.scatter(row["PC1"], row["PC2"], marker="*", s=620, c=bin_color[label],
                   edgecolors=NAVY, linewidths=1.6, zorder=6)

    # generated prefixes, bin-coloured, marker by group
    for group, marker, size in [("main", "D", 95), ("ablation", "^", 130)]:
        sub = generated[generated["group"] == group]
        for label in means.index:
            s2 = sub[sub["size_bin"] == label]
            if not len(s2):
                continue
            ax.scatter(s2["PC1"], s2["PC2"], marker=marker, s=size,
                       c=bin_color[label], edgecolors="black",
                       linewidths=1.1, alpha=0.95, zorder=7)

    # robust axis limits from the natural cloud
    x_lo, x_hi = np.percentile(natural["PC1"], [0.5, 99.5])
    y_lo, y_hi = np.percentile(natural["PC2"], [0.5, 99.5])
    pad_x, pad_y = 0.10 * (x_hi - x_lo), 0.10 * (y_hi - y_lo)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax.set_xlabel(f"PC1  ({evr[0] * 100:.0f}% var.)", fontsize=28)
    ax.set_ylabel(f"PC2  ({evr[1] * 100:.0f}% var.)", fontsize=28)
    ax.tick_params(labelsize=20)
    metric_str = "  ".join(METRIC_TEX[m] for m in LEGACY_METRICS)
    ax.set_title(f"Natural-book PCA ({metric_str})", fontsize=28, pad=14)

    bin_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=13,
               markerfacecolor=bin_color[b["label"]], markeredgecolor="none",
               label=f'natural {b["label"]}')
        for b in size_bins if b["label"] in set(natural["size_bin"])
    ]
    main_sub = generated[generated["group"] == "main"]
    n_main_runs = main_sub["file"].nunique() if "file" in main_sub.columns else 0
    kind_handles = [
        Line2D([], [], marker="D", linestyle="", markersize=12,
               markerfacecolor="white", markeredgecolor="black",
               label=f"generated prefix (hierarchical summaries mode, {n_main_runs} runs)"),
    ]
    if (generated["group"] == "ablation").any():
        kind_handles.append(
            Line2D([], [], marker="^", linestyle="", markersize=13,
                   markerfacecolor="white", markeredgecolor="black",
                   label="generated prefix (ablation modes)"))
    kind_handles.append(
        Line2D([], [], marker="*", linestyle="-", color=NAVY, markersize=20,
               markerfacecolor="white", markeredgecolor=NAVY,
               label="natural bin mean"))
    kind_handles.append(
        Line2D([], [], linestyle="-", color="#555555", linewidth=3.2,
               label="natural bin 50% region"))
    kind_handles.append(
        Line2D([], [], linestyle="-", color="#555555", linewidth=2.0,
               alpha=0.35, label="natural bin 95% region"))
    leg = ax.legend(handles=bin_handles + kind_handles, fontsize=18,
                    loc="upper left", framealpha=0.92, ncols=legend_cols)
    leg.set_zorder(10)

    fig.tight_layout()
    return fig


def robust_limits(natural: pd.DataFrame, pcx: str = "PC1", pcy: str = "PC2",
                  generated: pd.DataFrame | None = None, lower_q: float = 2.0,
                  upper_q: float = 98.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """2-98% percentile range of the combined natural+generated scores, so the
    heavy PC tails and the few extreme points are ignored (not allowed to
    stretch the axis)."""
    if generated is not None and len(generated):
        xs = np.concatenate([natural[pcx].to_numpy(), generated[pcx].to_numpy()])
        ys = np.concatenate([natural[pcy].to_numpy(), generated[pcy].to_numpy()])
    else:
        xs, ys = natural[pcx].to_numpy(), natural[pcy].to_numpy()
    x_lo, x_hi = np.percentile(xs, [lower_q, upper_q])
    y_lo, y_hi = np.percentile(ys, [lower_q, upper_q])
    pad_x, pad_y = 0.08 * (x_hi - x_lo), 0.08 * (y_hi - y_lo)
    return (x_lo - pad_x, x_hi + pad_x), (y_lo - pad_y, y_hi + pad_y)


def draw_bin_panel(ax, natural: pd.DataFrame, generated: pd.DataFrame,
                   label: str, color: str, mean_row: pd.Series,
                   x_lim: tuple[float, float], y_lim: tuple[float, float],
                   title_prefix: str = "", title_fs: int = 21,
                   tick_fs: int = 16, marker_scale: float = 1.0,
                   pcx: str = "PC1", pcy: str = "PC2") -> None:
    """One size-bin panel: bin cloud over gray corpus, regions, mean, prefixes."""
    sub = natural[natural["size_bin"] == label]
    gen = generated[generated["size_bin"] == label]
    # full corpus as gray context behind the bin of interest
    ax.scatter(natural[pcx], natural[pcy], s=7 * marker_scale, c="#90A4AE",
               alpha=0.30, linewidths=0, zorder=1)
    ax.scatter(sub[pcx], sub[pcy], s=13 * marker_scale, c=color,
               alpha=0.65, linewidths=0, zorder=2)
    if len(sub) >= 20:
        add_bin_density_contours(ax, sub[[pcx, pcy]].to_numpy(), color)
    ax.scatter(mean_row[pcx], mean_row[pcy], marker="*",
               s=560 * marker_scale, c=color,
               edgecolors=NAVY, linewidths=1.6, zorder=6)
    for group, marker, size in [("main", "D", 80), ("ablation", "^", 110)]:
        s2 = gen[gen["group"] == group]
        if not len(s2):
            continue
        ax.scatter(s2[pcx], s2[pcy], marker=marker, s=size * marker_scale,
                   c=color, edgecolors="black",
                   linewidths=1.1, alpha=0.95, zorder=7)
    ax.set_title(f"{title_prefix}{label}   ({len(sub):,} nat · {len(gen)} gen)",
                 fontsize=title_fs, pad=8, color=darken(color, 0.55),
                 fontweight="bold")
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)


def plot_scores_per_bin(natural: pd.DataFrame, generated: pd.DataFrame,
                        size_bins: list[dict], bin_color: dict[str, str],
                        evr: np.ndarray, figsize: tuple[float, float],
                        nrows: int = 2, ncols: int = 3) -> plt.Figure:
    """Small-multiples score plot: one panel per size bin, legend in the spare cell."""
    labels = [b["label"] for b in size_bins if b["label"] in set(natural["size_bin"])]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    x_lim, y_lim = robust_limits(natural)
    means = (natural.groupby("size_bin")[["PC1", "PC2"]].mean()
             .reindex(labels).dropna())

    for ax, label in zip(axes, labels):
        draw_bin_panel(ax, natural, generated, label, bin_color[label],
                       means.loc[label], x_lim, y_lim)

    # spare cells host the legend
    main_sub = generated[generated["group"] == "main"]
    n_main_runs = main_sub["file"].nunique() if "file" in main_sub.columns else 0
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=10,
               markerfacecolor="#B0BEC5", markeredgecolor="none",
               label="all 10 000 natural books"),
        Line2D([], [], marker="o", linestyle="", markersize=10,
               markerfacecolor=bin_color[labels[len(labels) // 2]],
               markeredgecolor="none", label="natural books in size bin"),
        Line2D([], [], marker="D", linestyle="", markersize=11,
               markerfacecolor="white", markeredgecolor="black",
               label="generated prefix\n"
                     f"(hierarchical summaries mode, {n_main_runs} runs)"),
    ]
    if (generated["group"] == "ablation").any():
        handles.append(
            Line2D([], [], marker="^", linestyle="", markersize=12,
                   markerfacecolor="white", markeredgecolor="black",
                   label="generated prefix (ablation modes)"))
    handles += [
        Line2D([], [], marker="*", linestyle="", markersize=19,
               markerfacecolor="white", markeredgecolor=NAVY,
               label="natural bin mean"),
        Line2D([], [], linestyle="-", color="#555555", linewidth=3.2,
               label="natural bin 50% region"),
        Line2D([], [], linestyle="-", color="#555555", linewidth=2.0,
               alpha=0.35, label="natural bin 95% region"),
    ]
    for ax in axes[len(labels):]:
        ax.axis("off")
    axes[len(labels)].legend(handles=handles, fontsize=19, loc="center",
                             framealpha=0)

    metric_str = "  ".join(METRIC_TEX[m] for m in LEGACY_METRICS)
    fig.suptitle(f"Natural-book PCA ({metric_str}) — one panel per size bin",
                 fontsize=27)
    fig.supxlabel(f"PC1  ({evr[0] * 100:.0f}% var.)", fontsize=25)
    fig.supylabel(f"PC2  ({evr[1] * 100:.0f}% var.)", fontsize=25)
    fig.tight_layout(rect=(0.015, 0.015, 1, 1))
    return fig


STAT_METRICS = ["beta", "alpha", "H_DFA1", "H_WT_db4"]
INFO_METRICS = ["entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"]
FAMILY_COLORS = {"statistical": "#00796B", "informational": "#E8710A"}
METRIC_NAMES = {
    "beta": "Beta", "alpha": "Alpha", "H_DFA1": "Hurst", "H_WT_db4": "Hurst (WT)",
    "entropy_bits": "Entropy", "perplexity": "Perplexity",
    "lz_complexity": "LZ Complexity", "mi_decay_gamma": "MI Decay",
}


def add_corpus_density(ax, xy: np.ndarray,
                       rng: np.random.Generator | None = None) -> None:
    """Filled gray KDE contours of the full corpus at 10% percentile steps."""
    rng = rng or np.random.default_rng(0)
    sample = xy
    if len(sample) > KDE_MAX_POINTS:
        sample = sample[rng.choice(len(sample), KDE_MAX_POINTS, replace=False)]
    kde = gaussian_kde(sample.T)
    dens_at_points = kde(sample.T)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    gx = np.linspace(*x_lim, KDE_GRID)
    gy = np.linspace(*y_lim, KDE_GRID)
    gxx, gyy = np.meshgrid(gx, gy)
    dens = kde(np.vstack([gxx.ravel(), gyy.ravel()])).reshape(gxx.shape)
    thresholds = np.quantile(dens_at_points, np.linspace(0.05, 0.95, 10))
    levels = np.append(thresholds, dens.max() * 1.05)
    greys = plt.cm.Greys(np.linspace(0.05, 0.32, len(thresholds)))
    ax.contourf(gxx, gyy, dens, levels=levels, colors=greys, zorder=1)
    ax.contour(gxx, gyy, dens, levels=thresholds, colors="#9E9E9E",
               linewidths=0.6, alpha=0.45, zorder=2)


def add_loading_arrows(ax, loadings: pd.DataFrame, fontsize: int = 17,
                       pcx: str = "PC1", pcy: str = "PC2") -> None:
    """Descriptor loading arrows for the given PC pair, coloured by family.

    Labels of nearly collinear arrows (same family) are staggered radially
    so they stack along the arrow direction instead of overlapping.
    """
    vecs = loadings[[pcx, pcy]].to_numpy()
    max_norm = np.linalg.norm(vecs, axis=1).max()
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    scale = 0.42 * min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) / 2 / max_norm

    angles = np.degrees(np.arctan2(vecs[:, 1], vecs[:, 0]))
    order = np.argsort(angles)
    label_mult = np.full(len(vecs), 1.14)
    group_start, group_pos = 0, 0
    for j, i in enumerate(order):
        if j and angles[i] - angles[order[group_start]] > 14:
            group_start, group_pos = j, 0
        label_mult[i] = 1.14 + 0.30 * group_pos
        group_pos += 1

    # manual nudges in data units for labels that collide with other elements
    # (tuned for the PC1/PC2 panel only)
    label_shift = {"mi_decay_gamma": (0.0, -0.45)} if (pcx, pcy) == ("PC1", "PC2") else {}

    for k, (metric, (vx, vy)) in enumerate(zip(loadings.index, vecs)):
        family = "statistical" if metric in STAT_METRICS else "informational"
        col = FAMILY_COLORS[family]
        x, y = vx * scale, vy * scale
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=2.6, mutation_scale=24), zorder=8)
        dx, dy = label_shift.get(metric, (0.0, 0.0))
        ax.text(x * label_mult[k] + dx, y * label_mult[k] + dy,
                METRIC_NAMES[metric],
                color="white", fontsize=fontsize, fontweight="bold",
                ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.28", facecolor=col,
                          edgecolor="none", alpha=0.92))


def plot_combined(natural: pd.DataFrame, generated: pd.DataFrame,
                  size_bins: list[dict], bin_color: dict[str, str],
                  evr: np.ndarray, loadings: pd.DataFrame,
                  figsize: tuple[float, float],
                  norm_note: str = "", pcx: str = "PC1", pcy: str = "PC2") -> plt.Figure:
    """Overview panel (corpus density + loadings + bin means) with one
    score panel per size bin, mirroring the genre-grid combined figure."""
    ix, iy = int(pcx[2:]) - 1, int(pcy[2:]) - 1
    labels = [b["label"] for b in size_bins if b["label"] in set(natural["size_bin"])]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.24, wspace=0.18,
                          left=0.06, right=0.985, top=0.915, bottom=0.05)
    ax_a = fig.add_subplot(gs[0:2, 0:2])
    small_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]),
                  fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                  fig.add_subplot(gs[2, 2])]

    x_lim, y_lim = robust_limits(natural, pcx, pcy, generated=generated)
    means = (natural.groupby("size_bin")[[pcx, pcy]].mean()
             .reindex(labels).dropna())

    # ── panel A: corpus scores + density, size-bin means, loadings ──────────
    ax_a.set_xlim(*x_lim)
    ax_a.set_ylim(*y_lim)
    add_corpus_density(ax_a, natural[[pcx, pcy]].to_numpy())
    ax_a.scatter(natural[pcx], natural[pcy], s=5, c="#546E7A",
                 alpha=0.25, linewidths=0, zorder=3)
    ax_a.plot(means[pcx], means[pcy], "-", color=NAVY, lw=2.0, zorder=6)
    for label, row in means.iterrows():
        ax_a.scatter(row[pcx], row[pcy], marker="*", s=700,
                     c=bin_color[label], edgecolors=NAVY, linewidths=1.6,
                     zorder=7)
    add_loading_arrows(ax_a, loadings, pcx=pcx, pcy=pcy)
    ax_a.set_title("(A)  Natural-book density, descriptor loadings, "
                   "and size-bin mean scores", fontsize=17, pad=10)
    ax_a.set_xlabel(f"{pcx}  ({evr[ix] * 100:.0f}% var.)", fontsize=19)
    ax_a.set_ylabel(f"{pcy}  ({evr[iy] * 100:.0f}% var.)", fontsize=19)
    ax_a.tick_params(labelsize=15)

    main_sub = generated[generated["group"] == "main"]
    n_main_runs = main_sub["file"].nunique() if "file" in main_sub.columns else 0
    star_handles = [
        Line2D([], [], marker="*", linestyle="", markersize=17,
               markerfacecolor=bin_color[label], markeredgecolor=NAVY,
               label=f"mean, {label}")
        for label in means.index
    ]
    family_handles = [
        Line2D([], [], linestyle="-", color=FAMILY_COLORS["statistical"],
               linewidth=3.0, label="statistical loadings"),
        Line2D([], [], linestyle="-", color=FAMILY_COLORS["informational"],
               linewidth=3.0, label="informational loadings"),
    ]
    gen_handles = [
        Line2D([], [], marker="D", linestyle="", markersize=10,
               markerfacecolor="white", markeredgecolor="black",
               label="generated prefix, panels (B)–(F)\n"
                     f"(hierarchical summaries mode, {n_main_runs} runs)"),
        Line2D([], [], linestyle="-", color="#555555", linewidth=3.6,
               label="natural bin 50% contour, panels (B)–(F)"),
        Line2D([], [], linestyle="-", color="#555555", linewidth=2.2,
               alpha=0.45, label="natural bin 95% contour, panels (B)–(F)"),
    ]
    ax_a.legend(handles=star_handles + family_handles + gen_handles,
                fontsize=14, loc="upper left", framealpha=0.92).set_zorder(11)

    # ── panels B–F: one score panel per size bin ────────────────────────────
    for i, (ax, label) in enumerate(zip(small_axes, labels)):
        draw_bin_panel(ax, natural, generated, label, bin_color[label],
                       means.loc[label], x_lim, y_lim,
                       title_prefix=f"({chr(ord('B') + i)})  ",
                       title_fs=17, tick_fs=13, marker_scale=0.75,
                       pcx=pcx, pcy=pcy)

    metric_str = "  ".join(METRIC_TEX[m] for m in LEGACY_METRICS)
    title = (f"Natural-book PCA ({metric_str}) — {pcx}×{pcy}, "
             "generated prefixes per size bin")
    if norm_note:
        title += f"   [{norm_note}]"
    fig.suptitle(title, fontsize=21)
    return fig


def plot_bin_distributions(natural: pd.DataFrame, generated: pd.DataFrame,
                           size_bins: list[dict], bin_color: dict[str, str],
                           evr: np.ndarray) -> plt.Figure:
    """Per-bin natural violins for PC1/PC2 with generated prefixes overlaid."""
    labels = [b["label"] for b in size_bins if b["label"] in set(natural["size_bin"])]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 13.5), sharex=True)
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        data = [natural.loc[natural["size_bin"] == lab, pc].to_numpy() for lab in labels]
        # raincloud layout: natural density as a left half-violin,
        # generated prefixes jittered on the right side only
        parts = ax.violinplot(data, positions=range(len(labels)), widths=1.10,
                              showextrema=False)
        for i, (body, lab) in enumerate(zip(parts["bodies"], labels)):
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.clip(verts[:, 0], -np.inf, i)
            body.set_facecolor(bin_color[lab])
            body.set_edgecolor(darken(bin_color[lab]))
            body.set_linewidth(1.4)
            body.set_alpha(0.85)
        for i, vals in enumerate(data):
            lo_o, lo_i, hi_i, hi_o = np.percentile(vals, [2.5, 25, 75, 97.5])
            col = darken(bin_color[labels[i]])
            ax.plot([i, i], [lo_o, hi_o], color=col, lw=2.0, alpha=0.35, zorder=3)
            ax.plot([i, i], [lo_i, hi_i], color=col, lw=5.5, alpha=0.95,
                    zorder=4, solid_capstyle="butt")
            ax.plot(i, vals.mean(), marker="*", ms=22, mec=NAVY, mew=1.4,
                    mfc=bin_color[labels[i]], zorder=6)
        for group, marker, size in [("main", "D", 70), ("ablation", "^", 95)]:
            sub = generated[generated["group"] == group]
            for i, lab in enumerate(labels):
                s2 = sub[sub["size_bin"] == lab]
                if not len(s2):
                    continue
                jitter = rng.uniform(0.08, 0.42, len(s2))
                ax.scatter(i + jitter, s2[pc], marker=marker, s=size,
                           c=bin_color[lab], edgecolors="black",
                           linewidths=1.0, alpha=0.95, zorder=7)
        all_nat = natural[pc].to_numpy()
        y_lo, y_hi = np.percentile(all_nat, [0.5, 99.5])
        y_lo = min(y_lo, generated[pc].min())
        y_hi = max(y_hi, generated[pc].max())
        pad = 0.08 * (y_hi - y_lo)
        ax.set_ylim(y_lo - pad, y_hi + pad)
        idx = int(pc[2:]) - 1
        ax.set_ylabel(f"{pc}  ({evr[idx] * 100:.0f}% var.)", fontsize=26)
        ax.tick_params(labelsize=20)
        ax.grid(axis="y", color="#CFD8DC", lw=0.8, alpha=0.6)

    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=22)
    axes[1].set_xlabel("token size bin", fontsize=26)
    axes[0].set_title("Natural score distributions per size bin\nvs generated prefixes",
                      fontsize=27, pad=12)

    handles = [
        Line2D([], [], marker="D", linestyle="", markersize=11,
               markerfacecolor="white", markeredgecolor="black",
               label="generated prefix"),
        Line2D([], [], marker="*", linestyle="", markersize=19,
               markerfacecolor="white", markeredgecolor=NAVY,
               label="natural mean"),
        Line2D([], [], linestyle="-", color="#555555", linewidth=5.5,
               label="natural 25–75%"),
        Line2D([], [], linestyle="-", color="#555555", linewidth=2.0,
               alpha=0.35, label="natural 2.5–97.5%"),
    ]
    axes[0].legend(handles=handles, fontsize=17, loc="upper right", framealpha=0.92)

    fig.tight_layout()
    return fig


def fit_per_bin_stats(natural: pd.DataFrame, metrics: list[str],
                      iqr_epsilon: float = 1e-6) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Per-size-bin robust center (median) and scale (IQR = q75-q25) of each
    metric, matching the per-bin robust z-scoring in pca_sc_metrics.py.

    Median centering is used deliberately rather than the mean: subtracting the
    mean would force every bin's mean score to exactly zero and erase the
    residual between-bin (length) trajectory, whereas the median leaves the
    realistic residual that skewed descriptor distributions carry. The IQR
    falls back to the std when it is degenerate.
    """
    stats = {}
    for label, g in natural.groupby("size_bin"):
        median = g[metrics].median()
        iqr = g[metrics].quantile(0.75) - g[metrics].quantile(0.25)
        std = g[metrics].std(ddof=0)
        scale = iqr.where(np.isfinite(iqr) & (iqr.abs() > iqr_epsilon), std)
        scale = scale.where(np.isfinite(scale) & (scale.abs() > iqr_epsilon), 1.0)
        stats[label] = (median, scale)
    return stats


def standardize_per_bin(df: pd.DataFrame, metrics: list[str],
                        stats: dict[str, tuple[pd.Series, pd.Series]]) -> np.ndarray:
    """Robust-z each row's metrics using the natural median/IQR of its size bin.

    This rescales away most of the size-driven offsets so the PCA emphasises
    within-bin descriptor structure, while the median centering keeps the small
    residual length trajectory; generated prefixes are standardised against the
    natural stats of their matching bin.
    """
    parts = []
    for label, g in df.groupby("size_bin"):
        center, scale = stats[label]
        parts.append((g[metrics] - center) / scale)
    z = pd.concat(parts).reindex(df.index)
    return z[metrics].to_numpy()


def plot_length_combined(natural: pd.DataFrame, generated: pd.DataFrame,
                         loadings: pd.DataFrame, evr: np.ndarray,
                         pcx: str, pcy: str, figsize: tuple[float, float],
                         q_labels: list[str], q_colors: list) -> plt.Figure:
    """Combined-figure layout, but coloured by log(n_tokens) instead of size bin.

    Panel A: corpus coloured by length + descriptor loadings + the length axis
    (red arrow = correlation of log N with this PC pair). Panels B–F: one per
    equal-count length quintile (no arbitrary token edges)."""
    ix, iy = int(pcx[2:]) - 1, int(pcy[2:]) - 1
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.24, wspace=0.18,
                          left=0.06, right=0.985, top=0.915, bottom=0.05)
    ax_a = fig.add_subplot(gs[0:2, 0:2])
    small_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]),
                  fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                  fig.add_subplot(gs[2, 2])]

    x_lim, y_lim = robust_limits(natural, pcx, pcy, generated=generated)
    means = (natural.groupby("length_q")[[pcx, pcy]].mean()
             .reindex(range(5)).dropna())

    # ── panel A ──────────────────────────────────────────────────────────────
    ax_a.set_xlim(*x_lim)
    ax_a.set_ylim(*y_lim)
    add_corpus_density(ax_a, natural[[pcx, pcy]].to_numpy())
    sc = ax_a.scatter(natural[pcx], natural[pcy], c=natural["length_pct"],
                      cmap="viridis", vmin=0, vmax=100, s=6, alpha=0.55,
                      linewidths=0, zorder=3)
    ax_a.plot(means[pcx], means[pcy], "-", color=NAVY, lw=2.0, zorder=6)
    for q, row in means.iterrows():
        ax_a.scatter(row[pcx], row[pcy], marker="*", s=700, c=[q_colors[q]],
                     edgecolors=NAVY, linewidths=1.6, zorder=7)
    add_loading_arrows(ax_a, loadings, pcx=pcx, pcy=pcy)

    # red length axis: correlation of log N with each PC, scaled to the panel
    cx = float(np.corrcoef(natural[pcx], natural["log10_n"])[0, 1])
    cy = float(np.corrcoef(natural[pcy], natural["log10_n"])[0, 1])
    half = 0.42 * min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) / 2
    ax_a.annotate("", xy=(cx * half, cy * half), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="red", lw=3.0,
                                  mutation_scale=26), zorder=9)
    ax_a.text(cx * half * 1.08, cy * half * 1.08,
              f"log N  (r=({cx:+.2f},{cy:+.2f}))", color="red", fontsize=13,
              fontweight="bold", ha="center", va="center", zorder=10,
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.9))
    cb = fig.colorbar(sc, ax=ax_a, shrink=0.7, pad=0.01)
    cb.set_label("length percentile (rank of n_tokens)", fontsize=12)
    ax_a.set_title("(A)  Corpus coloured by length · descriptor loadings · "
                   "length axis (red)", fontsize=15, pad=10)
    ax_a.set_xlabel(f"{pcx}  ({evr[ix] * 100:.0f}% var.)", fontsize=18)
    ax_a.set_ylabel(f"{pcy}  ({evr[iy] * 100:.0f}% var.)", fontsize=18)
    ax_a.tick_params(labelsize=14)

    # ── panels B–F: one per length quintile ──────────────────────────────────
    for i, (ax, q) in enumerate(zip(small_axes, range(5))):
        sub = natural[natural["length_q"] == q]
        gen = generated[generated["length_q"] == q]
        ax.scatter(natural[pcx], natural[pcy], s=5, c="#90A4AE",
                   alpha=0.28, linewidths=0, zorder=1)
        ax.scatter(sub[pcx], sub[pcy], s=10, c=[q_colors[q]],
                   alpha=0.7, linewidths=0, zorder=2)
        if len(sub) >= 20:
            add_bin_density_contours(ax, sub[[pcx, pcy]].to_numpy(), q_colors[q])
        m = sub[[pcx, pcy]].mean()
        ax.scatter(m[pcx], m[pcy], marker="*", s=420, c=[q_colors[q]],
                   edgecolors=NAVY, linewidths=1.4, zorder=6)
        if len(gen):
            ax.scatter(gen[pcx], gen[pcy], marker="D", s=60, c=[q_colors[q]],
                       edgecolors="black", linewidths=1.0, alpha=0.95, zorder=7)
        ax.set_title(f"({chr(ord('B') + i)})  {q_labels[q]}  "
                     f"({len(sub):,} nat · {len(gen)} gen)",
                     fontsize=13, pad=6, color=darken(q_colors[q], 0.55),
                     fontweight="bold")
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.tick_params(labelsize=11)

    metric_str = "  ".join(METRIC_TEX[m] for m in LEGACY_METRICS)
    fig.suptitle(f"Natural-book PCA ({metric_str}) — {pcx}×{pcy}, "
                 "length gradient (global normalization, equal-count quintiles)",
                 fontsize=20)
    return fig


def make_length_gradient(natural: pd.DataFrame, args, size_bins: list[dict]) -> None:
    """Fit the global PCA and emit the length-gradient combined figures."""
    natural = natural.copy()
    scaler = StandardScaler().fit(natural[LEGACY_METRICS].to_numpy())
    pca = PCA(n_components=4).fit(scaler.transform(natural[LEGACY_METRICS].to_numpy()))
    natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
        scaler.transform(natural[LEGACY_METRICS].to_numpy()))
    natural["log10_n"] = np.log10(natural["n_tokens"].astype(float))
    natural["length_pct"] = natural["n_tokens"].rank(pct=True) * 100.0
    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=LEGACY_METRICS,
                            columns=["PC1", "PC2", "PC3", "PC4"])

    generated = load_generated_main(args.generated_main)
    generated = generated.dropna(subset=LEGACY_METRICS + ["n_tokens"]).copy()
    generated[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
        scaler.transform(generated[LEGACY_METRICS].to_numpy()))
    generated["log10_n"] = np.log10(generated["n_tokens"].astype(float))

    # equal-count length quintiles (no arbitrary token edges)
    edges = np.quantile(natural["n_tokens"].to_numpy(), np.linspace(0, 1, 6))
    natural["length_q"] = np.clip(np.digitize(natural["n_tokens"], edges[1:-1]), 0, 4)
    generated["length_q"] = np.clip(np.digitize(generated["n_tokens"], edges[1:-1]), 0, 4)
    q_labels = [f"{format_token_label(int(edges[i]))}–{format_token_label(int(edges[i + 1]))}"
                for i in range(5)]
    # quintile colours at their percentile centres (10/30/50/70/90) so the
    # panel stars line up with the percentile colourbar
    q_colors = [plt.cm.viridis(p) for p in (0.10, 0.30, 0.50, 0.70, 0.90)]

    for pcx, pcy, sfx in [("PC1", "PC2", "pc1_pc2"), ("PC3", "PC4", "pc3_pc4")]:
        fig = plot_length_combined(natural, generated, loadings, evr, pcx, pcy,
                                   (16.5, 15.5), q_labels, q_colors)
        name = f"legacy_pca_length_gradient_{sfx}.png"
        fig.savefig(args.output_dir / name, dpi=200)
        print(f"figure  → {args.output_dir / name}")
        if args.poster_dir:
            fig.savefig(args.poster_dir / name, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-results", type=Path, default=DEFAULT_BATCH_RESULTS)
    parser.add_argument("--generated-main", type=Path, default=DEFAULT_GENERATED_MAIN)
    parser.add_argument("--generated-ablation", type=Path, default=DEFAULT_GENERATED_ABLATION)
    parser.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poster-dir", type=Path, default=DEFAULT_POSTER_DIR,
                        help="Directory receiving poster copies of the figures.")
    parser.add_argument("--include-ablation", action="store_true",
                        help="Also project the ablation-mode prefixes (excluded by "
                             "default: their full cohort data is on another machine).")
    parser.add_argument("--per-bin-normalization", action="store_true",
                        help="Robust-z each metric within its size bin (natural "
                             "per-bin median / IQR, matching pca_sc_metrics.py) before "
                             "the PCA, instead of a single global StandardScaler. "
                             "Outputs are written with a '_per_bin' suffix so both "
                             "variants can coexist.")
    parser.add_argument("--length-gradient", action="store_true",
                        help="Emit the combined-figure-style length-gradient plots "
                             "(global PCA, corpus coloured by log n_tokens, red length "
                             "axis, equal-count length quintiles) for PC1×PC2 and "
                             "PC3×PC4, then exit.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.poster_dir:
        args.poster_dir.mkdir(parents=True, exist_ok=True)
    size_bins = load_size_bins(args.size_stats)
    palette = size_bin_palette(len(size_bins))
    bin_color = {b["label"]: palette[i] for i, b in enumerate(size_bins)}

    natural = load_natural(args.batch_results)
    natural["size_bin"] = [assign_bin(t, size_bins) for t in natural["n_tokens"]]
    natural = natural.dropna(subset=["size_bin"])
    print(f"natural books: {len(natural)}")

    if args.length_gradient:
        make_length_gradient(natural, args, size_bins)
        return

    if args.per_bin_normalization:
        bin_stats = fit_per_bin_stats(natural, LEGACY_METRICS)
        nat_z = standardize_per_bin(natural, LEGACY_METRICS, bin_stats)
        pca = PCA(n_components=4).fit(nat_z)
        natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(nat_z)
    else:
        scaler = StandardScaler().fit(natural[LEGACY_METRICS].to_numpy())
        pca = PCA(n_components=4).fit(scaler.transform(natural[LEGACY_METRICS].to_numpy()))
        natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            scaler.transform(natural[LEGACY_METRICS].to_numpy()))
    evr = pca.explained_variance_ratio_

    generated = load_generated_main(args.generated_main)
    if args.include_ablation:
        ablation = load_generated_ablation(args.generated_ablation)
        if len(ablation):
            generated = pd.concat([generated, ablation], ignore_index=True)
    generated["size_bin"] = [assign_bin(t, size_bins, lower_edge_tolerance=0.10)
                             for t in generated["n_tokens"]]
    generated = generated.dropna(subset=["size_bin"])
    if args.per_bin_normalization:
        gen_scores = pca.transform(standardize_per_bin(generated, LEGACY_METRICS, bin_stats))
    else:
        gen_scores = pca.transform(scaler.transform(generated[LEGACY_METRICS].to_numpy()))
    generated[["PC1", "PC2", "PC3", "PC4"]] = gen_scores
    print(f"generated prefixes: {len(generated)} "
          f"({(generated['group'] == 'main').sum()} main, "
          f"{(generated['group'] == 'ablation').sum()} ablation)")

    loadings = pd.DataFrame(pca.components_.T,
                            index=LEGACY_METRICS,
                            columns=["PC1", "PC2", "PC3", "PC4"])

    # '_per_bin' suffix keeps the per-size-bin-normalised variant alongside the
    # global-StandardScaler one rather than overwriting it.
    suffix = "_per_bin" if args.per_bin_normalization else ""
    norm_note = "per-size-bin normalized" if args.per_bin_normalization else ""

    # plot_scores, plot_scores_per_bin, and plot_bin_distributions are kept as
    # standalone variants but no longer produced; the combined figure covers them
    figures = {
        f"legacy_pca_size_bins_combined{suffix}.png":
            plot_combined(natural, generated, size_bins, bin_color, evr,
                          loadings, (16.5, 15.5), norm_note=norm_note,
                          pcx="PC1", pcy="PC2"),
        f"legacy_pca_size_bins_combined{suffix}_pc3_pc4.png":
            plot_combined(natural, generated, size_bins, bin_color, evr,
                          loadings, (16.5, 15.5), norm_note=norm_note,
                          pcx="PC3", pcy="PC4"),
    }
    for name, fig in figures.items():
        path = args.output_dir / name
        fig.savefig(path, dpi=200)
        print(f"figure  → {path}")
        if args.poster_dir:
            fig.savefig(args.poster_dir / name, dpi=200)
        plt.close(fig)

    keep = ["file", "title", "group", "mode_label", "size_bin", "n_tokens",
            *LEGACY_METRICS, "PC1", "PC2", "PC3", "PC4"]
    generated[[c for c in keep if c in generated.columns]].to_csv(
        args.output_dir / f"generated_prefix_scores{suffix}.csv", index=False)

    loadings.to_csv(args.output_dir / f"natural_pca_loadings{suffix}.csv")

    summary = {
        "n_natural_books": int(len(natural)),
        "n_generated_prefixes": int(len(generated)),
        "metrics": LEGACY_METRICS,
        "normalization": "per_bin" if args.per_bin_normalization else "global",
        "explained_variance_ratio": [float(v) for v in evr],
        "size_bins": [{"label": b["label"],
                       "n_natural": int((natural["size_bin"] == b["label"]).sum()),
                       "n_generated": int((generated["size_bin"] == b["label"]).sum())}
                      for b in size_bins],
    }
    (args.output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
