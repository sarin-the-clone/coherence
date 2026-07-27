#!/usr/bin/env python3
"""
PCA of the canonical descriptor set (8 stat/info + 27 token-based) on the
natural-book corpus, colour-coded by token size bin, with generated-book
prefixes projected onto the natural score plane — the all-metrics analogue of
pca_legacy_size_bins.py.

Data sources (the legacy experimental_metrics.json has no token metrics):
  - natural stat/info 8 : natural_texts/Output/batch_results/batch_*.json
  - natural token 27    : gutenberg_benchmark_cache/token_naturalness_metrics.jsonl
                        gutenberg_benchmark_cache/token_predictability_metrics.jsonl
  - generated (46 SC runs x 5 size bins, all 35 non-SC features):
                        natural_texts/Output/pca/sc_combined/
                          sc_combined_pca_generated_scores.csv

Outputs (both PC pairs; '_per_bin' suffix when --per-bin-normalization):
  natural_texts/Output/pca/allmetrics_size_bins/
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

# reuse the metric-agnostic rendering helpers from the legacy size-bin script
from pca_legacy_size_bins import (  # noqa: E402
    NAVY, add_bin_density_contours, add_corpus_density, assign_bin, darken,
    fit_per_bin_stats, format_token_label, load_size_bins, robust_limits,
    size_bin_palette, standardize_per_bin,
)
from analysis.descriptor_schema import (  # noqa: E402
    STAT_INFO_METRICS,
    TOKEN_METRICS,
    available_features,
)

LEGACY_METRICS = list(STAT_INFO_METRICS)

DEFAULT_BATCH_RESULTS = ROOT / "natural_texts" / "Output" / "batch_results"
DEFAULT_TOKEN_NAT = (ROOT / "natural_texts" / "Output" / "gutenberg_benchmark_cache"
                     / "token_naturalness_metrics.jsonl")
DEFAULT_TOKEN_PRED = (ROOT / "natural_texts" / "Output" / "gutenberg_benchmark_cache"
                      / "token_predictability_metrics.jsonl")
DEFAULT_GENERATED = (ROOT / "natural_texts" / "Output" / "pca" / "sc_combined"
                     / "sc_combined_pca_generated_scores.csv")
DEFAULT_SIZE_STATS = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"
DEFAULT_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "allmetrics_size_bins"
DEFAULT_POSTER_DIR = ROOT / "ppt"

FAMILY_COLORS = {"sc": "#9B2226", "legacy": "#0D2545", "td": "#00796B",
                 "tr": "#E8710A", "tp": "#6A1B9A"}
FAMILY_LABEL = {"sc": "semantic coherence", "legacy": "stat/info (legacy)",
                "td": "token distribution", "tr": "token recurrence",
                "tp": "token predictability"}

# generated prefixes are coloured by conditioning (ablation) mode — same scheme
# as pca_sc_metrics.py (the sc_pca biplots)
MODE_COLORS = {
    "summaries_conditioning":         "#1f77b4",
    "extended_conditioning":          "#d62728",
    "extended_no_fixed_conditioning": "#00897B",
    "basic_conditioning":             "#2ca02c",
    "minimal_fixed_conditioning":     "#9467bd",
    "markov_rolling_conditioning":    "#8c564b",
    "minimal_rolling_conditioning":   "#8c564b",
}
MODE_LABELS = {
    "summaries_conditioning":         "summaries",
    "extended_conditioning":          "extended/full",
    "extended_no_fixed_conditioning": "extended no fixed",
    "basic_conditioning":             "basic",
    "minimal_fixed_conditioning":     "minimal fixed",
    "markov_rolling_conditioning":    "markov rolling",
}
MODE_ORDER = ["summaries_conditioning", "extended_conditioning",
              "extended_no_fixed_conditioning", "basic_conditioning",
              "minimal_fixed_conditioning", "markov_rolling_conditioning"]


def metric_family(m: str) -> str:
    if m.startswith(("W1k", "W2k", "W8k", "W40k")):
        return "sc"
    if m in LEGACY_METRICS:
        return "legacy"
    return m[:2]  # 'td', 'tr', 'tp'


def short_label(m: str) -> str:
    return m[3:] if m[:3] in ("td_", "tr_", "tp_") else m


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in open(path)])


def load_natural(batch_dir: Path, token_nat: Path, token_pred: Path) -> tuple[pd.DataFrame, list[str]]:
    recs = []
    for p in sorted(glob.glob(str(batch_dir / "batch_*.json"))):
        recs.extend(json.loads(Path(p).read_text()))
    bdf = pd.DataFrame(recs)
    bdf["book_id"] = bdf["book_id"].astype(str)

    tn = load_jsonl(token_nat); tn["book_id"] = tn["book_id"].astype(str)
    tp = load_jsonl(token_pred); tp["book_id"] = tp["book_id"].astype(str)
    tn_cols = [c for c in TOKEN_METRICS if c in tn.columns]
    tp_cols = [c for c in TOKEN_METRICS if c in tp.columns]

    nat = (bdf[["book_id", "n_tokens", *LEGACY_METRICS]]
           .merge(tn[["book_id", *tn_cols]], on="book_id", how="inner")
           .merge(tp[["book_id", *tp_cols]], on="book_id", how="inner"))
    features = available_features(nat.columns, include_sc=False)
    nat = nat.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["n_tokens"])
    return nat, features


def load_generated(path: Path, features: list[str]) -> pd.DataFrame:
    g = pd.read_csv(path)
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    g["group"] = "main"
    return g


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_bin_panel(ax, natural, generated, label, color, mean_row, x_lim, y_lim,
                   title_prefix="", title_fs=17, tick_fs=13, marker_scale=0.75,
                   pcx="PC1", pcy="PC2"):
    sub = natural[natural["size_bin"] == label]
    gen = generated[generated["size_bin"] == label]
    ax.scatter(natural[pcx], natural[pcy], s=5 * marker_scale, c="#90A4AE",
               alpha=0.25, linewidths=0, zorder=1)
    ax.scatter(sub[pcx], sub[pcy], s=11 * marker_scale, c=[color],
               alpha=0.6, linewidths=0, zorder=2)
    if len(sub) >= 20:
        add_bin_density_contours(ax, sub[[pcx, pcy]].to_numpy(), color)
    ax.scatter(mean_row[pcx], mean_row[pcy], marker="*", s=520 * marker_scale,
               c=[color], edgecolors=NAVY, linewidths=1.4, zorder=6)
    if len(gen):
        gc = [MODE_COLORS.get(mode, "#000000") for mode in gen["conditioning_mode"]]
        ax.scatter(gen[pcx], gen[pcy], marker="D", s=70 * marker_scale, c=gc,
                   edgecolors="black", linewidths=0.8, alpha=0.95, zorder=7)
    ax.set_title(f"{title_prefix}{label}   ({len(sub):,} nat · {len(gen)} gen)",
                 fontsize=title_fs, pad=6, color=darken(color, 0.55), fontweight="bold")
    ax.tick_params(labelsize=tick_fs)
    ax.set_xlim(*x_lim); ax.set_ylim(*y_lim)


def add_topk_loading_arrows(ax, loadings, pcx, pcy, k=12, fontsize=11):
    mag = np.sqrt(loadings[pcx] ** 2 + loadings[pcy] ** 2)
    top = mag.sort_values(ascending=False).index[:k]
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    max_norm = mag.loc[top].max()
    scale = 0.42 * min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) / 2 / max_norm
    for m in top:
        vx, vy = loadings.loc[m, pcx] * scale, loadings.loc[m, pcy] * scale
        col = FAMILY_COLORS[metric_family(m)]
        ax.annotate("", xy=(vx, vy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                    mutation_scale=16, alpha=0.9), zorder=8)
        ax.text(vx * 1.12, vy * 1.12, short_label(m), color="white", fontsize=fontsize,
                fontweight="bold", ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=col, edgecolor="none",
                          alpha=0.9))


def display_limits(natural, generated, pcx, pcy, lower_q=2.0, upper_q=98.0, pad=0.08):
    """Robust display limits: the 2-98% percentile range of the combined
    natural+generated scores, so the heavy PC tails and the few extreme
    generated points are ignored (not allowed to stretch the axis)."""
    if len(generated):
        xs = np.concatenate([natural[pcx].to_numpy(), generated[pcx].to_numpy()])
        ys = np.concatenate([natural[pcy].to_numpy(), generated[pcy].to_numpy()])
    else:
        xs, ys = natural[pcx].to_numpy(), natural[pcy].to_numpy()
    x_lo, x_hi = np.percentile(xs, [lower_q, upper_q])
    y_lo, y_hi = np.percentile(ys, [lower_q, upper_q])
    px, py = (x_hi - x_lo) * pad, (y_hi - y_lo) * pad
    return (x_lo - px, x_hi + px), (y_lo - py, y_hi + py)


def plot_combined(natural, generated, size_bins, bin_color, evr, loadings, figsize,
                  pcx="PC1", pcy="PC2", norm_note="", topk=12,
                  feature_desc="8 stat/info + 27 token"):
    ix, iy = int(pcx[2:]) - 1, int(pcy[2:]) - 1
    labels = [b["label"] for b in size_bins if b["label"] in set(natural["size_bin"])]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.24, wspace=0.18,
                          left=0.06, right=0.985, top=0.915, bottom=0.05)
    ax_a = fig.add_subplot(gs[0:2, 0:2])
    small_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]),
                  fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
                  fig.add_subplot(gs[2, 2])]
    x_lim, y_lim = display_limits(natural, generated, pcx, pcy)
    means = natural.groupby("size_bin")[[pcx, pcy]].mean().reindex(labels).dropna()

    ax_a.set_xlim(*x_lim); ax_a.set_ylim(*y_lim)
    add_corpus_density(ax_a, natural[[pcx, pcy]].to_numpy())
    ax_a.scatter(natural[pcx], natural[pcy], s=5, c="#546E7A", alpha=0.22,
                 linewidths=0, zorder=3)
    ax_a.plot(means[pcx], means[pcy], "-", color=NAVY, lw=2.0, zorder=6)
    for label, row in means.iterrows():
        ax_a.scatter(row[pcx], row[pcy], marker="*", s=700, c=[bin_color[label]],
                     edgecolors=NAVY, linewidths=1.6, zorder=7)
    add_topk_loading_arrows(ax_a, loadings, pcx, pcy, k=topk)
    ax_a.set_title(f"(A)  Natural density, size-bin means, top-{topk} loadings",
                   fontsize=16, pad=10)
    ax_a.set_xlabel(f"{pcx}  ({evr[ix] * 100:.0f}% var.)", fontsize=18)
    ax_a.set_ylabel(f"{pcy}  ({evr[iy] * 100:.0f}% var.)", fontsize=18)
    ax_a.tick_params(labelsize=14)

    star_handles = [Line2D([], [], marker="*", linestyle="", markersize=15,
                           markerfacecolor=bin_color[lab], markeredgecolor=NAVY,
                           label=f"mean, {lab}") for lab in means.index]
    present_fams = [f for f in ("sc", "legacy", "td", "tr", "tp")
                    if any(metric_family(m) == f for m in loadings.index)]
    fam_handles = [Line2D([], [], linestyle="-", color=FAMILY_COLORS[f], lw=3,
                          label=FAMILY_LABEL[f]) for f in present_fams]
    present_modes = [m for m in MODE_ORDER if m in set(generated["conditioning_mode"])]
    gen_handles = [Line2D([], [], marker="D", linestyle="", markersize=9,
                          markerfacecolor=MODE_COLORS[m], markeredgecolor="black",
                          label=f"gen: {MODE_LABELS[m]}") for m in present_modes]
    ax_a.legend(handles=star_handles + fam_handles + gen_handles, fontsize=11,
                loc="upper left", framealpha=0.92, ncol=2).set_zorder(11)

    for i, (ax, label) in enumerate(zip(small_axes, labels)):
        draw_bin_panel(ax, natural, generated, label, bin_color[label],
                       means.loc[label], x_lim, y_lim,
                       title_prefix=f"({chr(ord('B') + i)})  ", pcx=pcx, pcy=pcy)

    title = (f"PCA ({feature_desc}) — {pcx}×{pcy}, "
             "generated prefixes per size bin")
    if norm_note:
        title += f"   [{norm_note}]"
    fig.suptitle(title, fontsize=20)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-results", type=Path, default=DEFAULT_BATCH_RESULTS)
    p.add_argument("--token-naturalness", type=Path, default=DEFAULT_TOKEN_NAT)
    p.add_argument("--token-predictability", type=Path, default=DEFAULT_TOKEN_PRED)
    p.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    p.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--poster-dir", type=Path, default=DEFAULT_POSTER_DIR)
    p.add_argument("--feature-set", choices=["all", "legacy"], default="all",
                   help="'all' = 8 stat/info + 27 token; 'legacy' = the 8 legacy "
                        "descriptors only (same SC-run cohort, mode-coloured).")
    p.add_argument("--per-bin-normalization", action="store_true",
                   help="Robust-z each metric within its size bin (median/IQR) before "
                        "the PCA, instead of a global StandardScaler.")
    args = p.parse_args()

    if args.output_dir is None:
        sub = "allmetrics_size_bins" if args.feature_set == "all" else "legacy8_size_bins"
        args.output_dir = ROOT / "natural_texts" / "Output" / "pca" / sub
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.poster_dir:
        args.poster_dir.mkdir(parents=True, exist_ok=True)

    size_bins = load_size_bins(args.size_stats)
    palette = size_bin_palette(len(size_bins))
    bin_color = {b["label"]: palette[i] for i, b in enumerate(size_bins)}

    natural, features = load_natural(args.batch_results, args.token_naturalness,
                                     args.token_predictability)
    if args.feature_set == "legacy":
        features = [f for f in features if f in LEGACY_METRICS]
        feat_desc, name_prefix = f"{len(features)} legacy descriptors", "legacy8"
    else:
        feat_desc, name_prefix = f"8 legacy + {len(features) - 8} token", "allmetrics"
    natural["size_bin"] = [assign_bin(t, size_bins) for t in natural["n_tokens"]]
    natural = natural.dropna(subset=["size_bin"]).reset_index(drop=True)
    print(f"natural books: {len(natural)} | feature set '{args.feature_set}': "
          f"{len(features)} features")

    generated = load_generated(args.generated, features)
    generated["size_bin"] = [assign_bin(t, size_bins, lower_edge_tolerance=0.10)
                             for t in generated["n_tokens"]]
    generated = generated.dropna(subset=["size_bin"]).reset_index(drop=True)
    print(f"generated prefixes: {len(generated)} "
          f"({generated['conditioning_mode'].nunique()} modes)")

    if args.per_bin_normalization:
        stats = fit_per_bin_stats(natural, features)
        Zn = standardize_per_bin(natural, features, stats)
        pca = PCA(n_components=4).fit(Zn)
        natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(Zn)
        generated[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            standardize_per_bin(generated, features, stats))
    else:
        scaler = StandardScaler().fit(natural[features].to_numpy())
        pca = PCA(n_components=4).fit(scaler.transform(natural[features].to_numpy()))
        natural[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            scaler.transform(natural[features].to_numpy()))
        generated[["PC1", "PC2", "PC3", "PC4"]] = pca.transform(
            scaler.transform(generated[features].to_numpy()))
    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, index=features,
                            columns=["PC1", "PC2", "PC3", "PC4"])

    suffix = "_per_bin" if args.per_bin_normalization else ""
    norm_note = "per-size-bin normalized" if args.per_bin_normalization else ""
    for pcx, pcy, sfx in [("PC1", "PC2", "pc1_pc2"), ("PC3", "PC4", "pc3_pc4")]:
        fig = plot_combined(natural, generated, size_bins, bin_color, evr, loadings,
                            (16.5, 15.5), pcx=pcx, pcy=pcy, norm_note=norm_note,
                            feature_desc=feat_desc)
        name = f"{name_prefix}_pca_size_bins_combined{suffix}_{sfx}.png"
        fig.savefig(args.output_dir / name, dpi=200)
        print(f"figure  → {args.output_dir / name}")
        if args.poster_dir:
            fig.savefig(args.poster_dir / name, dpi=200)
        plt.close(fig)

    loadings.to_csv(args.output_dir / f"natural_pca_loadings{suffix}.csv")
    keep = ["run_id", "conditioning_mode", "size_bin", "n_tokens", "PC1", "PC2", "PC3", "PC4"]
    generated[[c for c in keep if c in generated.columns]].to_csv(
        args.output_dir / f"generated_scores{suffix}.csv", index=False)
    summary = {
        "n_natural_books": int(len(natural)),
        "n_generated_prefixes": int(len(generated)),
        "n_features": len(features),
        "features": features,
        "normalization": "per_bin" if args.per_bin_normalization else "global",
        "explained_variance_ratio": [float(v) for v in evr],
    }
    (args.output_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
