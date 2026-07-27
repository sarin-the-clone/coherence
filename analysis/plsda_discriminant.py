#!/usr/bin/env python3
"""
PLS-DA discriminant analysis: natural vs generated long-form text, at matched
size (per-bin median/IQR normalization, so length is controlled).

Runs on the SC-analysed subcorpus so every feature set is compared on the same
books. Reports cross-validated AUC + balanced accuracy (group-aware CV, prefixes
of one run kept together), VIP variable importance, the matched-size direction
of each metric (generated higher/lower), per-ablation-mode separation, and a
PLS-DA score plot with the per-size-bin natural means drawn as a star trajectory.

Feature sets (--feature-set):
  combined    9 SC + 8 stat + 27 token   (default)
  sc          9 SC only
  token_stat  8 stat + 27 token

Outputs → natural_texts/Output/pca/discriminant/<feature_set>/
  report.txt, vip.csv, plsda_scores.png, plsda_biplot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text as MplText
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import pca_sc_natural_combined as scn  # noqa: E402
import pca_sc_combined_size_bins as scs  # noqa: E402
import pca_allmetrics_size_bins as ams  # noqa: E402
from descriptor_schema import (  # noqa: E402
    EXCLUDED_EXACT_ALIASES,
    EXCLUDED_PROVENANCE_DUPLICATES,
)
from pca_legacy_size_bins import (  # noqa: E402
    NAVY, add_bin_density_contours, assign_bin, fit_per_bin_stats, load_size_bins,
    size_bin_palette, standardize_per_bin,
)

DEFAULT_OUTPUT_ROOT = ROOT / "natural_texts" / "Output" / "pca" / "discriminant"
DEFAULT_SIZE_STATS = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"

# readable SC feature names: W40k_off_diag_mean -> W40k_global_similarity, etc.
SC_SUFFIX_DISPLAY = {
    "off_diag_mean": "global_similarity",
    "novelty_mean":  "novelty",
    "sim_lag1_mean": "local_flow",
}
SC_WINDOWS = ("W1k", "W2k", "W8k", "W40k")


def display_name(metric: str) -> str:
    for w in SC_WINDOWS:
        if metric.startswith(w + "_"):
            suffix = metric[len(w) + 1:]
            return f"{w}_{SC_SUFFIX_DISPLAY.get(suffix, suffix)}"
    return metric


def is_sc(metric: str) -> bool:
    return metric.startswith(SC_WINDOWS)


# fine-family bar colours: blue / orange / aqua / violet / magenta
FAM_COLOR = {"stat": "#2a78d6", "td": "#eb6834", "tr": "#1baf7a",
             "tp": "#4a3aa7", "sc": "#c51b7d"}
# darker same-hue shades for the tick labels (readable as small text, ≥4.8:1)
FAM_TEXT_COLOR = {"stat": "#1c5cab", "td": "#bf4e1e", "tr": "#0e7a54",
                  "tp": "#4a3aa7", "sc": "#8d1459"}
FAM_LABEL = {"sc": "semantic coherence", "stat": "scaling / informational",
             "td": "token distribution", "tr": "token recurrence",
             "tp": "token predictability"}

# descriptor families for the VIP grouping (top -> bottom on the bar axis):
# scaling/informational, then token statistics (its three sub-families), then
# semantic. Fine families td/tr/tp all roll up to the token-statistics family.
FINE_ORDER = ["stat", "td", "tr", "tp", "sc"]
MAJOR_OF = {"stat": "scaling / informational",
            "td": "token statistics", "tr": "token statistics",
            "tp": "token statistics", "sc": "semantic"}

# readable descriptor labels for the plot (the CSV/report keep the raw names)
SC_PRETTY = {"global_similarity": "Global similarity", "novelty": "Novelty",
             "local_flow": "Local flow"}
PRETTY = {
    # scaling / informational
    "lz_complexity": "LZ complexity", "perplexity": "Perplexity",
    "H_DFA1": "Hurst exponent (DFA)", "entropy_bits": "Entropy (bits/token)",
    "H_WT_db4": "Hurst exponent (wavelet)", "mi_decay_gamma": "Mutual-info decay γ",
    "alpha": "Spectral exponent α", "beta": "Fluctuation scaling β",
    # token distribution
    "td_heaps_exponent": "Heaps’ exponent", "td_vocab_size": "Vocabulary size",
    "td_zipf_exponent": "Zipf exponent", "td_zipf_slope": "Zipf slope",
    "td_type_token_ratio": "Type–token ratio",
    "td_hapax_token_fraction": "Hapax share (of tokens)",
    "td_top1_token_fraction": "Top-1 token share",
    "td_top10_token_fraction": "Top-10 token share",
    "td_hapax_type_fraction": "Hapax share (of types)",
    # token recurrence
    "tr_repeat_gap_median_log2": "Repeat-gap median (log₂)",
    "tr_repeat_gap_mean_log2": "Repeat-gap mean (log₂)",
    "tr_repeat_gap_burstiness": "Repeat-gap burstiness",
    "tr_repeat_rate": "Repeat rate",
    "tr_repeat_gap_cv": "Repeat-gap variability (CV)",
    "tr_repeat_gap_logbin_entropy": "Repeat-gap entropy",
    "tr_repeat_gap_mean": "Repeat-gap mean",
    # token predictability (n = conditioning order)
    "tp_redundancy_1": "Redundancy (order 1)", "tp_redundancy_2": "Redundancy (order 2)",
    "tp_redundancy_3": "Redundancy (order 3)", "tp_redundancy_4": "Redundancy (order 4)",
    "tp_entropy_0": "Entropy (order 0)",
    "tp_entropy_gain_1": "Entropy gain (order 1)", "tp_entropy_gain_2": "Entropy gain (order 2)",
    "tp_entropy_gain_3": "Entropy gain (order 3)", "tp_entropy_gain_4": "Entropy gain (order 4)",
    "tp_entropy_drop_4": "Entropy drop (order 4)", "tp_entropy_slope": "Entropy-decay slope",
    "tp_cond_entropy_1": "Conditional entropy (order 1)",
    "tp_cond_entropy_2": "Conditional entropy (order 2)",
    "tp_cond_entropy_3": "Conditional entropy (order 3)",
    "tp_cond_entropy_4": "Conditional entropy (order 4)",
    "tp_context_efficiency_4": "Context efficiency (order 4)",
}
# One loading-label exception: the Robinson Crusoe landmark lies directly over
# this long recurrence label in the combined LV1/LV2 biplot.
ARROW_LABEL_OFFSETS = {"Repeat-gap burstiness": (0.0, 0.45)}

def pretty_name(name: str) -> str:
    """Human-readable label for a (display-name) descriptor, for plot axes."""
    for w in SC_WINDOWS:
        if name.startswith(w + "_"):
            suffix = name[len(w) + 1:]
            return f"{SC_PRETTY.get(suffix, suffix.replace('_', ' '))} ({w[1:]} window)"
    return PRETTY.get(name, name.replace("_", " "))


def descriptor_label(metric: str) -> str:
    """Single display label for VIP bars and PLS loading arrows."""
    return pretty_name(display_name(metric))


def fine_family(metric: str) -> str:
    if is_sc(metric):
        return "sc"
    if metric in scn.STAT_INFO_METRICS:
        return "stat"
    return metric[:2]  # td / tr / tp


def plot_vip(table: pd.DataFrame, out_path: Path, feature_set: str, auc: float) -> None:
    """Signed VIP bar chart, grouped by descriptor family (scaling/informational,
    token statistics {distribution, recurrence, predictability}, semantic) and
    ordered by VIP within each family. Bar side = matched-size direction
    (right = generated higher, left = lower), colour = fine family."""
    present = [f for f in FINE_ORDER if f in set(table["fam"])]
    # top -> bottom: family blocks in FINE_ORDER, VIP-descending inside each
    ordered = pd.concat(
        [table[table["fam"] == f].sort_values("VIP", ascending=False) for f in present],
        ignore_index=True,
    )
    n = len(ordered)
    fam_tb = list(ordered["fam"])                    # fine family, top -> bottom
    # barh puts index 0 at the bottom, so reverse to keep the first block on top
    t = ordered.iloc[::-1].reset_index(drop=True)
    # bar length = VIP (non-negative); side = matched-size direction (right =
    # generated higher than natural, left = lower)
    signed = (t["VIP"] * np.sign(t["gen_offset_iqr"]).replace(0, 1)).to_numpy()
    colors = [FAM_COLOR[f] for f in t["fam"]]

    fig, ax = plt.subplots(figsize=(8.8, max(4.5, 0.205 * n)))
    ax.barh(range(n), signed, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty_name(m) for m in t["metric"]], fontsize=8)
    ax.set_ylim(-0.6, n - 0.4)
    # colour each tick label to match its bar; bold the discriminating VIP > 1 rows
    for lab, fam, vip in zip(ax.get_yticklabels(), t["fam"], t["VIP"]):
        lab.set_color(FAM_TEXT_COLOR[fam])
        if vip > 1:
            lab.set_fontweight("bold")

    # contiguous major-family blocks (top -> bottom): rule off, label on the right
    # (no per-family shading -- the bar colours already carry family identity)
    blocks, i = [], 0
    while i < n:
        maj = MAJOR_OF[fam_tb[i]]
        j = i
        while j < n and MAJOR_OF[fam_tb[j]] == maj:
            j += 1
        y_top, y_bot = n - 1 - i, n - 1 - (j - 1)    # bar indices bounding the block
        blocks.append((maj, y_top, y_bot))
        i = j
    TOKEN_MAJOR = "token statistics"
    for k, (maj, y_top, y_bot) in enumerate(blocks):
        if k > 0:                                    # rule between families
            ax.axhline(y_top + 0.5, color="#222", lw=1.3, zorder=2)
        # push the token-statistics label to an outer tier so its three
        # sub-family labels fit on the inner tier
        x_major = 1.058 if maj == TOKEN_MAJOR else 1.012
        ax.text(x_major, (y_top + y_bot) / 2, maj, transform=ax.get_yaxis_transform(),
                rotation=90, va="center", ha="center", fontsize=9.5, fontweight="bold",
                color="#3a3a3a")
    # sub-family labels within token statistics (distribution / recurrence /
    # predictability) on the inner tier
    fine_blocks, i = [], 0
    while i < n:
        f = fam_tb[i]; j = i
        while j < n and fam_tb[j] == f:
            j += 1
        fine_blocks.append((f, n - 1 - i, n - 1 - (j - 1)))
        i = j
    for f, y_top, y_bot in fine_blocks:
        if MAJOR_OF[f] == TOKEN_MAJOR:
            ax.text(1.012, (y_top + y_bot) / 2, FAM_LABEL[f].replace("token ", ""),
                    transform=ax.get_yaxis_transform(), rotation=90, va="center",
                    ha="center", fontsize=8, style="italic", color="#555555")
    # faint dotted rules between the token-statistics sub-families
    for i in range(1, n):
        if fam_tb[i] != fam_tb[i - 1] and MAJOR_OF[fam_tb[i]] == MAJOR_OF[fam_tb[i - 1]]:
            ax.axhline((n - 1 - i) + 0.5, color="#888", lw=0.7, ls=":", alpha=0.7, zorder=2)

    # shade the discriminating zones (VIP > 1, either direction) in green
    xabs = float(np.abs(signed).max()) * 1.07
    ax.set_xlim(-xabs, xabs)
    ax.axvspan(-xabs, -1, color="#2E7D32", alpha=0.09, zorder=0)
    ax.axvspan(1, xabs, color="#2E7D32", alpha=0.09, zorder=0)
    for x in (-1, 1):
        ax.axvline(x, color="#1B5E20", ls=(0, (5, 3)), lw=1.5, alpha=0.9, zorder=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("signed VIP\n($\\mathbf{VIP > 1}$ = discriminating)", fontsize=11)
    ax.set_title(f"PLS-DA variable importance — {feature_set} "
                 f"(in-corpus CV AUC {auc:.3f})", fontsize=12)
    fams = [f for f in FINE_ORDER if f in set(t["fam"])]
    ax.legend(handles=[plt.Line2D([], [], marker="s", linestyle="", markersize=6.5,
                                  markerfacecolor=FAM_COLOR[f], markeredgecolor="black",
                                  label=FAM_LABEL[f]) for f in fams],
              fontsize=6.8, loc="upper right", framealpha=0.9,
              borderpad=0.35, labelspacing=0.3, handletextpad=0.45)
    fig.tight_layout(rect=[0, 0, 0.92, 1])          # reserve right margin for the two label tiers
    fig.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def vip_scores(pls: PLSRegression) -> np.ndarray:
    t = pls.x_scores_                       # (n, A)
    w = pls.x_weights_                      # (p, A)
    q = pls.y_loadings_                     # (1, A)
    p, _ = w.shape
    ssy = (q[0] ** 2) * np.sum(t ** 2, axis=0)
    total = ssy.sum()
    wn = w / np.linalg.norm(w, axis=0)
    return np.sqrt(p * np.sum((wn ** 2) * ssy[None, :], axis=1) / total)


def aggregate_group_scores(y: np.ndarray, scores: np.ndarray,
                           groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one target and mean score per independent CV group.

    A generated group is a complete run with five nested prefixes. Averaging its
    scores prevents those correlated prefixes from counting as five independent
    validation cases. Natural groups contain one book each.
    """
    grouped = (pd.DataFrame({"group": groups, "target": y, "score": scores})
               .groupby("group", sort=False)
               .agg(target=("target", "first"), n_target=("target", "nunique"),
                    score=("score", "mean")))
    if not (grouped["n_target"] == 1).all():
        raise ValueError("Each cross-validation group must contain one class only")
    return grouped["target"].to_numpy(), grouped["score"].to_numpy()


def pls_component_x_variance(pls: PLSRegression, X: np.ndarray) -> np.ndarray:
    """Incremental descriptor variance reconstructed by each PLS component."""
    scores = pls.x_scores_
    loadings = pls.x_loadings_
    scaled = (X - pls._x_mean) / pls._x_std
    total = np.sum(scaled ** 2)
    if total <= 0:
        return np.full(scores.shape[1], np.nan)

    reconstruction = np.zeros_like(scaled)
    previous_residual = total
    explained = []
    for component in range(scores.shape[1]):
        reconstruction += np.outer(scores[:, component], loadings[:, component])
        residual = np.sum((scaled - reconstruction) ** 2)
        explained.append(100.0 * (previous_residual - residual) / total)
        previous_residual = residual
    return np.asarray(explained)


# generated-mode legend labels (the manuscript's Fig-legend names). Used across
# all discriminant figures, matching the manuscript's mode names. report.txt
# keeps the lowercase ams.MODE_LABELS (the raw code keys).
FIG_MODE_LABELS = {
    "summaries_conditioning":        "Hierarchical summaries",
    "extended_conditioning":         "Extended",
    "extended_no_fixed_conditioning": "Extended variant",
    "basic_conditioning":            "Basic",
    "minimal_fixed_conditioning":    "Fixed contract",
    "markov_rolling_conditioning":   "Markov rolling",
}


class HandlerEllipse(HandlerBase):
    """Legend handler drawing a dashed ellipse (for the 2σ mode ellipses)."""
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        e = Ellipse((width / 2. - xdescent, height / 2. - ydescent),
                    width * 0.95, height * 1.15, facecolor="none",
                    edgecolor="0.35", ls="--", lw=1.3)
        e.set_transform(trans)
        return [e]


class BoxedInitialHandler(HandlerBase):
    """Legend handler drawing a boxed initial, matching metrics_vs_size_all.png."""
    def __init__(self, text, color, **kw):
        self._text, self._color = text, color
        super().__init__(**kw)

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        t = MplText(width / 2. - xdescent, height / 2. - ydescent, self._text,
                    ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color=self._color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=self._color, linewidth=1.1))
        t.set_transform(trans)
        return [t]


DUMMY_MODES = {"markov_rolling_conditioning", "minimal_fixed_conditioning"}


def score_figure(out_path, Tn, Tg, gen_modes, nat_bins, size_bins, n_feat, fs, auc,
                 arrows=None, title_extra="", landmarks=None, show_bin_means=True,
                 title=None, lv_var=None, components=(0, 1)):
    """PLS score plot for a selected LV pair (natural, generated by mode, per-bin star
    trajectory); with `arrows` it becomes a biplot overlaying loading vectors.
    `landmarks` overlays projected landmark books; `show_bin_means` toggles the
    per-bin star trajectory; `title` overrides the default title."""
    x_component, y_component = components
    Tn = Tn[:, [x_component, y_component]]
    Tg = Tg[:, [x_component, y_component]]
    palette = size_bin_palette(len(size_bins))
    bin_color = {b["label"]: palette[i] for i, b in enumerate(size_bins)}
    labels = [b["label"] for b in size_bins if b["label"] in set(nat_bins)]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ellipse_bounds = []
    ax.scatter(Tn[:, 0], Tn[:, 1], s=7, c="#B0BEC5", alpha=0.35, linewidths=0, label="natural books")
    # natural 50% / 95% highest-density contours (thin lines)
    if len(Tn) >= 20:
        add_bin_density_contours(ax, Tn[:, :2], "#37474F", lw_scale=0.5)
    ax.plot([], [], color="#37474F", lw=1.7, label="natural p25-p75")
    ax.plot([], [], color="#37474F", lw=1.0, alpha=0.5, label="natural p2.5-p97.5")
    # generated modes: "dummy" baselines (Markov rolling, Minimal fixed) as
    # diamonds, elaborate conditioning as X markers so the comparison stands out
    for mode in [m for m in ams.MODE_ORDER if m in set(gen_modes)]:
        mask = gen_modes == mode
        dummy = mode in DUMMY_MODES
        lbl = FIG_MODE_LABELS.get(mode, ams.MODE_LABELS[mode])
        ax.scatter(Tg[mask, 0], Tg[mask, 1], s=(58 if dummy else 46),
                   marker=("D" if dummy else "X"), c=ams.MODE_COLORS[mode],
                   edgecolors=("black" if dummy else "none"),
                   linewidths=(0.7 if dummy else 0), alpha=0.9,
                   zorder=(7 if dummy else 6), label=lbl)
    # 2σ ellipse around each baseline (dummy) mode, coloured to match its
    # diamonds, to make its region explicit against the elaborate-mode cloud
    if arrows is not None:
        for dm in ("markov_rolling_conditioning", "minimal_fixed_conditioning"):
            pts = Tg[gen_modes == dm][:, :2]
            if len(pts) < 5:
                continue
            vals, vecs = np.linalg.eigh(np.cov(pts.T))
            o = vals.argsort()[::-1]; vals, vecs = vals[o], vecs[:, o]
            ang_e = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * 2.0 * np.sqrt(np.maximum(vals, 0))
            col = ams.MODE_COLORS[dm]
            # bake the alpha into the fill only, so the dashed edge stays fully
            # opaque (patch-level alpha would fade the border too)
            ax.add_patch(Ellipse(pts.mean(0), w, h, angle=ang_e,
                                 facecolor=to_rgba(col, 0.10), edgecolor=col,
                                 lw=2.2, ls="--", zorder=4))
            # Include the full rotated 2-sigma ellipse in the final plot limits.
            theta = np.radians(ang_e)
            half_w, half_h = w / 2, h / 2
            radius_x = np.hypot(half_w * np.cos(theta), half_h * np.sin(theta))
            radius_y = np.hypot(half_w * np.sin(theta), half_h * np.cos(theta))
            center_x, center_y = pts.mean(0)
            ellipse_bounds.append((center_x - radius_x, center_x + radius_x,
                                   center_y - radius_y, center_y + radius_y))
    if show_bin_means:
        nat_df = pd.DataFrame({"size_bin": nat_bins, "LV1": Tn[:, 0], "LV2": Tn[:, 1]})
        means = nat_df.groupby("size_bin")[["LV1", "LV2"]].mean().reindex(labels).dropna()
        ax.plot(means["LV1"], means["LV2"], "-", color=NAVY, lw=1.8, zorder=7)
        for lab, row in means.iterrows():
            ax.scatter(row["LV1"], row["LV2"], marker="*", s=520, c=[bin_color[lab]],
                       edgecolors=NAVY, linewidths=1.5, zorder=8, label=f"natural mean, {lab}")

    LANDMARK_COLOR = "#B3261E"
    Ls = None
    arrow_texts = []
    if arrows is not None:
        allpts = np.vstack([Tn[:, :2], Tg[:, :2]])
        xr = np.percentile(allpts[:, 0], [2, 98]); yr = np.percentile(allpts[:, 1], [2, 98])
        span = min(xr[1] - xr[0], yr[1] - yr[0])
        L = arrows["lv"]
        scale = 0.45 * span / np.linalg.norm(L, axis=1).max()
        Ls = L * scale
        for (lv1, lv2), lab, col in zip(Ls, arrows["labels"], arrows["colors"]):
            ax.annotate("", xy=(lv1, lv2), xytext=(0, 0), zorder=9,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, mutation_scale=13))
            # Fixed offsets keep labels visually tied to their loading arrows.
            dx = 0.018 * span * (1 if lv1 >= 0 else -1)
            dy = 0.018 * span * (1 if lv2 >= 0 else -1)
            arrow_texts.append(ax.text(
                lv1 + dx, lv2 + dy, lab,
                color="white", fontsize=6.5, fontweight="bold",
                ha="left" if lv1 >= 0 else "right",
                va="bottom" if lv2 >= 0 else "top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.16", facecolor=col, edgecolor="none", alpha=0.95)))

    # landmark books, projected into the PLS space: boxed red initials
    # (same style as metrics_vs_size_all.png)
    landmark_texts = []
    if landmarks is not None:
        lx, ly = landmarks["xy"][:, 0], landmarks["xy"][:, 1]
        for x, y, lab in zip(lx, ly, landmarks["labels"]):
            dx, dy = LANDMARK_LABEL_OFFSETS.get(lab, (0.0, 0.0))
            landmark_texts.append(ax.text(
                x + dx, y + dy, lab, fontsize=7.5, fontweight="bold", ha="center", va="center",
                color=LANDMARK_COLOR, zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=LANDMARK_COLOR, linewidth=1.1)))

    if arrow_texts:
        adjust_text(
            arrow_texts,
            objects=landmark_texts,
            target_x=Ls[:, 0],
            target_y=Ls[:, 1],
            avoid_self=False,
            force_text=(0.35, 0.50),
            force_static=(0.12, 0.18),
            force_pull=(1.0, 1.0),
            force_explode=(0.15, 0.20),
            pull_threshold=8,
            expand=(1.08, 1.15),
            max_move=(7, 7),
            iter_lim=7,
            min_arrow_len=12,
            ax=ax,
        )
        max_label_distance = 0.08 * span
        for text, (tip_x, tip_y), label in zip(arrow_texts, Ls, arrows["labels"]):
            text_x, text_y = text.get_position()
            delta_x, delta_y = text_x - tip_x, text_y - tip_y
            distance = np.hypot(delta_x, delta_y)
            if distance > max_label_distance:
                ratio = max_label_distance / distance
                text.set_position((tip_x + delta_x * ratio, tip_y + delta_y * ratio))
            offset_x, offset_y = ARROW_LABEL_OFFSETS.get(label, (0.0, 0.0))
            if offset_x or offset_y:
                text_x, text_y = text.get_position()
                text.set_position((text_x + offset_x, text_y + offset_y))

    # Robust display limits suppress sparse score tails while retaining the full
    # baseline ellipses, loading arrows, and landmark projections below.
    allpts = np.vstack([Tn[:, :2], Tg[:, :2]])
    x_lo, x_hi = np.percentile(allpts[:, 0], [2, 98])
    y_lo, y_hi = np.percentile(allpts[:, 1], [2, 98])
    if Ls is not None:
        tips = Ls * 1.16
        x_lo, x_hi = min(x_lo, tips[:, 0].min()), max(x_hi, tips[:, 0].max())
        y_lo, y_hi = min(y_lo, tips[:, 1].min()), max(y_hi, tips[:, 1].max())
    if landmarks is not None:
        x_lo, x_hi = min(x_lo, lx.min()), max(x_hi, lx.max())
        y_lo, y_hi = min(y_lo, ly.min()), max(y_hi, ly.max())
    for ellipse_x_lo, ellipse_x_hi, ellipse_y_lo, ellipse_y_hi in ellipse_bounds:
        x_lo, x_hi = min(x_lo, ellipse_x_lo), max(x_hi, ellipse_x_hi)
        y_lo, y_hi = min(y_lo, ellipse_y_lo), max(y_hi, ellipse_y_hi)
    px, py = 0.10 * (x_hi - x_lo), 0.10 * (y_hi - y_lo)
    ax.set_xlim(x_lo - px, x_hi + px); ax.set_ylim(y_lo - py, y_hi + py)

    if lv_var is not None:
        ax.set_xlabel(f"LV{x_component + 1}  ({lv_var[x_component]:.0f}% of descriptor variance)")
        ax.set_ylabel(f"LV{y_component + 1}  ({lv_var[y_component]:.0f}% of descriptor variance)")
    else:
        ax.set_xlabel(f"LV{x_component + 1}"); ax.set_ylabel(f"LV{y_component + 1}")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"PLS-DA LV{x_component + 1} × LV{y_component + 1} "
                     f"({fs}, {n_feat} feat, per-bin normalized){title_extra}\n"
                     f"natural vs generated — in-corpus CV AUC {auc:.3f}")

    # ── three legends: descriptor-family colours · point types · landmark key ──
    # (1+2) one legend: point types (natural, contours, generated modes) plus
    # the descriptor-family colours of the loading arrows
    pt_handles, pt_labels = ax.get_legend_handles_labels()
    pt_hmap = {}
    if arrows is not None:
        present = set(arrows["colors"])
        for f in FINE_ORDER:
            if FAM_COLOR[f] in present:
                pt_handles.append(Line2D([], [], color=FAM_COLOR[f], lw=3.4))
                pt_labels.append(FAM_LABEL[f])
        # key for the dashed 2σ ellipses drawn around Fixed contract & Markov rolling
        ell = Ellipse((0, 0), 1, 1, facecolor="none", edgecolor="0.35", ls="--", lw=1.3)
        pt_handles.append(ell); pt_labels.append("2σ ellipse (Fixed contract, Markov rolling)")
        pt_hmap[ell] = HandlerEllipse()
    pt_leg = ax.legend(pt_handles, pt_labels, handler_map=pt_hmap,
                       loc="upper right", bbox_to_anchor=(0.995, 0.995),
                       fontsize=7.5, framealpha=0.9, ncol=2)
    ax.add_artist(pt_leg)
    # (3) landmark books key — boxed initials + titles, same style as
    # metrics_vs_size_all.png
    if landmarks is not None:
        lm_h, lm_l, hmap = [], [], {}
        for ini, ttl in LANDMARK_TITLES:
            h = Line2D([], [], linestyle="none", marker="", label=ttl)
            lm_h.append(h); lm_l.append(ttl); hmap[h] = BoxedInitialHandler(ini, LANDMARK_COLOR)
        ax.legend(lm_h, lm_l, handler_map=hmap, loc="lower right",
                  bbox_to_anchor=(0.995, 0.005), ncol=3, fontsize=7,
                  title="Landmark books", title_fontsize=7.5, framealpha=0.9,
                  handlelength=1.3, handletextpad=0.4, columnspacing=0.9, labelspacing=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _draw_arrows(ax, arrows, Tn, Tg):
    allpts = np.vstack([Tn[:, :2], Tg[:, :2]])
    xr = np.percentile(allpts[:, 0], [2, 98]); yr = np.percentile(allpts[:, 1], [2, 98])
    span = min(xr[1] - xr[0], yr[1] - yr[0])
    L = arrows["lv"]; scale = 0.45 * span / np.linalg.norm(L, axis=1).max()
    for (lv1, lv2), lab, col in zip(L * scale, arrows["labels"], arrows["colors"]):
        ax.annotate("", xy=(lv1, lv2), xytext=(0, 0), zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14))
        ax.text(lv1 * 1.10, lv2 * 1.10, lab, color="white", fontsize=7.5, fontweight="bold",
                ha="center", va="center", zorder=10,
                bbox=dict(boxstyle="round,pad=0.16", facecolor=col, edgecolor="none", alpha=0.92))
    return L * scale * 1.18


def plot_per_bin(out_path, Tn, Tg, nat_bins, gen_bins, gen_modes, size_bins,
                 n_feat, fs, auc, arrows):
    """3×3 combined figure: panel A overview (with loading arrows), panels B–F
    one per size bin showing that bin's natural 50/95% envelope + generated."""
    labels = [b["label"] for b in size_bins if b["label"] in set(nat_bins)]
    palette = size_bin_palette(len(size_bins))
    bin_color = {b["label"]: palette[i] for i, b in enumerate(size_bins)}

    allpts = np.vstack([Tn[:, :2], Tg[:, :2]])
    x_lo, x_hi = np.percentile(allpts[:, 0], [1, 99])
    y_lo, y_hi = np.percentile(allpts[:, 1], [1, 99])
    px, py = 0.08 * (x_hi - x_lo), 0.08 * (y_hi - y_lo)
    xlim, ylim = (x_lo - px, x_hi + px), (y_lo - py, y_hi + py)

    fig = plt.figure(figsize=(16.5, 15.5))
    gs = fig.add_gridspec(3, 3, hspace=0.22, wspace=0.16, left=0.05, right=0.99,
                          top=0.92, bottom=0.05)
    ax_a = fig.add_subplot(gs[0:2, 0:2])
    small = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]),
             fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]

    # ── panel A: overview ──
    ax_a.scatter(Tn[:, 0], Tn[:, 1], s=6, c="#B0BEC5", alpha=0.30, linewidths=0)
    if len(Tn) >= 20:
        add_bin_density_contours(ax_a, Tn[:, :2], "#37474F")
    for mode in [m for m in ams.MODE_ORDER if m in set(gen_modes)]:
        mask = gen_modes == mode
        ax_a.scatter(Tg[mask, 0], Tg[mask, 1], s=42, marker="D", c=ams.MODE_COLORS[mode],
                     edgecolors="black", linewidths=0.5, alpha=0.95, zorder=6,
                     label=FIG_MODE_LABELS.get(mode, ams.MODE_LABELS[mode]))
    nat_df = pd.DataFrame({"b": nat_bins, "LV1": Tn[:, 0], "LV2": Tn[:, 1]})
    means = nat_df.groupby("b")[["LV1", "LV2"]].mean().reindex(labels).dropna()
    ax_a.plot(means["LV1"], means["LV2"], "-", color=NAVY, lw=1.8, zorder=7)
    for lab, row in means.iterrows():
        ax_a.scatter(row["LV1"], row["LV2"], marker="*", s=560, c=[bin_color[lab]],
                     edgecolors=NAVY, linewidths=1.5, zorder=8)
    tips = _draw_arrows(ax_a, arrows, Tn, Tg)
    ax_a.set_xlim(min(xlim[0], tips[:, 0].min()), max(xlim[1], tips[:, 0].max()))
    ax_a.set_ylim(min(ylim[0], tips[:, 1].min()), max(ylim[1], tips[:, 1].max()))
    ax_a.set_title("(A)  all bins — natural 50/95% envelope, loadings, size-bin means", fontsize=14)
    ax_a.set_xlabel("LV1"); ax_a.set_ylabel("LV2")
    ax_a.legend(fontsize=9, loc="best", framealpha=0.9, ncol=2)

    # ── panels B–F: per bin ──
    for i, (ax, lab) in enumerate(zip(small, labels)):
        nb = nat_bins == lab; gb = gen_bins == lab
        ax.scatter(Tn[:, 0], Tn[:, 1], s=4, c="#CFD8DC", alpha=0.25, linewidths=0, zorder=1)
        ax.scatter(Tn[nb, 0], Tn[nb, 1], s=8, c=[bin_color[lab]], alpha=0.45, linewidths=0, zorder=2)
        if nb.sum() >= 20:
            add_bin_density_contours(ax, Tn[nb][:, :2], bin_color[lab])
        for mode in [m for m in ams.MODE_ORDER if m in set(gen_modes[gb])]:
            mm = gb & (gen_modes == mode)
            ax.scatter(Tg[mm, 0], Tg[mm, 1], s=46, marker="D", c=ams.MODE_COLORS[mode],
                       edgecolors="black", linewidths=0.5, alpha=0.95, zorder=6)
        ax.set_title(f"({chr(ord('B') + i)})  {lab}   ({int(nb.sum())} nat · {int(gb.sum())} gen)",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.tick_params(labelsize=9)

    fig.suptitle(f"PLS-DA ({fs}, {n_feat} feat, per-bin normalized) — per size bin "
                 f"(in-corpus CV AUC {auc:.3f})", fontsize=18)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _conf_ellipse(ax, pts, color, nstd=2.0, lw=1.8):
    """95%-ish (2-sigma) covariance ellipse for a group's CV scores."""
    if len(pts) < 5:
        return
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(pts.mean(0), w, h, angle=theta, facecolor="none",
                         edgecolor=color, lw=lw, alpha=0.85, zorder=5))


def plot_cva(out_path, X, groups, fs, n_feat):
    """Canonical Variate Analysis (regularized LDA) treating natural + each
    ablation mode as separate groups; plots CV1×CV2 with group ellipses and
    centroids."""
    present = ["natural"] + [m for m in ams.MODE_ORDER if m in set(groups)]
    # PCA-reduce first (orthogonalise the collinear features) so the CVA/LDA is
    # well-conditioned; plain LDA on 49 correlated features is numerically unstable.
    n_pca = min(15, X.shape[1], len(X) - 1)
    Xr = PCA(n_components=n_pca).fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=2)
    Z = lda.fit(Xr, groups).transform(Xr)
    evr = lda.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for g in present:
        mask = groups == g
        is_nat = g == "natural"
        col = "#90A4AE" if is_nat else ams.MODE_COLORS[g]
        lab = "natural" if is_nat else FIG_MODE_LABELS.get(g, ams.MODE_LABELS[g])
        ax.scatter(Z[mask, 0], Z[mask, 1], s=(6 if is_nat else 42),
                   marker=("o" if is_nat else "D"), c=col,
                   alpha=(0.3 if is_nat else 0.9), linewidths=(0 if is_nat else 0.5),
                   edgecolors="none" if is_nat else "black", zorder=(2 if is_nat else 6),
                   label=lab)
        _conf_ellipse(ax, Z[mask][:, :2], "#546E7A" if is_nat else col)
        c = Z[mask][:, :2].mean(0)
        ax.scatter(*c, marker="X", s=230, c=col, edgecolors="black", linewidths=1.3, zorder=9)
    x_lo, x_hi = np.percentile(Z[:, 0], [1, 99]); y_lo, y_hi = np.percentile(Z[:, 1], [1, 99])
    px, py = 0.10 * (x_hi - x_lo), 0.10 * (y_hi - y_lo)
    ax.set_xlim(x_lo - px, x_hi + px); ax.set_ylim(y_lo - py, y_hi + py)
    ax.set_xlabel(f"CV1 ({evr[0] * 100:.0f}% of between-group var.)")
    ax.set_ylabel(f"CV2 ({evr[1] * 100:.0f}%)")
    ax.set_title(f"Canonical variate analysis ({fs}, {n_feat} feat, per-bin normalized)\n"
                 "natural + ablation modes as groups  (X = group centroid, ellipse = 2σ)")
    ax.legend(fontsize=8, loc="best", framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return evr


def load_data(size_bins):
    natural, all_features = scs.build_natural()
    natural["size_bin"] = [assign_bin(t, size_bins) for t in natural["n_tokens"]]
    natural = natural.dropna(subset=["size_bin"]).reset_index(drop=True)

    generated = scn.load_generated_rows(all_features)
    for c in all_features:
        generated[c] = pd.to_numeric(generated[c], errors="coerce")
    generated = generated.replace([np.inf, -np.inf], np.nan).dropna(subset=all_features).copy()
    generated["size_bin"] = [assign_bin(t, size_bins, lower_edge_tolerance=0.10)
                             for t in generated["n_tokens"]]
    generated = generated.dropna(subset=["size_bin"]).reset_index(drop=True)
    return natural, generated, all_features


def select_features(all_features: list[str], feature_set: str) -> list[str]:
    if feature_set == "sc":
        return [f for f in all_features if is_sc(f)]
    if feature_set == "token_stat":
        return [f for f in all_features if not is_sc(f)]
    return list(all_features)  # combined


# ── landmark books: projected into the fitted PLS score space ────────────────
LANDMARK_CSV = (ROOT / "natural_texts" / "Output" / "pca" / "generated_prefixes"
                / "landmark_prefix_projection_scores.csv")
LANDMARK_SC_JSON = Path("/tmp/landmark_sc.json")   # written by compute_sc_landmarks.py
LANDMARK_INITIALS = {"Frankenstein": "F", "Sherlock": "SH", "PridePrejudice": "PP",
                     "TracedTracked": "TT", "OriginSpecies": "OS", "Crusoe": "RC",
                     "MobyDick": "MD", "Ulysses": "U", "DonQuixote": "DQ",
                     "MonteCristo": "MC", "WarPeace": "WP", "LesMis": "LM"}
# initials -> full title, ordered by book length (for the on-plot key)
LANDMARK_TITLES = [("F", "Frankenstein"), ("SH", "Sherlock Holmes"),
                   ("PP", "Pride & Prejudice"), ("TT", "Traced & Tracked"),
                   ("OS", "Origin of Species"), ("RC", "Robinson Crusoe"),
                   ("MD", "Moby Dick"), ("U", "Ulysses"), ("DQ", "Don Quixote"),
                   ("MC", "Count of Monte Cristo"), ("WP", "War and Peace"),
                   ("LM", "Les Misérables")]
# Frankenstein and Sherlock project almost identically on LV1/LV2. Offset only
# the visible Frankenstein initial; the point remains at its fitted score.
LANDMARK_LABEL_OFFSETS = {"F": (0.42, 0.04)}


def load_landmark_projection(feats, stats, pls, size_bins, components=(0, 1)):
    """Assemble the 35-feature vector for each landmark book (26 non-SC features
    from the endpoint CSV row + 9 SC features computed offline), per-bin
    standardize against the natural stats, and project through the fitted PLS.
    Returns {"xy": (k,2) array, "labels": [initials]} or None if SC missing."""
    if not LANDMARK_SC_JSON.exists():
        return None
    sc = json.loads(LANDMARK_SC_JSON.read_text())
    lm = pd.read_csv(LANDMARK_CSV)
    ep = lm.loc[lm.groupby("title")["n_tokens"].idxmax()].reset_index(drop=True)
    ep["gid"] = ep["file"].astype(str).str.extract(r"^(\d+)_")
    sc_feats = {f for f in feats if is_sc(f)}
    rows = []
    for _, r in ep.iterrows():
        gid = str(r["gid"])
        if gid not in sc:
            continue
        screc = {f"{w}_{m}": sc[gid].get(w, {}).get(m)
                 for w in ("W2k", "W8k", "W40k")
                 for m in ("off_diag_mean", "novelty_mean", "sim_lag1_mean")}
        rec = {f: (screc.get(f) if f in sc_feats else r.get(f)) for f in feats}
        if any(pd.isna(rec[f]) for f in feats):
            continue
        rec["n_tokens"] = r["n_tokens"]
        rec["size_bin"] = assign_bin(r["n_tokens"], size_bins)
        short = str(r["source_metric_file"]).replace("metrics_", "").replace(".json", "")
        rec["_ini"] = LANDMARK_INITIALS.get(short, "?")
        rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["size_bin"]).reset_index(drop=True)
    T = pls.transform(standardize_per_bin(df, feats, stats))
    return {"xy": T[:, list(components)], "labels": list(df["_ini"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-set", choices=["combined", "sc", "token_stat"], default="combined")
    p.add_argument("--n-components", type=int, default=3)
    p.add_argument("--biplot-vip-min", type=float, default=1.0,
                   help="draw loading arrows for every feature with VIP above this "
                        "(default 1.0, the standard importance threshold).")
    p.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = p.parse_args()

    out_dir = args.output_root / args.feature_set
    out_dir.mkdir(parents=True, exist_ok=True)

    size_bins = load_size_bins(args.size_stats)
    natural, generated, all_features = load_data(size_bins)
    feats = select_features(all_features, args.feature_set)

    # per-bin (matched-size) normalization, fit on natural
    stats = fit_per_bin_stats(natural, feats)
    Xn = standardize_per_bin(natural, feats, stats)
    Xg = standardize_per_bin(generated, feats, stats)
    X = np.vstack([Xn, Xg])
    y = np.r_[np.zeros(len(Xn)), np.ones(len(Xg))]
    groups = np.r_[[f"nat{b}" for b in natural["book_id"]],
                   ["gen_" + str(r) for r in generated["run_id"]]]
    raw_rows = pd.concat([natural, generated], ignore_index=True)

    # Group-aware CV: all five prefixes from a generated run stay together.
    # Refit the natural per-bin median/IQR transform inside each training fold
    # so held-out natural books cannot influence their own normalization.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    prefix_aucs, prefix_baccs = [], []
    group_aucs, group_baccs, group_oof = [], [], []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        natural_train = natural.iloc[tr[tr < len(natural)]]
        fold_stats = fit_per_bin_stats(natural_train, feats)
        Xtr = standardize_per_bin(raw_rows.iloc[tr], feats, fold_stats)
        Xte = standardize_per_bin(raw_rows.iloc[te], feats, fold_stats)
        m = PLSRegression(n_components=args.n_components).fit(Xtr, y[tr])
        p_tr, p_te = m.predict(Xtr).ravel(), m.predict(Xte).ravel()
        # Prefix-level scores are retained for reference only. The primary CV
        # result is one score per complete generated run (and natural book).
        fpr, tpr, thr = roc_curve(y[tr], p_tr)
        t_opt = thr[int(np.argmax(tpr - fpr))]
        prefix_aucs.append(roc_auc_score(y[te], p_te))
        prefix_baccs.append(balanced_accuracy_score(y[te], (p_te >= t_opt).astype(int)))

        y_tr_group, p_tr_group = aggregate_group_scores(y[tr], p_tr, groups[tr])
        y_te_group, p_te_group = aggregate_group_scores(y[te], p_te, groups[te])
        fpr, tpr, thr = roc_curve(y_tr_group, p_tr_group)
        t_opt_group = thr[int(np.argmax(tpr - fpr))]
        group_aucs.append(roc_auc_score(y_te_group, p_te_group))
        group_pred = (p_te_group >= t_opt_group).astype(int)
        group_baccs.append(balanced_accuracy_score(y_te_group, group_pred))
        group_oof.append(pd.DataFrame({
            "fold": fold,
            "group_id": pd.unique(groups[te]),
            "true_class": y_te_group.astype(int),
            "pls_score": p_te_group,
            "threshold": t_opt_group,
            "predicted_class": group_pred,
        }))

    # Use the independent run/book unit for all figure and report headlines.
    auc = float(np.mean(group_aucs))
    group_oof_df = pd.concat(group_oof, ignore_index=True)
    group_oof_df["true_label"] = np.where(group_oof_df["true_class"] == 1,
                                            "generated", "natural")
    group_oof_df["predicted_label"] = np.where(group_oof_df["predicted_class"] == 1,
                                                 "generated", "natural")
    group_oof_df.to_csv(out_dir / "cv_run_book_predictions.csv", index=False)
    tn, fp, fn, tp = confusion_matrix(
        group_oof_df["true_class"], group_oof_df["predicted_class"], labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    pls = PLSRegression(n_components=args.n_components).fit(X, y)
    vip = vip_scores(pls)
    lv_var = pls_component_x_variance(pls, X)
    coef = np.ravel(pls.coef_)
    if coef.size != len(feats):                       # sklearn version shape guard
        coef = np.ravel(pls.coef_.T)[:len(feats)]
    gen_offset = Xg.mean(axis=0)                       # matched-size mean offset of generated

    table = pd.DataFrame({
        "metric": [display_name(f) for f in feats],
        "family": ["SC" if is_sc(f) else ("stat" if f in scn.STAT_INFO_METRICS else "token")
                   for f in feats],
        "fam": [fine_family(f) for f in feats],
        "VIP": vip,
        "pls_coef": coef,
        "gen_offset_iqr": gen_offset,
        "direction": ["gen higher" if o > 0 else "gen lower" for o in gen_offset],
    }).sort_values("VIP", ascending=False).reset_index(drop=True)
    table.drop(columns=["fam"]).to_csv(out_dir / "vip.csv", index=False)
    plot_vip(table, out_dir / "vip_barplot.png", args.feature_set, auc)

    # per-mode separation (continuous PLS score; higher = more "generated-like")
    scores_all = pls.predict(X).ravel()
    gen_scores = scores_all[len(Xn):]
    nat_scores = scores_all[:len(Xn)]
    per_mode = (pd.DataFrame({"mode": generated["conditioning_mode"].values, "score": gen_scores})
                .groupby("mode")["score"].agg(["count", "mean", "std"])
                .sort_values("mean"))

    # ── text report ──────────────────────────────────────────────────────────
    lines = []
    lines.append(f"PLS-DA discriminant analysis — feature set: {args.feature_set}")
    lines.append("=" * 70)
    lines.append(f"natural (SC subcorpus): {len(Xn)}    generated: {len(Xg)} "
                 f"({generated['conditioning_mode'].nunique()} modes)")
    lines.append(f"features: {len(feats)}    latent variables: {args.n_components}")
    lines.append("normalization: per-size-bin median/IQR (matched size); "
                 "CV refits natural-bin statistics in each training fold")
    lines.append("CV: 5-fold StratifiedGroupKFold (all five prefixes of each "
                 "generated run held together)")
    lines.append("primary CV test unit: generated run (mean score across its five "
                 "held-out prefixes) versus natural book")
    lv_parts = [f"LV{i + 1} {value:.1f}%" for i, value in enumerate(lv_var)]
    lines.append("incremental descriptor variance by latent variable: " + ", ".join(lv_parts)
                 + f" (cumulative {np.nansum(lv_var):.1f}%)")
    if args.feature_set != "sc":
        lines.append("descriptor policy: preserve interpretively distinct descriptors; "
                     "exclude exact aliases and the overlapping unigram-entropy source")
        lines.append("excluded exact token aliases:")
        for metric, relationship in EXCLUDED_EXACT_ALIASES.items():
            lines.append(f"  {metric} = {relationship}")
        lines.append("excluded provenance duplicate:")
        for metric, relationship in EXCLUDED_PROVENANCE_DUPLICATES.items():
            lines.append(f"  {metric} = {relationship}")
    lines.append("")
    lines.append(f"In-corpus run/book CV AUC        : {np.mean(group_aucs):.3f} ± {np.std(group_aucs):.3f}")
    lines.append(f"Run/book balanced accuracy        : {np.mean(group_baccs):.3f} ± {np.std(group_baccs):.3f}")
    lines.append(f"Prefix-row CV AUC (correlated)    : {np.mean(prefix_aucs):.3f} ± {np.std(prefix_aucs):.3f}")
    lines.append(f"Prefix-row balanced accuracy      : {np.mean(prefix_baccs):.3f} ± {np.std(prefix_baccs):.3f}")
    lines.append("Pooled out-of-fold run/book confusion matrix "
                 "(fold-specific training thresholds):")
    lines.append("                         predicted natural  predicted generated")
    lines.append(f"  actual natural                  {tn:4d}                 {fp:4d}")
    lines.append(f"  actual generated                {fn:4d}                 {tp:4d}")
    lines.append(f"Pooled sensitivity {sensitivity:.3f} | pooled specificity {specificity:.3f}")
    lines.append("scope: discrimination of these 49 generated runs versus this natural-book "
                 "corpus; not an external-model, prompt, or corpus benchmark")
    lines.append(f"natural score mean {nat_scores.mean():+.2f} | generated score mean {gen_scores.mean():+.2f}")
    lines.append("")
    lines.append("Variable importance (VIP > 1 = important), with matched-size direction:")
    lines.append(f"  {'metric':32s} {'family':6s} {'VIP':>5s} {'offset':>7s}  direction")
    for _, r in table.iterrows():
        star = "*" if r.VIP > 1 else " "
        lines.append(f" {star}{r.metric:32s} {r.family:6s} {r.VIP:5.2f} "
                     f"{r.gen_offset_iqr:+7.2f}  {r.direction}")
    lines.append("")
    lines.append("Per-ablation-mode separation (mean PLS score; higher = more generated-like):")
    for mode, row in per_mode.iterrows():
        lines.append(f"  {ams.MODE_LABELS.get(mode, mode):20s} n={int(row['count']):3d}  "
                     f"mean={row['mean']:+.2f} ± {row['std']:.2f}")
    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report + "\n")
    print(report)

    # ── score plot + biplot (LV1 × LV2) ───────────────────────────────────────
    T = pls.x_scores_
    Tn, Tg = T[:len(Xn)], T[len(Xn):]
    gen_modes = generated["conditioning_mode"].to_numpy()
    nat_bins = natural["size_bin"].values
    score_figure(out_dir / "plsda_scores.png", Tn, Tg, gen_modes, nat_bins, size_bins,
                 len(feats), args.feature_set, auc, lv_var=lv_var, components=(0, 1))

    # biplot: overlay loading vectors for every important feature (VIP > threshold)
    ranked = np.argsort(vip)[::-1]
    order = ranked[vip[ranked] > args.biplot_vip_min]
    if len(order) == 0:                               # fallback if none clear the bar
        order = ranked[:8]
    loadings_lv = pls.x_loadings_[:, :2]
    arrows = {"lv": loadings_lv[order],
              "labels": [descriptor_label(feats[i]) for i in order],
              "colors": [FAM_COLOR[fine_family(feats[i])] for i in order]}
    landmarks = load_landmark_projection(feats, stats, pls, size_bins, components=(0, 1))
    if landmarks is not None:
        print(f"landmark books projected: {len(landmarks['labels'])}")
    biplot_title = (f"Discriminant descriptor distribution — natural, generated "
                    f"& landmark books\n(showing {len(order)} of {len(feats)} "
                    f"descriptors with VIP > {args.biplot_vip_min:g}  —  "
                    f"in-corpus CV AUC {auc:.3f})")
    score_figure(out_dir / "plsda_biplot.png", Tn, Tg, gen_modes, nat_bins, size_bins,
                 len(feats), args.feature_set, auc, arrows=arrows,
                 landmarks=landmarks, show_bin_means=False, title=biplot_title, lv_var=lv_var,
                 components=(0, 1))
    plot_per_bin(out_dir / "plsda_per_bin.png", Tn, Tg, nat_bins,
                 generated["size_bin"].values, gen_modes, size_bins,
                 len(feats), args.feature_set, auc, arrows)

    # canonical variate analysis: natural + each ablation mode as a group
    groups = np.r_[np.array(["natural"] * len(Xn)), gen_modes]
    cva_evr = plot_cva(out_dir / "cva_scores.png", np.vstack([Xn, Xg]), groups,
                       args.feature_set, len(feats))
    print(f"CVA: CV1 {cva_evr[0] * 100:.0f}% / CV2 {cva_evr[1] * 100:.0f}% of between-group variance")
    print(f"\noutputs → {out_dir}")


if __name__ == "__main__":
    main()
