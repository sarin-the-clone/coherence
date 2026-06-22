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
  combined    9 SC + 8 stat + 32 token   (default)
  sc          9 SC only
  token_stat  8 stat + 32 token

Outputs → natural_texts/Output/pca/discriminant/<feature_set>/
  report.txt, vip.csv, plsda_scores.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import pca_sc_natural_combined as scn  # noqa: E402
import pca_sc_combined_size_bins as scs  # noqa: E402
import pca_allmetrics_size_bins as ams  # noqa: E402
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


FAM_COLOR = {"sc": "#9B2226", "stat": "#0D2545", "td": "#00796B",
             "tr": "#E8710A", "tp": "#6A1B9A"}
FAM_LABEL = {"sc": "semantic coherence", "stat": "stat/info",
             "td": "token distribution", "tr": "token recurrence",
             "tp": "token predictability"}


def fine_family(metric: str) -> str:
    if is_sc(metric):
        return "sc"
    if metric in scn.STAT_INFO_METRICS:
        return "stat"
    return metric[:2]  # td / tr / tp


def plot_vip(table: pd.DataFrame, out_path: Path, feature_set: str, auc: float) -> None:
    """Signed VIP bar chart: length = VIP, side = matched-size direction
    (right = generated higher, left = lower), colour = metric family."""
    t = table.sort_values("VIP").reset_index(drop=True)   # ascending → top of axis = highest
    signed = t["VIP"] * np.sign(t["gen_offset_iqr"]).replace(0, 1)
    colors = [FAM_COLOR[f] for f in t["fam"]]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.30 * len(t))))
    ax.barh(range(len(t)), signed, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t["metric"], fontsize=8)
    for x in (-1, 1):
        ax.axvline(x, color="#555", ls="--", lw=0.9, alpha=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("← generated LOWER      signed VIP  (|VIP| > 1 = important)      generated HIGHER →",
                  fontsize=10)
    ax.set_title(f"PLS-DA variable importance — {feature_set} (CV AUC {auc:.3f})", fontsize=12)
    fams = [f for f in ["sc", "stat", "td", "tr", "tp"] if f in set(t["fam"])]
    ax.legend(handles=[plt.Line2D([], [], marker="s", linestyle="", markersize=9,
                                  markerfacecolor=FAM_COLOR[f], markeredgecolor="black",
                                  label=FAM_LABEL[f]) for f in fams],
              fontsize=8, loc="lower right", framealpha=0.9)
    ax.margins(y=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
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


def score_figure(out_path, Tn, Tg, gen_modes, nat_bins, size_bins, n_feat, fs, auc,
                 arrows=None, title_extra=""):
    """LV1×LV2 score plot (natural, generated by mode, per-bin natural-mean star
    trajectory); with `arrows` it becomes a biplot overlaying loading vectors."""
    palette = size_bin_palette(len(size_bins))
    bin_color = {b["label"]: palette[i] for i, b in enumerate(size_bins)}
    labels = [b["label"] for b in size_bins if b["label"] in set(nat_bins)]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    ax.scatter(Tn[:, 0], Tn[:, 1], s=7, c="#B0BEC5", alpha=0.35, linewidths=0, label="natural")
    # natural 50% / 95% highest-density contours
    if len(Tn) >= 20:
        add_bin_density_contours(ax, Tn[:, :2], "#37474F")
    ax.plot([], [], color="#37474F", lw=3.4, label="natural 50%")
    ax.plot([], [], color="#37474F", lw=2.0, alpha=0.5, label="natural 95%")
    for mode in [m for m in ams.MODE_ORDER if m in set(gen_modes)]:
        mask = gen_modes == mode
        ax.scatter(Tg[mask, 0], Tg[mask, 1], s=45, marker="D", c=ams.MODE_COLORS[mode],
                   edgecolors="black", linewidths=0.6, alpha=0.95, zorder=6,
                   label="gen: " + ams.MODE_LABELS[mode])
    nat_df = pd.DataFrame({"size_bin": nat_bins, "LV1": Tn[:, 0], "LV2": Tn[:, 1]})
    means = nat_df.groupby("size_bin")[["LV1", "LV2"]].mean().reindex(labels).dropna()
    ax.plot(means["LV1"], means["LV2"], "-", color=NAVY, lw=1.8, zorder=7)
    for lab, row in means.iterrows():
        ax.scatter(row["LV1"], row["LV2"], marker="*", s=520, c=[bin_color[lab]],
                   edgecolors=NAVY, linewidths=1.5, zorder=8, label=f"natural mean, {lab}")

    if arrows is not None:
        allpts = np.vstack([Tn[:, :2], Tg[:, :2]])
        xr = np.percentile(allpts[:, 0], [2, 98]); yr = np.percentile(allpts[:, 1], [2, 98])
        span = min(xr[1] - xr[0], yr[1] - yr[0])
        L = arrows["lv"]
        scale = 0.45 * span / np.linalg.norm(L, axis=1).max()
        for (lv1, lv2), lab, col in zip(L * scale, arrows["labels"], arrows["colors"]):
            ax.annotate("", xy=(lv1, lv2), xytext=(0, 0), zorder=9,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=14))
            ax.text(lv1 * 1.10, lv2 * 1.10, lab, color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=col, edgecolor="none", alpha=0.92))

    # robust display limits (ignore the ~1% score outliers), keeping arrows on-panel
    allpts = np.vstack([Tn[:, :2], Tg[:, :2]])
    x_lo, x_hi = np.percentile(allpts[:, 0], [1, 99])
    y_lo, y_hi = np.percentile(allpts[:, 1], [1, 99])
    if arrows is not None:
        tips = L * scale * 1.18
        x_lo, x_hi = min(x_lo, tips[:, 0].min()), max(x_hi, tips[:, 0].max())
        y_lo, y_hi = min(y_lo, tips[:, 1].min()), max(y_hi, tips[:, 1].max())
    px, py = 0.08 * (x_hi - x_lo), 0.08 * (y_hi - y_lo)
    ax.set_xlim(x_lo - px, x_hi + px); ax.set_ylim(y_lo - py, y_hi + py)

    ax.set_xlabel("LV1"); ax.set_ylabel("LV2")
    ax.set_title(f"PLS-DA ({fs}, {n_feat} feat, per-bin normalized){title_extra}\n"
                 f"natural vs generated — CV AUC {auc:.3f}")
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9, ncol=2)
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
                     label="gen: " + ams.MODE_LABELS[mode])
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
                 f"(CV AUC {auc:.3f})", fontsize=18)
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
        lab = "natural" if is_nat else "gen: " + ams.MODE_LABELS[g]
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

    # group-aware cross-validation
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    aucs, baccs = [], []
    for tr, te in cv.split(X, y, groups):
        m = PLSRegression(n_components=args.n_components).fit(X[tr], y[tr])
        p_tr, p_te = m.predict(X[tr]).ravel(), m.predict(X[te]).ravel()
        # threshold from the train fold (Youden's J) — fixed 0.5 is degenerate
        # under class imbalance
        fpr, tpr, thr = roc_curve(y[tr], p_tr)
        t_opt = thr[int(np.argmax(tpr - fpr))]
        aucs.append(roc_auc_score(y[te], p_te))
        baccs.append(balanced_accuracy_score(y[te], (p_te >= t_opt).astype(int)))

    pls = PLSRegression(n_components=args.n_components).fit(X, y)
    vip = vip_scores(pls)
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
    plot_vip(table, out_dir / "vip_barplot.png", args.feature_set, float(np.mean(aucs)))

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
                 "CV: 5-fold StratifiedGroupKFold (run-grouped)")
    lines.append("")
    lines.append(f"Cross-validated AUC              : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    lines.append(f"Cross-validated balanced accuracy: {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")
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
    auc = float(np.mean(aucs))
    score_figure(out_dir / "plsda_scores.png", Tn, Tg, gen_modes, nat_bins, size_bins,
                 len(feats), args.feature_set, auc)

    # biplot: overlay loading vectors for every important feature (VIP > threshold)
    ranked = np.argsort(vip)[::-1]
    order = ranked[vip[ranked] > args.biplot_vip_min]
    if len(order) == 0:                               # fallback if none clear the bar
        order = ranked[:8]
    loadings_lv = pls.x_loadings_[:, :2]
    arrows = {"lv": loadings_lv[order],
              "labels": [display_name(feats[i]) for i in order],
              "colors": [FAM_COLOR[fine_family(feats[i])] for i in order]}
    score_figure(out_dir / "plsda_biplot.png", Tn, Tg, gen_modes, nat_bins, size_bins,
                 len(feats), args.feature_set, auc, arrows=arrows,
                 title_extra=f" — biplot ({len(order)} feat, VIP>{args.biplot_vip_min:g})")
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
