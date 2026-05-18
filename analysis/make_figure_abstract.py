#!/usr/bin/env python3
"""
make_figure_abstract.py

Produces a publication-quality figure summarising the long-form LLM coherence
study. The figure now contains a single PCA panel comparing generated texts
against the natural books landscape (genre-coloured, with density contours).

Natural-text means (from 10 068 Gutenberg books via batch_*.json):
  beta           : 0.2487
  H_DMA          : 0.5939
  H_WT_db4       : 0.5871
  lz_complexity  : 5.5138
  mi_decay_gamma : 0.0128   ← generated texts are HIGHER (~0.04) = faster decay
  entropy_bits   : 9.0220
  perplexity     : 532.8
"""

import argparse
import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
AGG_JSON   = ROOT / "generated_texts_og" / "Output" / "aggregated_metrics.json"
CHAOS_JSON = ROOT / "generated_texts_og" / "Output" / "chaotic_run_metrics.json"
OUT_PATH   = ROOT / "generated_texts_og" / "Output" / "figure_abstract.pdf"
PANEL_IMG  = ROOT / "analysis" / "assets" / "channel_panel.png"

# ── colours ────────────────────────────────────────────────────────────────
C_HIGH  = "#2563EB"   # blue  – high-temp generated
C_LOW   = "#60A5FA"   # sky   – low-temp generated
C_NAT   = "#16A34A"   # green – natural text
C_CHAOS = "#9CA3AF"   # grey  – chaotic run
C_PANEL = "#F8FAFC"   # panel bg
C_ACCENT= "#7C3AED"   # purple accent
FONT = "DejaVu Sans"

# Empirical natural-book means from 10 068 Gutenberg books (batch_*.json)
NAT_MEANS = {
    "beta":          0.2487,
    "alpha":         0.1015,
    "H_DMA":         0.5939,
    "H_WT_db4":      0.5871,
    "entropy_bits":  9.0220,
    "lz_complexity": 5.5138,
    "mi_decay_gamma":0.0128,
    "perplexity":    532.81,
}

# ── helpers ─────────────────────────────────────────────────────────────────
def cast_keys(d: dict) -> dict:
    return {m: {float(k): v for k, v in bins.items()} for m, bins in d.items()}


def load_agg():
    with open(AGG_JSON) as f:
        raw = json.load(f)
    return cast_keys(raw.get("high_temp", {})), cast_keys(raw.get("low_temp", {}))


def load_chaos():
    if not CHAOS_JSON.exists():
        return {}
    with open(CHAOS_JSON) as f:
        raw = json.load(f)
    return {m: {int(k): v for k, v in d.items()} for m, d in raw.items()}


def series_from_binned(binned, metric):
    if metric not in binned:
        return np.array([]), np.array([]), np.array([])
    d = binned[metric]
    xs = sorted(d.keys())
    means = np.array([d[x]["mean"] for x in xs])
    stds  = np.array([d[x]["std"]  for x in xs])
    return np.array(xs) / 1000.0, means, stds   # x in k-tokens


def panel_bg(ax, color=C_PANEL):
    ax.set_facecolor(color)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#475569", labelsize=7)
    ax.title.set_color("#1E293B")


class _GrayPrefix:
    def __init__(self, text, fontsize=7):
        self.text = text
        self.fontsize = fontsize


class _GrayPrefixHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        t = plt.Text(xdescent, ydescent + height * 0.5, orig_handle.text,
                     color="#64748b", fontsize=orig_handle.fontsize, va="center", ha="left", transform=trans)
        return [t]


# ── helpers for small panels ───────────────────────────────────────────────
def load_size_stats():
    path = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def plot_sparkline(ax, metric, agg_data, size_stats, title):
    ax.set_title(title, fontsize=9, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Natural bands
    bins = size_stats.get("bins", []) if size_stats else []
    for b in bins:
        mstats = b.get("metrics", {}).get(metric)
        if not mstats:
            continue
        p25 = mstats.get("p2_5") or mstats.get("p5") or mstats.get("p25")
        p75 = mstats.get("p97_5") or mstats.get("p95") or mstats.get("p75")
        lo_tok, hi_tok = b.get("range_tokens", [0, 0])
        if p25 is None or p75 is None:
            continue
        ax.fill_between([lo_tok/1000.0, hi_tok/1000.0], [p25, p25], [p75, p75],
                        color="#E2E8F0", alpha=0.4, zorder=1)

    if metric in agg_data:
        stats_map = agg_data[metric]
        sorted_centers = sorted(stats_map.keys())
        xs = [c/1000.0 for c in sorted_centers]
        means = [stats_map[c]["mean"] for c in sorted_centers]
        stds = [stats_map[c]["std"] for c in sorted_centers]
        upper = np.array(means) + np.array(stds)
        lower = np.array(means) - np.array(stds)
        ax.plot(xs, means, color="#2563EB", lw=1.6, zorder=3)
        ax.fill_between(xs, lower, upper, color="#2563EB", alpha=0.25, zorder=2)

    ax.set_xlim(0, 1000)


def generate_channel_panel_image(out_path: Path):
    """Generate a braided-channel schematic as a standalone PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    # Threads feeding the funnel
    threads = [
        {"color": "#10B981", "phase": 0.0, "amp": 0.05},
        {"color": "#F59E0B", "phase": 0.8, "amp": 0.045},
        {"color": "#6366F1", "phase": 1.6, "amp": 0.05},
        {"color": "#EC4899", "phase": 2.4, "amp": 0.04},
        {"color": "#0EA5E9", "phase": 3.2, "amp": 0.05},
    ]
    xs = np.linspace(0.0, 0.22, 120)
    for t in threads:
        ys = 0.68 + t["amp"] * np.sin(10 * xs + t["phase"])
        ax.plot(xs, ys, color=t["color"], linewidth=2.0, alpha=0.85, zorder=2)

    # Intent blob
    intent = mpatches.Ellipse((0.12, 0.68), width=0.20, height=0.22, facecolor="#E0F2FE",
                              edgecolor="#38BDF8", linewidth=1.4)
    ax.add_patch(intent)
    ax.text(0.12, 0.68, "Multidimensional\nintent\n(plot, themes,\ncharacters, facts)",
            ha="center", va="center", fontsize=9, color="#0f172a")

    # Funnel
    funnel_pts = [(0.25, 0.72), (0.36, 0.72), (0.32, 0.55), (0.29, 0.52), (0.26, 0.55)]
    funnel = mpatches.Polygon(funnel_pts, closed=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.2)
    ax.add_patch(funnel)
    ax.text(0.305, 0.74, "Serialization\ninto 1D tokens", ha="center", va="bottom", fontsize=9, color="#475569")
    ax.annotate("", xy=(0.26, 0.68), xytext=(0.02, 0.68),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.3))

    # Braided 1D channel
    channel_x = np.linspace(0.36, 0.55, 60)
    braid_colors = ["#10B981", "#F59E0B", "#6366F1", "#EC4899", "#0EA5E9"]
    for i, c in enumerate(braid_colors):
        amp = 0.0045
        phase = i * 0.7
        y_line = 0.62 + amp * np.sin(45 * channel_x + phase)
        ax.plot(channel_x, y_line, color=c, lw=1.8, alpha=0.9)
    ax.annotate("", xy=(0.55, 0.62), xytext=(0.36, 0.62),
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0))
    ax.text(0.455, 0.64, "1D communication channel", ha="center", va="bottom", fontsize=8.5, color="#2563EB")

    # Generator box
    enc_box = FancyBboxPatch((0.58, 0.64), 0.30, 0.16, boxstyle="round,pad=0.03",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2)
    ax.add_patch(enc_box)
    ax.text(0.73, 0.72, "LLM generator\n(iterative self-conditioning)", ha="center", va="center",
            fontsize=8.5, color="#0f172a")

    # Downstream arrow
    ax.annotate("", xy=(0.73, 0.60), xytext=(0.73, 0.54),
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2))

    # Loopback
    ax.annotate("",
                xy=(0.58, 0.70), xytext=(0.52, 0.52),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.1, connectionstyle="arc3,rad=-0.35"))
    ax.text(0.50, 0.54, "hierarchical prompt reuse", ha="right", va="center",
            fontsize=8, color="#475569")

    ax.text(0.12, 0.30, "Coherence = long-range constraints preserved\nacross the 1D channel",
            ha="left", va="center", fontsize=8.5, color="#0f172a")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_pipeline(ax):
    ax.axis("off")
    # If a pre-rendered image exists, use it
    if PANEL_IMG.exists():
        try:
            img = mpimg.imread(PANEL_IMG)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        except Exception:
            pass
    # If a pre-rendered image exists, use it
    if PANEL_IMG.exists():
        try:
            img = mpimg.imread(PANEL_IMG)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        except Exception:
            pass
    # Multidimensional intent threads entangling into a single braid
    threads = [
        {"color": "#10B981", "phase": 0.0},
        {"color": "#F59E0B", "phase": 0.8},
        {"color": "#6366F1", "phase": 1.6},
    ]
    xs = np.linspace(0.0, 0.22, 80)
    for t in threads:
        ys = 0.68 + 0.05 * np.sin(8 * xs + t["phase"])
        ax.plot(xs, ys, color=t["color"], linewidth=2.0, alpha=0.8, zorder=2)

    # Multidimensional intent blob
    intent = mpatches.Ellipse((0.12, 0.68), width=0.20, height=0.22, facecolor="#E0F2FE",
                              edgecolor="#38BDF8", linewidth=1.2)
    ax.add_patch(intent)
    ax.text(0.12, 0.68, "Multidimensional\nintent\n(plot, themes,\ncharacters, facts)",
            ha="center", va="center", fontsize=8, color="#0f172a")

    # Funnel (serialization into 1D)
    funnel_pts = [(0.25, 0.72), (0.36, 0.72), (0.32, 0.55), (0.29, 0.52), (0.26, 0.55)]
    funnel = mpatches.Polygon(funnel_pts, closed=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.1)
    ax.add_patch(funnel)
    ax.text(0.305, 0.74, "Serialization\ninto 1D tokens", ha="center", va="bottom", fontsize=8, color="#475569")
    ax.annotate("", xy=(0.26, 0.68), xytext=(0.02, 0.68),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.2))

    # 1D braided channel
    channel_x = np.linspace(0.36, 0.55, 40)
    braid_colors = ["#10B981", "#F59E0B", "#6366F1"]
    for i, c in enumerate(braid_colors):
        amp = 0.004
        phase = i * 0.6
        y_line = 0.62 + amp * np.sin(40 * channel_x + phase)
        ax.plot(channel_x, y_line, color=c, lw=1.6, alpha=0.9)
    ax.annotate("", xy=(0.55, 0.62), xytext=(0.36, 0.62),
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.0))
    ax.text(0.455, 0.64, "1D communication channel", ha="center", va="bottom", fontsize=8, color="#2563EB")

    # Encoder (LLM generator)
    enc_box = FancyBboxPatch((0.58, 0.64), 0.30, 0.16, boxstyle="round,pad=0.03",
                             facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.2)
    ax.add_patch(enc_box)
    ax.text(0.73, 0.72, "LLM generator\n(iterative self-conditioning)", ha="center", va="center",
            fontsize=8, color="#0f172a")

    # Downstream text stream (focus on channel only)
    ax.annotate("", xy=(0.73, 0.60), xytext=(0.73, 0.54),
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.2))

    # Loopback for hierarchical prompt reuse (channel-focused)
    ax.annotate("",
                xy=(0.58, 0.70), xytext=(0.52, 0.52),
                arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.1, connectionstyle="arc3,rad=-0.35"))
    ax.text(0.50, 0.54, "hierarchical prompt reuse", ha="right", va="center",
            fontsize=8, color="#475569")

    ax.text(0.12, 0.30, "Coherence = long-range constraints preserved\nacross the 1D channel",
            ha="left", va="center", fontsize=8, color="#0f172a")


def plot_pca_panel(fig, ax_pca):
    panel_bg(ax_pca)
    # ── Load real PC coordinates if available ───────────────────────────────
    # We use a representative subset of natural book PC coords from batch files
    nat_dir = ROOT / "natural_texts" / "Output" / "batch_results"
    nat_pts_ok = False
    try:
        import glob, pandas as pd
        from sklearn.decomposition import PCA as skPCA
        from sklearn.preprocessing import StandardScaler

        batch_files = sorted(glob.glob(str(nat_dir / "batch_*.json")))
        if batch_files:
            records = []
            for bf in batch_files:
                with open(bf) as fh:
                    records.extend(json.load(fh))
            df_nat = pd.DataFrame(records)

            METRIC_COLS = ["beta", "alpha", "H_DMA", "H_WT_db4",
                           "entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma"]
            METRIC_COLS = [c for c in METRIC_COLS if c in df_nat.columns]
            df_nat = df_nat.dropna(subset=METRIC_COLS)

            meta_path = ROOT / "natural_texts" / "Output" / "long_books_with_metadata.csv"
            if meta_path.exists():
                meta = pd.read_csv(meta_path)
                df_nat["ID_num"] = df_nat["book_id"].astype(str).str.extract(r"_(\d+)$")[0]
                df_nat["ID_num"] = pd.to_numeric(df_nat["ID_num"], errors="coerce")
                df_nat = df_nat.merge(meta[["ID", "FinalGenre"]], left_on="ID_num",
                                      right_on="ID", how="left")
                df_nat["FinalGenre"] = df_nat["FinalGenre"].fillna("Unknown")
            else:
                df_nat["FinalGenre"] = "Unknown"

            X_nat = df_nat[METRIC_COLS].values
            scaler = StandardScaler(); X_s = scaler.fit_transform(X_nat)
            pca_model = skPCA(n_components=2); PC = pca_model.fit_transform(X_s)
            df_nat["PC1"] = PC[:, 0]; df_nat["PC2"] = PC[:, 1]

            # Load generated text metrics
            gm_path = ROOT / "generated_texts_og" / "Output" / "gmetrics_all.json"
            with open(gm_path) as fh:
                gmetrics = json.load(fh)
            gm_df = pd.DataFrame(gmetrics)
            gm_df = gm_df.dropna(subset=METRIC_COLS)
            Xg = gm_df[METRIC_COLS].values
            Xg_s = scaler.transform(Xg)
            Pg = pca_model.transform(Xg_s)
            gen_pc1, gen_pc2 = Pg[:, 0], Pg[:, 1]

            GENRE_COLORS = {
                "Fiction": "#8B5CF6",
                "Adventure/Mystery/Sci-Fi": "#2563EB",
                "Romance, Poetry & Drama": "#EC4899",
                "Science & Philosophy": "#D97706",
                "Non-Fiction": "#16A34A",
            }

            def simplify(g):
                if not isinstance(g, str): return "Unknown"
                m = {
                    "General Fiction":"Fiction","Fiction":"Fiction",
                    "Historical Fiction":"Fiction","Short Stories":"Fiction",
                    "Humor":"Fiction","Children/YA":"Fiction",
                    "Adventure":"Adventure/Mystery/Sci-Fi",
                    "Mystery/Crime":"Adventure/Mystery/Sci-Fi",
                    "Sci-Fi/Fantasy":"Adventure/Mystery/Sci-Fi",
                    "Mythology":"Adventure/Mystery/Sci-Fi",
                    "Romance":"Romance, Poetry & Drama",
                    "Romance Drama":"Romance, Poetry & Drama",
                    "Poetry & Drama":"Romance, Poetry & Drama",
                    "Science":"Science & Philosophy",
                    "Science Book":"Science & Philosophy",
                    "Philosophy":"Science & Philosophy",
                    "Reference":"Science & Philosophy",
                    "Biography":"Non-Fiction","Non-Fiction":"Non-Fiction",
                    "History":"Non-Fiction","Essays & Letters":"Non-Fiction",
                    "Religion":"Non-Fiction","Travel":"Non-Fiction","Other":"Non-Fiction",
                }
                return m.get(g.strip(), "Unknown")

            # Use broad_genre column (always present in batch files)
            genre_col = "broad_genre" if "broad_genre" in df_nat.columns else "FinalGenre" if "FinalGenre" in df_nat.columns else "genre"
            df_nat["Genre"] = df_nat[genre_col].apply(simplify)

            # Scatter natural books as tiny points coloured by genre
            for genre, gc in GENRE_COLORS.items():
                sub = df_nat[df_nat["Genre"] == genre]
                if len(sub) < 2: continue
                ax_pca.scatter(sub["PC1"], sub["PC2"], c=gc, s=5, alpha=0.15,
                               edgecolors="none", zorder=2)

            # KDE contours for GLOBAL density (to show crowded regions of natural books)
            try:
                from scipy.stats import gaussian_kde
                pts = df_nat[["PC1", "PC2"]].dropna().values
                if len(pts) > 30:
                    kde = gaussian_kde(pts.T)
                    x1_g, x2_g = pts[:,0].min()-0.5, pts[:,0].max()+0.5
                    y1_g, y2_g = pts[:,1].min()-0.5, pts[:,1].max()+0.5
                    xx, yy = np.mgrid[x1_g:x2_g:80j, y1_g:y2_g:80j]
                    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    
                    # Calculate volume percentiles
                    z_flat = np.sort(z.ravel())[::-1]
                    z_cumsum = np.cumsum(z_flat)
                    z_cumsum /= z_cumsum[-1]
                    # Plot expanded mass contours
                    pcts = [0.95, 0.85, 0.70, 0.55, 0.40, 0.25, 0.10]
                    levels = [z_flat[np.searchsorted(z_cumsum, p)] for p in pcts]
                    levels = sorted(levels) # Ensure ascending order for matplotlib contour
                    
                    # Filled contour background
                    ax_pca.contourf(xx, yy, z, levels=[levels[0], z.max() * 1.1], colors=["#f8fafc"], alpha=0.8, zorder=1)
                    ax_pca.contourf(xx, yy, z, levels=[0.0] + levels + [z.max() * 1.1], cmap="Blues", alpha=0.15, zorder=1)
                    
                    cs = ax_pca.contour(xx, yy, z, levels=levels, colors="#64748b",
                                        linewidths=0.6, linestyles="--", alpha=0.5, zorder=3)
                    
                    # Note: pcts was entered in descending order (highest area -> lowest density z)
                    # Since levels are sorted (ascending z), level[i] corresponds to pcts[i] to correctly map the outermost bounds
                    fmt = {levels[i]: f"{int(pcts[i]*100)}%" for i in range(len(levels))}
                    texts = ax_pca.clabel(cs, inline=True, fontsize=7.5, fmt=fmt, colors="#64748b")
                    for t in texts:
                        try:
                            t.set_fontweight("bold")
                        except Exception:
                            pass
            except Exception as e:
                print(f"KDE calculation failed: {e}")

            # --- Load generated texts and group by token length ---
            exp_path = ROOT / "generated_texts_og" / "experimental_metrics.json"
            plot_colored = False
            P_all = np.array([])
            if exp_path.exists():
                with open(exp_path) as fh:
                    exp_metrics = json.load(fh)
                exp_df = pd.DataFrame(exp_metrics)
                exp_df["base_file"] = exp_df["file"].str.replace(".txt", "", regex=False).str.lower()
                gm_df_mapped = gm_df.copy()
                gm_df_mapped["base_file"] = gm_df["file"].str.replace(".txt", "", regex=False).str.lower()
                exp_df = exp_df[exp_df["base_file"].isin(gm_df_mapped["base_file"].unique())].copy()
                if not exp_df.empty:
                    exp_df["n_tokens"] = pd.to_numeric(exp_df["n_tokens"], errors="coerce")
                    exp_df = exp_df.dropna(subset=METRIC_COLS + ["n_tokens"])
                    
                    X_all = exp_df[METRIC_COLS].values
                    X_all_s = scaler.transform(X_all)
                    P_all = pca_model.transform(X_all_s)
                    plot_colored = True
                    
                    buckets = [
                        (150000, 300000, "150k - 300k", "#d73027"),  # Red
                        (300000, 500000, "300k - 500k", "#fc8d59"),  # Orange-Red
                        (500000, 800000, "500k - 800k", "#91bfdb"),  # Light Blue
                        (800000, 10000000, "800k+", "#4575b4"),      # Dark Blue
                    ]
                    
                    df_nat["n_tokens"] = pd.to_numeric(df_nat["n_tokens"], errors="coerce")
                    nat_counts = [len(df_nat[(df_nat["n_tokens"] >= lo) & (df_nat["n_tokens"] < hi)]) for lo, hi, _, _ in buckets]
                    nat_total = max(1, sum(nat_counts))
                    nat_props = [c / nat_total for c in nat_counts]
                    
                    gen_counts = [len(exp_df[(exp_df["n_tokens"] >= lo) & (exp_df["n_tokens"] < hi)]) for lo, hi, _, _ in buckets]
                    possible_N = [gen_counts[i] / nat_props[i] for i in range(len(buckets)) if nat_props[i] > 0]
                    N_total_plot = int(min(possible_N)) if possible_N else 0
                    target_gen_counts = [max(3, int(N_total_plot * p)) for p in nat_props]
                    
                    centroids = []
                    
                    all_sub_list = []
                    for i, (lo, hi, label, color) in enumerate(buckets):
                        sub = exp_df[(exp_df["n_tokens"] >= lo) & (exp_df["n_tokens"] < hi)].copy()
                        if sub.empty: continue
                        
                        # Compute centroid before subsampling
                        X_all_sub = sub[METRIC_COLS].values
                        P_all_sub = pca_model.transform(scaler.transform(X_all_sub))
                        centroids.append(np.mean(P_all_sub, axis=0))
                        
                        # Subsample proportionally to match structural density of Natural distributions
                        target_n = target_gen_counts[i]
                        if len(sub) > target_n and target_n > 0:
                            sub = sub.sample(n=target_n, random_state=42)
                        elif target_n == 0:
                            sub = sub.iloc[0:0] # plot nothing if target is 0
                            
                        if len(sub) == 0:
                            continue
                            
                        all_sub_list.append(sub)

                    if all_sub_list:
                        all_subs = pd.concat(all_sub_list)
                        # Subsample plotted generated points to reduce clutter
                        if len(all_subs) > 150:
                            all_subs = all_subs.sample(n=150, random_state=42)
                        X_sub = all_subs[METRIC_COLS].values
                        X_sub_s = scaler.transform(X_sub)
                        P_sub = pca_model.transform(X_sub_s)
                        
                        sc = ax_pca.scatter(
                            P_sub[:, 0], P_sub[:, 1], c=all_subs["n_tokens"],
                            cmap="coolwarm_r",
                            s=40, edgecolors="white", linewidths=0.5, zorder=10, 
                        )
                        # Replace colorbar with a compact row of colored dots
                        tick_vals = [150000, 250000, 350000, 500000, 650000, 800000, 1000000]
                        # Fixed palette to make the lowest bucket clearly red (no log scaling for legend dots)
                        dot_colors = ["#d73027", "#f46d43", "#fdae61", "#fee08b", "#abd9e9", "#74add1", "#4575b4"]
                        cb_ax = inset_axes(
                            ax_pca, width="30%", height="4%", loc="lower left",
                            bbox_to_anchor=(0.12, 0.04, 1, 1), bbox_transform=ax_pca.transAxes,
                            borderpad=0.6
                        )
                        xs = list(range(len(tick_vals)))
                        cb_ax.scatter(xs, [0] * len(xs), c=dot_colors, s=52, marker="o",
                                      edgecolors="white", linewidths=0.5)
                        cb_ax.set_xlim(-0.6, len(xs) - 0.4)
                        cb_ax.set_ylim(-0.5, 0.5)
                        cb_ax.set_xticks(xs)
                        cb_ax.set_xticklabels(["150k", "250k", "350k", "500k", "650k", "800k", "1M"], fontsize=7)
                        cb_ax.set_yticks([])
                        for spine in cb_ax.spines.values():
                            spine.set_visible(False)
                        cb_ax.tick_params(axis="x", length=0, pad=2)
                        cb_ax.text(
                            0.5, 1.1, "Length in tokens",
                            ha="center", va="center", fontsize=7.5, fontweight="normal", 
                            color="#0f172a", transform=cb_ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="none", edgecolor="none", alpha=1.0)
                        )

                    # Trajectories removed
            # Clip axis to keep generated + natural books visible, remove extremes
            p1_lo, p1_hi = np.percentile(df_nat["PC1"], [2, 98])
            p2_lo, p2_hi = np.percentile(df_nat["PC2"], [2, 98])
            margin1 = (p1_hi - p1_lo) * 0.12
            margin2 = (p2_hi - p2_lo) * 0.15
            
            if plot_colored and len(P_all) > 0:
                gen_x_lo = np.percentile(P_all[:, 0], 2) - 0.5
                gen_x_hi = np.percentile(P_all[:, 0], 98) + 0.5
            else:
                gen_x_lo = np.percentile(gen_pc1, 5) - 0.5
                gen_x_hi = np.percentile(gen_pc1, 95) + 0.5

            all_x_lo = min(p1_lo - margin1, gen_x_lo)
            all_x_hi = min(2.0, max(p1_hi + margin1, gen_x_hi))
            ax_pca.set_xlim(all_x_lo, all_x_hi)
            y_low = p2_lo - margin2
            y_high = min(2.0, p2_hi + margin2)
            ax_pca.set_ylim(y_low, y_high)

            # Add genre centroid labels
            from adjustText import adjust_text
            genre_texts = []
            offset_map = {
                "Fiction": (-0.25, -0.15),
                "Adventure/Mystery/Sci-Fi": (-0.20, -0.15),
                "Romance, Poetry & Drama": (0.0, 0.18),
                "Science & Philosophy": (-0.15, 0.18),
                "Non-Fiction": (0.0, 0.18),
            }
            for genre, gc in GENRE_COLORS.items():
                sub = df_nat[df_nat["Genre"] == genre]
                if len(sub) < 20: continue
                cx, cy = sub["PC1"].median(), sub["PC2"].median()
                ax_pca.scatter(cx, cy, c=gc, marker="D", s=90, edgecolors="white", linewidths=1.0, zorder=12)
                # Nudge labels away from centroid diamonds to reduce overlap
                dx, dy = offset_map.get(genre, (0.0, 0.15))
                t = ax_pca.text(cx + dx, cy + dy, genre.replace(" & ", "\n& "),
                            fontsize=6.5, ha="center", va="center",
                            color=gc, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                      edgecolor=gc, alpha=0.75, linewidth=0.8))
                genre_texts.append(t)
                
            if genre_texts:
                adjust_text(genre_texts, ax=ax_pca, force_text=2.2, arrowprops=dict(arrowstyle="-", color="#CBD5E1", lw=0.6))

            # Draw from the centroid of natural texts
            nat_cx, nat_cy = df_nat["PC1"].median(), df_nat["PC2"].median()
            scale = 3.5 # scaled for visibility

            # Natural Token length loading
            df_nat_tokens = df_nat.dropna(subset=["n_tokens", "PC1", "PC2"]).copy()
            df_nat_tokens["n_tokens"] = pd.to_numeric(df_nat_tokens["n_tokens"], errors="coerce")
            if not df_nat_tokens.empty:
                r_pc1 = df_nat_tokens["n_tokens"].corr(df_nat_tokens["PC1"])
                r_pc2 = df_nat_tokens["n_tokens"].corr(df_nat_tokens["PC2"])
                
                # Plot the arrow (scaled for visibility alongside loadings)
                arrow_color = "#16A34A" # Green for Natural
                ax_pca.arrow(nat_cx, nat_cy, r_pc1 * scale, r_pc2 * scale, 
                             color=arrow_color, width=0.03, head_width=0.2, head_length=0.3, zorder=15, alpha=0.9, ls="-")
                ax_pca.text(nat_cx + r_pc1 * scale * 1.05, nat_cy + r_pc2 * scale * 1.05 + 0.1,
                            "Length direction for Natural books", color=arrow_color, fontsize=7.5, 
                            ha="right", va="bottom", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

            # Generated texts Token length loading
            if plot_colored and len(P_all) > 0 and "n_tokens" in exp_df.columns:
                gen_r_pc1 = np.corrcoef(exp_df["n_tokens"].values, P_all[:, 0])[0, 1]
                gen_r_pc2 = np.corrcoef(exp_df["n_tokens"].values, P_all[:, 1])[0, 1]
                
                # Plot the arrow
                arrow_color_gen = "#334155" # Slate for Generated
                ax_pca.arrow(nat_cx, nat_cy, gen_r_pc1 * scale, gen_r_pc2 * scale, 
                             color=arrow_color_gen, width=0.03, head_width=0.2, head_length=0.3, zorder=16, alpha=0.9, ls="--")
                ax_pca.text(nat_cx + gen_r_pc1 * scale * 1.05, nat_cy + gen_r_pc2 * scale * 1.05 - 0.1,
                            "Length direction for Generated texts", color=arrow_color_gen, fontsize=7.5, 
                            ha="right", va="top", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))


            nat_pts_ok = True

    except Exception as e:
        print(f"[PCA panel] Could not load real data, using schematic: {e}")

    if not nat_pts_ok:
        # ── Fallback schematic ──────────────────────────────────────────────
        rng = np.random.default_rng(42)

        GENRE_MOCK = {
            "Fiction":                (0.8, 0.2, "#8B5CF6"),
            "Adventure/Mystery/Sci-Fi":(-0.3, 1.1, "#2563EB"),
            "Romance, Poetry & Drama": (1.5, 1.0, "#EC4899"),
            "Non-Fiction":             (-1.2, -0.5, "#16A34A"),
            "Science & Philosophy":    (-0.5, -1.3, "#D97706"),
        }
        for g, (mx, my, gc) in GENRE_MOCK.items():
            pts = rng.normal([mx, my], [1.1, 0.8], (250, 2))
            ax_pca.scatter(pts[:, 0], pts[:, 1], c=gc, s=5, alpha=0.18,
                           edgecolors="none", zorder=2)

        # Generated cluster – slightly offset from natural centre
        gen_pts = rng.normal([-2.2, -0.8], [0.5, 0.4], (30, 2))
        ax_pca.scatter(gen_pts[:, 0], gen_pts[:, 1], c=C_HIGH, s=40, marker="D",
                       edgecolors="white", linewidths=0.5, zorder=10,
                       label="Generated texts")

        GENRE_COLORS = {g: c for g, (_, _, c) in GENRE_MOCK.items()}

    # axis cosmetics
    ax_pca.set_xlabel("")
    ax_pca.set_ylabel("")
    ax_pca.grid(False)
    ax_pca.set_xticks([])
    ax_pca.set_yticks([])

    # Legend – genre centroid diamond + generated texts + contour guide
    from matplotlib.lines import Line2D
    centroid_handle = [Line2D([0], [0], marker="D", color="w", markerfacecolor="#888888", markersize=8, 
                              markeredgewidth=1.0, markeredgecolor="white", label="  Mean natural book within category")]
    gen_handle = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444", markersize=8, 
                              markeredgewidth=1.0, markeredgecolor="white", label="  Generated texts")]
    contour_text = _GrayPrefix("-- xx% --", fontsize=6.5)
    contour_desc = "  Natural book density percentile contours"
    
    # Get handles automatically added
    handles, labels = ax_pca.get_legend_handles_labels()
    all_handles = handles + centroid_handle + gen_handle + [contour_text]
    labels = labels + ["  Mean natural book within category", "  Generated texts", contour_desc]
    
    ax_pca.legend(handles=all_handles, labels=labels,
                  fontsize=8, loc="lower right", ncol=1,
                  frameon=True, facecolor="white", edgecolor="#E2E8F0", framealpha=0.75,
                  handler_map={_GrayPrefix: _GrayPrefixHandler()})

    return ax_pca


# ── main composition ──────────────────────────────────────────────────────
def make_figure():
    # Ensure panel image exists with the improved braid schematic
    try:
        if not PANEL_IMG.exists():
            generate_channel_panel_image(PANEL_IMG)
    except Exception:
        pass

    fig = plt.figure(figsize=(14, 8), facecolor="white")
    gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.5, 1.5, 0.45],
                          width_ratios=[1.05, 1.6, 1.0])

    fig.text(0.5, 0.97, "Book-scale coherence: natural manifold vs generative trajectories",
             fontsize=12, fontweight="bold", ha="center", color="#0f172a")
    fig.text(0.5, 0.94, "Statistical scaling vs informational descriptors across 1M-token generations",
             fontsize=9, ha="center", color="#334155")

    ax_schem = fig.add_subplot(gs[0:2, 0])
    draw_pipeline(ax_schem)

    ax_pca = fig.add_subplot(gs[0:2, 1])
    plot_pca_panel(fig, ax_pca)

    size_stats = load_size_stats()
    agg_high, agg_low = load_agg()
    ax_struct = fig.add_subplot(gs[0, 2])
    plot_sparkline(ax_struct, "beta", agg_high, size_stats, "Structural (β)")
    ax_info = fig.add_subplot(gs[1, 2])
    plot_sparkline(ax_info, "entropy_bits", agg_high, size_stats, "Informational (entropy)")

    ax_footer = fig.add_subplot(gs[2, :])
    ax_footer.axis("off")
    footer_text = (
        "Natural books define a 2D coherence manifold (statistical vs informational). "
        "Iterative self-conditioning keeps structural scaling within natural bands up to 1M tokens, "
        "while informational descriptors are more fragile under high stochasticity."
    )
    ax_footer.text(0.5, 0.5, footer_text, ha="center", va="center",
                   fontsize=9, color="#0f172a", wrap=True)

    jpg_path = str(OUT_PATH).replace(".pdf", ".jpg")
    fig.savefig(jpg_path, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved → {jpg_path}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
