#!/usr/bin/env python3
"""
PCA on SC-analysed natural books combining all feature families:
  - SC metrics  (W2k/W8k/W40k × off_diag_mean/novelty_mean/sim_lag1_mean)
  - Stat/info   (8 canonical scaling/information descriptors)
  - Token       (27 canonical td_/tr_/tp_ descriptors)

Fits on natural books with complete canonical descriptors in the loaded
tables, projects them, and colours points by SC bin (token-size bin).
"""

from __future__ import annotations

import glob
import re
import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.descriptor_schema import (
    SC_FEATURES,
    SC_METRICS,
    SC_WINDOWS,
    STAT_INFO_METRICS,
    TOKEN_METRICS,
    available_features,
)

# ── paths ─────────────────────────────────────────────────────────────────────
NATURAL_SC_PATHS = (
    [Path("/tmp/natural_books_sc_median.json")]
    + sorted(Path("/tmp").glob("natural_sc_extra_*.json"))
)
BATCH_RESULTS_DIR  = ROOT / "natural_texts" / "Output" / "batch_results"
TOKEN_NAT_JSONL    = ROOT / "natural_texts" / "Output" / "gutenberg_benchmark_cache" / "token_naturalness_metrics.jsonl"
TOKEN_PRED_JSONL   = ROOT / "natural_texts" / "Output" / "gutenberg_benchmark_cache" / "token_predictability_metrics.jsonl"
OUTPUT_DIR         = ROOT / "natural_texts" / "Output" / "pca" / "sc_combined"
GENERATED_SC_FILE  = ROOT / "generated_texts_og" / "Output" / "sc_prefix_bins.json"
PROMPTS_DIR        = ROOT / "generated_texts_og" / "prompts"

# token counts at which SC was computed for each bin (from compute_sc_prefix_bins.py)
BIN_TOKEN_TARGETS = {
    "0":  121_140,
    "1":  193_795,
    "2":  353_917,
    "3":  593_869,
    "4": 1_072_000,
}

TOKEN_PREFIXES = ("td_", "tr_", "tp_")

BIN_LABELS = {
    "0": "100–150k tok",
    "1": "150–300k tok",
    "2": "300–500k tok",
    "3": "500–800k tok",
    "4": "800k–1M tok",
}
BIN_COLORS = {
    "0": "#FDE68A",
    "1": "#6EE7B7",
    "2": "#67E8F9",
    "3": "#60A5FA",
    "4": "#1E3A8A",
}

LOADING_FAMILY_COLORS = {
    "SC":         "#9B2226",
    "stat/info":  "#005F73",
    "additional": "#3D405B",
}

MODE_COLORS = {
    "summaries_conditioning":      "#1565C0",
    "basic_conditioning":          "#00897B",
    "extended_conditioning":       "#C62828",
    "markov_rolling_conditioning": "#E65100",
    "minimal_fixed_conditioning":  "#6A1B9A",
    "minimal_rolling_conditioning":"#E65100",
}
MODE_LABELS = {
    "summaries_conditioning":      "summaries",
    "basic_conditioning":          "basic",
    "extended_conditioning":       "extended",
    "markov_rolling_conditioning": "markov",
    "minimal_fixed_conditioning":  "minimal",
    "minimal_rolling_conditioning":"markov",
}
MODE_ORDER = [
    "summaries_conditioning", "basic_conditioning", "extended_conditioning",
    "markov_rolling_conditioning", "minimal_fixed_conditioning",
]


# ── loaders ────────────────────────────────────────────────────────────────────

def load_sc_books() -> pd.DataFrame:
    rows = []
    for path in NATURAL_SC_PATHS:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        for book_id, rec in data.items():
            row = {
                "book_id": int(book_id),
                "title":   rec.get("title", ""),
                "bin":     str(rec.get("bin", "?")),
                "n_tokens": rec.get("n_tokens"),
            }
            for w in SC_WINDOWS:
                wrec = rec.get(w, {}) or {}
                for m in SC_METRICS:
                    row[f"{w}_{m}"] = wrec.get(m)
            rows.append(row)
    return pd.DataFrame(rows)


def load_batch_metrics() -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(BATCH_RESULTS_DIR)):
        if fname.startswith("batch_") and fname.endswith(".json"):
            data = json.loads((BATCH_RESULTS_DIR / fname).read_text(encoding="utf-8"))
            rows.extend(data)
    df = pd.DataFrame(rows)
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
    return df


def load_token_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
    return df


def feature_family(name: str) -> str:
    if any(name.startswith(p) for p in TOKEN_PREFIXES):
        return "additional"
    if name in SC_FEATURES:
        return "SC"
    return "stat/info"


# ── generated book loader ─────────────────────────────────────────────────────

GENERATED_TOKEN_METRICS_CACHE = Path("/tmp/sc_generated_prefix_token_metrics.json")


def build_marker_sidecar(folder: Path) -> list[dict]:
    """Reconstruct sidecar-like records (chunk_id, accepted_attempt, n_tokens)
    from chunk_NNN_accepted.txt markers + each accepted chunk's
    accumulated_metrics, for runs that have no token_naturalness sidecar."""
    chunks: dict[int, list[int]] = {}
    for f in folder.iterdir():
        m = re.match(r"chunk_(\d+)_attempt_(\d+)\.txt$", f.name)
        if m:
            chunks.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    recs = []
    for cid in sorted(chunks):
        marker = folder / f"chunk_{cid:03d}_accepted.txt"
        att = None
        if marker.exists():
            try:
                att = json.loads(marker.read_text()).get("attempt")
            except Exception:
                att = None
        if att is None:
            att = 1 if 1 in chunks[cid] else min(chunks[cid])
        p = folder / f"chunk_{cid:03d}_attempt_{att}.txt"
        try:
            acc = (json.loads(p.read_text(encoding="utf-8")).get("accumulated_metrics") or {})
            nt = acc.get("n_tokens")
        except Exception:
            nt = None
        if nt is not None:
            recs.append({"chunk_id": cid, "accepted_attempt": att, "n_tokens": nt})
    return recs


def load_generated_rows(features: list[str]) -> pd.DataFrame:
    """
    For each (run, bin) in sc_prefix_bins.json, combine:
      - SC metrics from sc_prefix_bins.json
      - stat/info from accumulated_metrics of the nearest sidecar chunk
      - td_/tr_/tp_ from precomputed cache (compute_generated_prefix_token_metrics.py)
        with sidecar as fallback for any features not in the cache
    Returns one row per (run_id, bin) with all feature columns.
    """
    if not GENERATED_SC_FILE.exists():
        print(f"WARNING: {GENERATED_SC_FILE} not found; skipping generated projection.")
        return pd.DataFrame()

    with open(GENERATED_SC_FILE) as f:
        sc_data = json.load(f)

    token_cache: dict = {}
    if GENERATED_TOKEN_METRICS_CACHE.exists():
        with open(GENERATED_TOKEN_METRICS_CACHE) as f:
            token_cache = json.load(f)
    else:
        print(f"WARNING: {GENERATED_TOKEN_METRICS_CACHE} not found; td_/tr_/tp_ from sidecar only.")

    rows = []
    missing_counts: dict[str, int] = {}

    for run_id, run_rec in sc_data.items():
        folder = PROMPTS_DIR / run_id
        if not folder.is_dir():
            continue
        mode   = run_rec.get("conditioning_mode", "unknown")
        if mode == "minimal_rolling_conditioning":
            mode = "markov_rolling_conditioning"

        sidecar_path = folder / "token_naturalness_by_chunk.jsonl"
        if sidecar_path.exists():
            sidecar_recs = [json.loads(l) for l in sidecar_path.read_text().splitlines() if l.strip()]
        else:
            sidecar_recs = build_marker_sidecar(folder)  # no-sidecar runs (marker fallback)
        if not sidecar_recs:
            continue

        for bin_id, bin_sc in run_rec.get("bins", {}).items():
            target_tokens = BIN_TOKEN_TARGETS.get(bin_id)
            if target_tokens is None:
                continue

            nearest = min(sidecar_recs,
                          key=lambda r: abs(float(r.get("n_tokens", 0)) - target_tokens))
            chunk_id = int(nearest["chunk_id"])
            attempt  = int(nearest.get("accepted_attempt") or 1)
            attempt_path = folder / f"chunk_{chunk_id:03d}_attempt_{attempt}.txt"
            if not attempt_path.exists():
                continue
            try:
                att_data = json.loads(attempt_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            acc = att_data.get("accumulated_metrics") or {}

            row = {
                "run_id":            run_id,
                "conditioning_mode": mode,
                "bin":               str(bin_id),
                "n_tokens":          float(nearest.get("n_tokens", target_tokens)),
            }
            # SC features
            for w in SC_WINDOWS:
                wrec = bin_sc.get(w, {}) or {}
                for m in SC_METRICS:
                    row[f"{w}_{m}"] = wrec.get(m)
            # stat/info from accumulated_metrics
            for k, v in acc.items():
                row[k] = v
            # td_/tr_/tp_: precomputed cache first, sidecar as fallback
            cached = token_cache.get(run_id, {}).get(bin_id, {})
            for k, v in cached.items():
                if k.startswith(TOKEN_PREFIXES):
                    row[k] = v
            for k, v in nearest.items():
                if k.startswith(TOKEN_PREFIXES) and k not in row:
                    row[k] = v

            ok = True
            for feat in features:
                val = row.get(feat)
                if val is None or (isinstance(val, float) and not math.isfinite(val)):
                    missing_counts[feat] = missing_counts.get(feat, 0) + 1
                    ok = False
                    break
            if ok:
                rows.append(row)

    if missing_counts:
        top = sorted(missing_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  Rows dropped due to missing features (top blockers): {top}")

    return pd.DataFrame(rows)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load SC rows
    sc_df = load_sc_books()
    print(f"SC books loaded: {len(sc_df)}")

    # 2. Load stat/info from batch_results
    batch_df = load_batch_metrics()
    keep_stat = ["book_id"] + [c for c in STAT_INFO_METRICS if c in batch_df.columns]
    batch_df = batch_df[keep_stat].copy()

    # 3. Load additional token metrics
    tn_df   = load_token_jsonl(TOKEN_NAT_JSONL)
    tp_df   = load_token_jsonl(TOKEN_PRED_JSONL)
    tn_cols = [c for c in TOKEN_METRICS if c in tn_df.columns]
    tp_cols = [c for c in TOKEN_METRICS if c in tp_df.columns]
    tn_df   = tn_df[["book_id"] + tn_cols]
    tp_df   = tp_df[["book_id"] + tp_cols]

    # 4. Merge everything on book_id
    df = sc_df.merge(batch_df, on="book_id", how="inner")
    df = df.merge(tn_df,    on="book_id", how="inner")
    df = df.merge(tp_df,    on="book_id", how="inner")
    print(f"After merge: {len(df)} rows")

    # 5. Build feature matrix — drop near-zero-variance and rows with any NaN
    all_features = available_features(df.columns)

    for col in all_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    clean = df.dropna(subset=all_features).reset_index(drop=True)
    print(f"Complete rows: {len(clean)}")

    # drop near-zero-variance
    stds = clean[all_features].std()
    dropped = [f for f in all_features if stds[f] < 1e-9]
    features = [f for f in all_features if stds[f] >= 1e-9]
    if dropped:
        print(f"Dropped zero-variance: {dropped}")
    print(f"Features used: {len(features)}")

    X = clean[features].to_numpy(dtype=float)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(6, len(features), len(clean))
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.tolist()
    print(f"Explained variance: {[round(v*100,1) for v in explained]}%")

    score_df = clean[["book_id", "title", "bin", "n_tokens"]].copy()
    for i in range(n_components):
        score_df[f"PC{i+1}"] = scores[:, i]
    score_df.to_csv(OUTPUT_DIR / "sc_combined_pca_scores.csv", index=False)

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    loadings_df.index.name = "feature"
    loadings_df.to_csv(OUTPUT_DIR / "sc_combined_pca_loadings.csv")

    summary = {
        "n_books": int(len(clean)),
        "n_features": len(features),
        "features": features,
        "dropped_zero_variance": dropped,
        "explained_variance_ratio": explained,
        "bins": {k: int((clean["bin"] == k).sum()) for k in sorted(clean["bin"].unique())},
    }
    (OUTPUT_DIR / "sc_combined_pca_summary.json").write_text(json.dumps(summary, indent=2))

    # ── plots ─────────────────────────────────────────────────────────────────

    def scatter_by_bin(ax, pc_x, pc_y, alpha=0.6, s=26):
        xcol, ycol = f"PC{pc_x}", f"PC{pc_y}"
        for bin_id in sorted(score_df["bin"].unique()):
            sub = score_df[score_df["bin"] == bin_id]
            ax.scatter(
                sub[xcol], sub[ycol],
                s=s, c=BIN_COLORS.get(bin_id, "gray"),
                alpha=alpha, edgecolors="none",
                label=f"{BIN_LABELS.get(bin_id, bin_id)} (n={len(sub)})",
            )
        # mean trajectory
        means = (
            score_df.assign(bin_num=pd.to_numeric(score_df["bin"], errors="coerce"))
            .dropna(subset=["bin_num"])
            .groupby("bin_num")[[xcol, ycol]]
            .mean()
            .sort_index()
        )
        if len(means) > 1:
            ax.plot(means[xcol], means[ycol],
                    color="#111827", lw=2.0, marker="*", markersize=9,
                    alpha=0.9, label="Mean trajectory", zorder=5)
        ax.set_xlabel(f"PC{pc_x} ({explained[pc_x-1]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y} ({explained[pc_y-1]*100:.1f}%)")
        ax.grid(True, alpha=0.22)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.axvline(0, color="#aaa", lw=0.6)

    # scores PC1xPC2
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter_by_bin(ax, 1, 2)
    ax.legend(framealpha=0.9, fontsize=8, loc="best")
    ax.set_title(
        f"Natural books (n={len(clean)}) — combined SC + stat/info + additional PCA\n"
        f"{len(features)} features · PC1={explained[0]*100:.1f}% · PC2={explained[1]*100:.1f}%"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sc_combined_pca_scores_pc1_pc2.jpg", dpi=200)
    plt.close(fig)

    # scores PC3xPC4
    if n_components >= 4:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter_by_bin(ax, 3, 4)
        ax.legend(framealpha=0.9, fontsize=8, loc="best")
        ax.set_title(
            f"Natural books (n={len(clean)}) — combined SC + stat/info + additional PCA\n"
            f"{len(features)} features · PC3={explained[2]*100:.1f}% · PC4={explained[3]*100:.1f}%"
        )
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "sc_combined_pca_scores_pc3_pc4.jpg", dpi=200)
        plt.close(fig)

    # loadings PC1xPC2
    def plot_loadings(pc_x, pc_y, out_path):
        xcol, ycol = f"PC{pc_x}", f"PC{pc_y}"
        fig, ax = plt.subplots(figsize=(9, 8))
        for feat, row in loadings_df.iterrows():
            family = feature_family(feat)
            color  = LOADING_FAMILY_COLORS[family]
            ax.scatter(row[xcol], row[ycol], color=color, s=55, alpha=0.82)
            ax.text(row[xcol] + 0.004, row[ycol] + 0.004, feat, fontsize=7, color=color)
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=l) for l, c in LOADING_FAMILY_COLORS.items()]
        ax.legend(handles=handles, frameon=False, fontsize=8)
        ax.axhline(0, color="#aaa", lw=0.6)
        ax.axvline(0, color="#aaa", lw=0.6)
        ax.set_xlabel(f"PC{pc_x} ({explained[pc_x-1]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y} ({explained[pc_y-1]*100:.1f}%)")
        ax.set_title(f"PCA loadings (PC{pc_x} vs PC{pc_y}) — {len(features)} features")
        ax.grid(True, alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    plot_loadings(1, 2, OUTPUT_DIR / "sc_combined_pca_loadings_pc1_pc2.jpg")
    if n_components >= 4:
        plot_loadings(3, 4, OUTPUT_DIR / "sc_combined_pca_loadings_pc3_pc4.jpg")

    # biplot PC1xPC2
    def plot_biplot(pc_x, pc_y, out_path, top_n=16):
        xcol, ycol = f"PC{pc_x}", f"PC{pc_y}"
        fig, ax = plt.subplots(figsize=(11, 9))
        scatter_by_bin(ax, pc_x, pc_y, alpha=0.45, s=22)

        # scale loadings to ~18% of score spread
        score_arr = score_df[[xcol, ycol]].to_numpy(dtype=float)
        spread    = np.nanpercentile(np.abs(score_arr), 98, axis=0)
        load_arr  = loadings_df[[xcol, ycol]].to_numpy(dtype=float)
        max_load  = np.sqrt((load_arr**2).sum(axis=1)).max()
        scale     = 0.18 * spread.min() / max_load if max_load > 0 else 1.0

        # show only top_n by loading magnitude
        magnitudes = np.sqrt(loadings_df[xcol]**2 + loadings_df[ycol]**2)
        top_feats  = magnitudes.nlargest(top_n).index

        for feat in top_feats:
            row   = loadings_df.loc[feat]
            family = feature_family(feat)
            color  = LOADING_FAMILY_COLORS[family]
            x, y   = float(row[xcol]) * scale, float(row[ycol]) * scale
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.3,
                                        alpha=0.8, shrinkA=0, shrinkB=0,
                                        mutation_scale=7))
            ax.text(x * 1.07, y * 1.07, feat, fontsize=7, color=color)

        ax.axhline(0, color="#aaa", lw=0.6)
        ax.axvline(0, color="#aaa", lw=0.6)
        ax.set_xlabel(f"PC{pc_x} ({explained[pc_x-1]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y} ({explained[pc_y-1]*100:.1f}%)")
        ax.set_title(
            f"Biplot PC{pc_x}×PC{pc_y} — SC + stat/info + additional\n"
            f"Arrows: top {top_n} features by loading magnitude"
        )
        ax.grid(True, alpha=0.22)
        ax.legend(framealpha=0.9, fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

    plot_biplot(1, 2, OUTPUT_DIR / "sc_combined_pca_biplot_pc1_pc2.jpg")
    if n_components >= 4:
        plot_biplot(3, 4, OUTPUT_DIR / "sc_combined_pca_biplot_pc3_pc4.jpg")

    # ── generated book projection ─────────────────────────────────────────────
    gen_df = load_generated_rows(features)
    if gen_df.empty:
        print("No generated rows could be projected.")
    else:
        X_gen    = gen_df[features].to_numpy(dtype=float)
        X_gen_sc = scaler.transform(X_gen)
        gen_scores = pca.transform(X_gen_sc)
        for i in range(n_components):
            gen_df[f"PC{i+1}"] = gen_scores[:, i]
        gen_df.to_csv(OUTPUT_DIR / "sc_combined_pca_generated_scores.csv", index=False)
        print(f"Generated rows projected: {len(gen_df)}  "
              f"(runs={gen_df['run_id'].nunique()}, bins={sorted(gen_df['bin'].unique())})")

        def plot_with_generated(pc_x, pc_y, out_path):
            xcol, ycol = f"PC{pc_x}", f"PC{pc_y}"
            fig, ax = plt.subplots(figsize=(12, 9))

            # natural background
            scatter_by_bin(ax, pc_x, pc_y, alpha=0.35, s=18)

            # generated trajectories
            modes_present = [m for m in MODE_ORDER if m in gen_df["conditioning_mode"].values]
            for mode in modes_present:
                color = MODE_COLORS.get(mode, "#333")
                label = MODE_LABELS.get(mode, mode)
                sub   = gen_df[gen_df["conditioning_mode"] == mode]

                # individual run trajectories (thin, semi-transparent)
                for _, run_sub in sub.groupby("run_id"):
                    run_sub = run_sub.sort_values("n_tokens")
                    ax.plot(run_sub[xcol], run_sub[ycol],
                            color=color, lw=0.9, alpha=0.28, zorder=6)

                # mean trajectory per bin
                mean_traj = (
                    sub.assign(bin_num=pd.to_numeric(sub["bin"], errors="coerce"))
                    .dropna(subset=["bin_num"])
                    .groupby("bin_num")[[xcol, ycol]]
                    .mean()
                    .sort_index()
                )
                if len(mean_traj) > 1:
                    ax.plot(mean_traj[xcol], mean_traj[ycol],
                            color=color, lw=2.2, alpha=0.9,
                            marker="o", markersize=5,
                            label=f"{label} (n={sub['run_id'].nunique()} runs)",
                            zorder=8)
                    # diamond at last bin
                    ax.scatter(mean_traj[xcol].iloc[-1], mean_traj[ycol].iloc[-1],
                               color=color, marker="D", s=75,
                               edgecolors="black", linewidths=0.5, zorder=9)

            ax.axhline(0, color="#aaa", lw=0.6)
            ax.axvline(0, color="#aaa", lw=0.6)
            ax.set_xlabel(f"PC{pc_x} ({explained[pc_x-1]*100:.1f}%)")
            ax.set_ylabel(f"PC{pc_y} ({explained[pc_y-1]*100:.1f}%)")
            ax.set_title(
                f"Generated books projected onto natural SC+all-metrics PCA\n"
                f"PC{pc_x}×PC{pc_y} — {len(features)} features — "
                f"{gen_df['run_id'].nunique()} generated runs × up to 5 bins each"
            )
            ax.grid(True, alpha=0.22)
            handles, labels = ax.get_legend_handles_labels()
            dedup = {}
            for h, l in zip(handles, labels):
                dedup.setdefault(l, h)
            ax.legend(dedup.values(), dedup.keys(),
                      framealpha=0.9, fontsize=8, loc="best", ncol=2)
            fig.tight_layout()
            fig.savefig(out_path, dpi=200)
            plt.close(fig)
            print(f"Saved {out_path}")

        plot_with_generated(1, 2, OUTPUT_DIR / "sc_combined_pca_with_generated_pc1_pc2.jpg")
        if n_components >= 4:
            plot_with_generated(3, 4, OUTPUT_DIR / "sc_combined_pca_with_generated_pc3_pc4.jpg")

    print(f"\nAll outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
