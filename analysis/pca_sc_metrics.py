#!/usr/bin/env python3
"""
PCA synthesis for semantic-coherence (SC) metrics.

Inputs:
  - generated_texts_og/Output/sc_prefix_bins.json
  - natural SC JSON files produced by compute_sc_natural_*.py

By default, the script robust-standardizes each SC feature within token-size bin
using the natural reference rows:

    z = (value - median_natural_bin) / IQR_natural_bin

Then it fits PCA on natural rows and projects generated prefix-bin rows.

Use ``--standardization global`` to preserve the natural size trajectory by
standardizing each feature once across all natural rows instead of recentering
each token-size bin separately.

Use ``--standardization none`` to fit PCA directly on raw SC feature values
without robust pre-standardization. PCA still centers features as part of the
covariance calculation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENERATED_SC = ROOT / "generated_texts_og" / "Output" / "sc_prefix_bins.json"
DEFAULT_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "sc"
DEFAULT_GLOBAL_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "sc_global"
DEFAULT_RAW_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "sc_raw"
DEFAULT_NATURAL_MEDIAN_SC = Path("/tmp/natural_books_sc_median.json")
DEFAULT_NATURAL_EXTRA_GLOB = "/tmp/natural_sc_extra_*.json"

WINDOWS = ["W2k", "W8k", "W40k"]
SC_METRICS = ["off_diag_mean", "novelty_mean", "sim_lag1_mean"]
FEATURES = [f"{window}_{metric}" for window in WINDOWS for metric in SC_METRICS]

MODE_LABELS = {
    "natural": "Natural",
    "summaries_conditioning": "Summaries",
    "extended_conditioning": "Extended/full",
    "extended_no_fixed_conditioning": "Extended no fixed",
    "basic_conditioning": "Basic",
    "minimal_fixed_conditioning": "Minimal fixed",
    "markov_rolling_conditioning": "Markov rolling",
    "minimal_rolling_conditioning": "Markov rolling",
}

MODE_COLORS = {
    "natural": "#6B7280",
    "summaries_conditioning": "#1f77b4",
    "extended_conditioning": "#d62728",
    "extended_no_fixed_conditioning": "#00897B",
    "basic_conditioning": "#2ca02c",
    "minimal_fixed_conditioning": "#9467bd",
    "markov_rolling_conditioning": "#8c564b",
    "minimal_rolling_conditioning": "#8c564b",
}
NATURAL_TRAJECTORY_COLOR = "#111827"
NATURAL_BIN_LABELS = {
    "0": "Natural 100-150k",
    "1": "Natural 150-300k",
    "2": "Natural 300-500k",
    "3": "Natural 500-800k",
    "4": "Natural 800k-1M",
}
NATURAL_BIN_COLORS = {
    "0": "#FDE68A",
    "1": "#A7F3D0",
    "2": "#67E8F9",
    "3": "#60A5FA",
    "4": "#1E3A8A",
}
BIPLOT_LOADING_RADIUS_FRACTION = 0.18
BIPLOT_ARROW_MUTATION_SCALE = 7

MODE_ORDER = [
    "summaries_conditioning",
    "extended_conditioning",
    "extended_no_fixed_conditioning",
    "basic_conditioning",
    "minimal_fixed_conditioning",
    "markov_rolling_conditioning",
]


@dataclass
class PCAFit:
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    mean_: np.ndarray
    n_components_: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) @ self.components_.T


def default_natural_sc_paths() -> list[Path]:
    paths = []
    if DEFAULT_NATURAL_MEDIAN_SC.exists():
        paths.append(DEFAULT_NATURAL_MEDIAN_SC)
    paths.extend(sorted(Path("/tmp").glob("natural_sc_extra_*.json")))
    return paths


def finite_float(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def normalize_mode(mode: str) -> str:
    if mode == "minimal_rolling_conditioning":
        return "markov_rolling_conditioning"
    return mode


def load_natural_rows(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        if not path.exists():
            print(f"WARNING: natural SC file missing, skipping: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        for book_id, rec in data.items():
            bin_id = rec.get("bin")
            if bin_id is None:
                continue
            row = {
                "source": "natural",
                "id": str(book_id),
                "run_id": str(book_id),
                "conditioning_mode": "natural",
                "label": str(rec.get("title") or book_id),
                "bin": str(bin_id),
                "n_tokens": rec.get("n_tokens"),
            }
            for window in WINDOWS:
                wrec = rec.get(window, {})
                for metric in SC_METRICS:
                    row[f"{window}_{metric}"] = finite_float(wrec.get(metric))
            rows.append(row)
    if not rows:
        raise SystemExit("No natural SC rows found.")
    return pd.DataFrame(rows)


def load_generated_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Generated SC file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    rows = []
    for run_id, rec in data.items():
        mode = normalize_mode(str(rec.get("conditioning_mode") or "unknown"))
        for bin_id, bin_rec in rec.get("bins", {}).items():
            row = {
                "source": "generated",
                "id": f"{run_id}::bin{bin_id}",
                "run_id": run_id,
                "conditioning_mode": mode,
                "label": run_id,
                "bin": str(bin_id),
                "n_tokens": None,
            }
            for window in WINDOWS:
                wrec = bin_rec.get(window, {})
                for metric in SC_METRICS:
                    row[f"{window}_{metric}"] = finite_float(wrec.get(metric))
            rows.append(row)
    if not rows:
        raise SystemExit(f"No generated SC rows found in {path}")
    return pd.DataFrame(rows)


def complete_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for feature in FEATURES:
        out[feature] = pd.to_numeric(out[feature], errors="coerce")
    return out.dropna(subset=FEATURES).reset_index(drop=True)


def robust_bin_standardize(
    natural_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    iqr_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generated_bins = set(generated_df["bin"].astype(str))
    natural_df = natural_df[natural_df["bin"].astype(str).isin(generated_bins)].copy()
    if natural_df.empty:
        raise SystemExit("No natural rows overlap generated prefix bins.")

    stats_rows = []
    nat_z = natural_df.copy()
    gen_z = generated_df.copy()

    for bin_id in sorted(generated_bins, key=lambda value: int(value) if str(value).isdigit() else str(value)):
        mask_nat = nat_z["bin"].astype(str) == str(bin_id)
        mask_gen = gen_z["bin"].astype(str) == str(bin_id)
        if not mask_nat.any():
            print(f"WARNING: no natural rows for bin {bin_id}; generated rows in this bin will be dropped.")
            gen_z = gen_z[~mask_gen].copy()
            continue
        for feature in FEATURES:
            vals = natural_df.loc[natural_df["bin"].astype(str) == str(bin_id), feature].astype(float)
            median = float(vals.median())
            q25 = float(vals.quantile(0.25))
            q75 = float(vals.quantile(0.75))
            iqr = q75 - q25
            std = float(vals.std(ddof=0))
            scale = iqr if math.isfinite(iqr) and abs(iqr) > iqr_epsilon else std
            if not math.isfinite(scale) or abs(scale) <= iqr_epsilon:
                scale = 1.0
            nat_z.loc[mask_nat, feature] = (nat_z.loc[mask_nat, feature].astype(float) - median) / scale
            gen_z.loc[mask_gen, feature] = (gen_z.loc[mask_gen, feature].astype(float) - median) / scale
            stats_rows.append(
                {
                    "bin": bin_id,
                    "feature": feature,
                    "natural_n": int(vals.size),
                    "median": median,
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "std": std,
                    "scale_used": scale,
                }
            )

    return (
        complete_feature_rows(nat_z),
        complete_feature_rows(gen_z),
        pd.DataFrame(stats_rows),
    )


def robust_global_standardize(
    natural_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    iqr_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats_rows = []
    nat_z = natural_df.copy()
    gen_z = generated_df.copy()

    for feature in FEATURES:
        vals = natural_df[feature].astype(float)
        median = float(vals.median())
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        iqr = q75 - q25
        std = float(vals.std(ddof=0))
        scale = iqr if math.isfinite(iqr) and abs(iqr) > iqr_epsilon else std
        if not math.isfinite(scale) or abs(scale) <= iqr_epsilon:
            scale = 1.0

        nat_z[feature] = (nat_z[feature].astype(float) - median) / scale
        gen_z[feature] = (gen_z[feature].astype(float) - median) / scale
        stats_rows.append(
            {
                "bin": "all",
                "feature": feature,
                "natural_n": int(vals.size),
                "median": median,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "std": std,
                "scale_used": scale,
            }
        )

    return (
        complete_feature_rows(nat_z),
        complete_feature_rows(gen_z),
        pd.DataFrame(stats_rows),
    )


def no_standardize(
    natural_df: pd.DataFrame,
    generated_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats_rows = []
    for feature in FEATURES:
        vals = natural_df[feature].astype(float)
        stats_rows.append(
            {
                "bin": "all",
                "feature": feature,
                "natural_n": int(vals.size),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "std": float(vals.std(ddof=0)),
                "scale_used": 1.0,
            }
        )
    return (
        complete_feature_rows(natural_df),
        complete_feature_rows(generated_df),
        pd.DataFrame(stats_rows),
    )


def equal_bin_weights(natural_z: pd.DataFrame) -> np.ndarray:
    bin_counts = natural_z["bin"].astype(str).value_counts().to_dict()
    return natural_z["bin"].astype(str).map(lambda bin_id: 1.0 / bin_counts[bin_id]).to_numpy(dtype=float)


def fit_weighted_pca(values: np.ndarray, weights: np.ndarray, n_components: int) -> PCAFit:
    if values.ndim != 2 or values.shape[0] < 2:
        raise SystemExit("Need at least 2 natural rows for PCA.")
    n_components = min(n_components, values.shape[0], values.shape[1])
    if n_components < 2:
        raise SystemExit("Need at least 2 PCA components; add more natural SC rows.")

    weights = np.asarray(weights, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones(values.shape[0], dtype=float)

    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    components = eigvecs[:, :n_components].T
    explained = eigvals[:n_components]
    total = float(eigvals.sum())
    ratio = explained / total if total > 0 else np.zeros_like(explained)
    return PCAFit(
        components_=components,
        explained_variance_=explained,
        explained_variance_ratio_=ratio,
        mean_=mean,
        n_components_=n_components,
    )


def fit_project_pca(
    natural_z: pd.DataFrame,
    generated_z: pd.DataFrame,
    n_components: int = 4,
    balance_bins: bool = True,
):
    x_nat = natural_z[FEATURES].to_numpy(dtype=float)
    x_gen = generated_z[FEATURES].to_numpy(dtype=float)
    weights = equal_bin_weights(natural_z) if balance_bins else np.ones(x_nat.shape[0], dtype=float)
    pca = fit_weighted_pca(x_nat, weights, n_components=n_components)
    nat_scores = pca.transform(x_nat)
    gen_scores = pca.transform(x_gen)
    return pca, nat_scores, gen_scores


def attach_scores(df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for i in range(scores.shape[1]):
        out[f"PC{i + 1}"] = scores[:, i]
    return out


def write_loadings(pca: PCAFit, output_dir: Path) -> pd.DataFrame:
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    loadings.index.name = "feature"
    loadings.to_csv(output_dir / "sc_pca_loadings.csv")
    return loadings


def rgba_with_alpha(color: str, alpha: float):
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)


def plot_natural_scores_by_bin(
    ax,
    natural: pd.DataFrame,
    xcol: str,
    ycol: str,
    *,
    point_size: float,
    alpha: float,
    zorder: int,
) -> None:
    for bin_id in sorted(natural["bin"].astype(str).unique(), key=lambda value: int(value) if value.isdigit() else value):
        sub = natural[natural["bin"].astype(str) == bin_id]
        if sub.empty:
            continue
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=point_size,
            c=NATURAL_BIN_COLORS.get(bin_id, MODE_COLORS["natural"]),
            alpha=alpha,
            edgecolors="none",
            label=f"{NATURAL_BIN_LABELS.get(bin_id, f'Natural bin {bin_id}')} (n={len(sub)})",
            zorder=zorder,
        )


def plot_scores(scores: pd.DataFrame, output_path: Path, pcx: int, pcy: int, explained: list[float]) -> None:
    xcol = f"PC{pcx}"
    ycol = f"PC{pcy}"
    if xcol not in scores or ycol not in scores:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    natural = scores[scores["source"] == "natural"]
    generated = scores[scores["source"] == "generated"]

    plot_natural_scores_by_bin(ax, natural, xcol, ycol, point_size=42, alpha=0.45, zorder=1)
    plot_natural_mean_trajectory(ax, natural, xcol, ycol, zorder=4)

    for mode in MODE_ORDER:
        sub = generated[generated["conditioning_mode"] == mode]
        if sub.empty:
            continue
        color = MODE_COLORS[mode]
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=26,
            facecolors=[rgba_with_alpha(color, 0.28)],
            edgecolors=color,
            linewidths=0.75,
            label=f"{MODE_LABELS[mode]} rows",
            zorder=3,
        )
        mean_by_bin = (
            sub.assign(bin_num=pd.to_numeric(sub["bin"], errors="coerce"))
            .dropna(subset=["bin_num"])
            .groupby("bin_num")[[xcol, ycol]]
            .mean()
            .sort_index()
        )
        if len(mean_by_bin) > 1:
            ax.plot(
                mean_by_bin[xcol],
                mean_by_bin[ycol],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=f"{MODE_LABELS[mode]} mean trajectory",
                zorder=5,
            )
            ax.scatter(
                mean_by_bin[xcol].iloc[-1],
                mean_by_bin[ycol].iloc[-1],
                color=color,
                marker="D",
                s=70,
                edgecolors="black",
                linewidths=0.4,
                zorder=6,
            )

    ax.axhline(0, color="#999999", linewidth=0.7, alpha=0.45)
    ax.axvline(0, color="#999999", linewidth=0.7, alpha=0.45)
    ax.set_xlabel(f"{xcol} ({explained[pcx - 1] * 100:.1f}% natural variance)")
    ax.set_ylabel(f"{ycol} ({explained[pcy - 1] * 100:.1f}% natural variance)")
    ax.set_title(f"SC PCA Projection ({xcol} vs {ycol})")
    ax.grid(True, alpha=0.22)
    deduplicate_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_natural_mean_trajectory(ax, natural: pd.DataFrame, xcol: str, ycol: str, zorder: int = 3) -> None:
    mean_by_bin = (
        natural.assign(bin_num=pd.to_numeric(natural["bin"], errors="coerce"))
        .dropna(subset=["bin_num"])
        .groupby("bin_num")[[xcol, ycol]]
        .mean()
        .sort_index()
    )
    if len(mean_by_bin) <= 1:
        return
    ax.plot(
        mean_by_bin[xcol],
        mean_by_bin[ycol],
        color=NATURAL_TRAJECTORY_COLOR,
        linewidth=2.2,
        marker="*",
        markersize=9,
        alpha=0.92,
        label="Natural mean trajectory",
        zorder=zorder,
    )
    ax.scatter(
        mean_by_bin[xcol].iloc[-1],
        mean_by_bin[ycol].iloc[-1],
        color=NATURAL_TRAJECTORY_COLOR,
        marker="D",
        s=82,
        edgecolors="white",
        linewidths=0.6,
        zorder=zorder + 1,
    )


def feature_family(feature: str) -> str:
    if "off_diag_mean" in feature:
        return "global similarity"
    if "novelty_mean" in feature:
        return "novelty"
    if "sim_lag1_mean" in feature:
        return "local flow"
    return "other"


def plot_loadings(loadings: pd.DataFrame, output_path: Path, pcx: int, pcy: int, explained: list[float]) -> None:
    xcol = f"PC{pcx}"
    ycol = f"PC{pcy}"
    if xcol not in loadings or ycol not in loadings:
        return
    colors = {
        "global similarity": "#1f77b4",
        "novelty": "#ff7f0e",
        "local flow": "#2ca02c",
        "other": "#666666",
    }
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    for feature, row in loadings.iterrows():
        family = feature_family(feature)
        x = row[xcol]
        y = row[ycol]
        ax.scatter(x, y, color=colors[family], s=58, alpha=0.85, label=family)
        ax.text(x + 0.01, y + 0.01, feature, fontsize=8)

    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, label in zip(handles, labels):
        dedup.setdefault(label, h)
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8)
    ax.axhline(0, color="#999999", linewidth=0.7, alpha=0.45)
    ax.axvline(0, color="#999999", linewidth=0.7, alpha=0.45)
    ax.set_xlabel(f"{xcol} ({explained[pcx - 1] * 100:.1f}%)")
    ax.set_ylabel(f"{ycol} ({explained[pcy - 1] * 100:.1f}%)")
    ax.set_title(f"SC PCA Loadings ({xcol} vs {ycol})")
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def deduplicate_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        if label and label not in dedup:
            dedup[label] = handle
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8, ncol=2)


def loading_scale_for_scores(scores: pd.DataFrame, loadings: pd.DataFrame, xcol: str, ycol: str) -> float:
    score_points = scores[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    loading_points = loadings[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if score_points.size == 0 or loading_points.size == 0:
        return 1.0

    xlo, xhi = np.nanpercentile(score_points[:, 0], [2, 98])
    ylo, yhi = np.nanpercentile(score_points[:, 1], [2, 98])
    score_radius = BIPLOT_LOADING_RADIUS_FRACTION * min(max(xhi - xlo, 1e-9), max(yhi - ylo, 1e-9))
    loading_radius = float(np.nanmax(np.sqrt(np.sum(loading_points**2, axis=1))))
    if not math.isfinite(loading_radius) or loading_radius <= 0:
        return 1.0
    return score_radius / loading_radius


def plot_biplot(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    output_path: Path,
    pcx: int,
    pcy: int,
    explained: list[float],
) -> None:
    xcol = f"PC{pcx}"
    ycol = f"PC{pcy}"
    if xcol not in scores or ycol not in scores or xcol not in loadings or ycol not in loadings:
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))
    natural = scores[scores["source"] == "natural"]
    generated = scores[scores["source"] == "generated"]

    plot_natural_scores_by_bin(ax, natural, xcol, ycol, point_size=36, alpha=0.28, zorder=1)
    plot_natural_mean_trajectory(ax, natural, xcol, ycol, zorder=4)

    for mode in MODE_ORDER:
        sub = generated[generated["conditioning_mode"] == mode]
        if sub.empty:
            continue
        color = MODE_COLORS[mode]
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=22,
            facecolors=[rgba_with_alpha(color, 0.18)],
            edgecolors=color,
            linewidths=0.7,
            label=f"{MODE_LABELS[mode]} rows",
            zorder=3,
        )
        mean_by_bin = (
            sub.assign(bin_num=pd.to_numeric(sub["bin"], errors="coerce"))
            .dropna(subset=["bin_num"])
            .groupby("bin_num")[[xcol, ycol]]
            .mean()
            .sort_index()
        )
        if len(mean_by_bin) > 1:
            ax.plot(
                mean_by_bin[xcol],
                mean_by_bin[ycol],
                color=color,
                linewidth=2.1,
                marker="o",
                markersize=4,
                label=f"{MODE_LABELS[mode]} mean trajectory",
                zorder=5,
            )
            ax.scatter(
                mean_by_bin[xcol].iloc[-1],
                mean_by_bin[ycol].iloc[-1],
                color=color,
                marker="D",
                s=72,
                edgecolors="black",
                linewidths=0.45,
                zorder=6,
            )

    scale = loading_scale_for_scores(scores, loadings, xcol, ycol)
    family_colors = {
        "global similarity": "#1f77b4",
        "novelty": "#ff7f0e",
        "local flow": "#2ca02c",
        "other": "#666666",
    }
    for feature, row in loadings.iterrows():
        family = feature_family(feature)
        color = family_colors[family]
        x = float(row[xcol]) * scale
        y = float(row[ycol]) * scale
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                alpha=0.78,
                linewidth=1.4,
                mutation_scale=BIPLOT_ARROW_MUTATION_SCALE,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=7,
        )
        ax.text(
            x * 1.06 + 0.02 * np.sign(x if x != 0 else 1),
            y * 1.06 + 0.02 * np.sign(y if y != 0 else 1),
            feature,
            fontsize=8,
            color=color,
            zorder=8,
        )

    ax.axhline(0, color="#999999", linewidth=0.8, alpha=0.55)
    ax.axvline(0, color="#999999", linewidth=0.8, alpha=0.55)
    ax.set_xlabel(f"{xcol} ({explained[pcx - 1] * 100:.1f}% natural variance)")
    ax.set_ylabel(f"{ycol} ({explained[pcy - 1] * 100:.1f}% natural variance)")
    ax.set_title(f"SC PCA Biplot ({xcol} vs {ycol})")
    ax.grid(True, alpha=0.22)
    deduplicate_legend(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-sc", type=Path, default=DEFAULT_GENERATED_SC)
    parser.add_argument(
        "--natural-sc",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Natural SC JSON files. Defaults to /tmp/natural_books_sc_median.json "
            "plus all /tmp/natural_sc_extra_*.json files."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument(
        "--standardization",
        choices=("bin", "global", "none"),
        default="bin",
        help=(
            "Feature scaling before PCA. 'bin' uses natural median/IQR within each "
            "token-size bin; 'global' uses one natural median/IQR across all rows; "
            "'none' uses raw SC values."
        ),
    )
    parser.add_argument(
        "--no-bin-balance",
        action="store_true",
        help="Disable equal token-bin weighting when fitting PCA on natural rows.",
    )
    args = parser.parse_args()

    if args.output_dir == DEFAULT_OUTPUT_DIR:
        if args.standardization == "global":
            args.output_dir = DEFAULT_GLOBAL_OUTPUT_DIR
        elif args.standardization == "none":
            args.output_dir = DEFAULT_RAW_OUTPUT_DIR
    args.output_dir.mkdir(parents=True, exist_ok=True)
    natural_paths = args.natural_sc if args.natural_sc is not None else default_natural_sc_paths()

    natural = complete_feature_rows(load_natural_rows(natural_paths))
    generated = complete_feature_rows(load_generated_rows(args.generated_sc))
    if args.standardization == "global":
        natural_z, generated_z, scaling_stats = robust_global_standardize(natural, generated)
    elif args.standardization == "none":
        natural_z, generated_z, scaling_stats = no_standardize(natural, generated)
    else:
        natural_z, generated_z, scaling_stats = robust_bin_standardize(natural, generated)

    balance_bins = not args.no_bin_balance
    pca, nat_scores, gen_scores = fit_project_pca(
        natural_z,
        generated_z,
        args.components,
        balance_bins=balance_bins,
    )
    explained = pca.explained_variance_ratio_.tolist()
    loadings = write_loadings(pca, args.output_dir)

    scores = pd.concat(
        [
            attach_scores(natural_z, nat_scores),
            attach_scores(generated_z, gen_scores),
        ],
        ignore_index=True,
    )
    scores.to_csv(args.output_dir / "sc_pca_scores.csv", index=False)
    scaling_stats.to_csv(args.output_dir / "sc_pca_robust_scaling.csv", index=False)
    if args.standardization == "bin":
        scaling_stats.to_csv(args.output_dir / "sc_pca_bin_robust_scaling.csv", index=False)
    pd.concat([natural_z, generated_z], ignore_index=True).to_csv(
        args.output_dir / "sc_pca_features_robust_z.csv",
        index=False,
    )

    summary = {
        "features": FEATURES,
        "windows": WINDOWS,
        "metrics": SC_METRICS,
        "n_natural_rows": int(len(natural_z)),
        "n_generated_rows": int(len(generated_z)),
        "natural_rows_by_bin": {
            str(bin_id): int(count)
            for bin_id, count in natural_z["bin"].astype(str).value_counts().sort_index().items()
        },
        "standardization_mode": args.standardization,
        "pca_bin_balanced": bool(balance_bins),
        "pca_bin_balance_method": (
            "weighted covariance with each token-size bin contributing equal total weight"
            if balance_bins
            else "unweighted covariance across natural rows"
        ),
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": explained,
        "natural_files": [str(path) for path in natural_paths],
        "natural_extra_glob": DEFAULT_NATURAL_EXTRA_GLOB,
        "generated_file": str(args.generated_sc),
        "standardization": (
            "global robust z = (x - natural global median) / natural global IQR; std fallback if IQR is zero"
            if args.standardization == "global"
            else "none; raw SC feature values are used, with only PCA covariance centering"
            if args.standardization == "none"
            else "per-bin robust z = (x - natural bin median) / natural bin IQR; std fallback if IQR is zero"
        ),
    }
    (args.output_dir / "sc_pca_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    plot_scores(scores, args.output_dir / "sc_pca_scores_pc1_pc2.jpg", 1, 2, explained)
    plot_loadings(loadings, args.output_dir / "sc_pca_loadings_pc1_pc2.jpg", 1, 2, explained)
    plot_biplot(scores, loadings, args.output_dir / "sc_pca_biplot_pc1_pc2.jpg", 1, 2, explained)
    if pca.n_components_ >= 4:
        plot_scores(scores, args.output_dir / "sc_pca_scores_pc3_pc4.jpg", 3, 4, explained)
        plot_loadings(loadings, args.output_dir / "sc_pca_loadings_pc3_pc4.jpg", 3, 4, explained)
        plot_biplot(scores, loadings, args.output_dir / "sc_pca_biplot_pc3_pc4.jpg", 3, 4, explained)

    print(
        "SC PCA complete: "
        f"{len(natural_z)} natural rows, {len(generated_z)} generated rows, "
        f"explained={', '.join(f'{v * 100:.1f}%' for v in explained)}"
    )
    print(f"Outputs -> {args.output_dir}")


if __name__ == "__main__":
    main()
