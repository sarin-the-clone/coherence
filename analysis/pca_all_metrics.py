#!/usr/bin/env python3
"""
PCA using original coherence descriptors plus curated token-naturalness metrics.

Natural books:
  - original metrics from natural_texts/Output/batch_results/batch_*.json
  - selected token metrics from natural_texts/Output/gutenberg_benchmark_cache/*.jsonl

Generated books:
  - prompt folders that contain token_naturalness_metrics.json
  - old final accumulated metrics recovered from the final accepted attempt file
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import gaussian_kde
except ImportError:  # pragma: no cover
    gaussian_kde = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import (  # noqa: E402
    CURATED_TOKEN_NATURALNESS_METRICS,
    OPTIONAL_CHUNK_CONTEXT_METRICS,
)

DEFAULT_BATCH_RESULTS = ROOT / "natural_texts" / "Output" / "batch_results"
DEFAULT_NATURAL_TOKEN_METRICS = (
    ROOT
    / "natural_texts"
    / "Output"
    / "gutenberg_benchmark_cache"
    / "token_naturalness_metrics.jsonl"
)
DEFAULT_NATURAL_TOKEN_PREDICTABILITY = (
    ROOT
    / "natural_texts"
    / "Output"
    / "gutenberg_benchmark_cache"
    / "token_predictability_metrics.jsonl"
)
DEFAULT_NATURAL_TOKEN_CONTEXT_DYNAMICS = (
    ROOT
    / "natural_texts"
    / "Output"
    / "gutenberg_benchmark_cache"
    / "token_context_dynamics_metrics.jsonl"
)
DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
DEFAULT_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca" / "all_metrics"
DEFAULT_SIZE_STATS = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"

# Canonical 8 stat/info descriptors, kept consistent with the figures
# (make_figure_abstract.py / plot_metrics_vs_size.py). H_DFA1 + H_WT_db4 are
# the retained Hurst estimators; H_DMA, D, delta, mi_tau_max, mi_n_lags are
# intentionally excluded so this set matches earlier results.
OLD_DESCRIPTOR_METRICS = [
    "beta",
    "alpha",
    "H_DFA1",
    "H_WT_db4",
    "entropy_bits",
    "perplexity",
    "lz_complexity",
    "mi_decay_gamma",
]
TOKEN_PREFIXES = ("td_", "tr_", "tp_", "tc_")


def format_token_label(n_tokens: int) -> str:
    if n_tokens >= 1_000_000 and n_tokens % 1_000_000 == 0:
        return f"{n_tokens // 1_000_000}M"
    if n_tokens >= 1_000 and n_tokens % 1_000 == 0:
        return f"{n_tokens // 1_000}k"
    if n_tokens >= 1_000:
        return f"{n_tokens / 1_000:.0f}k"
    return str(n_tokens)


def load_size_bins(path: Path) -> list[dict]:
    if not path.exists():
        return [
            {"label": "100-150k", "range_tokens": [100_000, 150_000]},
            {"label": "150-300k", "range_tokens": [150_000, 300_000]},
            {"label": "300-500k", "range_tokens": [300_000, 500_000]},
            {"label": "500-800k", "range_tokens": [500_000, 800_000]},
            {"label": ">800k", "range_tokens": [800_000, float("inf")]},
        ]

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    bins = []
    for item in data.get("bins", []):
        lo, hi = item.get("range_tokens", [None, None])
        if lo is None or hi is None:
            continue
        label = f">{format_token_label(int(lo))}" if hi > 5_000_000 else f"{format_token_label(int(lo))}-{format_token_label(int(hi))}"
        bins.append({
            "name": item.get("name"),
            "label": label,
            "range_tokens": [int(lo), float(hi)],
        })
    return bins


def assign_token_size_bins(df: pd.DataFrame, size_bins: list[dict]) -> pd.DataFrame:
    out = df.copy()

    def assign(value) -> str:
        try:
            n_tokens = int(value)
        except Exception:
            return "unknown"
        for item in size_bins:
            lo, hi = item["range_tokens"]
            if lo <= n_tokens < hi:
                return item["label"]
        return "unknown"

    out["token_size_bin"] = out["n_tokens"].map(assign) if "n_tokens" in out else "unknown"
    return out


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_natural_batch_metrics(batch_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(batch_dir.glob("batch_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No batch_*.json files found in {batch_dir}")
    df = pd.DataFrame(rows)
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
    return df


def load_natural_token_metrics(path: Path) -> pd.DataFrame:
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"No token metric rows found in {path}")
    df = pd.DataFrame(rows)
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce").astype("Int64")
    return df


def load_and_merge_natural_token_metrics(
    paths: list[Path],
    token_metric_set: str = "curated",
    include_chunk_context_metrics: bool = False,
) -> pd.DataFrame:
    merged = None
    for path in paths:
        df = load_natural_token_metrics(path)
        token_metrics = discover_token_metrics(
            df,
            token_metric_set=token_metric_set,
            include_chunk_context_metrics=include_chunk_context_metrics,
        )
        keep = ["book_id"] + token_metrics
        if merged is None:
            merged = df[keep].copy()
        else:
            merged = merged.merge(df[keep], on="book_id", how="inner")
    if merged is None or merged.empty:
        raise SystemExit(f"No token metric rows found in {paths}")
    return merged


def discover_token_metrics(
    df: pd.DataFrame,
    token_metric_set: str = "curated",
    include_chunk_context_metrics: bool = False,
) -> list[str]:
    available = sorted(
        col
        for col in df.columns
        if col.startswith(TOKEN_PREFIXES) and not col.endswith("_r2")
        and pd.api.types.is_numeric_dtype(df[col])
    )
    if token_metric_set == "all":
        if include_chunk_context_metrics:
            return available
        return [col for col in available if not col.startswith("tc_")]

    if token_metric_set != "curated":
        raise ValueError(f"unknown token metric set: {token_metric_set}")

    keep = set(CURATED_TOKEN_NATURALNESS_METRICS)
    if include_chunk_context_metrics:
        keep.update(OPTIONAL_CHUNK_CONTEXT_METRICS)
    return [col for col in available if col in keep]


def build_natural_dataframe(
    batch_dir: Path,
    token_jsonls: list[Path],
    token_metric_set: str = "curated",
    include_chunk_context_metrics: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    old_df = load_natural_batch_metrics(batch_dir)
    token_df = load_and_merge_natural_token_metrics(
        token_jsonls,
        token_metric_set=token_metric_set,
        include_chunk_context_metrics=include_chunk_context_metrics,
    )

    token_metrics = discover_token_metrics(
        token_df,
        token_metric_set=token_metric_set,
        include_chunk_context_metrics=include_chunk_context_metrics,
    )
    keep_token = ["book_id"] + token_metrics
    merged = old_df.merge(token_df[keep_token], on="book_id", how="inner", suffixes=("", "_token"))

    old_metrics = [m for m in OLD_DESCRIPTOR_METRICS if m in merged.columns]
    metric_cols = old_metrics + token_metrics
    for col in metric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    return merged, metric_cols


def finite_metric_frame(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    clean = df.copy()
    clean[metric_cols] = clean[metric_cols].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=metric_cols)
    return clean


def nonconstant_metric_cols(df: pd.DataFrame, metric_cols: list[str], min_std: float = 1e-12) -> tuple[list[str], list[str]]:
    kept = []
    dropped = []
    for metric in metric_cols:
        std = float(df[metric].std(ddof=0))
        if math.isfinite(std) and std > min_std:
            kept.append(metric)
        else:
            dropped.append(metric)
    return kept, dropped


def final_accepted_attempt(folder: Path) -> tuple[int, int, Path] | None:
    candidates = []
    for path in folder.glob("chunk_*_accepted.txt"):
        match = re.fullmatch(r"chunk_(\d+)_accepted\.txt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    chunk_id, marker_path = max(candidates)
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        marker = {}
    attempt = int(marker.get("accepted_attempt") or 1)
    return chunk_id, attempt, folder / f"chunk_{chunk_id:03d}_attempt_{attempt}.txt"


def load_generated_metric_rows(prompts_dir: Path, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for token_path in sorted(prompts_dir.glob("*/token_naturalness_metrics.json")):
        folder = token_path.parent
        final_attempt = final_accepted_attempt(folder)
        if final_attempt is None:
            continue
        chunk_id, attempt, attempt_path = final_attempt
        if not attempt_path.exists():
            continue
        try:
            token_metrics = json.loads(token_path.read_text(encoding="utf-8"))
            attempt_data = json.loads(attempt_path.read_text(encoding="utf-8"))
            metadata_path = folder / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        except Exception:
            continue

        row = {}
        row.update(attempt_data.get("accumulated_metrics") or {})
        row.update({k: v for k, v in token_metrics.items() if k.startswith(TOKEN_PREFIXES)})
        row.update({
            "folder_name": folder.name,
            "title": metadata.get("title") or token_metrics.get("folder_name") or folder.name,
            "conditioning_mode": metadata.get("conditioning_mode") or "unknown",
            "chunk_id": chunk_id,
            "accepted_attempt": attempt,
            "n_tokens": row.get("n_tokens") or token_metrics.get("n_tokens"),
        })
        if all(isinstance(row.get(col), (int, float)) and math.isfinite(row[col]) for col in metric_cols):
            rows.append(row)

    return pd.DataFrame(rows)


def fit_pca(df: pd.DataFrame, metric_cols: list[str], n_components: int) -> tuple[np.ndarray, PCA, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[metric_cols].to_numpy(dtype=float))
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(x_scaled)
    return coords, pca, scaler


def project_generated(df_generated: pd.DataFrame, metric_cols: list[str], pca: PCA, scaler: StandardScaler) -> pd.DataFrame:
    if df_generated.empty:
        return df_generated
    x_scaled = scaler.transform(df_generated[metric_cols].to_numpy(dtype=float))
    coords = pca.transform(x_scaled)
    out = df_generated.copy()
    for i in range(coords.shape[1]):
        out[f"PC{i + 1}"] = coords[:, i]
    return out


def plot_kde(ax, points: np.ndarray, color: str) -> None:
    if gaussian_kde is None or len(points) < 20:
        return
    try:
        kde = gaussian_kde(points.T)
        x_min, x_max = np.percentile(points[:, 0], [1, 99])
        y_min, y_max = np.percentile(points[:, 1], [1, 99])
        x_pad = (x_max - x_min) * 0.2
        y_pad = (y_max - y_min) * 0.2
        xx, yy = np.mgrid[
            x_min - x_pad:x_max + x_pad:120j,
            y_min - y_pad:y_max + y_pad:120j,
        ]
        pos = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kde(pos).T, xx.shape)
        levels = np.linspace(np.percentile(density[density > 0], 20), density.max(), 8)
        ax.contour(xx, yy, density, levels=levels, colors=color, linewidths=1.0, alpha=0.45)
    except Exception:
        return


def robust_score_limits(
    points: np.ndarray,
    lower_q: float = 0.5,
    upper_q: float = 99.5,
    pad_fraction: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return quantile-based score limits for display only."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 5 or points.shape[1] < 2:
        return None
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] < 5:
        return None

    limits = []
    for dim in range(2):
        lo, hi = np.percentile(points[:, dim], [lower_q, upper_q])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return None
        pad = (hi - lo) * pad_fraction
        limits.append((float(lo - pad), float(hi + pad)))
    return limits[0], limits[1]


def apply_robust_score_limits(
    ax,
    points: np.ndarray,
    lower_q: float = 2.0,
    upper_q: float = 98.0,
    pad_fraction: float = 0.08,
    extra_points: np.ndarray | None = None,
) -> None:
    """Set display limits from the natural-score quantiles, then expand just
    enough to keep ``extra_points`` (e.g. projected generated books) on-panel.

    The tighter default quantiles trim natural's heavy PC tails so the bulk
    (and the loading arrows) is not crushed; the union with extra_points keeps
    the generated books visible without letting them stretch the whole axis.
    """
    limits = robust_score_limits(points, lower_q=lower_q, upper_q=upper_q, pad_fraction=pad_fraction)
    if limits is None:
        return
    (x_min, x_max), (y_min, y_max) = limits
    if extra_points is not None:
        ep = np.asarray(extra_points, dtype=float)
        ep = ep[np.isfinite(ep).all(axis=1)] if ep.ndim == 2 else ep
        if ep.ndim == 2 and len(ep):
            px = (x_max - x_min) * 0.06
            py = (y_max - y_min) * 0.06
            x_min = min(x_min, float(ep[:, 0].min()) - px)
            x_max = max(x_max, float(ep[:, 0].max()) + px)
            y_min = min(y_min, float(ep[:, 1].min()) - py)
            y_max = max(y_max, float(ep[:, 1].max()) + py)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def scale_scores_to_loading_frame(
    natural: pd.DataFrame,
    loadings: pd.DataFrame,
    pc_x: str,
    pc_y: str,
) -> np.ndarray | None:
    """Map PCA score density into loading coordinates for a visual backdrop."""
    if pc_x not in natural or pc_y not in natural:
        return None

    score_points = natural[[pc_x, pc_y]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if score_points.shape[0] < 20:
        return None

    return scale_score_points_to_loading_frame(score_points, score_points, loadings, pc_x, pc_y)


def scale_score_points_to_loading_frame(
    reference_points: np.ndarray,
    points: np.ndarray,
    loadings: pd.DataFrame,
    pc_x: str,
    pc_y: str,
) -> np.ndarray | None:
    """Map arbitrary score points using the same robust score-to-loading transform."""
    if reference_points.shape[0] < 2 or points.shape[0] < 1:
        return None

    loading_points = loadings[[pc_x, pc_y]].to_numpy(dtype=float)
    loading_span = np.nanmax(np.abs(loading_points), axis=0)
    loading_span = np.where(np.isfinite(loading_span) & (loading_span > 0), loading_span, 1.0)

    scaled = np.zeros_like(points)
    for dim in range(2):
        center = np.median(reference_points[:, dim])
        lo, hi = np.percentile(reference_points[:, dim], [1, 99])
        robust_half_span = max(abs(hi - center), abs(center - lo), 1e-12)
        scaled[:, dim] = (points[:, dim] - center) / robust_half_span * loading_span[dim]

    return scaled


def mean_score_trajectory_by_size_bin(natural: pd.DataFrame, size_bins: list[dict], pc_cols: list[str]) -> pd.DataFrame:
    if not size_bins or "token_size_bin" not in natural:
        return pd.DataFrame()
    rows = []
    for idx, item in enumerate(size_bins):
        label = item["label"]
        group = natural[natural["token_size_bin"] == label]
        if group.empty:
            continue
        row = {
            "token_size_bin": label,
            "bin_order": idx,
            "count": int(len(group)),
            "mean_n_tokens": float(pd.to_numeric(group["n_tokens"], errors="coerce").mean()),
        }
        for pc in pc_cols:
            if pc in group:
                row[pc] = float(group[pc].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_supplementary_loadings(
    natural: pd.DataFrame,
    variables: dict[str, pd.Series],
    pc_cols: list[str],
) -> pd.DataFrame:
    """Project passive variables by correlation with existing PC scores."""
    rows = []
    for name, values in variables.items():
        frame = natural[pc_cols].copy()
        frame[name] = pd.to_numeric(values, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=pc_cols + [name])
        row = {"metric": name, "group": "Supplementary"}
        for pc in pc_cols:
            if len(frame) < 3 or frame[name].std(ddof=0) <= 0 or frame[pc].std(ddof=0) <= 0:
                row[pc] = np.nan
            else:
                row[pc] = float(frame[name].corr(frame[pc]))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["metric", "group"] + pc_cols)
    return pd.DataFrame(rows).set_index("metric")


def metric_group(metric: str) -> str:
    if metric.startswith("td_"):
        return "Token distribution"
    if metric.startswith("tr_"):
        return "Token recurrence"
    if metric.startswith("tp_"):
        return "Token predictability"
    if metric.startswith("tc_"):
        return "Token context dynamics"
    if metric in {"entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma", "mi_tau_max", "mi_n_lags"}:
        return "Information"
    return "Structural"


def plot_pca(
    natural: pd.DataFrame,
    generated: pd.DataFrame,
    pca: PCA,
    metric_cols: list[str],
    output_path: Path,
    max_loading_labels: int,
) -> None:
    colors = {
        "Structural": "#1f77b4",
        "Information": "#ff7f0e",
        "Token distribution": "#17becf",
        "Token recurrence": "#2ca02c",
        "Token predictability": "#bc6c25",
        "Token context dynamics": "#e377c2",
    }

    fig, ax = plt.subplots(figsize=(15, 11))
    points = natural[["PC1", "PC2"]].to_numpy()
    plot_kde(ax, points, "gray")
    ax.scatter(points[:, 0], points[:, 1], s=8, color="gray", alpha=0.10, edgecolors="none", label="Natural books")

    if not generated.empty:
        mode_colors = {
            "minimal_fixed_conditioning": "#d00000",
            "basic_conditioning": "#1d4ed8",
            "markov_rolling_conditioning": "#f59e0b",
            "minimal_rolling_conditioning": "#f59e0b",
            "summaries_conditioning": "#059669",
            "extended_conditioning": "#7c3aed",
            "unknown": "black",
        }
        for _, row in generated.iterrows():
            color = mode_colors.get(row.get("conditioning_mode"), "black")
            ax.scatter(row["PC1"], row["PC2"], s=180, marker="*", color=color, edgecolors="black", linewidths=1.0, zorder=20)
            label = f"{row.get('conditioning_mode', 'unknown').replace('_conditioning', '')}\n{str(row.get('title', 'generated'))[:32]}"
            ax.text(row["PC1"] + 0.12, row["PC2"], label, fontsize=9, color=color, va="center",
                    bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.2"))

    loadings = pca.components_[:2].T
    strength = np.sqrt(np.sum(loadings**2, axis=1))
    top_indices = np.argsort(strength)[::-1][:max_loading_labels]
    scale = 5.0
    for idx in top_indices:
        metric = metric_cols[idx]
        group = metric_group(metric)
        color = colors[group]
        x, y = loadings[idx] * scale
        ax.arrow(0, 0, x, y, color=color, alpha=0.70, linewidth=1.3, head_width=0.10, length_includes_head=True)
        ax.text(x * 1.08, y * 1.08, metric, color=color, fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor=color, alpha=0.80, boxstyle="round,pad=0.12"))

    handles = [
        plt.Line2D([], [], color=color, marker="o", linestyle="None", label=group)
        for group, color in colors.items()
        if any(metric_group(m) == group for m in metric_cols)
    ]
    handles.append(plt.Line2D([], [], color="gray", marker="o", linestyle="None", label="Natural books"))
    if not generated.empty:
        handles.append(plt.Line2D([], [], color="black", marker="*", linestyle="None", markersize=12, label="Generated books"))
    ax.legend(handles=handles, loc="best", framealpha=0.9)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(
        "Natural-books PCA using original coherence + token-naturalness metrics\n"
        f"{len(metric_cols)} metrics, {len(natural)} natural books; loadings show top {len(top_indices)} contributors\n"
        "Display axes use natural-score 2-98% quantiles (+ generated) to reduce outlier compression"
    )
    ax.grid(True, alpha=0.25)
    gen_pts = generated[["PC1", "PC2"]].to_numpy() if not generated.empty else None
    apply_robust_score_limits(ax, points, extra_points=gen_pts)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_score_scatter(
    natural: pd.DataFrame,
    generated: pd.DataFrame,
    pca: PCA,
    output_path: Path,
    pc_x: int,
    pc_y: int,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 10))
    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"
    points = natural[[x_col, y_col]].to_numpy()
    plot_kde(ax, points, "gray")
    ax.scatter(points[:, 0], points[:, 1], s=8, color="gray", alpha=0.10, edgecolors="none", label="Natural books")

    if not generated.empty and x_col in generated and y_col in generated:
        mode_colors = {
            "minimal_fixed_conditioning": "#d00000",
            "basic_conditioning": "#1d4ed8",
            "markov_rolling_conditioning": "#f59e0b",
            "minimal_rolling_conditioning": "#f59e0b",
            "summaries_conditioning": "#059669",
            "extended_conditioning": "#7c3aed",
            "unknown": "black",
        }
        for _, row in generated.iterrows():
            color = mode_colors.get(row.get("conditioning_mode"), "black")
            ax.scatter(row[x_col], row[y_col], s=180, marker="*", color=color, edgecolors="black", linewidths=1.0, zorder=20)
            label = f"{row.get('conditioning_mode', 'unknown').replace('_conditioning', '')}\n{str(row.get('title', 'generated'))[:32]}"
            ax.text(row[x_col] + 0.12, row[y_col], label, fontsize=9, color=color, va="center",
                    bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.2"))

    ax.set_xlabel(f"{x_col} ({pca.explained_variance_ratio_[pc_x - 1] * 100:.1f}%)")
    ax.set_ylabel(f"{y_col} ({pca.explained_variance_ratio_[pc_y - 1] * 100:.1f}%)")
    ax.set_title(f"Natural-books PCA scores: {x_col} x {y_col}\nDisplay axes: natural-score 2-98% quantiles (+ generated)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)
    gen_pts = (generated[[x_col, y_col]].to_numpy()
               if (not generated.empty and x_col in generated and y_col in generated) else None)
    apply_robust_score_limits(ax, points, extra_points=gen_pts)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_loading_scatter(
    loadings: pd.DataFrame,
    output_path: Path,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    natural: pd.DataFrame | None = None,
    supplementary: pd.DataFrame | None = None,
    mean_score_trajectory: pd.DataFrame | None = None,
) -> None:
    colors = {
        "Structural": "#1f77b4",
        "Information": "#ff7f0e",
        "Token distribution": "#17becf",
        "Token recurrence": "#2ca02c",
        "Token predictability": "#bc6c25",
        "Token context dynamics": "#e377c2",
        "Supplementary": "#111827",
    }
    markers = {
        "Structural": "s",
        "Information": "s",
        "Token distribution": "o",
        "Token recurrence": "o",
        "Token predictability": "o",
        "Token context dynamics": "o",
        "Supplementary": "D",
    }
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.35)

    density_points = None
    reference_score_points = None
    if natural is not None:
        reference_score_points = natural[[pc_x, pc_y]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        density_points = scale_scores_to_loading_frame(natural, loadings, pc_x, pc_y)
    if density_points is not None:
        plot_kde(ax, density_points, "gray")

    if (
        mean_score_trajectory is not None
        and not mean_score_trajectory.empty
        and reference_score_points is not None
        and pc_x in mean_score_trajectory
        and pc_y in mean_score_trajectory
    ):
        trajectory = mean_score_trajectory.sort_values("bin_order").replace([np.inf, -np.inf], np.nan).dropna(subset=[pc_x, pc_y])
        raw_points = trajectory[[pc_x, pc_y]].to_numpy(dtype=float)
        scaled_points = scale_score_points_to_loading_frame(reference_score_points, raw_points, loadings, pc_x, pc_y)
        if scaled_points is not None and len(scaled_points):
            ax.plot(
                scaled_points[:, 0],
                scaled_points[:, 1],
                color="#111827",
                linewidth=2.1,
                alpha=0.85,
                zorder=11,
                label="Mean natural score trajectory by token bin",
            )
            ax.scatter(
                scaled_points[:, 0],
                scaled_points[:, 1],
                marker="*",
                s=230,
                color="#ffd166",
                edgecolors="#111827",
                linewidths=1.1,
                zorder=12,
            )
            for (_, row), (x, y) in zip(trajectory.iterrows(), scaled_points):
                ax.text(
                    x + 0.010,
                    y + 0.010,
                    str(row["token_size_bin"]),
                    fontsize=8,
                    color="#111827",
                    bbox=dict(facecolor="white", edgecolor="#111827", alpha=0.82, boxstyle="round,pad=0.13"),
                    zorder=13,
                )

    for group, group_df in loadings.groupby("group"):
        ax.scatter(
            group_df[pc_x],
            group_df[pc_y],
            s=90,
            marker=markers.get(group, "o"),
            color=colors.get(group, "gray"),
            edgecolors="black",
            linewidths=0.5,
            label=group,
            alpha=0.9,
        )
        for metric, row in group_df.iterrows():
            ax.text(
                row[pc_x] + 0.006,
                row[pc_y] + 0.006,
                metric,
                fontsize=8,
                color=colors.get(group, "gray"),
            )

    if supplementary is not None and not supplementary.empty:
        active_radius = np.sqrt(loadings[pc_x] ** 2 + loadings[pc_y] ** 2)
        active_radius = float(np.nanmax(active_radius)) if len(active_radius) else 1.0
        if not np.isfinite(active_radius) or active_radius <= 0:
            active_radius = 1.0

        supp_points = supplementary[[pc_x, pc_y]].replace([np.inf, -np.inf], np.nan).dropna()
        supp_radius = np.sqrt(supp_points[pc_x] ** 2 + supp_points[pc_y] ** 2)
        supp_radius = float(np.nanmax(supp_radius)) if len(supp_radius) else 1.0
        if not np.isfinite(supp_radius) or supp_radius <= 0:
            supp_radius = 1.0

        scale = 0.95 * active_radius / supp_radius
        for metric, row in supp_points.iterrows():
            x = float(row[pc_x]) * scale
            y = float(row[pc_y]) * scale
            raw_r = float(np.sqrt(row[pc_x] ** 2 + row[pc_y] ** 2))
            ax.arrow(
                0,
                0,
                x,
                y,
                color=colors["Supplementary"],
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                head_width=0.012,
                length_includes_head=True,
                zorder=8,
            )
            ax.scatter(
                [x],
                [y],
                s=115,
                marker=markers["Supplementary"],
                color="white",
                edgecolors=colors["Supplementary"],
                linewidths=1.4,
                zorder=9,
            )
            ax.text(
                x + 0.008,
                y + 0.008,
                f"{metric}\n|r|={raw_r:.2f}",
                fontsize=8,
                color=colors["Supplementary"],
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=colors["Supplementary"], alpha=0.85, boxstyle="round,pad=0.15"),
                zorder=10,
            )

    ax.set_xlabel(f"{pc_x} loading")
    ax.set_ylabel(f"{pc_y} loading")
    title = f"PCA Metric Loadings: {pc_x} vs {pc_y}"
    if density_points is not None:
        title += "\nGray contours: natural-book PCA score density, robustly scaled into loading coordinates"
        ax.plot([], [], color="gray", linewidth=1.0, alpha=0.45, label="Natural score density (scaled)")
    if mean_score_trajectory is not None and not mean_score_trajectory.empty:
        title += "\nStars: mean natural-book PCA scores by token-size bin, robustly scaled into loading coordinates"
    if supplementary is not None and not supplementary.empty:
        title += "\nDashed vector: supplementary variable projected by correlation with PC scores"
        ax.plot([], [], color=colors["Supplementary"], linestyle="--", linewidth=1.8, label="Supplementary variable (scaled)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_loading_bars(loadings: pd.DataFrame, output_path: Path, pcs: tuple[str, str] = ("PC1", "PC2"), top_n: int = 18) -> None:
    colors = {
        "Structural": "#1f77b4",
        "Information": "#ff7f0e",
        "Token distribution": "#17becf",
        "Token recurrence": "#2ca02c",
        "Token predictability": "#bc6c25",
        "Token context dynamics": "#e377c2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, top_n * 0.38)))
    for ax, pc in zip(axes, pcs):
        subset = loadings.reindex(loadings[pc].abs().sort_values(ascending=False).head(top_n).index)
        subset = subset.iloc[::-1]
        bar_colors = [colors.get(group, "gray") for group in subset["group"]]
        ax.barh(subset.index, subset[pc], color=bar_colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Top {top_n} absolute loadings on {pc}")
        ax.set_xlabel("Loading")
        ax.grid(axis="x", alpha=0.25)
    handles = [
        plt.Line2D([], [], color=color, marker="s", linestyle="None", label=group)
        for group, color in colors.items()
        if group in set(loadings["group"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-results", type=Path, default=DEFAULT_BATCH_RESULTS)
    parser.add_argument(
        "--natural-token-metrics",
        type=Path,
        nargs="+",
        default=[
            DEFAULT_NATURAL_TOKEN_METRICS,
            DEFAULT_NATURAL_TOKEN_PREDICTABILITY,
            DEFAULT_NATURAL_TOKEN_CONTEXT_DYNAMICS,
        ],
    )
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    parser.add_argument(
        "--token-metric-set",
        choices=["curated", "all"],
        default="curated",
        help="curated uses the selected td/tr/tp metrics; all uses every available td/tr/tp metric.",
    )
    parser.add_argument(
        "--include-chunk-context-metrics",
        action="store_true",
        help="Include optional tc_* chunk-context metrics. Off by default because natural-book boundaries are synthetic.",
    )
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--max-loading-labels", type=int, default=24)
    parser.add_argument("--loading-bar-top-n", type=int, default=18)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    natural_raw, metric_cols_all = build_natural_dataframe(
        args.batch_results,
        args.natural_token_metrics,
        token_metric_set=args.token_metric_set,
        include_chunk_context_metrics=args.include_chunk_context_metrics,
    )
    natural = finite_metric_frame(natural_raw, metric_cols_all)
    metric_cols, dropped_metrics = nonconstant_metric_cols(natural, metric_cols_all)
    coords, pca, scaler = fit_pca(natural, metric_cols, args.components)
    pc_cols = [f"PC{i + 1}" for i in range(args.components)]
    pc_frame = pd.DataFrame(coords, columns=pc_cols, index=natural.index)
    natural = pd.concat([natural.copy(), pc_frame], axis=1)
    size_bins = load_size_bins(args.size_stats)
    natural = assign_token_size_bins(natural, size_bins)
    mean_score_trajectory = mean_score_trajectory_by_size_bin(natural, size_bins, pc_cols)
    natural["log10_n_tokens"] = np.log10(pd.to_numeric(natural["n_tokens"], errors="coerce"))
    supplementary = compute_supplementary_loadings(
        natural,
        {"log10_n_tokens": natural["log10_n_tokens"]},
        pc_cols,
    )

    generated_raw = load_generated_metric_rows(args.prompts_dir, metric_cols)
    generated = project_generated(generated_raw, metric_cols, pca, scaler)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(args.components)],
        index=metric_cols,
    )
    loadings["group"] = [metric_group(m) for m in metric_cols]
    loadings["pc1_pc2_strength"] = np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2)

    if not mean_score_trajectory.empty:
        mean_score_trajectory = mean_score_trajectory.copy()
        for pc_x, pc_y in (("PC1", "PC2"), ("PC3", "PC4")):
            if pc_x not in mean_score_trajectory or pc_y not in mean_score_trajectory:
                continue
            reference_points = natural[[pc_x, pc_y]].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            raw_points = mean_score_trajectory[[pc_x, pc_y]].to_numpy(dtype=float)
            scaled_points = scale_score_points_to_loading_frame(reference_points, raw_points, loadings, pc_x, pc_y)
            if scaled_points is None:
                continue
            mean_score_trajectory[f"{pc_x}_{pc_y}_loading_x"] = scaled_points[:, 0]
            mean_score_trajectory[f"{pc_x}_{pc_y}_loading_y"] = scaled_points[:, 1]

    natural_scores_out = args.output_dir / "natural_pca_all_metrics_scores.csv"
    mean_trajectory_out = args.output_dir / "natural_pca_all_metrics_mean_score_trajectory_by_token_bin.csv"
    generated_out = args.output_dir / "generated_pca_all_metrics_projection.csv"
    loadings_out = args.output_dir / "natural_pca_all_metrics_loadings.csv"
    supplementary_out = args.output_dir / "natural_pca_all_metrics_supplementary_loadings.csv"
    summary_out = args.output_dir / "natural_pca_all_metrics_summary.json"
    plot_out = args.output_dir / "natural_pca_all_metrics_pc1_pc2.jpg"
    plot_pc3_pc4_out = args.output_dir / "natural_pca_all_metrics_pc3_pc4.jpg"
    loading_scatter_out = args.output_dir / "natural_pca_all_metrics_loadings_pc1_pc2.jpg"
    loading_scatter_pc3_pc4_out = args.output_dir / "natural_pca_all_metrics_loadings_pc3_pc4.jpg"
    loading_bars_out = args.output_dir / "natural_pca_all_metrics_loading_bars.jpg"
    loading_bars_pc3_pc4_out = args.output_dir / "natural_pca_all_metrics_loading_bars_pc3_pc4.jpg"

    keep_natural = ["book_id", "title", "author", "genre", "broad_genre", "n_tokens", "token_size_bin"] + [f"PC{i + 1}" for i in range(args.components)]
    natural[[c for c in keep_natural if c in natural.columns]].to_csv(natural_scores_out, index=False)
    if not mean_score_trajectory.empty:
        mean_score_trajectory.to_csv(mean_trajectory_out, index=False)
    elif mean_trajectory_out.exists():
        mean_trajectory_out.unlink()
    if not generated.empty:
        generated.to_csv(generated_out, index=False)
    elif generated_out.exists():
        generated_out.unlink()
    loadings.sort_values("pc1_pc2_strength", ascending=False).to_csv(loadings_out)
    supplementary.to_csv(supplementary_out)
    summary_out.write_text(json.dumps({
        "natural_input_rows": int(len(natural_raw)),
        "natural_clean_rows": int(len(natural)),
        "generated_projected_rows": int(len(generated)),
        "metric_count": len(metric_cols),
        "dropped_zero_variance_metrics": dropped_metrics,
        "metrics": metric_cols,
        "token_metric_set": args.token_metric_set,
        "include_chunk_context_metrics": args.include_chunk_context_metrics,
        "size_stats": str(args.size_stats),
        "mean_score_trajectory_bins": mean_score_trajectory.to_dict(orient="records"),
        "supplementary_variables": supplementary.reset_index().to_dict(orient="records"),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
    }, indent=2), encoding="utf-8")

    plot_pca(natural, generated, pca, metric_cols, plot_out, args.max_loading_labels)
    plot_score_scatter(natural, generated, pca, plot_pc3_pc4_out, 3, 4)
    plot_loading_scatter(
        loadings,
        loading_scatter_out,
        "PC1",
        "PC2",
        natural=natural,
        supplementary=supplementary,
        mean_score_trajectory=mean_score_trajectory,
    )
    plot_loading_scatter(
        loadings,
        loading_scatter_pc3_pc4_out,
        "PC3",
        "PC4",
        natural=natural,
        supplementary=supplementary,
        mean_score_trajectory=mean_score_trajectory,
    )
    plot_loading_bars(loadings, loading_bars_out, ("PC1", "PC2"), top_n=args.loading_bar_top_n)
    plot_loading_bars(loadings, loading_bars_pc3_pc4_out, ("PC3", "PC4"), top_n=args.loading_bar_top_n)

    print(f"Natural rows: {len(natural)} / {len(natural_raw)}")
    print(f"Generated projected rows: {len(generated)}")
    print(f"Metrics: {len(metric_cols)}")
    if dropped_metrics:
        print(f"Dropped zero-variance metrics: {dropped_metrics}")
    print(f"Explained variance: {[round(float(x), 4) for x in pca.explained_variance_ratio_]}")
    print(f"Saved plot -> {plot_out}")
    print(f"Saved PC3xPC4 plot -> {plot_pc3_pc4_out}")
    print(f"Saved loading scatter -> {loading_scatter_out}")
    print(f"Saved PC3xPC4 loading scatter -> {loading_scatter_pc3_pc4_out}")
    print(f"Saved loading bars -> {loading_bars_out}")
    print(f"Saved PC3xPC4 loading bars -> {loading_bars_pc3_pc4_out}")
    print(f"Saved loadings -> {loadings_out}")
    print(f"Saved supplementary loadings -> {supplementary_out}")
    print(f"Saved natural scores -> {natural_scores_out}")
    if not mean_score_trajectory.empty:
        print(f"Saved mean score trajectory -> {mean_trajectory_out}")
    if not generated.empty:
        print(f"Saved generated projections -> {generated_out}")


if __name__ == "__main__":
    main()
