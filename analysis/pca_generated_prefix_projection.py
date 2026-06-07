#!/usr/bin/env python3
"""
Project selected generated-book prefixes onto a natural-corpus PCA score plot.

This intentionally fits a PCA on metrics available for generated prefixes:
original stat/info descriptors plus the curated token-naturalness metrics.
The full raw metric files may contain more td_*, tr_*, tp_*, and tc_* metrics,
but those are excluded by default because they add compute/storage cost and
were less informative in the natural-book PCA.

Generated prefixes are sampled at landmark token targets by default:
100k, 200k, 400k, 600k, and 1M tokens. The fast default uses the nearest
available generated sidecar chunk to each target. Exact prefix recomputation is
available with --exact-prefix-metrics, but it can be slow without CUDA.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from pca_all_metrics import (  # noqa: E402
    DEFAULT_BATCH_RESULTS,
    DEFAULT_NATURAL_TOKEN_METRICS,
    DEFAULT_NATURAL_TOKEN_PREDICTABILITY,
    build_natural_dataframe,
    finite_metric_frame,
    fit_pca,
    apply_robust_score_limits,
    mean_score_trajectory_by_size_bin,
    nonconstant_metric_cols,
    plot_kde,
)
from add_token_naturalness_to_generated import (  # noqa: E402
    reconstruct_generated_text,
    tokenize_reconstruction,
)
from metrics import CoherenceMetrics  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402


DEFAULT_FOLDERS = [
    ROOT / "generated_texts_og" / "prompts" / "20260522_043453_preoom_repair_the_inventor_s_dilemma_a_guide_to_robotics_and_the_ethics_of_artificial_life_science_minimal_fixed_conditioning",
    ROOT / "generated_texts_og" / "prompts" / "20260601_001605_unveiling_mars_monoliths_mysteries_and_the_new_cosmos_science_basic_conditioning",
    ROOT / "generated_texts_og" / "prompts" / "20260521_144550_stellar_exodus_the_final_frontier_space_opera_markov_rolling_conditioning",
]
DEFAULT_OUTPUT_DIR = ROOT / "natural_texts" / "Output" / "pca_generated_prefixes"
DEFAULT_LANDMARK_METRICS_DIR = ROOT / "natural_texts" / "Selected_Books" / "metrics_with_token_naturalness"
DEFAULT_SIZE_STATS = ROOT / "natural_texts" / "Output" / "analysis" / "size_binned_metrics.json"
SIZE_BIN_COLORS = [
    "#d7eef8",  # light cyan for the densest short-book bin
    "#9ed9d3",
    "#5bb7a5",
    "#b88b12",
    "#7a174f",  # dark magenta for rare long books
    "#332288",  # dark fallback
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def load_sidecar(folder: Path) -> list[dict]:
    path = folder / "token_naturalness_by_chunk.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing sidecar: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"Empty sidecar: {path}")
    return records


def infer_conditioning_mode(folder: Path, metadata: dict) -> str:
    mode = metadata.get("conditioning_mode")
    if mode:
        return mode
    match = re.search(
        r"(minimal_fixed_conditioning|markov_rolling_conditioning|minimal_rolling_conditioning|basic_conditioning|summaries_conditioning|extended_conditioning)",
        folder.name,
    )
    return match.group(1) if match else "unknown"


def label_for_mode(mode: str) -> str:
    return mode.replace("_conditioning", "").replace("_", " ")


def parse_prefix_tokens(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip().lower().replace("_", "")
        if not item:
            continue
        multiplier = 1
        if item.endswith("k"):
            multiplier = 1_000
            item = item[:-1]
        elif item.endswith("m"):
            multiplier = 1_000_000
            item = item[:-1]
        values.append(int(float(item) * multiplier))
    if not values:
        raise ValueError("empty --prefix-tokens")
    return sorted(set(values))


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
        if hi > 5_000_000:
            label = f">{format_token_label(int(lo))}"
        else:
            label = f"{format_token_label(int(lo))}-{format_token_label(int(hi))}"
        bins.append({
            "name": item.get("name"),
            "label": label,
            "range_tokens": [int(lo), float(hi)],
        })
    return bins


def size_bin_colors(size_bins: list[dict]) -> dict[str, str]:
    return {
        item["label"]: SIZE_BIN_COLORS[idx % len(SIZE_BIN_COLORS)]
        for idx, item in enumerate(size_bins)
    }


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


def resolve_prefix_targets(final_tokens: int, requested_tokens: list[int]) -> list[int]:
    """Use exact requested targets up to final length; include final only if shorter."""
    targets = [target for target in requested_tokens if target <= final_tokens]
    larger_targets = [target for target in requested_tokens if target > final_tokens]
    if larger_targets and final_tokens not in targets:
        targets.append(final_tokens)
    return sorted(set(targets))


def closest_token_prefix_records(records: list[dict], prefix_tokens: list[int]) -> list[dict]:
    final_tokens = max(int(rec["n_tokens"]) for rec in records if rec.get("n_tokens") is not None)
    selected: dict[int, dict] = {}
    for target in resolve_prefix_targets(final_tokens, prefix_tokens):
        rec = min(records, key=lambda r: abs(float(r.get("n_tokens", 0)) - target))
        out = dict(rec)
        out["prefix_target_tokens"] = int(target)
        out["prefix_token_error"] = int(out.get("n_tokens", 0)) - int(target)
        out["prefix_label"] = format_token_label(int(target))
        chunk_id = int(out["chunk_id"])
        previous = selected.get(chunk_id)
        if previous is None or abs(out["prefix_token_error"]) < abs(previous["prefix_token_error"]):
            selected[chunk_id] = out
    return sorted(selected.values(), key=lambda rec: int(rec["prefix_target_tokens"]))


def chunk_info_for_token_target(reconstruction: dict, tokenized: dict, target_tokens: int) -> dict:
    cumulative = len(tokenized["seed_tokens"])
    if target_tokens <= cumulative:
        return {
            "chunk_id": 0,
            "accepted_attempt": 0,
            "selection_reason": "seed_prefix",
            "chunk_end_tokens": cumulative,
            "chunk_n_tokens": 0,
        }

    for chunk, chunk_tokens in zip(reconstruction["chunks"], tokenized["chunk_token_parts"]):
        cumulative += len(chunk_tokens)
        if target_tokens <= cumulative:
            return {
                "chunk_id": int(chunk["chunk_id"]),
                "accepted_attempt": int(chunk["attempt"]),
                "selection_reason": chunk["selection_reason"],
                "chunk_end_tokens": cumulative,
                "chunk_n_tokens": len(chunk_tokens),
            }

    if not reconstruction["chunks"]:
        return {
            "chunk_id": 0,
            "accepted_attempt": 0,
            "selection_reason": "empty",
            "chunk_end_tokens": cumulative,
            "chunk_n_tokens": 0,
        }
    last = reconstruction["chunks"][-1]
    last_tokens = tokenized["chunk_token_parts"][-1] if tokenized["chunk_token_parts"] else []
    return {
        "chunk_id": int(last["chunk_id"]),
        "accepted_attempt": int(last["attempt"]),
        "selection_reason": last["selection_reason"],
        "chunk_end_tokens": cumulative,
        "chunk_n_tokens": len(last_tokens),
    }


def compute_exact_prefix_metrics(tokens: list[int], device: str, max_order: int) -> dict:
    cm = CoherenceMetrics(tokens, device=device)
    return cm.compute_all(
        include_token_naturalness=True,
        token_predictability_max_order=max_order,
        include_token_context_dynamics=False,
    )


def load_prefix_metric_rows(
    folders: list[Path],
    metric_cols: list[str],
    prefix_tokens: list[int],
    tokenizer_name: str,
    device: str,
    max_order: int,
    include_seed: bool,
) -> pd.DataFrame:
    rows = []
    tokenizer = Tokenizer(kind=tokenizer_name)
    for folder in folders:
        metadata_path = folder / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        mode = infer_conditioning_mode(folder, metadata)
        title = metadata.get("title") or folder.name
        reconstruction = reconstruct_generated_text(folder, include_seed=include_seed)
        tokenized = tokenize_reconstruction(reconstruction, tokenizer, include_seed=include_seed)
        full_tokens = tokenized["full_tokens"]
        final_tokens = len(full_tokens)

        for target in resolve_prefix_targets(final_tokens, prefix_tokens):
            prefix = full_tokens[:target]
            metrics = compute_exact_prefix_metrics(prefix, device=device, max_order=max_order)
            chunk_info = chunk_info_for_token_target(reconstruction, tokenized, target)
            row = dict(metrics)
            row.update({
                "folder_name": folder.name,
                "title": title,
                "conditioning_mode": mode,
                "condition_label": label_for_mode(mode),
                "chunk_id": chunk_info["chunk_id"],
                "accepted_attempt": chunk_info["accepted_attempt"],
                "selection_reason": chunk_info["selection_reason"],
                "chunk_end_tokens": chunk_info["chunk_end_tokens"],
                "n_tokens": target,
                "prefix_target_tokens": target,
                "prefix_label": format_token_label(target),
                "prefix_fraction": target / final_tokens if final_tokens else np.nan,
                "n_tokens_full_book": final_tokens,
                "include_seed": include_seed,
            })
            if all(isinstance(row.get(col), (int, float)) and math.isfinite(row[col]) for col in metric_cols):
                rows.append(row)
    return pd.DataFrame(rows)


def load_sidecar_prefix_metric_rows(
    folders: list[Path],
    metric_cols: list[str],
    prefix_tokens: list[int],
) -> pd.DataFrame:
    rows = []
    for folder in folders:
        sidecar_records = load_sidecar(folder)
        selected = closest_token_prefix_records(sidecar_records, prefix_tokens)
        metadata_path = folder / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}
        mode = infer_conditioning_mode(folder, metadata)
        title = metadata.get("title") or folder.name

        final_tokens = max(int(rec["n_tokens"]) for rec in sidecar_records if rec.get("n_tokens") is not None)
        for rec in selected:
            chunk_id = int(rec["chunk_id"])
            attempt = int(rec.get("accepted_attempt") or 1)
            attempt_path = folder / f"chunk_{chunk_id:03d}_attempt_{attempt}.txt"
            if not attempt_path.exists():
                continue
            attempt_data = read_json(attempt_path)
            row = {}
            row.update(attempt_data.get("accumulated_metrics") or {})
            row.update({k: v for k, v in rec.items() if k.startswith(("td_", "tr_", "tp_"))})
            row.update({
                "folder_name": folder.name,
                "title": title,
                "conditioning_mode": mode,
                "condition_label": label_for_mode(mode),
                "chunk_id": chunk_id,
                "accepted_attempt": attempt,
                "selection_reason": rec.get("selection_reason"),
                "n_tokens": rec.get("n_tokens"),
                "prefix_target_tokens": rec.get("prefix_target_tokens"),
                "prefix_token_error": rec.get("prefix_token_error"),
                "prefix_label": rec.get("prefix_label") or format_token_label(int(rec.get("n_tokens", 0))),
                "prefix_fraction": float(rec.get("n_tokens", 0)) / final_tokens if final_tokens else np.nan,
                "n_tokens_full_book": final_tokens,
            })
            if all(isinstance(row.get(col), (int, float)) and math.isfinite(row[col]) for col in metric_cols):
                rows.append(row)
    return pd.DataFrame(rows)


def load_landmark_prefix_rows(
    landmark_metrics_dir: Path,
    metric_cols: list[str],
    landmark_filter: str = "outlier",
) -> pd.DataFrame:
    rows = []
    if not landmark_metrics_dir.exists():
        return pd.DataFrame(rows)

    for path in sorted(landmark_metrics_dir.glob("metrics_*.json")):
        try:
            records = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(records, list):
            continue
        if landmark_filter == "outlier" and not bool(records[0].get("is_outlier")):
            continue

        final_tokens = max((int(rec.get("n_tokens") or 0) for rec in records), default=0)
        for rec in records:
            row = dict(rec)
            n_tokens = int(row.get("n_tokens") or 0)
            row.update({
                "landmark_metric_file": path.name,
                "source_kind": "selected_natural_landmark_prefix",
                "prefix_target_tokens": n_tokens,
                "prefix_label": format_token_label(n_tokens),
                "prefix_fraction": n_tokens / final_tokens if final_tokens else np.nan,
            })
            if all(isinstance(row.get(col), (int, float)) and math.isfinite(row[col]) for col in metric_cols):
                rows.append(row)

    return pd.DataFrame(rows)


def add_endpoint_arrow(
    ax,
    group: pd.DataFrame,
    x_col: str,
    y_col: str,
    color: str,
    linewidth: float = 2.2,
    alpha: float = 0.88,
    mutation_scale: float = 16,
    shrink: float = 12,
    zorder: float = 9,
) -> None:
    """Add a single arrowhead on the final segment of a trajectory."""
    xs = group[x_col].to_numpy(dtype=float)
    ys = group[y_col].to_numpy(dtype=float)
    if len(xs) < 2:
        return

    x0, y0 = xs[-2], ys[-2]
    x1, y1 = xs[-1], ys[-1]
    if not all(np.isfinite([x0, y0, x1, y1])):
        return
    if np.hypot(x1 - x0, y1 - y0) < 1e-9:
        return

    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=linewidth,
            alpha=alpha,
            shrinkA=shrink,
            shrinkB=shrink,
            mutation_scale=mutation_scale,
        ),
        zorder=zorder,
    )


def plot_prefix_scores(
    natural: pd.DataFrame,
    generated: pd.DataFrame,
    pca,
    output_path: Path,
    pc_x: int,
    pc_y: int,
    landmarks: pd.DataFrame | None = None,
    size_bins: list[dict] | None = None,
    mean_score_trajectory: pd.DataFrame | None = None,
) -> None:
    x_col = f"PC{pc_x}"
    y_col = f"PC{pc_y}"
    fig, ax = plt.subplots(figsize=(14, 10.5))

    points = natural[[x_col, y_col]].to_numpy()
    plot_kde(ax, points, "gray")
    size_bins = size_bins or []
    bin_colors = size_bin_colors(size_bins)
    if size_bins and "token_size_bin" in natural:
        for item in size_bins:
            label = item["label"]
            group = natural[natural["token_size_bin"] == label]
            if group.empty:
                continue
            group_points = group[[x_col, y_col]].to_numpy()
            ax.scatter(
                group_points[:, 0],
                group_points[:, 1],
                s=12,
                color=bin_colors[label],
                alpha=0.38,
                edgecolors="none",
                label=f"Natural {label} ({len(group)})",
                zorder=2,
            )
        unknown = natural[natural["token_size_bin"] == "unknown"]
        if not unknown.empty:
            unknown_points = unknown[[x_col, y_col]].to_numpy()
            ax.scatter(
                unknown_points[:, 0],
                unknown_points[:, 1],
                s=8,
                color="gray",
                alpha=0.12,
                edgecolors="none",
                label=f"Natural unknown ({len(unknown)})",
                zorder=2,
            )
    else:
        ax.scatter(points[:, 0], points[:, 1], s=8, color="gray", alpha=0.09, edgecolors="none", label="Natural books")

    if (
        mean_score_trajectory is not None
        and not mean_score_trajectory.empty
        and x_col in mean_score_trajectory
        and y_col in mean_score_trajectory
    ):
        trajectory = mean_score_trajectory.sort_values("bin_order").replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
        if not trajectory.empty:
            ax.plot(
                trajectory[x_col],
                trajectory[y_col],
                color="#111827",
                linewidth=2.3,
                alpha=0.88,
                label="Mean natural score trajectory",
                zorder=6,
            )
            ax.scatter(
                trajectory[x_col],
                trajectory[y_col],
                marker="*",
                s=260,
                color="#ffd166",
                edgecolors="#111827",
                linewidths=1.15,
                zorder=7,
            )
            for _, row in trajectory.iterrows():
                ax.text(
                    row[x_col] + 0.12,
                    row[y_col] + 0.12,
                    str(row["token_size_bin"]),
                    fontsize=8,
                    color="#111827",
                    bbox=dict(facecolor="white", edgecolor="#111827", alpha=0.84, boxstyle="round,pad=0.14"),
                    zorder=8,
                )

    if landmarks is not None and not landmarks.empty:
        for _, group in landmarks.groupby("landmark_metric_file", sort=False):
            group = group.sort_values("n_tokens")
            ax.plot(
                group[x_col],
                group[y_col],
                color="#374151",
                linewidth=1.0,
                alpha=0.34,
                zorder=4,
            )
            add_endpoint_arrow(
                ax,
                group,
                x_col,
                y_col,
                "#374151",
                linewidth=1.05,
                alpha=0.48,
                mutation_scale=9,
                shrink=5,
                zorder=4.5,
            )
        if size_bins and "token_size_bin" in landmarks:
            first = True
            for item in size_bins:
                label = item["label"]
                group = landmarks[landmarks["token_size_bin"] == label]
                if group.empty:
                    continue
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    s=34,
                    marker="D",
                    facecolors="none",
                    edgecolors=bin_colors[label],
                    linewidths=1.35,
                    alpha=0.9,
                    label="Selected natural landmarks" if first else None,
                    zorder=5,
                )
                first = False
        else:
            ax.scatter(
                landmarks[x_col],
                landmarks[y_col],
                s=32,
                marker="D",
                facecolors="none",
                edgecolors="#111827",
                linewidths=0.8,
                alpha=0.55,
                label="Selected natural landmarks",
                zorder=5,
            )

    colors = {
        "minimal_fixed_conditioning": "#d00000",
        "basic_conditioning": "#1d4ed8",
        "markov_rolling_conditioning": "#f59e0b",
        "minimal_rolling_conditioning": "#f59e0b",
        "unknown": "black",
    }
    for (mode, folder_name), group in generated.groupby(["conditioning_mode", "folder_name"], sort=False):
        group = group.sort_values("n_tokens")
        color = colors.get(mode, "black")
        ax.plot(group[x_col], group[y_col], color=color, linewidth=2.0, alpha=0.9, label=label_for_mode(mode))
        add_endpoint_arrow(ax, group, x_col, y_col, color)
        ax.scatter(group[x_col], group[y_col], color=color, s=70, edgecolors="black", linewidths=0.6, zorder=10)
        for _, row in group.iterrows():
            ax.text(
                row[x_col],
                row[y_col],
                str(row.get("prefix_label") or format_token_label(int(row["n_tokens"]))),
                fontsize=8,
                color="black",
                ha="center",
                va="center",
                zorder=11,
            )
        final = group.iloc[-1]
        ax.scatter(final[x_col], final[y_col], color=color, s=145, marker="D", edgecolors="black", linewidths=0.9, zorder=12)

    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax.legend(dedup.values(), dedup.keys(), loc="best", framealpha=0.92, fontsize=8)
    ax.set_xlabel(f"{x_col} ({pca.explained_variance_ratio_[pc_x - 1] * 100:.1f}%)")
    ax.set_ylabel(f"{y_col} ({pca.explained_variance_ratio_[pc_y - 1] * 100:.1f}%)")
    ax.set_title(
        "Generated-prefix projections on natural-book PCA score density\n"
        "Metric subset: original stat/info + curated token-naturalness metrics\n"
        "Display axes use natural-score 0.5-99.5% quantiles to reduce outlier compression"
    )
    ax.grid(True, alpha=0.25)
    apply_robust_score_limits(ax, points)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-results", type=Path, default=DEFAULT_BATCH_RESULTS)
    parser.add_argument(
        "--natural-token-metrics",
        type=Path,
        nargs="+",
        default=[DEFAULT_NATURAL_TOKEN_METRICS, DEFAULT_NATURAL_TOKEN_PREDICTABILITY],
    )
    parser.add_argument("--folder", type=Path, action="append", default=None)
    parser.add_argument("--landmark-metrics-dir", type=Path, default=DEFAULT_LANDMARK_METRICS_DIR)
    parser.add_argument(
        "--landmark-filter",
        choices=["outlier", "all"],
        default="outlier",
        help="Which selected natural-book trajectories to show as landmarks.",
    )
    parser.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument(
        "--prefix-tokens",
        default="100000,200000,400000,600000,1000000",
        help="Comma-separated exact generated prefix lengths. Supports k/m suffixes.",
    )
    parser.add_argument("--tokenizer", default="gpt4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--include-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exact-prefix-metrics",
        action="store_true",
        help="Recompute all generated metrics on exact token prefixes instead of using nearest sidecar chunks.",
    )
    args = parser.parse_args()

    folders = args.folder or DEFAULT_FOLDERS
    prefix_tokens = parse_prefix_tokens(args.prefix_tokens)
    size_bins = load_size_bins(args.size_stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    natural_raw, metric_cols_all = build_natural_dataframe(args.batch_results, args.natural_token_metrics)
    natural = finite_metric_frame(natural_raw, metric_cols_all)
    metric_cols, dropped_metrics = nonconstant_metric_cols(natural, metric_cols_all)
    coords, pca, scaler = fit_pca(natural, metric_cols, args.components)
    coord_df = pd.DataFrame(coords, columns=[f"PC{i + 1}" for i in range(coords.shape[1])], index=natural.index)
    natural = pd.concat([natural, coord_df], axis=1)
    natural = assign_token_size_bins(natural, size_bins)
    pc_cols = [f"PC{i + 1}" for i in range(coords.shape[1])]
    mean_score_trajectory = mean_score_trajectory_by_size_bin(natural, size_bins, pc_cols)

    if args.exact_prefix_metrics:
        generated_raw = load_prefix_metric_rows(
            folders,
            metric_cols,
            prefix_tokens,
            tokenizer_name=args.tokenizer,
            device=args.device,
            max_order=args.max_order,
            include_seed=args.include_seed,
        )
        generated_prefix_metric_mode = "exact_token_prefix_recomputed"
    else:
        generated_raw = load_sidecar_prefix_metric_rows(folders, metric_cols, prefix_tokens)
        generated_prefix_metric_mode = "nearest_sidecar_chunk_to_token_target"
    if generated_raw.empty:
        raise SystemExit("No generated prefix rows could be projected.")
    gen_scaled = scaler.transform(generated_raw[metric_cols].to_numpy(dtype=float))
    gen_coords = pca.transform(gen_scaled)
    generated = pd.concat(
        [
            generated_raw.reset_index(drop=True),
            pd.DataFrame(gen_coords, columns=[f"PC{i + 1}" for i in range(gen_coords.shape[1])]),
        ],
        axis=1,
    )

    if args.no_landmarks:
        landmarks = pd.DataFrame()
    else:
        landmarks_raw = load_landmark_prefix_rows(
            args.landmark_metrics_dir,
            metric_cols,
            landmark_filter=args.landmark_filter,
        )
        if not landmarks_raw.empty:
            landmark_scaled = scaler.transform(landmarks_raw[metric_cols].to_numpy(dtype=float))
            landmark_coords = pca.transform(landmark_scaled)
            landmarks = pd.concat(
                [
                    landmarks_raw.reset_index(drop=True),
                    pd.DataFrame(
                        landmark_coords,
                        columns=[f"PC{i + 1}" for i in range(landmark_coords.shape[1])],
                    ),
                ],
                axis=1,
            )
        else:
            landmarks = pd.DataFrame()
    if not landmarks.empty:
        landmarks = assign_token_size_bins(landmarks, size_bins)

    scores_out = args.output_dir / "generated_prefix_projection_scores.csv"
    landmark_scores_out = args.output_dir / "landmark_prefix_projection_scores.csv"
    mean_trajectory_out = args.output_dir / "natural_mean_score_trajectory_by_token_bin.csv"
    summary_out = args.output_dir / "generated_prefix_projection_summary.json"
    plot_pc1_pc2 = args.output_dir / "generated_prefix_projection_pc1_pc2.jpg"
    plot_pc3_pc4 = args.output_dir / "generated_prefix_projection_pc3_pc4.jpg"

    generated.to_csv(scores_out, index=False)
    if not landmarks.empty:
        landmarks.to_csv(landmark_scores_out, index=False)
    if not mean_score_trajectory.empty:
        mean_score_trajectory.to_csv(mean_trajectory_out, index=False)
    elif mean_trajectory_out.exists():
        mean_trajectory_out.unlink()
    summary_out.write_text(json.dumps({
        "natural_input_rows": int(len(natural_raw)),
        "natural_clean_rows": int(len(natural)),
        "generated_prefix_rows": int(len(generated)),
        "landmark_prefix_rows": int(len(landmarks)),
        "landmark_metrics_dir": str(args.landmark_metrics_dir),
        "landmark_filter": args.landmark_filter,
        "size_stats": str(args.size_stats),
        "size_bins": size_bins,
        "mean_score_trajectory_bins": mean_score_trajectory.to_dict(orient="records"),
        "folders": [str(folder) for folder in folders],
        "prefix_tokens_requested": prefix_tokens,
        "generated_prefix_metric_mode": generated_prefix_metric_mode,
        "include_seed": args.include_seed,
        "tokenizer": args.tokenizer,
        "device": args.device,
        "max_order": args.max_order,
        "metric_count": len(metric_cols),
        "token_metric_set": "curated",
        "include_chunk_context_metrics": False,
        "metrics": metric_cols,
        "dropped_zero_variance_metrics": dropped_metrics,
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "natural_token_metric_files": [str(path) for path in args.natural_token_metrics],
    }, indent=2), encoding="utf-8")

    plot_prefix_scores(
        natural,
        generated,
        pca,
        plot_pc1_pc2,
        1,
        2,
        landmarks=landmarks,
        size_bins=size_bins,
        mean_score_trajectory=mean_score_trajectory,
    )
    plot_prefix_scores(
        natural,
        generated,
        pca,
        plot_pc3_pc4,
        3,
        4,
        landmarks=landmarks,
        size_bins=size_bins,
        mean_score_trajectory=mean_score_trajectory,
    )

    print(f"Natural rows: {len(natural)} / {len(natural_raw)}")
    print(f"Generated prefix rows: {len(generated)}")
    print(f"Landmark prefix rows: {len(landmarks)}")
    print(f"Metrics: {len(metric_cols)}")
    if dropped_metrics:
        print(f"Dropped zero-variance metrics: {dropped_metrics}")
    print(f"Explained variance: {[round(float(x), 4) for x in pca.explained_variance_ratio_]}")
    print(f"Saved PC1xPC2 plot -> {plot_pc1_pc2}")
    print(f"Saved PC3xPC4 plot -> {plot_pc3_pc4}")
    print(f"Saved projections -> {scores_out}")
    if not landmarks.empty:
        print(f"Saved landmark projections -> {landmark_scores_out}")
    if not mean_score_trajectory.empty:
        print(f"Saved mean score trajectory -> {mean_trajectory_out}")
    print(f"Saved summary -> {summary_out}")


if __name__ == "__main__":
    main()
