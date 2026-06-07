#!/usr/bin/env python3
"""
Build natural benchmark envelopes for retrospective token-naturalness metrics.

Inputs:
  natural_texts/Output/gutenberg_benchmark_cache/token_naturalness_metrics.jsonl
  natural_texts/Output/gutenberg_benchmark_cache/token_predictability_metrics.jsonl
  natural_texts/Output/gutenberg_benchmark_cache/token_context_dynamics_metrics.jsonl

Outputs:
  natural_texts/Output/analysis/token_naturalness_global_metrics.json
  natural_texts/Output/analysis/token_naturalness_size_binned_metrics.json

By default this also merges the token-naturalness metrics into the existing
natural_texts/Output/analysis/size_binned_metrics.json, which is what
plot_chunk_metrics.py reads for natural reference bands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import (  # noqa: E402
    CURATED_TOKEN_NATURALNESS_METRICS,
    OPTIONAL_CHUNK_CONTEXT_METRICS,
)

DEFAULT_METRICS_JSONL = (
    ROOT
    / "natural_texts"
    / "Output"
    / "gutenberg_benchmark_cache"
    / "token_naturalness_metrics.jsonl"
)
DEFAULT_TOKEN_PREDICTABILITY_JSONL = (
    ROOT
    / "natural_texts"
    / "Output"
    / "gutenberg_benchmark_cache"
    / "token_predictability_metrics.jsonl"
)
DEFAULT_TOKEN_CONTEXT_DYNAMICS_JSONL = (
    ROOT
    / "natural_texts"
    / "Output"
    / "gutenberg_benchmark_cache"
    / "token_context_dynamics_metrics.jsonl"
)
DEFAULT_ANALYSIS_DIR = ROOT / "natural_texts" / "Output" / "analysis"
DEFAULT_SIZE_STATS = DEFAULT_ANALYSIS_DIR / "size_binned_metrics.json"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_and_merge_jsonls(paths: list[Path]) -> list[dict]:
    merged_by_id: dict[int, dict] = {}
    for path in paths:
        records = load_jsonl(path)
        if not records:
            raise SystemExit(f"No records loaded from {path}")
        for record in records:
            book_id = record.get("book_id")
            if book_id is None:
                raise SystemExit(f"Missing book_id in {path}")
            target = merged_by_id.setdefault(int(book_id), {})
            target.update(record)
    return list(merged_by_id.values())


def finite_values(records: Iterable[dict], metric: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p97_5": float(np.percentile(arr, 97.5)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def discover_token_metrics(
    records: list[dict],
    token_metric_set: str = "curated",
    include_chunk_context_metrics: bool = False,
) -> list[str]:
    prefixes = ("td_", "tr_", "tp_", "tc_")
    metrics = set()
    for record in records:
        metrics.update(
            key
            for key, value in record.items()
            if key.startswith(prefixes)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )
    if token_metric_set == "all":
        if include_chunk_context_metrics:
            return sorted(metrics)
        return sorted(metric for metric in metrics if not metric.startswith("tc_"))

    if token_metric_set != "curated":
        raise ValueError(f"unknown token metric set: {token_metric_set}")

    keep = set(CURATED_TOKEN_NATURALNESS_METRICS)
    if include_chunk_context_metrics:
        keep.update(OPTIONAL_CHUNK_CONTEXT_METRICS)
    return sorted(metric for metric in metrics if metric in keep)


def default_bins(records: list[dict]) -> list[dict]:
    max_tokens = max(int(r.get("n_tokens", 0)) for r in records)
    specs = [
        ("0-(100-150k)", 100_000, 150_000),
        ("1-(150-300k)", 150_000, 300_000),
        ("2-(300-500k)", 300_000, 500_000),
        ("3-(500-800k)", 500_000, 800_000),
        ("4-(>800k)", 800_000, max_tokens + 1),
    ]
    return [
        {
            "name": name,
            "range_tokens": [lo, hi],
            "metrics": {},
            "count": 0,
        }
        for name, lo, hi in specs
    ]


def load_or_create_size_stats(records: list[dict], path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    bins = default_bins(records)
    return {
        "edges": [b["range_tokens"][0] for b in bins] + [bins[-1]["range_tokens"][1]],
        "metrics": [],
        "bins": bins,
    }


def compute_size_binned(records: list[dict], metrics: list[str], template: dict) -> dict:
    bins = []
    for template_bin in template["bins"]:
        lo, hi = template_bin["range_tokens"]
        bucket_records = [
            record
            for record in records
            if record.get("n_tokens") is not None and lo <= int(record["n_tokens"]) < hi
        ]
        bucket = {
            "name": template_bin["name"],
            "range_tokens": template_bin["range_tokens"],
            "metrics": {},
            "count": len(bucket_records),
        }
        for metric in metrics:
            stats = summarize(finite_values(bucket_records, metric))
            if stats:
                bucket["metrics"][metric] = stats
        bins.append(bucket)

    return {
        "edges": template.get("edges"),
        "metrics": metrics,
        "bins": bins,
    }


def merge_size_binned(existing: dict, token_bench: dict, source_path: Path) -> dict:
    token_prefixes = ("td_", "tr_", "tp_", "tc_")
    current_token_metrics = set(token_bench["metrics"])
    existing_metrics = [
        metric
        for metric in existing.get("metrics", [])
        if not str(metric).startswith(token_prefixes) or metric in current_token_metrics
    ]
    for metric in token_bench["metrics"]:
        if metric not in existing_metrics:
            existing_metrics.append(metric)
    existing["metrics"] = existing_metrics

    existing_bins = {b["name"]: b for b in existing.get("bins", [])}
    for token_bin in token_bench["bins"]:
        target = existing_bins.get(token_bin["name"])
        if target is None:
            existing.setdefault("bins", []).append(token_bin)
            continue
        target_metrics = target.setdefault("metrics", {})
        for metric in list(target_metrics):
            if str(metric).startswith(token_prefixes) and metric not in current_token_metrics:
                target_metrics.pop(metric, None)
        target.setdefault("metrics", {}).update(token_bin.get("metrics", {}))
        target["count_token_naturalness"] = token_bin.get("count", 0)
    existing["token_naturalness_benchmark"] = {
        "source": str(source_path),
        "metric_count": len(token_bench["metrics"]),
    }
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        nargs="+",
        default=[
            DEFAULT_METRICS_JSONL,
            DEFAULT_TOKEN_PREDICTABILITY_JSONL,
            DEFAULT_TOKEN_CONTEXT_DYNAMICS_JSONL,
        ],
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--size-stats", type=Path, default=DEFAULT_SIZE_STATS)
    parser.add_argument("--token-metric-set", choices=["curated", "all"], default="curated")
    parser.add_argument(
        "--include-chunk-context-metrics",
        action="store_true",
        help="Include optional tc_* metrics. Off by default because natural-book boundaries are synthetic.",
    )
    parser.add_argument("--no-merge-size-stats", action="store_true")
    args = parser.parse_args()

    records = load_and_merge_jsonls(args.metrics_jsonl)
    if not records:
        raise SystemExit(f"No records loaded from {args.metrics_jsonl}")

    metrics = discover_token_metrics(
        records,
        token_metric_set=args.token_metric_set,
        include_chunk_context_metrics=args.include_chunk_context_metrics,
    )
    if not metrics:
        raise SystemExit("No token-naturalness metrics found.")

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    template = load_or_create_size_stats(records, args.size_stats)
    size_binned = compute_size_binned(records, metrics, template)
    global_stats = {
        "source": [str(path) for path in args.metrics_jsonl],
        "record_count": len(records),
        "metrics": metrics,
        "token_metric_set": args.token_metric_set,
        "include_chunk_context_metrics": args.include_chunk_context_metrics,
        "n_tokens": summarize(finite_values(records, "n_tokens")),
        "metric_stats": {
            metric: summarize(finite_values(records, metric))
            for metric in metrics
        },
    }

    global_out = args.analysis_dir / "token_naturalness_global_metrics.json"
    size_out = args.analysis_dir / "token_naturalness_size_binned_metrics.json"
    global_out.write_text(json.dumps(global_stats, indent=2), encoding="utf-8")
    size_out.write_text(json.dumps(size_binned, indent=2), encoding="utf-8")

    print(f"Loaded natural token metrics: {len(records)} records")
    print(f"Token-naturalness metrics: {len(metrics)}")
    print(f"Saved global benchmark -> {global_out}")
    print(f"Saved size-binned benchmark -> {size_out}")

    if not args.no_merge_size_stats:
        existing = load_or_create_size_stats(records, args.size_stats)
        merged = merge_size_binned(existing, size_binned, args.metrics_jsonl[0])
        merged["token_naturalness_benchmark"]["sources"] = [str(path) for path in args.metrics_jsonl]
        args.size_stats.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(f"Merged token benchmarks into -> {args.size_stats}")


if __name__ == "__main__":
    main()
