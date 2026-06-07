#!/usr/bin/env python3
"""
Compute token-naturalness metrics for selected natural-book landmark prefixes.

Inputs are the existing Selected_Books/metrics_*.json trajectory files. Each
record already defines a landmark prefix through n_tokens. This script keeps
those original files unchanged and writes:

  - per-book sidecars under Selected_Books/token_naturalness_prefixes/
  - merged per-book copies under Selected_Books/metrics_with_token_naturalness/
  - one aggregate JSONL under natural_texts/Output/selected_landmark_token_naturalness/

The default metric set is td_* + tr_* + tp_*. TokenContextDynamics (tc_*) is
optional because it is much more expensive on many prefixes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import (  # noqa: E402
    CoherenceMetrics,
    filter_token_naturalness_metrics,
    token_context_dynamics_metrics,
    token_distribution_scaling_metrics,
    token_recurrence_metrics,
)
from tokenizer import Tokenizer  # noqa: E402


PROJECT_ROOT = ROOT
DEFAULT_SELECTED_DIR = PROJECT_ROOT / "natural_texts" / "Selected_Books"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "natural_texts" / "Output" / "selected_landmark_token_naturalness"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json_value(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json_value(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json_value(payload), indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(clean_json_value(record), ensure_ascii=False) + "\n")


def metric_record_sort_key(path: Path) -> str:
    return path.name.lower()


def selected_text_path(selected_dir: Path, metric_records: list[dict[str, Any]]) -> Path:
    source = metric_records[0].get("file") if metric_records else None
    if not source:
        raise ValueError("metric file has no 'file' key in its first record")
    path = selected_dir / f"{source}.txt"
    if not path.exists():
        raise FileNotFoundError(f"source text not found: {path}")
    return path


def compute_prefix_metrics(tokens: list[int], args_dict: dict[str, Any]) -> dict[str, Any]:
    metric_set = args_dict["metric_set"]
    max_order = int(args_dict["max_order"])
    device = args_dict["device"]
    metrics: dict[str, Any] = {}

    cm = CoherenceMetrics(tokens, device=device)
    if metric_set == "fast":
        metrics.update(token_distribution_scaling_metrics(tokens))
        metrics.update(token_recurrence_metrics(tokens))
    elif metric_set == "token_predictability":
        metrics.update(cm.compute_token_predictability(max_order=max_order))
    elif metric_set in {"all", "curated"}:
        metrics.update(cm.compute_token_naturalness(token_predictability_max_order=max_order))
    elif metric_set == "context_dynamics":
        pass
    else:
        raise ValueError(f"unknown metric_set: {metric_set}")

    if args_dict["include_context_dynamics"] or metric_set == "context_dynamics":
        pseudo_chunk_tokens = args_dict["pseudo_chunk_tokens"]
        if pseudo_chunk_tokens and pseudo_chunk_tokens > 0:
            boundaries = list(range(int(pseudo_chunk_tokens), len(tokens), int(pseudo_chunk_tokens)))
        else:
            boundaries = None
        metrics.update(
            token_context_dynamics_metrics(
                tokens=tokens,
                boundaries=boundaries,
                max_order=max_order,
                boundary_window=int(args_dict["boundary_window"]),
                n_checkpoints=int(args_dict["n_checkpoints"]),
                rank_order=int(args_dict["rank_order"]),
            )
        )

    if metric_set == "curated":
        metrics = filter_token_naturalness_metrics(
            metrics,
            include_chunk_context=bool(args_dict["include_context_dynamics"]),
        )

    return metrics


def process_metric_file(task: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]:
    metric_path = Path(task["metric_path"])
    selected_dir = Path(task["selected_dir"])
    tokenizer = Tokenizer(
        kind=task["tokenizer"],
        lowercase=task["lowercase"],
        normalize_digits=task["normalize_digits"],
        add_space_punct=task["add_space_punct"],
        pretokenize=task["pretokenize"],
    )

    records = json.loads(metric_path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"empty or invalid metric trajectory: {metric_path}")

    text_path = selected_text_path(selected_dir, records)
    text = text_path.read_text(encoding="utf-8", errors="replace")
    tokens = tokenizer.encode(text)

    prefix_records: list[dict[str, Any]] = []
    merged_records: list[dict[str, Any]] = []
    computed_at = utc_now_iso()

    args_dict = {
        "metric_set": task["metric_set"],
        "max_order": task["max_order"],
        "device": task["device"],
        "include_context_dynamics": task["include_context_dynamics"],
        "pseudo_chunk_tokens": task["pseudo_chunk_tokens"],
        "boundary_window": task["boundary_window"],
        "n_checkpoints": task["n_checkpoints"],
        "rank_order": task["rank_order"],
    }

    for record in records:
        target_tokens = int(record.get("n_tokens") or 0)
        prefix_len = min(max(target_tokens, 0), len(tokens))
        prefix_tokens = tokens[:prefix_len]
        new_metrics = compute_prefix_metrics(prefix_tokens, args_dict)

        metadata = {
            "source_metric_file": metric_path.name,
            "source_text_file": text_path.name,
            "file": record.get("file"),
            "title": record.get("title"),
            "genre": record.get("genre"),
            "model": record.get("model"),
            "chunk_id": record.get("chunk_id"),
            "is_outlier": record.get("is_outlier"),
            "computed_at": computed_at,
            "tokenizer": task["tokenizer"],
            "lowercase": task["lowercase"],
            "normalize_digits": task["normalize_digits"],
            "add_space_punct": task["add_space_punct"],
            "pretokenize": task["pretokenize"],
            "device": task["device"],
            "metric_set": task["metric_set"],
            "max_order": task["max_order"],
            "include_context_dynamics": task["include_context_dynamics"],
            "pseudo_chunk_tokens": task["pseudo_chunk_tokens"],
            "n_tokens_target": target_tokens,
            "n_tokens": prefix_len,
            "n_tokens_full_book": len(tokens),
            "new_metric_family_version": "token_naturalness_v1",
        }
        prefix_record = {**metadata, **new_metrics}
        prefix_records.append(prefix_record)

        merged = copy.deepcopy(record)
        merged.update(prefix_record)
        merged_records.append(merged)

    return metric_path.name, text_path.name, prefix_records, merged_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-dir", type=Path, default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer", default="gpt4")
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--normalize-digits", action="store_true")
    parser.add_argument("--no-add-space-punct", dest="add_space_punct", action="store_false")
    parser.set_defaults(add_space_punct=True)
    parser.add_argument("--pretokenize", default="none")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument(
        "--metric-set",
        choices=["fast", "token_predictability", "context_dynamics", "curated", "all"],
        default="curated",
        help="curated=selected td/tr/tp metrics; all=td_*+tr_*+tp_*; context_dynamics computes tc_* only.",
    )
    parser.add_argument("--include-context-dynamics", action="store_true")
    parser.add_argument("--pseudo-chunk-tokens", type=int, default=900)
    parser.add_argument("--boundary-window", type=int, default=64)
    parser.add_argument("--n-checkpoints", type=int, default=24)
    parser.add_argument("--rank-order", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--write-merged-copy",
        action="store_true",
        help="Write old+new merged metric trajectories under Selected_Books/metrics_with_token_naturalness.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_files = sorted(args.selected_dir.glob("metrics_*.json"), key=metric_record_sort_key)
    sidecar_dir = args.selected_dir / "token_naturalness_prefixes"
    merged_dir = args.selected_dir / "metrics_with_token_naturalness"
    aggregate_jsonl = args.output_dir / "landmark_prefix_token_naturalness.jsonl"
    errors_jsonl = args.output_dir / "landmark_prefix_token_naturalness_errors.jsonl"

    if not metric_files:
        raise SystemExit(f"No metrics_*.json files found in {args.selected_dir}")

    if args.force and aggregate_jsonl.exists():
        aggregate_jsonl.unlink()
    if args.force and errors_jsonl.exists():
        errors_jsonl.unlink()

    tasks = []
    skipped = 0
    for metric_path in metric_files:
        sidecar_path = sidecar_dir / metric_path.name
        merged_path = merged_dir / metric_path.name
        if sidecar_path.exists() and (not args.write_merged_copy or merged_path.exists()) and not args.force:
            skipped += 1
            continue
        tasks.append(
            {
                "metric_path": str(metric_path),
                "selected_dir": str(args.selected_dir),
                "tokenizer": args.tokenizer,
                "lowercase": args.lowercase,
                "normalize_digits": args.normalize_digits,
                "add_space_punct": args.add_space_punct,
                "pretokenize": args.pretokenize,
                "device": args.device,
                "max_order": args.max_order,
                "metric_set": args.metric_set,
                "include_context_dynamics": args.include_context_dynamics,
                "pseudo_chunk_tokens": args.pseudo_chunk_tokens,
                "boundary_window": args.boundary_window,
                "n_checkpoints": args.n_checkpoints,
                "rank_order": args.rank_order,
            }
        )

    processed = 0
    failed = 0

    def handle_result(result: tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]]) -> None:
        nonlocal processed
        metric_name, text_name, prefix_records, merged_records = result
        sidecar_path = sidecar_dir / metric_name
        write_json(sidecar_path, prefix_records)
        if args.write_merged_copy:
            write_json(merged_dir / metric_name, merged_records)
        append_jsonl(aggregate_jsonl, prefix_records)
        processed += 1
        print(
            f"[{processed}/{len(tasks)}] {metric_name}: {len(prefix_records)} prefixes "
            f"from {text_name}"
        )

    print(
        "landmark token metrics: "
        f"metric_files={len(metric_files)}, tasks={len(tasks)}, skipped={skipped}, "
        f"workers={args.workers}, metric_set={args.metric_set}, "
        f"include_context_dynamics={args.include_context_dynamics}"
    )

    if args.workers and args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(process_metric_file, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    handle_result(future.result())
                except Exception as exc:
                    failed += 1
                    append_jsonl(
                        errors_jsonl,
                        [{"metric_path": task["metric_path"], "error": str(exc), "failed_at": utc_now_iso()}],
                    )
                    print(f"ERROR {Path(task['metric_path']).name}: {exc}")
    else:
        for task in tasks:
            try:
                handle_result(process_metric_file(task))
            except Exception as exc:
                failed += 1
                append_jsonl(
                    errors_jsonl,
                    [{"metric_path": task["metric_path"], "error": str(exc), "failed_at": utc_now_iso()}],
                )
                print(f"ERROR {Path(task['metric_path']).name}: {exc}")

    print(
        "landmark token metrics complete: "
        f"processed={processed}, skipped={skipped}, failed={failed}, "
        f"sidecars={sidecar_dir}, aggregate={aggregate_jsonl}"
    )


if __name__ == "__main__":
    main()
