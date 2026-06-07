#!/usr/bin/env python3
"""
Cache and recompute metrics for the Gutenberg natural benchmark.

The benchmark identity is recovered from:
    natural_texts/Output/batch_results/batch_*.json

The cache stores trimmed Gutenberg texts in a compressed zip archive without
leaving extracted books on disk:
    natural_texts/Output/gutenberg_benchmark_cache/books.zip

Typical usage:
    python natural_texts/cache_gutenberg_benchmark.py status
    python natural_texts/cache_gutenberg_benchmark.py build-cache --limit 10
    python natural_texts/cache_gutenberg_benchmark.py compute-metrics --device cuda
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from metrics import (  # noqa: E402
    CoherenceMetrics,
    filter_token_naturalness_metrics,
    token_distribution_scaling_metrics,
    token_recurrence_metrics,
)
from retrieve_gutenberg import get_gutenberg_text  # noqa: E402
from tokenizer import Tokenizer  # noqa: E402


DEFAULT_BATCH_RESULTS_DIR = SCRIPT_DIR / "Output" / "batch_results"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "Output" / "gutenberg_benchmark_cache"
DEFAULT_ARCHIVE = DEFAULT_CACHE_DIR / "books.zip"
DEFAULT_CACHE_MANIFEST = DEFAULT_CACHE_DIR / "cache_manifest.jsonl"
DEFAULT_ERROR_LOG = DEFAULT_CACHE_DIR / "cache_errors.jsonl"
DEFAULT_METRICS_OUTPUT = DEFAULT_CACHE_DIR / "token_naturalness_metrics.jsonl"


ZIP_COMPRESSION = {
    "deflated": zipfile.ZIP_DEFLATED,
    "lzma": zipfile.ZIP_LZMA,
    "stored": zipfile.ZIP_STORED,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl_by_id(path: Path, id_key: str = "book_id") -> dict[int, dict]:
    if not path.exists():
        return {}

    records: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if id_key in record:
                records[int(record[id_key])] = record
    return records


def batch_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"batch_(\d+)\.json$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def load_benchmark_rows(batch_results_dir: Path = DEFAULT_BATCH_RESULTS_DIR) -> list[dict]:
    rows = []
    seen = set()
    batch_files = sorted(batch_results_dir.glob("batch_*.json"), key=batch_sort_key)

    for batch_file in batch_files:
        batch_num = batch_sort_key(batch_file)[0]
        with batch_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for batch_index, record in enumerate(data):
            book_id = int(record["book_id"])
            if book_id in seen:
                raise ValueError(f"Duplicate book_id in benchmark batch results: {book_id}")
            seen.add(book_id)
            rows.append({
                "benchmark_index": len(rows),
                "batch_num": batch_num,
                "batch_index": batch_index,
                "book_id": book_id,
                "title": record.get("title"),
                "author": record.get("author"),
                "genre": record.get("genre"),
                "broad_genre": record.get("broad_genre"),
                "n_tokens_old": record.get("n_tokens"),
            })

    return rows


def member_name(book_id: int) -> str:
    return f"books/{int(book_id)}.txt"


def zip_members(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with zipfile.ZipFile(path, "r") as zf:
        return set(zf.namelist())


def cached_book_ids(path: Path) -> set[int]:
    ids = set()
    for name in zip_members(path):
        match = re.fullmatch(r"books/(\d+)\.txt", name)
        if match:
            ids.add(int(match.group(1)))
    return ids


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_cached_book(
    archive_path: Path,
    record: dict,
    text: str,
    compression: int,
    compresslevel: int | None,
) -> dict:
    data = text.encode("utf-8")
    name = member_name(record["book_id"])

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"mode": "a", "compression": compression, "allowZip64": True}
    if compression == zipfile.ZIP_DEFLATED and compresslevel is not None:
        kwargs["compresslevel"] = compresslevel

    with zipfile.ZipFile(archive_path, **kwargs) as zf:
        zf.writestr(name, data)

    return {
        **record,
        "member": name,
        "source_url": f"https://www.gutenberg.org/cache/epub/{record['book_id']}/pg{record['book_id']}.txt",
        "cached_at": utc_now_iso(),
        "bytes_utf8": len(data),
        "sha256": sha256_bytes(data),
    }


def manifest_record_from_existing_member(archive_path: Path, record: dict) -> dict:
    name = member_name(record["book_id"])
    with zipfile.ZipFile(archive_path, "r") as zf:
        data = zf.read(name)
    return {
        **record,
        "member": name,
        "source_url": f"https://www.gutenberg.org/cache/epub/{record['book_id']}/pg{record['book_id']}.txt",
        "cached_at": None,
        "bytes_utf8": len(data),
        "sha256": sha256_bytes(data),
        "manifest_recovered_from_archive": True,
    }


def build_cache(args: argparse.Namespace) -> None:
    rows = load_benchmark_rows(args.batch_results_dir)
    if args.only_clean_core:
        bad_h_dma = {64634, 45450, 508, 70630, 42965}
        rows = [row for row in rows if row["book_id"] not in bad_h_dma]

    if args.limit is not None:
        rows = rows[:args.limit]

    compression = ZIP_COMPRESSION[args.compression]
    existing_members = zip_members(args.archive)
    manifest = load_jsonl_by_id(args.manifest)

    cached = 0
    recovered_manifest = 0
    downloaded = 0
    failed = 0

    for i, record in enumerate(rows, start=1):
        book_id = int(record["book_id"])
        name = member_name(book_id)

        if name in existing_members:
            cached += 1
            if book_id not in manifest:
                recovered = manifest_record_from_existing_member(args.archive, record)
                append_jsonl(args.manifest, recovered)
                manifest[book_id] = recovered
                recovered_manifest += 1
            if args.verbose:
                print(f"[{i}/{len(rows)}] cached {book_id}")
            continue

        print(f"[{i}/{len(rows)}] downloading {book_id}: {str(record.get('title') or '')[:70]}")
        try:
            text = get_gutenberg_text(book_id)
            if not text:
                raise RuntimeError("download returned empty text")

            cache_record = write_cached_book(
                args.archive,
                record,
                text,
                compression=compression,
                compresslevel=args.compresslevel,
            )
            append_jsonl(args.manifest, cache_record)
            manifest[book_id] = cache_record
            existing_members.add(name)
            downloaded += 1
        except Exception as exc:
            failed += 1
            append_jsonl(args.errors, {**record, "error": str(exc), "failed_at": utc_now_iso()})
            print(f"   failed {book_id}: {exc}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print(
        "cache build complete: "
        f"target={len(rows)}, already_cached={cached}, downloaded={downloaded}, "
        f"manifest_recovered={recovered_manifest}, failed={failed}"
    )


def status(args: argparse.Namespace) -> None:
    rows = load_benchmark_rows(args.batch_results_dir)
    target_ids = {int(row["book_id"]) for row in rows}
    cached_ids = cached_book_ids(args.archive)
    manifest = load_jsonl_by_id(args.manifest)

    missing_from_archive = target_ids - cached_ids
    missing_from_manifest = target_ids - set(manifest)
    extra_archive = cached_ids - target_ids

    print(f"benchmark rows:        {len(rows)}")
    print(f"benchmark unique IDs:  {len(target_ids)}")
    print(f"archive:              {args.archive}")
    print(f"cached archive IDs:   {len(cached_ids)}")
    print(f"manifest IDs:         {len(manifest)}")
    print(f"missing archive IDs:  {len(missing_from_archive)}")
    print(f"missing manifest IDs: {len(missing_from_manifest)}")
    print(f"extra archive IDs:    {len(extra_archive)}")
    if args.archive.exists():
        print(f"archive size:         {args.archive.stat().st_size / 1024**3:.2f} GiB")
    if missing_from_archive:
        print(f"missing sample:       {sorted(missing_from_archive)[:20]}")


def load_completed_metric_ids(output_path: Path) -> set[int]:
    return set(load_jsonl_by_id(output_path))


def pseudo_chunk_boundaries(n_tokens: int, pseudo_chunk_tokens: int | None) -> list[int] | None:
    if not pseudo_chunk_tokens or pseudo_chunk_tokens <= 0:
        return None
    return list(range(pseudo_chunk_tokens, n_tokens, pseudo_chunk_tokens))


def compute_metric_record_for_book(task: dict) -> tuple[str, dict]:
    book_id = int(task["book_id"])
    record = task["record"]
    cache_record = task["cache_record"]

    try:
        with zipfile.ZipFile(task["archive"], "r") as zf:
            data = zf.read(cache_record["member"])
        if sha256_bytes(data) != cache_record.get("sha256"):
            raise RuntimeError("cached text SHA256 does not match manifest")

        text = data.decode("utf-8", errors="replace")
        tokenizer = Tokenizer(kind=task["tokenizer"])
        tokens = tokenizer.encode(text)
        cm = CoherenceMetrics(tokens, device=task["device"])
        metrics = {}

        if task["metric_set"] == "fast":
            metrics.update(token_distribution_scaling_metrics(tokens))
            metrics.update(token_recurrence_metrics(tokens))
        elif task["metric_set"] == "token_predictability":
            metrics.update(cm.compute_token_predictability(max_order=task["max_order"]))
        elif task["metric_set"] in {"all", "curated"}:
            metrics.update(cm.compute_token_naturalness(
                token_predictability_max_order=task["max_order"]
            ))
        elif task["metric_set"] == "context_dynamics":
            pass
        if task["include_context_dynamics"] or task["metric_set"] == "context_dynamics":
            boundaries = pseudo_chunk_boundaries(len(tokens), task["pseudo_chunk_tokens"])
            metrics.update(cm.compute_token_context_dynamics(
                boundaries=boundaries,
                max_order=task["max_order"],
                boundary_window=task["boundary_window"],
                n_checkpoints=task["n_checkpoints"],
                rank_order=task["rank_order"],
            ))
        if task["metric_set"] == "curated":
            metrics = filter_token_naturalness_metrics(
                metrics,
                include_chunk_context=task["include_context_dynamics"],
            )

        out = {
            **record,
            "n_tokens": len(tokens),
            "computed_at": utc_now_iso(),
            "tokenizer": task["tokenizer"],
            "device": task["device"],
            "max_order": task["max_order"],
            "metric_set": task["metric_set"],
            **metrics,
        }
        return "ok", out
    except Exception as exc:
        return "error", {**record, "book_id": book_id, "error": str(exc), "failed_at": utc_now_iso()}


def compute_metrics(args: argparse.Namespace) -> None:
    rows = load_benchmark_rows(args.batch_results_dir)
    row_by_id = {int(row["book_id"]): row for row in rows}
    manifest = load_jsonl_by_id(args.manifest)
    cached_ids_in_order = [int(row["book_id"]) for row in rows if int(row["book_id"]) in manifest]

    if args.limit is not None:
        cached_ids_in_order = cached_ids_in_order[:args.limit]

    completed = load_completed_metric_ids(args.output)
    tokenizer = Tokenizer(kind=args.tokenizer)
    processed = 0
    skipped = 0
    failed = 0
    tasks = []

    for book_id in cached_ids_in_order:
        if book_id in completed and not args.force:
            skipped += 1
            continue
        tasks.append({
            "archive": str(args.archive),
            "book_id": book_id,
            "record": row_by_id[book_id],
            "cache_record": manifest[book_id],
            "tokenizer": args.tokenizer,
            "device": args.device,
            "metric_set": args.metric_set,
            "max_order": args.max_order,
            "include_context_dynamics": args.include_context_dynamics,
            "pseudo_chunk_tokens": args.pseudo_chunk_tokens,
            "boundary_window": args.boundary_window,
            "n_checkpoints": args.n_checkpoints,
            "rank_order": args.rank_order,
        })

    if args.workers and args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(compute_metric_record_for_book, task): task
                for task in tasks
            }
            for n_done, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
                task = future_to_task[future]
                book_id = int(task["book_id"])
                record = task["record"]
                print(f"[{n_done}/{len(tasks)}] metrics {book_id}: {str(record.get('title') or '')[:70]}")
                try:
                    status, out = future.result()
                except Exception as exc:
                    status = "error"
                    out = {**record, "error": str(exc), "failed_at": utc_now_iso()}

                if status == "ok":
                    append_jsonl(args.output, out)
                    completed.add(book_id)
                    processed += 1
                else:
                    append_jsonl(args.errors, out)
                    failed += 1

        print(
            "metrics complete: "
            f"target_cached={len(cached_ids_in_order)}, processed={processed}, skipped={skipped}, failed={failed}"
        )
        return

    with zipfile.ZipFile(args.archive, "r") as zf:
        for i, task in enumerate(tasks, start=1):
            book_id = int(task["book_id"])
            record = row_by_id[book_id]
            cache_record = manifest[book_id]
            print(f"[{i}/{len(tasks)}] metrics {book_id}: {str(record.get('title') or '')[:70]}")

            try:
                data = zf.read(cache_record["member"])
                if sha256_bytes(data) != cache_record.get("sha256"):
                    raise RuntimeError("cached text SHA256 does not match manifest")

                text = data.decode("utf-8", errors="replace")
                tokens = tokenizer.encode(text)
                cm = CoherenceMetrics(tokens, device=args.device)
                metrics = {}

                if args.metric_set == "fast":
                    metrics.update(token_distribution_scaling_metrics(tokens))
                    metrics.update(token_recurrence_metrics(tokens))
                elif args.metric_set == "token_predictability":
                    metrics.update(cm.compute_token_predictability(max_order=args.max_order))
                elif args.metric_set in {"all", "curated"}:
                    metrics.update(cm.compute_token_naturalness(
                        token_predictability_max_order=args.max_order
                    ))
                elif args.metric_set == "context_dynamics":
                    pass
                if args.include_context_dynamics or args.metric_set == "context_dynamics":
                    boundaries = pseudo_chunk_boundaries(len(tokens), args.pseudo_chunk_tokens)
                    metrics.update(cm.compute_token_context_dynamics(
                        boundaries=boundaries,
                        max_order=args.max_order,
                        boundary_window=args.boundary_window,
                        n_checkpoints=args.n_checkpoints,
                        rank_order=args.rank_order,
                    ))
                if args.metric_set == "curated":
                    metrics = filter_token_naturalness_metrics(
                        metrics,
                        include_chunk_context=args.include_context_dynamics,
                    )

                out = {
                    **record,
                    "n_tokens": len(tokens),
                    "computed_at": utc_now_iso(),
                    "tokenizer": args.tokenizer,
                    "device": args.device,
                    "max_order": args.max_order,
                    "metric_set": args.metric_set,
                    **metrics,
                }
                append_jsonl(args.output, out)
                completed.add(book_id)
                processed += 1
            except Exception as exc:
                failed += 1
                append_jsonl(args.errors, {**record, "error": str(exc), "failed_at": utc_now_iso()})
                print(f"   failed {book_id}: {exc}")

    print(
        "metrics complete: "
        f"target_cached={len(cached_ids_in_order)}, processed={processed}, skipped={skipped}, failed={failed}"
    )


def add_common_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-results-dir", type=Path, default=DEFAULT_BATCH_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERROR_LOG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show cache coverage against benchmark IDs.")
    add_common_cache_args(p_status)
    p_status.set_defaults(func=status)

    p_build = sub.add_parser("build-cache", help="Download benchmark texts into compressed zip cache.")
    add_common_cache_args(p_build)
    p_build.add_argument("--compression", choices=sorted(ZIP_COMPRESSION), default="deflated")
    p_build.add_argument("--compresslevel", type=int, default=6)
    p_build.add_argument("--sleep", type=float, default=0.2, help="Delay between Gutenberg requests.")
    p_build.add_argument("--limit", type=int, default=None)
    p_build.add_argument("--only-clean-core", action="store_true", help="Skip the 5 old rows dropped by PCA due invalid H_DMA.")
    p_build.add_argument("--verbose", action="store_true")
    p_build.set_defaults(func=build_cache)

    p_metrics = sub.add_parser("compute-metrics", help="Compute new metrics from cached zip members.")
    add_common_cache_args(p_metrics)
    p_metrics.add_argument("--output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    p_metrics.add_argument("--device", default="cuda")
    p_metrics.add_argument("--tokenizer", default="gpt4")
    p_metrics.add_argument("--max-order", type=int, default=4)
    p_metrics.add_argument("--limit", type=int, default=None)
    p_metrics.add_argument("--force", action="store_true")
    p_metrics.add_argument("--workers", type=int, default=1, help="Parallel book-level workers for CPU-heavy metrics.")
    p_metrics.add_argument(
        "--metric-set",
        choices=["fast", "token_predictability", "context_dynamics", "curated", "all"],
        default="curated",
        help=(
            "curated=selected td/tr/tp metrics; fast=td_*+tr_* only; token_predictability=tp_* only; "
            "context_dynamics=tc_* only; all=td_*+tr_*+tp_*."
        ),
    )
    p_metrics.add_argument(
        "--include-token-naturalness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility flag. Use --metric-set instead.",
    )
    p_metrics.add_argument("--include-context-dynamics", action="store_true")
    p_metrics.add_argument("--pseudo-chunk-tokens", type=int, default=900)
    p_metrics.add_argument("--boundary-window", type=int, default=64)
    p_metrics.add_argument("--n-checkpoints", type=int, default=24)
    p_metrics.add_argument("--rank-order", type=int, default=2)
    p_metrics.set_defaults(func=compute_metrics)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "include_token_naturalness", None) is False:
        args.metric_set = "fast"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
