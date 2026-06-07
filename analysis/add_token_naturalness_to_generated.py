#!/usr/bin/env python3
"""
Compute retrospective token-naturalness metrics for generated prompt folders.

This reconstructs the accepted generated text from chunk attempt files, computes
the new token-only metric families, then writes:

  <prompt_folder>/token_naturalness_metrics.json
  <prompt_folder>/token_naturalness_by_chunk.jsonl
  generated_texts_og/Output/generated_token_naturalness/generated_books_token_naturalness.jsonl

The script is resumable: existing folder JSONs are skipped unless --force.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


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


ATTEMPT_RE = re.compile(r"chunk_(\d+)_attempt_(\d+)\.txt$")
DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
DEFAULT_OUTPUT_DIR = ROOT / "generated_texts_og" / "Output" / "generated_token_naturalness"
DEFAULT_JSONL = DEFAULT_OUTPUT_DIR / "generated_books_token_naturalness.jsonl"
DEFAULT_CHUNK_JSONL = DEFAULT_OUTPUT_DIR / "generated_chunks_token_naturalness.jsonl"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_existing_jsonl_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            folder_name = rec.get("folder_name")
            if folder_name:
                ids.add(folder_name)
    return ids


def parse_attempt_file(path: Path) -> tuple[int, int] | None:
    if "prompt" in path.name:
        return None
    match = ATTEMPT_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def load_attempts(folder: Path) -> dict[int, dict[int, dict]]:
    attempts: dict[int, dict[int, dict]] = {}
    for path in folder.glob("chunk_*_attempt_*.txt"):
        parsed = parse_attempt_file(path)
        if parsed is None:
            continue
        chunk_id, attempt_id = parsed
        attempts.setdefault(chunk_id, {})[attempt_id] = {"path": path}
    return attempts


def read_attempt_text(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return content

    if isinstance(data, dict):
        return (
            data.get("new_text")
            or data.get("full_output")
            or data.get("text")
            or data.get("content")
            or data.get("response")
            or ""
        )
    return content


def accepted_attempt_from_marker(folder: Path, chunk_id: int, attempts: dict[int, dict]) -> int | None:
    marker = folder / f"chunk_{chunk_id:03d}_accepted.txt"
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

    accepted_attempt = data.get("accepted_attempt")
    if accepted_attempt is None:
        return None

    try:
        accepted_attempt = int(accepted_attempt)
    except Exception:
        return None

    return accepted_attempt if accepted_attempt in attempts else None


def accepted_attempt_from_next_prompt(folder: Path, chunk_id: int, attempts: dict[int, dict]) -> int | None:
    next_chunk = chunk_id + 1
    prompts = sorted(folder.glob(f"chunk_{next_chunk:03d}_attempt_*_prompt.txt"))
    if not prompts:
        return None

    try:
        prompt_text = prompts[0].read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    for attempt_id in sorted(attempts, reverse=True):
        text = read_attempt_text(attempts[attempt_id]["path"]).strip()
        if not text:
            continue
        snippet = text[:64] if len(text) >= 10 else text
        if snippet and snippet in prompt_text:
            return attempt_id
    return None


def load_narrative_seed(folder: Path) -> str:
    meta = folder / "metadata.json"
    if not meta.exists():
        return ""
    try:
        data = json.loads(meta.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    return data.get("narrative_seed", "") or ""


def reconstruct_generated_text(folder: Path, include_seed: bool = True) -> dict:
    attempts_by_chunk = load_attempts(folder)
    if not attempts_by_chunk:
        raise RuntimeError("no chunk attempt files found")

    seed = load_narrative_seed(folder) if include_seed else ""
    chunks = []
    selected = {}
    selection_counts = {
        "explicit_marker": 0,
        "next_prompt_match": 0,
        "fallback_max_attempt": 0,
    }

    for chunk_id in sorted(attempts_by_chunk):
        attempts = attempts_by_chunk[chunk_id]

        accepted_attempt = accepted_attempt_from_marker(folder, chunk_id, attempts)
        reason = "explicit_marker"
        if accepted_attempt is None:
            accepted_attempt = accepted_attempt_from_next_prompt(folder, chunk_id, attempts)
            reason = "next_prompt_match"
        if accepted_attempt is None:
            accepted_attempt = max(attempts)
            reason = "fallback_max_attempt"

        text = read_attempt_text(attempts[accepted_attempt]["path"])
        chunks.append({
            "chunk_id": chunk_id,
            "attempt": accepted_attempt,
            "selection_reason": reason,
            "text": text,
            "path": str(attempts[accepted_attempt]["path"]),
        })
        selected[chunk_id] = accepted_attempt
        selection_counts[reason] += 1

    full_text_parts = []
    if seed:
        full_text_parts.append(seed + "\n\n")
    full_text_parts.extend(chunk["text"] for chunk in chunks)

    return {
        "seed": seed,
        "chunks": chunks,
        "selected_attempts": selected,
        "selection_counts": selection_counts,
        "full_text": "".join(full_text_parts),
    }


def tokenize_reconstruction(reconstruction: dict, tokenizer: Tokenizer, include_seed: bool = True) -> dict:
    full_tokens = []
    seed_tokens = []
    if include_seed and reconstruction["seed"]:
        seed_tokens = tokenizer.encode(reconstruction["seed"] + "\n\n")
        full_tokens.extend(seed_tokens)

    chunk_token_parts = []
    for chunk in reconstruction["chunks"]:
        toks = tokenizer.encode(chunk["text"])
        chunk_token_parts.append(toks)
        full_tokens.extend(toks)

    return {
        "full_tokens": full_tokens,
        "seed_tokens": seed_tokens,
        "chunk_token_parts": chunk_token_parts,
    }


def compute_for_folder(
    folder: Path,
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    reconstruction: dict | None = None,
    tokenized: dict | None = None,
) -> dict:
    if reconstruction is None:
        reconstruction = reconstruct_generated_text(folder, include_seed=args.include_seed)
    if tokenized is None:
        tokenized = tokenize_reconstruction(reconstruction, tokenizer, include_seed=args.include_seed)
    full_tokens = tokenized["full_tokens"]
    chunk_token_parts = tokenized["chunk_token_parts"]

    metrics = {}
    cm = CoherenceMetrics(full_tokens, device=args.device)
    if args.metric_set == "fast":
        metrics.update(token_distribution_scaling_metrics(full_tokens))
        metrics.update(token_recurrence_metrics(full_tokens))
    elif args.metric_set == "token_predictability":
        metrics.update(cm.compute_token_predictability(max_order=args.max_order))
    elif args.metric_set in {"all", "curated"}:
        metrics.update(cm.compute_token_naturalness(token_predictability_max_order=args.max_order))

    if args.include_context_dynamics:
        metrics.update(token_context_dynamics_metrics(
            chunks=chunk_token_parts,
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

    generated_token_count = sum(len(part) for part in chunk_token_parts)
    result = {
        "folder_name": folder.name,
        "folder": str(folder),
        "computed_at": utc_now_iso(),
        "tokenizer": args.tokenizer,
        "device": args.device,
        "max_order": args.max_order,
        "include_seed": args.include_seed,
        "include_context_dynamics": args.include_context_dynamics,
        "n_chunks": len(reconstruction["chunks"]),
        "n_tokens": len(full_tokens),
        "n_seed_tokens": len(tokenized["seed_tokens"]),
        "n_generated_tokens": generated_token_count,
        "selection_counts": reconstruction["selection_counts"],
        "accepted_attempts_available": reconstruction["selection_counts"]["explicit_marker"],
        "new_metric_family_version": "token_naturalness_v1",
        **metrics,
    }
    return result


def compute_metrics_for_tokens(tokens: list[int], args: argparse.Namespace) -> dict:
    metrics = {}
    cm = CoherenceMetrics(tokens, device=args.device)
    if args.metric_set == "fast":
        metrics.update(token_distribution_scaling_metrics(tokens))
        metrics.update(token_recurrence_metrics(tokens))
    elif args.metric_set == "token_predictability":
        metrics.update(cm.compute_token_predictability(max_order=args.max_order))
    elif args.metric_set in {"all", "curated"}:
        metrics.update(cm.compute_token_naturalness(token_predictability_max_order=args.max_order))
    else:
        raise ValueError(f"Unsupported metric set: {args.metric_set}")
    if args.metric_set == "curated":
        metrics = filter_token_naturalness_metrics(metrics)
    return metrics


def compute_prefix_record_worker(task: dict) -> dict:
    args = argparse.Namespace(
        device=task["device"],
        metric_set=task["metric_set"],
        max_order=task["max_order"],
    )
    metrics = compute_metrics_for_tokens(task["tokens"], args)
    return {
        "folder_name": task["folder_name"],
        "folder": task["folder"],
        "chunk_id": task["chunk_id"],
        "accepted_attempt": task["accepted_attempt"],
        "selection_reason": task["selection_reason"],
        "computed_at": utc_now_iso(),
        "tokenizer": task["tokenizer"],
        "device": task["device"],
        "max_order": task["max_order"],
        "include_seed": task["include_seed"],
        "n_tokens": len(task["tokens"]),
        "chunk_n_tokens": task["chunk_n_tokens"],
        "new_metric_family_version": "token_naturalness_v1",
        **metrics,
    }


def compute_chunk_trajectory_for_folder(
    folder: Path,
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    reconstruction: dict | None = None,
    tokenized: dict | None = None,
    chunk_stride: int = 1,
) -> list[dict]:
    if reconstruction is None:
        reconstruction = reconstruct_generated_text(folder, include_seed=args.include_seed)
    if tokenized is None:
        tokenized = tokenize_reconstruction(reconstruction, tokenizer, include_seed=args.include_seed)

    cumulative_tokens = list(tokenized["seed_tokens"])
    records = []
    tasks = []

    final_chunk_id = int(reconstruction["chunks"][-1]["chunk_id"]) if reconstruction["chunks"] else None
    chunk_stride = max(1, int(chunk_stride))

    for chunk, chunk_tokens in zip(reconstruction["chunks"], tokenized["chunk_token_parts"]):
        cumulative_tokens.extend(chunk_tokens)
        chunk_id = int(chunk["chunk_id"])
        should_compute = (
            chunk_stride == 1
            or chunk_id % chunk_stride == 0
            or chunk_id == final_chunk_id
        )
        if not should_compute:
            continue

        if getattr(args, "prefix_workers", 1) and args.prefix_workers > 1:
            tasks.append({
                "tokens": list(cumulative_tokens),
                "folder_name": folder.name,
                "folder": str(folder),
                "chunk_id": chunk_id,
                "accepted_attempt": int(chunk["attempt"]),
                "selection_reason": chunk["selection_reason"],
                "tokenizer": args.tokenizer,
                "device": args.device,
                "metric_set": args.metric_set,
                "max_order": args.max_order,
                "include_seed": args.include_seed,
                "chunk_n_tokens": len(chunk_tokens),
            })
            continue

        metrics = compute_metrics_for_tokens(cumulative_tokens, args)

        record = {
            "folder_name": folder.name,
            "folder": str(folder),
            "chunk_id": chunk_id,
            "accepted_attempt": int(chunk["attempt"]),
            "selection_reason": chunk["selection_reason"],
            "computed_at": utc_now_iso(),
            "tokenizer": args.tokenizer,
            "device": args.device,
            "max_order": args.max_order,
            "include_seed": args.include_seed,
            "n_tokens": len(cumulative_tokens),
            "chunk_n_tokens": len(chunk_tokens),
            "new_metric_family_version": "token_naturalness_v1",
            **metrics,
        }
        records.append(record)

    if tasks:
        max_workers = max(1, int(args.prefix_workers))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for record in executor.map(compute_prefix_record_worker, tasks):
                records.append(record)
        records.sort(key=lambda rec: int(rec["chunk_id"]))

    return records


def iter_prompt_folders(prompts_dir: Path, pattern: str | None = None):
    folders = [p for p in prompts_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if pattern:
        folders = [p for p in folders if pattern in p.name]
    return sorted(folders, key=lambda p: p.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--folder", type=Path, default=None, help="Process one prompt folder only.")
    parser.add_argument("--pattern", default=None, help="Only process folders whose name contains this string.")
    parser.add_argument("--tokenizer", default="gpt4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--metric-set",
        choices=["fast", "token_predictability", "curated", "all"],
        default="curated",
        help="curated=selected td/tr/tp metrics; fast=td_*+tr_* only; token_predictability=tp_* only; all=td_*+tr_*+tp_*.",
    )
    parser.add_argument(
        "--include-token-naturalness",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility flag. Use --metric-set instead.",
    )
    parser.add_argument("--include-context-dynamics", action="store_true")
    parser.add_argument("--by-chunk", action="store_true", help="Also compute cumulative token-naturalness trajectory per accepted chunk.")
    parser.add_argument(
        "--by-chunk-only",
        action="store_true",
        help="Only write token_naturalness_by_chunk.jsonl; do not rewrite final token_naturalness_metrics.json.",
    )
    parser.add_argument("--chunk-stride", type=int, default=25, help="For --by-chunk, keep every Nth chunk plus final chunk. Use 1 for every chunk.")
    parser.add_argument("--prefix-workers", type=int, default=1, help="Parallel workers for per-prefix --by-chunk metric computation.")
    parser.add_argument("--chunk-jsonl", type=Path, default=DEFAULT_CHUNK_JSONL)
    parser.add_argument("--boundary-window", type=int, default=64)
    parser.add_argument("--n-checkpoints", type=int, default=24)
    parser.add_argument("--rank-order", type=int, default=2)
    args = parser.parse_args()

    if args.include_token_naturalness is False:
        args.metric_set = "fast"
    if args.by_chunk_only:
        args.by_chunk = True

    if args.folder is not None:
        folders = [args.folder]
    else:
        folders = iter_prompt_folders(args.prompts_dir, pattern=args.pattern)

    if args.limit is not None:
        folders = folders[:args.limit]

    print(f"candidate folders: {len(folders)}")
    if args.dry_run:
        reconstructable = 0
        for folder in folders:
            try:
                attempts = load_attempts(folder)
                if attempts:
                    reconstructable += 1
                    print(f"ok {folder.name}: {len(attempts)} chunks")
                else:
                    print(f"skip {folder.name}: no attempt files")
            except Exception as exc:
                print(f"fail {folder.name}: {exc}")
        print(f"dry run complete: reconstructable={reconstructable}/{len(folders)}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed_jsonl = load_existing_jsonl_ids(args.jsonl)
    tokenizer = Tokenizer(kind=args.tokenizer)

    processed = 0
    skipped = 0
    failed = 0

    for i, folder in enumerate(folders, start=1):
        folder_out = folder / "token_naturalness_metrics.json"
        chunk_out = folder / "token_naturalness_by_chunk.jsonl"
        needs_final = not args.by_chunk_only
        needs_chunk = args.by_chunk and (args.force or not chunk_out.exists())
        final_exists = folder_out.exists() or folder.name in completed_jsonl
        if not args.force and (not needs_chunk) and (not needs_final or final_exists):
            skipped += 1
            print(f"[{i}/{len(folders)}] skip existing {folder.name}")
            continue

        print(f"[{i}/{len(folders)}] compute {folder.name}")
        try:
            reconstruction = reconstruct_generated_text(folder, include_seed=args.include_seed)
            tokenized = tokenize_reconstruction(reconstruction, tokenizer, include_seed=args.include_seed)
            if not args.by_chunk_only:
                result = compute_for_folder(
                    folder,
                    args,
                    tokenizer,
                    reconstruction=reconstruction,
                    tokenized=tokenized,
                )
                folder_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                append_jsonl(args.jsonl, result)

            if args.by_chunk:
                chunk_records = compute_chunk_trajectory_for_folder(
                    folder,
                    args,
                    tokenizer,
                    reconstruction=reconstruction,
                    tokenized=tokenized,
                    chunk_stride=args.chunk_stride,
                )
                with chunk_out.open("w", encoding="utf-8") as f:
                    for rec in chunk_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        append_jsonl(args.chunk_jsonl, rec)

            processed += 1
        except Exception as exc:
            failed += 1
            error = {
                "folder_name": folder.name,
                "folder": str(folder),
                "failed_at": utc_now_iso(),
                "error": str(exc),
            }
            append_jsonl(args.output_dir / "generated_books_token_naturalness_errors.jsonl", error)
            print(f"   failed {folder.name}: {exc}")

    print(f"complete: processed={processed}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
