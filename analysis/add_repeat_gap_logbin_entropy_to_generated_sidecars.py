#!/usr/bin/env python3
"""
Patch generated token-naturalness sidecars with cumulative repeat-gap log-bin entropy.

This avoids recomputing the full curated metric set. It reconstructs accepted
chunks, tokenizes each chunk once, then maintains cumulative repeat-gap log2
bin counts to add:

  - tr_repeat_gap_logbin_entropy

to <prompt_folder>/token_naturalness_by_chunk.jsonl.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.add_token_naturalness_to_generated import (  # noqa: E402
    accepted_attempt_from_marker,
    load_attempts,
    load_narrative_seed,
    read_attempt_text,
)
from tokenizer import Tokenizer  # noqa: E402


DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
METRIC_KEY = "tr_repeat_gap_logbin_entropy"


def has_attempt_files(folder: Path) -> bool:
    return any("prompt" not in path.name for path in folder.glob("chunk_*_attempt_*.txt"))


def load_sidecar(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_sidecar_atomic(path: Path, records: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for record in records:
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def entropy_bits_from_counts(counts: Counter[int], total: int) -> float:
    if total <= 0:
        return float("nan")
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return float(entropy)


def cumulative_logbin_entropy_by_chunk(
    folder: Path,
    tokenizer: Tokenizer,
    include_seed: bool,
    sidecar_records: list[dict],
) -> dict[int, float]:
    attempts_by_chunk = load_attempts(folder)
    if not attempts_by_chunk:
        raise RuntimeError("no chunk attempt files found")

    sidecar_chunks = {int(record["chunk_id"]) for record in sidecar_records}
    sidecar_attempts = {
        int(record["chunk_id"]): int(record["accepted_attempt"])
        for record in sidecar_records
        if record.get("accepted_attempt") is not None
    }
    max_sidecar_chunk = max(sidecar_chunks)

    last_pos: dict[int, int] = {}
    logbin_counts: Counter[int] = Counter()
    total_gaps = 0
    pos = 0

    def add_tokens(tokens: list[int]) -> None:
        nonlocal pos, total_gaps
        for token in tokens:
            token = int(token)
            previous = last_pos.get(token)
            if previous is not None:
                gap = pos - previous
                if gap > 0:
                    logbin_counts[int(math.floor(math.log2(gap)))] += 1
                    total_gaps += 1
            last_pos[token] = pos
            pos += 1

    if include_seed:
        seed = load_narrative_seed(folder)
        if seed:
            add_tokens(tokenizer.encode(seed + "\n\n"))

    out: dict[int, float] = {}
    for chunk_id in sorted(attempts_by_chunk):
        if chunk_id > max_sidecar_chunk:
            break
        attempts = attempts_by_chunk[chunk_id]
        accepted_attempt = sidecar_attempts.get(chunk_id)
        if accepted_attempt not in attempts:
            accepted_attempt = accepted_attempt_from_marker(folder, chunk_id, attempts)
        if accepted_attempt not in attempts:
            accepted_attempt = max(attempts)

        text = read_attempt_text(attempts[accepted_attempt]["path"])
        add_tokens(tokenizer.encode(text))
        if chunk_id in sidecar_chunks:
            out[chunk_id] = entropy_bits_from_counts(logbin_counts, total_gaps)
    return out


def patch_folder(folder: Path, tokenizer_kind: str, force: bool) -> dict:
    sidecar = folder / "token_naturalness_by_chunk.jsonl"
    if not sidecar.exists():
        return {"folder_name": folder.name, "status": "skip_no_sidecar"}

    records = load_sidecar(sidecar)
    if not records:
        return {"folder_name": folder.name, "status": "skip_empty_sidecar"}

    if not force and all(METRIC_KEY in record for record in records):
        return {"folder_name": folder.name, "status": "skip_existing"}

    include_seed = bool(records[0].get("include_seed", True))
    tokenizer = Tokenizer(kind=records[0].get("tokenizer") or tokenizer_kind)
    values_by_chunk = cumulative_logbin_entropy_by_chunk(
        folder,
        tokenizer,
        include_seed=include_seed,
        sidecar_records=records,
    )

    patched = 0
    missing_chunks = 0
    for record in records:
        chunk_id = int(record["chunk_id"])
        value = values_by_chunk.get(chunk_id)
        if value is None:
            missing_chunks += 1
            continue
        record[METRIC_KEY] = value
        patched += 1

    write_sidecar_atomic(sidecar, records)
    return {
        "folder_name": folder.name,
        "status": "ok",
        "records": len(records),
        "patched": patched,
        "missing_chunks": missing_chunks,
    }


def iter_folders(prompts_dir: Path, pattern: str | None, force: bool) -> list[Path]:
    folders = []
    for folder in sorted(prompts_dir.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        if pattern and pattern not in folder.name:
            continue
        if not has_attempt_files(folder):
            continue
        sidecar = folder / "token_naturalness_by_chunk.jsonl"
        if not sidecar.exists():
            continue
        if not force:
            try:
                first = next((line for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()), "")
                if first and METRIC_KEY in json.loads(first):
                    continue
            except Exception:
                pass
        folders.append(folder)
    return folders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--tokenizer", default="gpt4")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.folder:
        folders = [args.folder]
    else:
        folders = iter_folders(args.prompts_dir, args.pattern, force=args.force)
    if args.limit is not None:
        folders = folders[: args.limit]

    print(f"repeat-gap logbin patch candidates: {len(folders)}, workers={args.workers}")
    if not folders:
        return

    ok = 0
    skipped = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(patch_folder, folder, args.tokenizer, args.force): folder for folder in folders}
        for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            folder = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                print(f"[{idx}/{len(folders)}] fail {folder.name}: {exc}")
                continue
            status = result["status"]
            if status == "ok":
                ok += 1
                print(f"[{idx}/{len(folders)}] ok {folder.name}: patched={result['patched']}/{result['records']}")
            elif status.startswith("skip"):
                skipped += 1
                print(f"[{idx}/{len(folders)}] {status} {folder.name}")
            else:
                failed += 1
                print(f"[{idx}/{len(folders)}] {status} {folder.name}")

    print(f"complete: ok={ok}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
