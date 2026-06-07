#!/usr/bin/env python3
"""
Patch generated token-naturalness sidecars with cumulative hapax metrics.

This avoids recomputing the full curated metric set. It reconstructs accepted
chunks, tokenizes each chunk once, then maintains cumulative token counts to add:

  - td_hapax_type_fraction
  - td_hapax_token_fraction

to <prompt_folder>/token_naturalness_by_chunk.jsonl.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.add_token_naturalness_to_generated import (  # noqa: E402
    reconstruct_generated_text,
    tokenize_reconstruction,
)
from tokenizer import Tokenizer  # noqa: E402


DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
HAPAX_KEYS = ("td_hapax_type_fraction", "td_hapax_token_fraction")


def has_attempt_files(folder: Path) -> bool:
    return any("prompt" not in path.name for path in folder.glob("chunk_*_attempt_*.txt"))


def load_sidecar(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_sidecar_atomic(path: Path, records: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for record in records:
            tmp.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def cumulative_hapax_by_chunk(folder: Path, tokenizer: Tokenizer, include_seed: bool) -> dict[int, dict[str, float]]:
    reconstruction = reconstruct_generated_text(folder, include_seed=include_seed)
    tokenized = tokenize_reconstruction(reconstruction, tokenizer, include_seed=include_seed)

    counts: dict[int, int] = {}
    hapax_types = 0
    n_tokens = 0

    def add_tokens(tokens: list[int]) -> None:
        nonlocal hapax_types, n_tokens
        for token in tokens:
            old = counts.get(token, 0)
            if old == 0:
                counts[token] = 1
                hapax_types += 1
            elif old == 1:
                counts[token] = 2
                hapax_types -= 1
            else:
                counts[token] = old + 1
            n_tokens += 1

    add_tokens(tokenized["seed_tokens"])

    out: dict[int, dict[str, float]] = {}
    for chunk, chunk_tokens in zip(reconstruction["chunks"], tokenized["chunk_token_parts"]):
        add_tokens(chunk_tokens)
        n_types = len(counts)
        chunk_id = int(chunk["chunk_id"])
        out[chunk_id] = {
            "td_hapax_type_fraction": float(hapax_types / n_types) if n_types else float("nan"),
            "td_hapax_token_fraction": float(hapax_types / n_tokens) if n_tokens else float("nan"),
        }
    return out


def patch_folder(folder: Path, tokenizer_kind: str, force: bool) -> dict:
    sidecar = folder / "token_naturalness_by_chunk.jsonl"
    if not sidecar.exists():
        return {"folder": str(folder), "folder_name": folder.name, "status": "skip_no_sidecar"}

    records = load_sidecar(sidecar)
    if not records:
        return {"folder": str(folder), "folder_name": folder.name, "status": "skip_empty_sidecar"}

    if not force and all(all(key in record for key in HAPAX_KEYS) for record in records):
        return {"folder": str(folder), "folder_name": folder.name, "status": "skip_existing"}

    include_seed = bool(records[0].get("include_seed", True))
    tokenizer = Tokenizer(kind=records[0].get("tokenizer") or tokenizer_kind)
    hapax_by_chunk = cumulative_hapax_by_chunk(folder, tokenizer, include_seed=include_seed)

    patched = 0
    missing_chunks = 0
    for record in records:
        chunk_id = int(record["chunk_id"])
        values = hapax_by_chunk.get(chunk_id)
        if values is None:
            missing_chunks += 1
            continue
        record.update(values)
        patched += 1

    write_sidecar_atomic(sidecar, records)
    return {
        "folder": str(folder),
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
                if first:
                    record = json.loads(first)
                    if all(key in record for key in HAPAX_KEYS):
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

    print(f"hapax patch candidates: {len(folders)}, workers={args.workers}")
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
