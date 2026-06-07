#!/usr/bin/env python3
"""
Compute curated token-naturalness trajectories for all generated prompt folders.

This is a thin parallel driver around add_token_naturalness_to_generated.py.
Each worker handles one prompt folder and writes:

  <prompt_folder>/token_naturalness_by_chunk.jsonl

Per-folder aggregate JSONL and logs are written under
generated_texts_og/Output/generated_token_naturalness/ to avoid concurrent
append races on a shared file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
DEFAULT_OUTPUT_DIR = ROOT / "generated_texts_og" / "Output" / "generated_token_naturalness"
DEFAULT_SCRIPT = ROOT / "analysis" / "add_token_naturalness_to_generated.py"


def has_attempt_files(folder: Path) -> bool:
    return any("prompt" not in path.name for path in folder.glob("chunk_*_attempt_*.txt"))


def iter_folders(prompts_dir: Path, pattern: str | None, excludes: list[str], force: bool) -> list[Path]:
    folders = []
    for folder in sorted(prompts_dir.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        if pattern and pattern not in folder.name:
            continue
        if any(exclude and exclude in folder.name for exclude in excludes):
            continue
        if not has_attempt_files(folder):
            continue
        if not force and (folder / "token_naturalness_by_chunk.jsonl").exists():
            continue
        folders.append(folder)
    return folders


def safe_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(keep)


def run_folder(folder: Path, args: argparse.Namespace) -> dict:
    log_path = args.logs_dir / f"{safe_name(folder.name)}.log"
    chunk_jsonl = args.per_folder_jsonl_dir / f"{safe_name(folder.name)}.jsonl"
    cmd = [
        sys.executable,
        str(args.script),
        "--folder",
        str(folder),
        "--by-chunk-only",
        "--chunk-stride",
        str(args.chunk_stride),
        "--metric-set",
        "curated",
        "--device",
        args.device,
        "--chunk-jsonl",
        str(chunk_jsonl),
    ]
    if args.force:
        cmd.append("--force")
    if not args.include_seed:
        cmd.append("--no-include-seed")

    start = datetime.now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={start.isoformat(timespec='seconds')}\n")
        log.write("command=" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT.parent), stdout=log, stderr=subprocess.STDOUT)
        end = datetime.now()
        log.write(f"\nfinished_at={end.isoformat(timespec='seconds')}\n")
        log.write(f"elapsed_seconds={(end - start).total_seconds():.1f}\n")
        log.write(f"returncode={proc.returncode}\n")

    return {
        "folder": str(folder),
        "folder_name": folder.name,
        "returncode": proc.returncode,
        "log": str(log_path),
        "chunk_jsonl": str(chunk_jsonl),
        "sidecar": str(folder / "token_naturalness_by_chunk.jsonl"),
        "elapsed_seconds": (end - start).total_seconds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-stride", type=int, default=1)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-seed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.logs_dir = args.output_dir / "curated_by_chunk_logs"
    args.per_folder_jsonl_dir = args.output_dir / "curated_by_chunk_jsonl"
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.per_folder_jsonl_dir.mkdir(parents=True, exist_ok=True)

    folders = iter_folders(args.prompts_dir, args.pattern, args.exclude, args.force)
    if args.limit is not None:
        folders = folders[: args.limit]

    print(
        "curated by-chunk candidates: "
        f"{len(folders)} folders, workers={args.workers}, stride={args.chunk_stride}, device={args.device}"
    )
    if not folders:
        return

    successes = 0
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_folder = {executor.submit(run_folder, folder, args): folder for folder in folders}
        for idx, future in enumerate(concurrent.futures.as_completed(future_to_folder), start=1):
            folder = future_to_folder[future]
            try:
                result = future.result()
            except Exception as exc:
                failures += 1
                print(f"[{idx}/{len(folders)}] FAIL {folder.name}: {exc}")
                continue
            if result["returncode"] == 0:
                successes += 1
                status = "ok"
            else:
                failures += 1
                status = f"fail rc={result['returncode']}"
            print(
                f"[{idx}/{len(folders)}] {status} {folder.name} "
                f"elapsed={result['elapsed_seconds'] / 60:.1f}m log={result['log']}"
            )

    print(f"complete: successes={successes}, failures={failures}")


if __name__ == "__main__":
    main()
