#!/usr/bin/env python3
"""
Generate chunk-metric evolution plots for all generated prompt folders.

For each reconstructable folder, this runs plot_chunk_metrics.py for the
manuscript-facing combined descriptor layout:

  - _legacy_vs_additional_descriptors_evolution_natural_benchmarks.jpg
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_DIR = ROOT / "generated_texts_og" / "prompts"
DEFAULT_SCRIPT = ROOT / "analysis" / "plot_chunk_metrics.py"


PLOTS = (
    ("legacy-vs-additional", "_legacy_vs_additional_descriptors_evolution_natural_benchmarks.jpg"),
)


def has_attempt_files(folder: Path) -> bool:
    return any("prompt" not in path.name for path in folder.glob("chunk_*_attempt_*.txt"))


def iter_folders(prompts_dir: Path, pattern: str | None, force: bool) -> list[Path]:
    folders = []
    for folder in sorted(prompts_dir.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        if pattern and pattern not in folder.name:
            continue
        if not has_attempt_files(folder):
            continue
        if not force and all((folder / filename).exists() for _, filename in PLOTS):
            continue
        folders.append(folder)
    return folders


def run_folder(folder: Path, args: argparse.Namespace) -> dict:
    start = datetime.now()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    commands = []
    for preset, filename in PLOTS:
        out_path = folder / filename
        if out_path.exists() and not args.force:
            continue
        cmd = [
            sys.executable,
            str(args.script),
            "--folder",
            str(folder),
            "--out",
            str(out_path),
            "--preset",
            preset,
            "--y-axis-ignore-first-n",
            str(args.y_axis_ignore_first_n),
        ]
        commands.append(cmd)
        subprocess.run(cmd, cwd=str(ROOT.parent), env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    end = datetime.now()
    return {
        "folder_name": folder.name,
        "folder": str(folder),
        "plots": len(commands),
        "elapsed_seconds": (end - start).total_seconds(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--y-axis-ignore-first-n",
        type=int,
        default=50,
        help="Draw early chunks but exclude first N chunks from y-axis autoscaling. Use 0 to disable.",
    )
    args = parser.parse_args()

    folders = iter_folders(args.prompts_dir, args.pattern, force=args.force)
    if args.limit is not None:
        folders = folders[: args.limit]

    print(f"plot candidates: {len(folders)} folders, workers={args.workers}")
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
            except subprocess.CalledProcessError as exc:
                failures += 1
                print(f"[{idx}/{len(folders)}] fail {folder.name}: rc={exc.returncode}")
                if exc.stdout:
                    print(exc.stdout[-2000:])
                continue
            except Exception as exc:
                failures += 1
                print(f"[{idx}/{len(folders)}] fail {folder.name}: {exc}")
                continue
            successes += 1
            print(
                f"[{idx}/{len(folders)}] ok {result['folder_name']} "
                f"plots={result['plots']} elapsed={result['elapsed_seconds']:.1f}s"
            )

    print(f"complete: successes={successes}, failures={failures}")


if __name__ == "__main__":
    main()
