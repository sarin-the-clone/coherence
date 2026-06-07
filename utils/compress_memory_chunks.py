#!/usr/bin/env python3
from __future__ import annotations

"""
Compress memory_chunk_* folders for one generated-book prompt folder.

The script archives each memory_chunk_* folder independently into
memory_chunks_xz/memory_chunk_XXX.tar.xz, verifies the archive, then deletes the
original folder. This avoids needing enough temporary disk space for one large
archive and keeps the memory snapshots recoverable if needed.

Usage:
    python3 utils/compress_memory_chunks.py generated_texts_og/prompts/<book_folder>
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def memory_chunk_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit("_", 1)[-1]
    if suffix.isdigit():
        return int(suffix), path.name
    return sys.maxsize, path.name


def archive_chunk(run_dir: Path, chunk_dir: Path, archive_dir: Path, xz_threads: str, dry_run: bool) -> bool:
    archive_path = archive_dir / f"{chunk_dir.name}.tar.xz"
    if archive_path.exists():
        print(f"  skip existing archive: {archive_path.name}")
        return False

    if dry_run:
        print(f"  [dry-run] {chunk_dir.name} -> {archive_path.relative_to(run_dir)}")
        return True

    archive_dir.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(archive_path.suffix + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    relative_partial = partial_path.relative_to(run_dir)
    cmd = ["tar", "-I", f"xz -T{xz_threads}", "-cf", str(relative_partial), chunk_dir.name]
    subprocess.run(cmd, cwd=run_dir, check=True)
    subprocess.run(["xz", "-t", str(relative_partial)], cwd=run_dir, check=True)

    partial_path.rename(archive_path)
    shutil.rmtree(chunk_dir)
    return True


def compress_folder(
    run_dir: Path,
    archive_dir_name: str,
    xz_threads: str,
    dry_run: bool,
    max_chunks: int | None,
    allow_run_prefix: bool,
) -> int:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")

    if run_dir.name.startswith("run_") and not allow_run_prefix:
        raise RuntimeError(
            f"Refusing to compress temporary run folder '{run_dir.name}'. "
            "Use --allow-run-prefix only if the run is complete or you do not need resume state."
        )

    memory_chunks = sorted(
        [p for p in run_dir.glob("memory_chunk_*") if p.is_dir()],
        key=memory_chunk_sort_key,
    )
    if not memory_chunks:
        print(f"{run_dir.name}: no memory_chunk_* folders found")
        return 0

    archive_dir = run_dir / archive_dir_name
    print(f"{run_dir.name}: {len(memory_chunks)} memory chunks -> {archive_dir.name}/")

    compressed = 0
    for chunk_dir in memory_chunks:
        if max_chunks is not None and compressed >= max_chunks:
            break
        if archive_chunk(run_dir, chunk_dir, archive_dir, xz_threads, dry_run):
            compressed += 1
            if compressed % 50 == 0:
                print(f"  compressed {compressed} chunks")

    print(f"{run_dir.name}: compressed {compressed} chunks")
    return compressed


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress memory_chunk_* folders for one prompt folder.")
    parser.add_argument("folder", type=Path, help="Generated-book prompt folder to compress.")
    parser.add_argument("--archive-dir-name", default="memory_chunks_xz")
    parser.add_argument("--xz-threads", default="0", help="xz thread count; 0 means all cores.")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-run-prefix",
        action="store_true",
        help="Allow compressing temporary run_* folders. Avoid this before resume/finalization.",
    )
    args = parser.parse_args()

    try:
        compress_folder(
            run_dir=args.folder,
            archive_dir_name=args.archive_dir_name,
            xz_threads=args.xz_threads,
            dry_run=args.dry_run,
            max_chunks=args.max_chunks,
            allow_run_prefix=args.allow_run_prefix,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
