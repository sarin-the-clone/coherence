#!/usr/bin/env python3
"""
compress_memory_chunks.py

Scans all experiment run folders in 'generated_texts_og/prompts'.
For each COMPLETED run (skipping active ones like Run 51):
1. Identifies all `memory_chunk_*` folders.
2. Compresses them into a single `memory_chunks.tar.xz` archive.
3. Verifies the archive integrity.
4. Deletes the original `memory_chunk_*` folders if verification succeeds.

Usage:
    python3 compress_memory_chunks.py
"""

import os
import sys
import glob
import subprocess
import shutil
from pathlib import Path

import argparse

# Configuration
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = SCRIPT_DIR.parent / "generated_texts_og" / "prompts"
SKIP_KEYWORDS = ["test"] # Safety skip active runs

def main():
    parser = argparse.ArgumentParser(description="Compress memory_chunk folders in old runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing them.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs to process.")
    args = parser.parse_args()

    if not PROMPTS_DIR.exists():
        print(f"Error: {PROMPTS_DIR} does not exist.")
        sys.exit(1)

    # Find all run directories
    run_dirs = sorted([d for d in PROMPTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    print(f"Found {len(run_dirs)} run directories.")

    processed_count = 0
    
    for run_dir in run_dirs:
        if args.limit is not None and processed_count >= args.limit:
            print(f"\nLimit of {args.limit} reached. Stopping.")
            break

        print(f"\nProcessing: {run_dir.name}")

        # Safety Check: Skip active runs
        if any(k in run_dir.name for k in SKIP_KEYWORDS):
            print(f"  SKIPPING: Active/Test run detected ({run_dir.name})")
            continue

        # Find memory chunks
        memory_chunks = sorted(list(run_dir.glob("memory_chunk_*")))
        if not memory_chunks:
            print("  No memory_chunk_* folders found to compress.")
            continue
            
        print(f"  Found {len(memory_chunks)} memory chunk folders.")
        
        # Define archive name
        archive_path = run_dir / "memory_chunks.tar.xz"
        
        if archive_path.exists():
            print(f"  WARNING: Archive {archive_path.name} already exists. Skipping to avoid overwrite.")
            continue

        # Prepare tar command
        # tar -cJf archive.tar.xz -C run_dir memory_chunk_1 memory_chunk_2 ...
        # We use relative paths to keep archive clean
        
        chunk_names = [p.name for p in memory_chunks]
        cmd = ["tar", "-I", "xz -T0", "-cf", str(archive_path.name)] + chunk_names
        
        if args.dry_run:
            print(f"  [DRY RUN] Would compress {len(chunk_names)} folders into {archive_path.name}")
            print(f"  [DRY RUN] Would delete originals after verification.")
            processed_count += 1
            continue

        print(f"  Compressing to {archive_path.name}...")
        
        try:
            # Run tar in the run_dir to use relative paths
            subprocess.run(cmd, cwd=run_dir, check=True)
            print("  Compression complete.")
            
            # Verify integrity
            print("  Verifying archive...")
            subprocess.run(["xz", "-t", str(archive_path.name)], cwd=run_dir, check=True)
            print("  Verification SUCCESS.")
            
            # Delete originals
            print("  Deleting original memory folders...")
            for p in memory_chunks:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    os.remove(p)
            print(f"  Cleaned up {len(memory_chunks)} folders.")
            processed_count += 1
            
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Compression or Verification failed! Originals NOT deleted.")
            print(f"  Details: {e}")
            # Try to cleanup partial archive
            if archive_path.exists():
                os.remove(archive_path)
                print("  Deleted partial archive.")
        except Exception as e:
            print(f"  ERROR: Unexpected error: {e}")

if __name__ == "__main__":
    main()
