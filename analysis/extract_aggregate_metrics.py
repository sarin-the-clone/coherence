#!/usr/bin/env python3
"""
Extract aggregated metric evolution (Mean +/- Std) across ALL generated books and save to JSON.
This cleanly separates the heavy processing from the visualization phase.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Ensure analysis dir is on path for sibling import
ANALYSIS_DIR = Path(__file__).parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))

# Import the core utility functions from the existing script
import plot_aggregate_metrics as pam

def convert_np_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_np_types(obj.tolist())
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Extract text metrics aggregated across all books into a JSON.")
    parser.add_argument("--out", "-o", default="generated_texts_og/Output/aggregated_metrics.json")
    parser.add_argument("--bin-size", type=int, default=20000, help="Bin size in tokens for aggregation")
    args = parser.parse_args()

    prompts_dir = ROOT / "generated_texts_og" / "prompts"
    if not prompts_dir.exists():
        print(f"Error: {prompts_dir} does not exist")
        return

    folders = pam.get_book_folders(prompts_dir)
    
    high_folders = []
    low_folders = []
    for folder in folders:
        pattern = folder / "chunk_001_attempt_1.txt"
        temp = 1.0 # default
        if pattern.exists():
            try:
                with open(pattern, "r") as f:
                    data = json.load(f)
                    temp = data.get("temperature", 1.0)
            except Exception:
                pass
        if temp < 0.8:
            low_folders.append(folder)
        else:
            high_folders.append(folder)
            
    print(f"Found {len(high_folders)} high temp runs, {len(low_folders)} low temp runs")
    
    print("Aggregating high temperature runs...")
    binned_stats_high = pam.aggregate_data_by_tokens(high_folders, bin_size=args.bin_size, max_tokens=1000000)
    print("Aggregating low temperature runs...")
    binned_stats_low = pam.aggregate_data_by_tokens(low_folders, bin_size=args.bin_size, max_tokens=1000000)
    
    output_path = ROOT / args.out
    
    result = {
        "high_temp": binned_stats_high,
        "low_temp": binned_stats_low
    }
    
    # Clean numpy floats before dumping
    clean_result = convert_np_types(result)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_result, f, indent=2)
        
    print(f"Saved aggregated metrics to {output_path}")

if __name__ == "__main__":
    main()
