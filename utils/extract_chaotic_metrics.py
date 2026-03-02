import json
import tarfile
import re
import sys
from pathlib import Path
import os

# Put root on path if necessary to run standalone
# Put root on path to import analysis scripts
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import analysis.plot_chunk_metrics as pcm

OUT_FILE = ROOT / "generated_texts_og" / "Output" / "chaotic_run_metrics.json"

def main():
    tar_path = ROOT / "generated_texts_og" / "prompts" / "_chaos4m_20251230_012534_the_irresistible_storm_thriller.tar.xz"
    if not tar_path.exists():
        print(f"Error: Could not find chaotic run tarball at {tar_path}")
        sys.exit(1)

    print(f"Reading chaotic run from tar archive: {tar_path}")
    print("This may take some time as we decompress on the fly...")

    # Pattern for attempts
    pattern = re.compile(r"chunk_(\d+)_attempt_(\d+)\.txt$")
    prompt_pattern = re.compile(r"chunk_(\d+)_attempt_\d+_prompt\.txt$")

    attempts_by_chunk = {}
    
    # Optional performance: we can just manually open the python tarfile library
    with tarfile.open(tar_path, "r:xz") as tar:
        for member in tar:
            if not member.isfile():
                continue
                
            name = Path(member.name).name
            
            # Check for chunk attempt file
            m = pattern.search(name)
            if not m:
                continue
                
            chunk = int(m.group(1))
            attempt = int(m.group(2))
            
            f = tar.extractfile(member)
            if f:
                try:
                    data = json.load(f)
                    data["_file_path"] = f"tar://{member.name}" # mock path
                    attempts_by_chunk.setdefault(chunk, {})[attempt] = data
                except Exception as e:
                    print(f"Warning: Could not parse JSON for {name}: {e}")
                    
    print(f"Finished extracting. Parsed {len(attempts_by_chunk)} chunks.")
    
    if not attempts_by_chunk:
        print("No attempts found.")
        return

    # To use pcm.find_accepted_attempt, it expects prompt files to exist on disk. 
    # Since we are reading from tar, and chaotic runs usually are forced accepted, we will just use the max attempt.
    # A cleaner way is to mock it out but for the chaotic run we just pick max attempt as the accepted one
    # or the only one since it's typically highly forced
    series_data = {}
    cumulative_tokens = pcm.estimate_cumulative_tokens(attempts_by_chunk)
    
    chunk_indices = sorted(attempts_by_chunk.keys())
    for chunk in chunk_indices:
        # Override the find_accepted_attempt here since it relies on the filesystem
        best_attempt = max(attempts_by_chunk[chunk].keys())
        rec = attempts_by_chunk[chunk][best_attempt]
        
        sources = [
            rec.get("accumulated_metrics", {}),
            rec.get("chunk_metrics", {}),
            rec.get("metrics", {})
        ]
        
        combined = {}
        for s in reversed(sources):
            if s:
                combined.update(s)
                
        for m, val in combined.items():
            if isinstance(val, (int, float)):
                if m not in series_data:
                    series_data[m] = {}
                # Map using tokens as X axis for the plot_aggregate script
                t = cumulative_tokens.get(chunk)
                if t is not None:
                    series_data[m][t] = val

    # Save to JSON
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(series_data, f, indent=2)
        
    print(f"Saved chaotic run metric time series to {OUT_FILE}")

if __name__ == "__main__":
    main()
