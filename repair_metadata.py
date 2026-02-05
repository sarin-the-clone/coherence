"""
repair_metadata.py

One-off script to repair a specific metadata corruption issue in Run 13/Run 6
where accumulated token counts became inconsistent after a specific chunk (452).
Recalculates correct token counts and updates subsequent chunk metadata.
"""
import os
import json
import glob
from tokenizer import Tokenizer

# Path to the run
RUN_DIR = "generated_texts_og/prompts/20251220_041136_fused_shadows_the_rising_of_kush_fantasy"

def main():
    print(f"Repairing metadata in: {RUN_DIR}")

    # 1. Load Chunk 451 (Baseline)
    path_451 = os.path.join(RUN_DIR, "chunk_451_attempt_1.txt")
    if not os.path.exists(path_451):
        print(f"Error: Chunk 451 not found at {path_451}")
        return

    with open(path_451, 'r') as f:
        data_451 = json.load(f)
    
    accumulated_451 = data_451["accumulated_metrics"]["n_tokens"]
    print(f"Chunk 451 Accumulated: {accumulated_451}")

    # 2. Re-tokenize Chunk 452 (The Repaired One)
    enc = Tokenizer(kind="gpt4")
    path_452 = os.path.join(RUN_DIR, "chunk_452_attempt_1.txt")
    with open(path_452, 'r') as f:
        data_452 = json.load(f)
    
    new_text_452 = data_452.get("new_text", "")
    if not new_text_452:
        print("Error: No new_text in Chunk 452")
        return

    tokens_452 = enc.encode(new_text_452)
    len_452 = len(tokens_452)
    print(f"Chunk 452 Actual Length: {len_452}")

    new_accumulated_452 = accumulated_451 + len_452
    print(f"Chunk 452 New Accumulated: {new_accumulated_452}")
    
    old_accumulated_452 = data_452["accumulated_metrics"]["n_tokens"]
    print(f"Chunk 452 Old Accumulated: {old_accumulated_452}")
    
    offset = old_accumulated_452 - new_accumulated_452
    print(f"Offset to subtract: {offset}")

    if offset == 0:
        print("No repair needed (offset is 0).")
        return

    # 3. Update Chunk 452
    data_452["accumulated_metrics"]["n_tokens"] = new_accumulated_452
    data_452["chunk_metrics"]["n_tokens"] = len_452
    
    with open(path_452, 'w') as f:
        json.dump(data_452, f, indent=2)
    print("Fixed Chunk 452.")

    # 4. Iterate and Fix Subsequent Chunks
    all_chunks = glob.glob(os.path.join(RUN_DIR, "chunk_*_attempt_*.txt"))
    
    count = 0
    for p in all_chunks:
        basename = os.path.basename(p)
        if "_prompt.txt" in basename: continue
        
        try:
            parts = basename.split("_")
            # Handle variable length splits if needed, but standard is chunk_XXX_attempt_Y
            # chunk_453_attempt_1.txt -> ['chunk', '453', 'attempt', '1.txt']
            cid = int(parts[1])
        except IndexError:
            continue
        except ValueError:
            continue
            
        if cid > 452:
            with open(p, 'r') as f:
                d = json.load(f)
            
            old_val = d.get("accumulated_metrics", {}).get("n_tokens", 0)
            if old_val == 0: continue

            d["accumulated_metrics"]["n_tokens"] = old_val - offset
            
            with open(p, 'w') as f:
                json.dump(d, f, indent=2)
            count += 1
            
    print(f"Fixed {count} subsequent chunks.")

if __name__ == "__main__":
    main()
