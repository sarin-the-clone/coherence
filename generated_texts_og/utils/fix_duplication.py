import os
import json
import re
import sys

def find_repeating_block(text, min_len=100):
    """
    Finds a substring of at least min_len that repeats consecutively.
    Returns the block and the total repeated count.
    """
    n = len(text)
    # Simple heuristic: Look for large blocks repeating
    # We scan for the first block of 'min_len' chars and see if it repeats immediately
    
    # Actually, the repetition in chunk 452 is likely paragraph-based.
    # "On that fateful day... (content) ... On that fateful day..."
    
    # Let's try to find a repeating sequence of paragraphs.
    paragraphs = text.split('\n\n')
    
    if len(paragraphs) < 5:
        return None

    # Check for repeating sequence of N paragraphs
    for i in range(len(paragraphs)):
        # Candidate start block
        block_start = paragraphs[i]
        if len(block_start) < 50: continue
        
        # Search for this block again later
        try:
           next_idx = paragraphs.index(block_start, i + 1)
        except ValueError:
            continue
            
        # Possible period P = next_idx - i
        P = next_idx - i
        if P <= 0: continue
        
        # Verify repetition
        is_repeating = True
        num_reps = 1
        
        curr = i
        while True:
            # Check if next block matches
            next_seq_start = curr + P
            if next_seq_start + P > len(paragraphs):
                break # Not enough left for a full rep
            
            p1 = paragraphs[curr : curr + P]
            p2 = paragraphs[next_seq_start : next_seq_start + P]
            
            if p1 == p2:
                num_reps += 1
                curr += P
            else:
                break
        
        if num_reps >= 2:
            return (i, P, num_reps)

    return None

def fix_file(filepath):
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {filepath} (JSON error)")
            return

    new_text = data.get("full_output") or data.get("new_text") # Depending on structure
    # Based on file view, it's 'full_output' inside the JSON usually, but generated text is 'new_text' in code?
    # Inspecting chunk 452:
    # "full_output": "\nIn the months that followed..."
    # "new_text" key is not seen in view_file 452, but "full_output" is.
    # Actually "new_text" might be extracted from "full_output".
    # Wait, in the JSON shown in view_file 452 (Step 454), keys are: "prompt", "temperature", "full_output", "chunk_metrics", "accumulated_metrics".
    # So the text is in "full_output".
    
    text = data.get("full_output", "")
    if not text:
        return

    # Check size first
    if len(text) < 50000: # 50k chars is plenty for a 16k tokens chunk? 
        # 16384 tokens * 4 chars = 65k chars.
        # But normally chunks are smaller (e.g. 4k tokens).
        # Chunk 452 was 150KB file size (incl prompt/metrics). Text likely ~100k chars.
        pass

    rep_info = find_repeating_block(text)
    if rep_info:
        start_idx, period, num_reps = rep_info
        print(f"  Found repetition in {os.path.basename(filepath)}: Start Para {start_idx}, Period {period}, Reps {num_reps}")
        
        # Construct corrected text
        paragraphs = text.split('\n\n')
        
        # Keep paragraphs up to end of first repetition
        # Start + Period = End of first instance
        keep_paras = paragraphs[:start_idx + period]
        
        corrected_text = '\n\n'.join(keep_paras)
        
        print(f"    Original len: {len(text)}, New len: {len(corrected_text)}")
        
        data["full_output"] = corrected_text
        # Reset metrics? Calculating metrics is hard without model. 
        # Fix text primarily. Metrics might be temporarily inaccurate but better than duplication.
        # We can set metrics to None or leave them (they reflect the bad text).
        # Leaving them is safer for "resuming" unless resumed engine re-checks.
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print("    Fixed.")
    else:
        # print(f"  No repetition found in {os.path.basename(filepath)}")
        pass

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix.py <directory>")
        return
        
    target_dir = sys.argv[1]
    print(f"Scanning {target_dir}...")
    
    files = sorted([f for f in os.listdir(target_dir) if f.startswith("chunk_") and f.endswith(".txt")])
    
    for fname in files:
        # Filter for suspected large files or just check all (fast enough)
        path = os.path.join(target_dir, fname)
        size = os.path.getsize(path)
        if size > 100000: # Check files > 100KB
            print(f"Checking large file: {fname} ({size/1024:.1f} KB)")
            fix_file(path)

if __name__ == "__main__":
    main()
