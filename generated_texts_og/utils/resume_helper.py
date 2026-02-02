import os
import re
import json
import glob
from natsort import natsorted

def find_latest_chunk_id(run_dir):
    """
    Finds the highest chunk ID that has a corresponding memory snapshot.
    """
    memory_dirs = glob.glob(os.path.join(run_dir, "memory_chunk_*"))
    if not memory_dirs:
        return 0
    
    # Extract IDs: memory_chunk_123 -> 123
    ids = []
    for d in memory_dirs:
        try:
            name = os.path.basename(d)
            cid = int(name.replace("memory_chunk_", ""))
            ids.append(cid)
        except ValueError:
            continue
    
    return max(ids) if ids else 0

def load_memory_snapshot(run_dir, chunk_id):
    """
    Loads memory.json from the specific memory_chunk_XXX folder.
    """
    path = os.path.join(run_dir, f"memory_chunk_{chunk_id:03d}", "memory.json")
    if not os.path.exists(path):
        return None
    
    with open(path, 'r') as f:
        return json.load(f)

def reconstruct_text_history(run_dir, last_chunk_id):
    """
    Reconstructs full_text and local_context from chunk files 1 to last_chunk_id.
    Heuristic: For each chunk ID, pick the file with the highest attempt number.
    Returns:
        full_text (str)
        last_chunk_text (str) - for setting local context
    """
    full_text_parts = []
    last_chunk_text = ""
    
    # We also need the NARRATIVE SEED which is in full_text but usually not in chunk files.
    # The 'initial_prompt.md' exists in the folder.
    # But `LongBookGenerator` splits it. 
    # For simplicity, we might just join the chunks. 
    # However, `LongBookGenerator.full_text` starts with `narrative_seed`.
    # Let's try to load `metadata.json` to get `narrative_seed`.
    
    seed = ""
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            seed = meta.get("narrative_seed", "")
    
    full_text_parts.append(seed)
    
    for i in range(1, last_chunk_id + 1):
        # Pattern: chunk_001_attempt_*.txt (exclude prompt files)
        # Note: glob pattern needs 3 digits if formatted that way, but list_dir showed `chunk_001`.
        # Code uses `f"chunk_{chunk_id:03d}_attempt_{attempt}"`.
        
        pattern = os.path.join(run_dir, f"chunk_{i:03d}_attempt_*.txt")
        files = glob.glob(pattern)
        # Exclude prompts
        files = [f for f in files if "_prompt.txt" not in f]
        
        if not files:
            print(f"Warning: No text file found for chunk {i}. Skipping text reconstruction for this chunk.")
            continue
            
        # Sort by attempt number
        # Filename format: ...chunk_XXX_attempt_Y.txt
        def get_attempt(path):
            m = re.search(r"attempt_(\d+)", os.path.basename(path))
            return int(m.group(1)) if m else 0
            
        best_file = max(files, key=get_attempt)
        
        with open(best_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # FIX: The file is a JSON dump (attempt_dump), not raw text.
        try:
            data = json.loads(content)
            # Try to get pure text
            text = data.get("new_text") 
            if not text:
                # Fallback to full output if new_text missing
                text = data.get("full_output", "")
                # If full output, we might need to strip prompt? 
                # But typically new_text is present.
                
            if not text:
                 # Fallback: maybe it IS raw text? (Legacy format)
                 text = content
        except json.JSONDecodeError:
            # It's raw text
            text = content
            
        full_text_parts.append(text)
        last_chunk_text = text

    return "\n\n".join(full_text_parts), last_chunk_text, full_text_parts
