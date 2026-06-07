"""
resume_helper.py

Utilities to assist in resuming interrupted long-book generation runs.
Includes functions to identify the latest valid checkpoint, load memory snapshots,
and reconstruct text history from individual chunk files.
"""
import os
import re
import json
import glob
from natsort import natsorted


def _find_accepted_attempt_file(run_dir, chunk_id, candidate_files):
    """
    Infer the accepted attempt.

    New runs write an explicit chunk_XXX_accepted.txt marker. Older runs did
    not, so we fall back to checking whether the next prompt contains the
    candidate continuation. That fallback only works for modes that re-inject
    rolling text; minimal_fixed_conditioning cannot be reconstructed perfectly
    without accepted markers.
    """
    if not candidate_files:
        return None

    accepted_path = os.path.join(run_dir, f"chunk_{chunk_id:03d}_accepted.txt")
    if os.path.exists(accepted_path):
        try:
            with open(accepted_path, "r", encoding="utf-8") as f:
                accepted_data = json.load(f)
            accepted_attempt = accepted_data.get("accepted_attempt")
            if accepted_attempt is not None:
                for path in candidate_files:
                    if _get_attempt(path) == int(accepted_attempt):
                        return path
        except Exception:
            pass

    next_chunk = chunk_id + 1
    next_prompts = glob.glob(os.path.join(run_dir, f"chunk_{next_chunk:03d}_attempt_*_prompt.txt"))
    if not next_prompts:
        return max(candidate_files, key=_get_attempt)

    try:
        with open(next_prompts[0], "r", encoding="utf-8") as f:
            next_prompt_content = f.read()
    except Exception:
        return max(candidate_files, key=_get_attempt)

    for path in sorted(candidate_files, key=_get_attempt, reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                data = json.loads(content)
                text = data.get("new_text") or data.get("full_output") or ""
            except json.JSONDecodeError:
                text = content
        except Exception:
            continue

        text = text.strip()
        if not text:
            continue
        snippet = text[:64] if len(text) >= 10 else text
        if snippet and snippet in next_prompt_content:
            return path

    return max(candidate_files, key=_get_attempt)


def _get_attempt(path):
    m = re.search(r"attempt_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0

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
    Prefer the accepted attempt by checking which chunk text is carried into the
    next chunk's saved prompt. Fall back to the highest attempt only when the
    accepted attempt cannot be inferred.
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
            
        best_file = _find_accepted_attempt_file(run_dir, i, files)
        
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

    full_text = "".join(f"{part}\n\n" for part in full_text_parts)
    return full_text, last_chunk_text, full_text_parts
