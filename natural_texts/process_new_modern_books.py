"""
process_new_modern_books.py

Parallel processing of new modern books (converted txt/mobi) to compute
coherence metrics at specified token targets.
Stores results in 'Selected_Books' for inclusion in PCA analysis.
"""
import os
import sys
import json
import glob
import pandas as pd

# Allow imports from coherence root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tokenizer import Tokenizer
from metrics import CoherenceMetrics
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

TARGETS = [100000, 200000, 400000, 600000, 1000000]

def guess_genre_from_filename(fname):
    """
    Guess genre from filenames/folder names encoded in the path if possible.
    For this batch, we know they are mostly modern fiction/historical.
    """
    f_lower = fname.lower()
    if "historical" in f_lower: return "Natural Historical Fiction"
    if "fantasy" in f_lower or "magic" in f_lower or "ice and fire" in f_lower: return "Natural Sci-Fi/Fantasy"
    if "mystery" in f_lower or "detective" in f_lower: return "Natural Mystery/Crime"
    return "Natural Fiction" # Default

def process_single_book(tf, metrics_out_dir):
    fname = os.path.basename(tf)
    # Create a safe ID from the filename
    safe_name = fname.replace('.txt', '').replace(' ', '_').replace('-', '_').replace(',', '')
    # Truncate if too long
    if len(safe_name) > 50: safe_name = safe_name[:50]
    
    genre = guess_genre_from_filename(tf) # You can pass full path to help guess?
        
    try:
        tokenizer = Tokenizer(kind="gpt4", add_space_punct=True, lowercase=False)
        with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
        tokens = tokenizer.encode(text)
        total_toks = len(tokens)
        
        # Skip very small books
        if total_toks < 10000:
            return "Skipped (too small)"
        
        selected_metrics = []
        for target in TARGETS:
            if target > total_toks:
                # Add the full book point if not already added
                if not any(m['n_tokens'] == total_toks for m in selected_metrics):
                       cm = CoherenceMetrics(tokens)
                       m = cm.compute_all()
                       m['n_tokens'] = total_toks
                       m['chunk_id'] = f"full_{total_toks}"
                       m['file'] = safe_name
                       m['genre'] = genre
                       m['model'] = 'Human'
                       m['is_outlier'] = False
                       selected_metrics.append(m)
                break
            
            sub_tokens = tokens[:target]
            cm = CoherenceMetrics(sub_tokens)
            m = cm.compute_all()
            m['n_tokens'] = target
            m['chunk_id'] = f"step_{target}"
            m['file'] = safe_name
            m['genre'] = genre
            m['model'] = 'Human'
            m['is_outlier'] = False
            selected_metrics.append(m)

        if selected_metrics:
            out_path = os.path.join(metrics_out_dir, f"metrics_{safe_name}.json")
            with open(out_path, 'w') as f:
                json.dump(selected_metrics, f, indent=2)
            return f"Success: {safe_name}"
    except Exception as e:
        return f"Error {fname}: {e}"
    return "Skipped"

def main():
    # Input folders
    converted_dirs = [
        os.path.join(SCRIPT_DIR, "Converted_Txt"),
        os.path.join(SCRIPT_DIR, "Converted_Mobi_Txt")
    ]
    
    # Output metrics directly to specific folder or general pool
    # Let's put them in 'Selected_Books' so they get picked up automatically
    metrics_out_dir = os.path.join(SCRIPT_DIR, "Selected_Books")
    os.makedirs(metrics_out_dir, exist_ok=True)
    
    txt_files = []
    for d in converted_dirs:
        txt_files.extend(glob.glob(os.path.join(d, "*.txt")))
    
    print(f"Propelling parallel metric calculation for {len(txt_files)} Modern Books on CUDA (2 workers)...")

    worker_func = partial(process_single_book, metrics_out_dir=metrics_out_dir)
    
    # Use 2 workers to stay safe with 30GB VRAM
    workers = 2 
    with Pool(workers) as p:
        results = list(tqdm(p.imap(worker_func, txt_files), total=len(txt_files)))
        
    for r in results:
        if r and r.startswith("Error"):
            print(r)

if __name__ == "__main__":
    main()
