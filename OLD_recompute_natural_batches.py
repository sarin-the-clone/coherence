"""
recompute_natural_batches.py

Reprocesses existing batch result files by locating the source text, 
re-tokenizing, and re-computing all coherence metrics.
Useful for updating metrics across the entire natural text corpus when metric definitions change.
"""
import os
import glob
import json
import logging
import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
import pandas as pd
from tokenizer import Tokenizer
from metrics import CoherenceMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recompute_metrics.log"),
        logging.StreamHandler()
    ]
)

def process_batch_file(batch_file, output_dir):
    """Process a single batch file."""
    try:
        with open(batch_file, 'r') as f:
            data = json.load(f)
            
        updated_data = []
        tokenizer = Tokenizer(kind="gpt4", add_space_punct=False, lowercase=False)
        
        for entry in data:
            # We need the full text which might not be in the batch file...
            # The batch files in natural_texts/Output/batch_results usually contain:
            # {file, n_tokens, beta, alpha...}
            # BUT NOT THE TEXT. 
            # We need to find the source text file. 
            # Assuming 'file' key holds the relative path from 'natural_texts/' or absolute path.
            
            file_path = entry.get('file')
            if not file_path:
                continue
                
            # Resolve path: Try absolute, then relative to project root, then relative to natural_texts
            possible_paths = [
                file_path,
                os.path.join(os.getcwd(), 'natural_texts', file_path),
                os.path.join(os.getcwd(), file_path)
            ]
            
            text_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    text_path = p
                    break
            
            if not text_path:
                logging.warning(f"Could not find text file for {file_path}")
                continue
                
            try:
                with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    
                tokens = tokenizer.encode(text)
                if len(tokens) < 100:
                    continue
                    
                cm = CoherenceMetrics(tokens)
                metrics = cm.compute_all()
                metrics['file'] = file_path # Keep original ID
                metrics['n_tokens'] = len(tokens)
                updated_data.append(metrics)
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                
        # Save updated batch
        batch_name = os.path.basename(batch_file)
        out_path = os.path.join(output_dir, batch_name)
        with open(out_path, 'w') as f:
            json.dump(updated_data, f, indent=2)
            
        return len(updated_data)
        
    except Exception as e:
        logging.error(f"Failed to process batch {batch_file}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='natural_texts/Output/batch_results')
    parser.add_argument('--output-dir', default='natural_texts/Output/batch_results_v2')
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    batch_files = sorted(glob.glob(os.path.join(args.input_dir, 'batch_*.json')))
    logging.info(f"Found {len(batch_files)} batch files to reprocess.")
    
    process_func = partial(process_batch_file, output_dir=args.output_dir)
    
    with Pool(args.workers) as p:
        results = list(tqdm(p.imap(process_func, batch_files), total=len(batch_files)))
        
    total_books = sum(results)
    logging.info(f"Recomputed metrics for {total_books} books.")
    logging.info(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
