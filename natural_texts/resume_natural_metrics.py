import os
import sys
import glob
import re

# Ensure project root is on path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR)) # src/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from coherence.natural_texts.batch_processor import run_batch

OUTPUT_DIR = os.path.join(THIS_DIR, 'Output')
BATCH_RESULTS_DIR = os.path.join(OUTPUT_DIR, 'batch_results')

def get_last_batch_num():
    if not os.path.exists(BATCH_RESULTS_DIR):
        return 0
    
    files = glob.glob(os.path.join(BATCH_RESULTS_DIR, "batch_*.json"))
    if not files:
        return 0
        
    max_batch = 0
    for f in files:
        match = re.search(r'batch_(\d+)\.json', os.path.basename(f))
        if match:
            num = int(match.group(1))
            if num > max_batch:
                max_batch = num
    return max_batch

def resume_processing(batch_size=100, device='cuda'):
    last_batch = get_last_batch_num()
    start_batch = last_batch + 1
    
    print(f"\n🔄 Resuming processing from batch {start_batch} (last found: {last_batch})...")
    
    batch_num = start_batch
    while True:
        # run_batch returns False when no books are left
        has_more = run_batch(batch_num, batch_size=batch_size, device=device)
        if not has_more:
            break
        batch_num += 1

if __name__ == "__main__":
    resume_processing()
