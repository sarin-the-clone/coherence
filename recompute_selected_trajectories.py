"""
recompute_selected_trajectories.py

Recalculates metrics for a subset of "selected" natural books (defined by text files in the Selected_Books directory).
Used to generate high-resolution trajectories for reference books in PCA plots.
"""
import os
import json
import glob
import pandas as pd
from tokenizer import Tokenizer
from metrics import CoherenceMetrics
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# Famous outliers for tagging (IDs from previous lists)
FAMOUS_IDS = ["2701", "2489", "135", "1342", "108", "84", "1228", "12623", "2600", "4300", "1184", "996", "58711"]

NAME_MAP = {
    # Outliers / Classics
    "2701": "MobyDick", "2489": "MobyDick", "135": "LesMis", "1342": "PridePrejudice",
    "108": "Sherlock", "84": "Frankenstein", "1228": "OriginSpecies", "12623": "Crusoe",
    "2600": "WarPeace", "4300": "Ulysses", "1184": "MonteCristo", "996": "DonQuixote",
    "58711": "TracedTracked", 
    # Current Centrals -> Updating to TRUE CENTRALS
    "54824": "SecretDoctrine", "65901": "HeartWorld",
    "76058": "Aspasia", "3008": "JargonFile", "11323": "CalebWilliams",
    "40475": "IndianWigwam", "12669": "Marriage", "67090": "WormOuroboros",
    "12360": "TopWorld", "50801": "NarHist", "7469": "Deronda", "30486": "Shirley",
    "24527": "FragmentsOfScience",
    # NEW TRUE SUPER-CENTRALS
    "2568": "TrentsCase",
    "1263": "GlimpsesMoon",
    "25439": "LookingBackward",
    "45962": "OrganismWhole",
    "40414": "SophyKravonia",
    "22158": "LureMask",
    "32843": "SunMaid"
}

TARGETS = [100000, 200000, 400000, 600000, 1000000]

def process_single_book(tf, selected_dir, meta_df):
    fname = os.path.basename(tf)
    bid = fname.split('_')[0]
    
    row = meta_df[meta_df['ID'] == bid]
    if row.empty:
        genre = "Natural Unknown"
    else:
        final_g = str(row.iloc[0]['FinalGenre'])
        genre = "Natural " + final_g
        
    try:
        tokenizer = Tokenizer(kind="gpt4", add_space_punct=True, lowercase=False)
        with open(tf, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
        tokens = tokenizer.encode(text)
        total_toks = len(tokens)
        
        selected_metrics = []
        for target in TARGETS:
            if target > total_toks:
                if not any(m['n_tokens'] == total_toks for m in selected_metrics):
                     if total_toks > 10000:
                           cm = CoherenceMetrics(tokens)
                           m = cm.compute_all()
                           m['n_tokens'] = total_toks
                           m['chunk_id'] = f"full_{total_toks}"
                           m['file'] = fname.replace('.txt', '')
                           m['genre'] = genre
                           m['model'] = 'Human'
                           m['is_outlier'] = (bid in FAMOUS_IDS)
                           selected_metrics.append(m)
                break
            
            sub_tokens = tokens[:target]
            cm = CoherenceMetrics(sub_tokens)
            m = cm.compute_all()
            m['n_tokens'] = target
            m['chunk_id'] = f"step_{target}"
            m['file'] = fname.replace('.txt', '')
            m['genre'] = genre
            m['model'] = 'Human'
            m['is_outlier'] = (bid in FAMOUS_IDS)
            selected_metrics.append(m)

        if selected_metrics:
            short_name = NAME_MAP.get(bid, bid)
            out_path = os.path.join(selected_dir, f"metrics_{short_name}.json")
            with open(out_path, 'w') as f:
                json.dump(selected_metrics, f, indent=2)
            return f"Success: {short_name}"
    except Exception as e:
        return f"Error {fname}: {e}"
    return "Skipped"

def main():
    base_dir = os.getcwd()
    selected_dir = os.path.join(base_dir, "natural_texts", "Selected_Books")
    meta_df = pd.read_csv(os.path.join(base_dir, "natural_texts", "Output", "long_books_with_metadata.csv"))
    meta_df['ID'] = meta_df['ID'].astype(str)
    
    txt_files = glob.glob(os.path.join(selected_dir, "*.txt"))
    
    print(f"Propelling parallel recalculation for {len(txt_files)} selected natural books on CUDA (2 workers)...")

    worker_func = partial(process_single_book, selected_dir=selected_dir, meta_df=meta_df)
    
    # Use 2 workers to stay safe with 30GB VRAM
    workers = 2 
    with Pool(workers) as p:
        results = list(tqdm(p.imap(worker_func, txt_files), total=len(txt_files)))
        
    for r in results:
        if r and r.startswith("Error"):
            print(r)

if __name__ == "__main__":
    main()
