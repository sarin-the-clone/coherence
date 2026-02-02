#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_on_natural_texts.py — Coherence analysis on natural long texts (Gutenberg classics)
"""


import os
import sys
# Add the coherence root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import requests
import multiprocessing as mp
from tqdm import tqdm
from tokenizer import Tokenizer
from coherence_metrics import CoherenceMetrics



def run_task(args):
    """Wrapper for multiprocessing to unpack arguments."""
    return process_book(*args)

def sort_result_keys(d):
    order = [
        "book_id", "book_title", "author", "genre", "n_tokens",
        "beta", "alpha", "H_DFA1", "H_DFA2", "H_DFA3", "H_DMA", 
        "H_WT_Haar", "H_WT_Haar_short", "H_WT_Haar_long",
        "H_WT_db4", "H_WT_db4_short", "H_WT_db4_long", "delta",
        "entropy_bits", "perplexity", "lz_complexity", "mi_decay_gamma",
        "mi_tau_max", "mi_n_lags", "tau_to_length_ratio"
    ]
    # Create a dict in that order
    return {k: d[k] for k in order if k in d}



# ----------------------------------------------------------------------
# 1. Retrieve Gutenberg text
# ----------------------------------------------------------------------
def check_book_available(book_id):
    """Check if a book is downloadable from Project Gutenberg."""
    base = f"https://www.gutenberg.org/files/{book_id}/"
    variants = [
        f"{book_id}-0.txt",
        f"{book_id}-8.txt",
        f"{book_id}.txt",
        f"{book_id}.txt.utf-8",
    ]
    
    for filename in variants:
        url = base + filename
        try:
            r = requests.head(url, timeout=10)  # HEAD request is faster
            if r.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False

def get_gutenberg_text(book_id):
    """Download text from Project Gutenberg with fallback patterns."""
    base = f"https://www.gutenberg.org/files/{book_id}/"
    variants = [
        f"{book_id}-0.txt",   # UTF-8 (new format)
        f"{book_id}-8.txt",   # UTF-8 (legacy)
        f"{book_id}.txt",     # plain
        f"{book_id}.txt.utf-8",  # rare variant
    ]

    for filename in variants:
        url = base + filename
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                text = r.text
                break
        except requests.RequestException:
            continue
    else:
        raise RuntimeError(f"⚠️ Could not fetch any valid version for book {book_id}")

    # crude trim to skip license header/footer
    start = text.find("*** START")
    end = text.find("*** END")
    if start == -1 or end == -1:
        print(f"⚠️ Warning: could not trim boilerplate for ID {book_id}")
        return text.strip()
    return text[start:end].strip()


# ----------------------------------------------------------------------
# 2. Author list
# ----------------------------------------------------------------------
# authors = {
#     "austen": {"id": 1342, "title": "Pride and Prejudice"},
#     "melville": {"id": 2701, "title": "Moby Dick"},
#     "dostoevsky": {"id": 600, "title": "Crime and Punishment"},
#     "tolstoy": {"id": 2600, "title": "War and Peace"},
#     "hugo": {"id": 135, "title": "Les Misérables"},
#     "dickens": {"id": 766, "title": "David Copperfield"},
#     "eliot": {"id": 145, "title": "Middlemarch"},
# }
# authors.update({
#     "shelley":   {"id": 84, "title": "Frankenstein"},
#     "carroll":   {"id": 11, "title": "Alice in Wonderland"},
#     "bronte":    {"id": 1260, "title": "Jane Eyre"},
#     "wilde":     {"id": 174, "title": "Dorian Gray"},
#     "conrad":    {"id": 526, "title": "Heart of Darkness"},
#     "joyce":     {"id": 4300, "title": "Ulysses"},
#     "orwell":    {"id": 7370, "title": "1984"},
#     "woolf":     {"id": 14558, "title": "Mrs Dalloway"},
#     "hesse":     {"id": 2500, "title": "Siddhartha"},
#     "stoker":    {"id": 345, "title": "Dracula"},
# })



# authors = {
#     "austen_1342": {"id": 1342, "title": "Pride and Prejudice"},
#     "austen_158":  {"id": 158,  "title": "Sense and Sensibility"},
#     "austen_141":  {"id": 141,  "title": "Emma"},
#     "austen_161":  {"id": 161,  "title": "Northanger Abbey"},
#     "austen_105":  {"id": 105,  "title": "Mansfield Park"},
#     "austen_121":  {"id": 121,  "title": "Persuasion"},
# }

# authors = {
#     "dickens_766":  {"id": 766,  "title": "David Copperfield"},
#     "dickens_730":  {"id": 730,  "title": "Great Expectations"},
#     "dickens_98":   {"id": 98,   "title": "A Tale of Two Cities"},
#     "dickens_580":  {"id": 580,  "title": "Oliver Twist"},
#     "dickens_1023": {"id": 1023, "title": "Bleak House"},
#     "dickens_786":  {"id": 786,  "title": "Hard Times"},
#     "dickens_967":  {"id": 967,  "title": "Little Dorrit"},
#     "dickens_1400": {"id": 1400, "title": "Our Mutual Friend"},
# }


# authors = {
#     "tolstoy_2600":  {"id": 2600,  "title": "War and Peace"},
#     "tolstoy_1399":  {"id": 1399,  "title": "Anna Karenina"},
#     "tolstoy_985":   {"id": 985,   "title": "The Kreutzer Sonata"},
#     "tolstoy_19333": {"id": 19333, "title": "Resurrection"},
#     "tolstoy_24450": {"id": 24450, "title": "Childhood, Boyhood and Youth"},
# }


authors = {
    # Mark Twain (>100k tokens only)
    "twain_76":   {"id": 76,   "title": "Adventures of Huckleberry Finn", "author": "Mark Twain", "genre": "Novel"},
    "twain_74":   {"id": 74,   "title": "The Adventures of Tom Sawyer", "author": "Mark Twain", "genre": "Novel"},
    "twain_142":  {"id": 142,  "title": "The Prince and the Pauper", "author": "Mark Twain", "genre": "Novel"},
    "twain_86":   {"id": 86,   "title": "A Connecticut Yankee in King Arthur's Court", "author": "Mark Twain", "genre": "Novel"},
    "twain_3177": {"id": 3177, "title": "Life on the Mississippi", "author": "Mark Twain", "genre": "Autobiography"},
    "twain_3176": {"id": 3176, "title": "The Innocents Abroad", "author": "Mark Twain", "genre": "Travel"},
    "twain_3178": {"id": 3178, "title": "Roughing It", "author": "Mark Twain", "genre": "Autobiography"},
    
    # Charles Dickens (similar era, prolific novelist)
    "dickens_766":  {"id": 766,  "title": "David Copperfield", "author": "Charles Dickens", "genre": "Novel"},
    "dickens_730":  {"id": 730,  "title": "Great Expectations", "author": "Charles Dickens", "genre": "Novel"},
    "dickens_98":   {"id": 98,   "title": "A Tale of Two Cities", "author": "Charles Dickens", "genre": "Novel"},
    "dickens_580":  {"id": 580,  "title": "Oliver Twist", "author": "Charles Dickens", "genre": "Novel"},
    "dickens_1023": {"id": 1023, "title": "Bleak House", "author": "Charles Dickens", "genre": "Novel"},
    "dickens_786":  {"id": 786,  "title": "Hard Times", "author": "Charles Dickens", "genre": "Novel"},
    "dickens_1400": {"id": 1400, "title": "Our Mutual Friend", "author": "Charles Dickens", "genre": "Novel"},
    
    # Scientific/Technical Documents (>100k tokens only)
    "darwin_1228":  {"id": 1228, "title": "On the Origin of Species", "author": "Charles Darwin", "genre": "Science"},
    "darwin_944":   {"id": 944,  "title": "The Descent of Man", "author": "Charles Darwin", "genre": "Science"},
    "darwin_2087":  {"id": 2087, "title": "The Expression of the Emotions", "author": "Charles Darwin", "genre": "Science"},
    "newton_28233": {"id": 28233, "title": "Principia Mathematica", "author": "Isaac Newton", "genre": "Science"},
    
    # Victor Hugo (expanding to 5+)
    "hugo_135":     {"id": 135,   "title": "Les Misérables", "author": "Victor Hugo", "genre": "Novel"},
    "hugo_2610":    {"id": 2610,  "title": "Notre-Dame de Paris", "author": "Victor Hugo", "genre": "Novel"},
    "hugo_32338":   {"id": 32338, "title": "Toilers of the Sea", "author": "Victor Hugo", "genre": "Novel"},
    "hugo_12587":   {"id": 12587, "title": "The Man Who Laughs", "author": "Victor Hugo", "genre": "Novel"},
    
    # Jane Austen (already 3, add more)
    "austen_1342":  {"id": 1342, "title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Novel"},
    "austen_158":   {"id": 158,  "title": "Sense and Sensibility", "author": "Jane Austen", "genre": "Novel"},
    "austen_141":   {"id": 141,  "title": "Emma", "author": "Jane Austen", "genre": "Novel"},
    "austen_105":   {"id": 105,  "title": "Persuasion", "author": "Jane Austen", "genre": "Novel"},
    "austen_121":   {"id": 121,  "title": "Northanger Abbey", "author": "Jane Austen", "genre": "Novel"},
    "austen_946":   {"id": 946,  "title": "Lady Susan", "author": "Jane Austen", "genre": "Novel"},
    
    # Holy Books (expanding to 5+)
    "bible_10":     {"id": 10,    "title": "The King James Bible", "author": "Various", "genre": "Religious"},
    "quran_2800":   {"id": 2800,  "title": "The Quran", "author": "Various", "genre": "Religious"},
    "gita_2388":    {"id": 2388,  "title": "The Bhagavad Gita", "author": "Various", "genre": "Religious"},
    "dhammapada_2017": {"id": 2017, "title": "Dhammapada (Buddhist)", "author": "Various", "genre": "Religious"},
    "confucius_3330": {"id": 3330, "title": "The Analects of Confucius", "author": "Confucius", "genre": "Religious"},
    
    # Additional Science/Technical Books (expanding to 8+)
    "lavoisier_30775": {"id": 30775, "title": "Elements of Chemistry", "author": "Antoine Lavoisier", "genre": "Science"},
    "faraday_14986":   {"id": 14986, "title": "Experimental Researches in Electricity", "author": "Michael Faraday", "genre": "Science"},
    "galileo_39031":   {"id": 39031, "title": "Sidereal Messenger", "author": "Galileo Galilei", "genre": "Science"},
    "lyell_33224":     {"id": 33224, "title": "Principles of Geology", "author": "Charles Lyell", "genre": "Science"},
}



id_to_title = {v["id"]: v["title"] for v in authors.values()}

# ----------------------------------------------------------------------
# 3. Worker function (runs on one GPU)
# ----------------------------------------------------------------------
def process_book(name, book_id, device):
    print(f"[{device.upper()}] 📖 {name.capitalize()} (ID {book_id})")
    book_title = id_to_title.get(book_id, "unknown")
    book_author = authors.get(name, {}).get("author", "Unknown")
    book_genre = authors.get(name, {}).get("genre", "Unknown")
    
    text = get_gutenberg_text(book_id)
    print(f"[{device.upper()}] → {len(text):,} characters")

    # Tokenize
    tok = Tokenizer("gpt4")
    tokens = tok.encode(text)
    n_tokens = len(tokens)
    print(f"[{device.upper()}] → {n_tokens:,} tokens")

    # Compute metrics
    cm = CoherenceMetrics(tokens, device=device)
    res = cm.compute_all()

    # Add metadata
    res["book_id"] = book_id
    res["book_title"] = book_title
    res["author"] = book_author
    res["genre"] = book_genre
    res["n_tokens"] = n_tokens
    res["tau_to_length_ratio"] = (
        res["mi_tau_max"] / n_tokens if "mi_tau_max" in res else None
    )

    print(f"[{device.upper()}] ✅ Done {name}")
    return (name, res)



# ----------------------------------------------------------------------
# 4. Parallel execution with hybrid GPU/CPU strategy
# ----------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(BASE_DIR, "Output"), exist_ok=True)

    ngpu = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(ngpu)] if ngpu > 0 else ["cpu"]
    print(f"🧮 Using devices: {devices}")

    # Assign one book per device in round robin
    # First, check which books are available
    print("🔍 Checking book availability...")
    available_books = {}
    for name, info in authors.items():
        book_id = info["id"]
        if check_book_available(book_id):
            available_books[name] = info
            print(f"  ✓ {name} (ID {book_id})")
        else:
            print(f"  ✗ {name} (ID {book_id}) - NOT AVAILABLE")
    
    print(f"\n📚 {len(available_books)}/{len(authors)} books available for processing\n")
    
    tasks = []
    for i, (name, info) in enumerate(available_books.items()):
        device = devices[i % len(devices)]
        tasks.append((name, info["id"], device))

    # Set CPU threading env to use all cores per process
    n_cores = os.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(max(1, n_cores // max(1, ngpu)))
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    os.environ["NUMBA_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

    print(f"🧩 Each analysis process may use up to {os.environ['OMP_NUM_THREADS']} CPU threads for CPU-bound metrics")

    results = {}
    with mp.get_context("spawn").Pool(processes=len(devices)) as pool:
        for name, res in tqdm(
            pool.imap_unordered(run_task, tasks),
            total=len(tasks),
            desc="📊 Processing books",
            dynamic_ncols=True,
        ):
            results[name] = res

    # Save results
    sorted_results = {k: sort_result_keys(v) for k, v in results.items()}
    with open(os.path.join(BASE_DIR, "Output/coherence_results.json"), "w") as f:
        json.dump(sorted_results, f, indent=2)

    print("\n✅ Results saved to Output/coherence_results.json")
    
    # Export to CSV for PCA analysis
    import csv
    csv_path = os.path.join(BASE_DIR, "Output/coherence_results.csv")
    if results:
        all_keys = set()
        for res in results.values():
            all_keys.update(res.keys())
        fieldnames = sorted(all_keys)
        
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for name, res in results.items():
                writer.writerow(res)
        
        print(f"✅ Results exported to {csv_path} for PCA analysis")

    # ----------------------------------------------------------------------
    # 5. Print summary table (aligned columns)
    # ----------------------------------------------------------------------
    print("\n📘 Summary (tokens, τ/length %, key metrics):\n")

    # Define fixed column widths with descriptive headers
    header = (
        f"{'Book':<25}"
        f"{'Author':<17}"
        f"{'Genre':<14}"
        f"{'Tokens':>10}"
        f"{'DFA1':>7}"
        f"{'DFA2':>7}"
        f"{'DFA3':>7}"
        f"{'DMA':>7}"
        f"{'Haar':>7}"
        f"{'Hr_sh':>7}"
        f"{'Hr_lg':>7}"
        f"{'db4':>7}"
        f"{'db_sh':>7}"
        f"{'db_lg':>7}"
        f"{'DEA':>7}"
        f"{'ACF(β)':>8}"
        f"{'PSD(α)':>8}"
    )
    print(header)
    print("-" * len(header))

    for name, res in results.items():
        title = (res.get("book_title", name) + " ")[:25]
        author = (res.get("author", "Unknown") + " ")[:17]
        genre = (res.get("genre", "Unknown") + " ")[:14]
        n_tokens = res.get("n_tokens", 0)
        
        # Structural
        beta = res.get("beta", float('nan'))
        alpha = res.get("alpha", float('nan'))
        H1 = res.get("H_DFA1", float('nan'))
        H2 = res.get("H_DFA2", float('nan'))
        H3 = res.get("H_DFA3", float('nan'))
        H_dma = res.get("H_DMA", float('nan'))
        H_haar = res.get("H_WT_Haar", float('nan'))
        H_haar_short = res.get("H_WT_Haar_short", float('nan'))
        H_haar_long = res.get("H_WT_Haar_long", float('nan'))
        H_db4 = res.get("H_WT_db4", float('nan'))
        H_db4_short = res.get("H_WT_db4_short", float('nan'))
        H_db4_long = res.get("H_WT_db4_long", float('nan'))
        delta = res.get("delta", float('nan'))

        print(
            f"{title:<25}"
            f"{author:<17}"
            f"{genre:<14}"
            f"{n_tokens:>10,.0f}"
            f"{H1:>7.3f}"
            f"{H2:>7.3f}"
            f"{H3:>7.3f}"
            f"{H_dma:>7.3f}"
            f"{H_haar:>7.3f}"
            f"{H_haar_short:>7.3f}"
            f"{H_haar_long:>7.3f}"
            f"{H_db4:>7.3f}"
            f"{H_db4_short:>7.3f}"
            f"{H_db4_long:>7.3f}"
            f"{delta:>7.3f}"
            f"{beta:>8.3f}"
            f"{alpha:>8.3f}"
        )

    print("\nLegend:")
    print("  DFA1-3   : Detrended Fluctuation Analysis (Order 1, 2, 3)")
    print("  DMA      : Detrending Moving Average (Centered)")
    print("  Haar     : Wavelet DFA (Haar - Global)")
    print("  Hr_sh    : Haar - Short-range scaling (< 100 tokens)")
    print("  Hr_lg    : Haar - Long-range scaling (>= 100 tokens)")
    print("  db4      : Wavelet DFA (Daubechies-4 - Global)")
    print("  db_sh    : db4 - Short-range scaling (< 100 tokens)")
    print("  db_lg    : db4 - Long-range scaling (>= 100 tokens)")
    print("  DEA      : Diffusion Entropy Analysis")
    print("  ACF(β)   : Autocorrelation Decay")
    print("  PSD(α)   : Spectral Exponent")
    print("\n")