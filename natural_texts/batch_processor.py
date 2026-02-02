#!/usr/bin/env python3
"""
batch_processor.py

Process books in stratified batches with progress tracking.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import torch
from datetime import datetime
from tokenizer import Tokenizer
from metrics import CoherenceMetrics

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOKS_CSV = os.path.join(BASE_DIR, 'Output/long_books_with_metadata.csv')
PROGRESS_FILE = os.path.join(BASE_DIR, 'Output/processed_books.json')
RESULTS_DIR = os.path.join(BASE_DIR, 'Output/batch_results')
BATCH_SIZE = 100

def load_progress():
    """Load list of already processed book IDs."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

def save_progress(processed_ids):
    """Save list of processed book IDs."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(list(processed_ids), f)

def get_stratified_batch(df, processed_ids, batch_size=100):
    """Sample batch stratified by BroadGenre."""
    # Filter out already processed books
    available = df[~df['ID'].isin(processed_ids)].copy()
    
    if len(available) == 0:
        return None
    
    print(f"📊 Available books: {len(available)}")
    print(f"   Already processed: {len(processed_ids)}")
    
    # Calculate samples per genre (proportional)
    genre_counts = available['FinalGenre'].value_counts()
    total = len(available)
    
    samples = {}
    for genre, count in genre_counts.items():
        proportion = count / total
        n_samples = max(1, int(batch_size * proportion))
        samples[genre] = min(n_samples, count)
    
    # Adjust to exactly batch_size
    total_samples = sum(samples.values())
    if total_samples > batch_size:
        # Reduce largest categories
        while total_samples > batch_size:
            max_genre = max(samples, key=samples.get)
            if samples[max_genre] > 1:
                samples[max_genre] -= 1
                total_samples -= 1
            else:
                break
    
    # Sample from each genre
    batch_dfs = []
    for genre, n in samples.items():
        genre_books = available[available['FinalGenre'] == genre]
        sampled = genre_books.sample(n=min(n, len(genre_books)), random_state=42)
        batch_dfs.append(sampled)
    
    batch = pd.concat(batch_dfs, ignore_index=True)
    
    print(f"\n📦 Batch composition ({len(batch)} books):")
    print(batch['FinalGenre'].value_counts())
    
    return batch

def process_book(row, device='cuda'):
    """Process a single book and compute coherence metrics."""
    book_id = row['ID']
    title = row['Title']
    author = row['Author']
    genre = row['Genre']
    broad_genre = row['FinalGenre'] # Using FinalGenre as broad_genre for compatibility
    n_tokens = row['Tokens']
    
    print(f"\n📖 Processing: {title[:50]}... (ID {book_id})")
    print(f"   Author: {author[:50]}")
    print(f"   Genre: {genre} → {broad_genre}")
    print(f"   Tokens: {n_tokens:,}")
    
    # Download text
    from retrieve_gutenberg import get_gutenberg_text
    try:
        text = get_gutenberg_text(book_id)
        if text is None:
            print(f"   ⚠️ Download returned None")
            return None
    except Exception as e:
        print(f"   ⚠️ Failed to download: {e}")
        return None
    
    # Tokenize
    tok = Tokenizer("gpt4")
    tokens = tok.encode(text)
    
    # Compute metrics
    print(f"   🔬 Computing coherence metrics...")
    cm = CoherenceMetrics(tokens, device=device)
    metrics = cm.compute_all()
    
    # Add metadata
    result = {
        'book_id': int(book_id),
        'title': title,
        'author': author,
        'genre': genre,
        'broad_genre': broad_genre,
        'n_tokens': int(n_tokens),
        **metrics
    }
    
    print(f"   ✅ Done!")
    return result

def run_batch(batch_num, batch_size=100, device='cuda'):
    """Run one batch of processing."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"⏱️  Batch {batch_num} started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📚 Loading books...")
    df = pd.read_csv(BOOKS_CSV)
    processed_ids = load_progress()
    
    # Get stratified batch
    batch = get_stratified_batch(df, processed_ids, batch_size)
    
    if batch is None:
        print("\n🎉 All books have been processed!")
        return False
    
    # Process each book
    results = []
    for idx, row in batch.iterrows():
        result = process_book(row, device=device)
        if result:
            results.append(result)
            # Update progress immediately
            processed_ids.add(int(row['ID']))
            save_progress(processed_ids)
    
    # Save batch results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    batch_file = f"{RESULTS_DIR}/batch_{batch_num:03d}.json"
    with open(batch_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"✅ Batch {batch_num} completed!")
    print(f"   Processed: {len(results)} / {len(batch)} books")
    print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"   Avg per book: {elapsed/len(results):.1f} seconds")
    print(f"   Saved to: {batch_file}")
    print(f"⏱️  Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    return True

if __name__ == "__main__":
    import sys
    
    batch_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    device = sys.argv[3] if len(sys.argv) > 3 else 'cuda'
    
    run_batch(batch_num, batch_size, device)
