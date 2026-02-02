#!/usr/bin/env python3
"""
retrieve_gutenberg.py

Uses the Gutenberg catalog CSV to find books with valid metadata (Author, Genre),
downloads them, counts tokens, and filters for > 100,000 tokens.
"""

import requests
import time
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from tokenizer import Tokenizer

def get_gutenberg_text(book_id):
    """Download text from Project Gutenberg."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and len(r.text) > 1000:
            raw_text = r.text
            
            # Trim boilerplate
            start = raw_text.find("*** START")
            end = raw_text.find("*** END")
            if start != -1 and end != -1:
                body = raw_text[start:end].strip()
            else:
                body = raw_text.strip()
                
            return body
    except requests.RequestException:
        pass
    return None

def extract_primary_genre(bookshelves_str):
    """Extract primary genre from Bookshelves string."""
    if pd.isna(bookshelves_str) or not bookshelves_str:
        return None
    
    # Split by semicolon and take first non-category shelf
    shelves = [s.strip() for s in bookshelves_str.split(';')]
    for shelf in shelves:
        # Skip generic "Category:" entries
        if not shelf.startswith('Category:'):
            return shelf
    
    # If all are categories, take the first Category entry
    if shelves and shelves[0].startswith('Category:'):
        return shelves[0].replace('Category:', '').strip()
    
    return None

def main():
    start_time = time.time()
    print(f"⏱️  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("📚 Loading Gutenberg catalog...", flush=True)
    
    # Load catalog
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    catalog = pd.read_csv(os.path.join(BASE_DIR, 'Output/pg_catalog.csv'))
    print(f"   Loaded {len(catalog)} books from catalog", flush=True)
    
    # Filter for books with valid metadata
    # - Must have Author
    # - Must have Bookshelves (for genre)
    # - Must be in English
    # - Type must be 'Text'
    
    filtered = catalog[
        (catalog['Type'] == 'Text') &
        (catalog['Language'] == 'en') &
        (catalog['Authors'].notna()) &
        (catalog['Authors'] != 'Various') &
        (catalog['Authors'] != 'Anonymous') &
        (catalog['Bookshelves'].notna())
    ].copy()
    
    print(f"   Filtered to {len(filtered)} books with valid metadata", flush=True)
    
    # Extract primary genre
    print("   Extracting genres...", flush=True)
    filtered['Genre'] = filtered['Bookshelves'].apply(extract_primary_genre)
    filtered = filtered[filtered['Genre'].notna()]
    
    print(f"   {len(filtered)} books with extractable genres", flush=True)
    
    # Scan all eligible books (no sampling)
    print(f"   Preparing to scan all {len(filtered)} eligible books...", flush=True)
    sampled = filtered
    print(f"   Ready to scan {len(sampled)} books", flush=True)
    
    print(f"\n🔍 Scanning {len(sampled)} books for > 100k tokens...")
    print("-" * 90)
    print(f"{'ID':<6} {'Tokens':<10} {'Status':<10} {'Author':<25} {'Genre':<20} {'Title'}")
    print("-" * 90)

    tok = Tokenizer("gpt4")
    found_count = 0
    rejected_count = 0
    failed_count = 0
    results = []
    
    # Open CSV file for incremental writing
    csv_file = open(os.path.join(BASE_DIR, "Output/long_books_list.csv"), "w")
    csv_file.write("ID,Tokens,Title,Author,Genre\n")
    csv_file.flush()

    for idx, row in sampled.iterrows():
        book_id = row['Text#']
        title = row['Title']
        author = row['Authors']
        genre = row['Genre']
        
        text = get_gutenberg_text(book_id)
        
        if text:
            tokens = tok.encode(text)
            n_tokens = len(tokens)
            
            if n_tokens > 100_000:
                status = "✅ KEEP"
                found_count += 1
                
                # Clean up for display
                auth_clean = str(author).split(';')[0][:24]
                genre_clean = str(genre)[:19]
                title_clean = str(title)[:35]
                
                results.append((book_id, title, author, genre, n_tokens))
                print(f"{book_id:<6} {n_tokens:<10,.0f} {status:<10} {auth_clean:<25} {genre_clean:<20} {title_clean}")
                
                # Write to CSV immediately
                t = str(title).replace('"', '""')
                a = str(author).replace('"', '""')
                g = str(genre).replace('"', '""')
                csv_file.write(f'{book_id},{n_tokens},"{t}","{a}","{g}"\n')
                csv_file.flush()
            else:
                rejected_count += 1
        else:
            failed_count += 1
        
        # Be nice to the server
        time.sleep(0.2)

    
    # Close CSV file
    csv_file.close()
    
    end_time = time.time()
    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    
    print("-" * 90)
    print(f"🎉 Found {found_count} books with > 100,000 tokens and valid metadata.")
    print(f"❌ Rejected {rejected_count} books (< 100k tokens)")
    print(f"⚠️  Failed to download {failed_count} books")
    print(f"📊 Total scanned: {found_count + rejected_count + failed_count} / {len(sampled)}")
    print(f"⏱️  Total time: {hours}h {minutes}m {seconds}s ({elapsed:.1f} seconds)")
    print(f"⏱️  Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📝 List saved to Output/long_books_list.csv")

if __name__ == "__main__":
    main()
