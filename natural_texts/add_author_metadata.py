#!/usr/bin/env python3
"""
add_author_metadata.py

Extract author metadata (era, language, location) from author strings and add to book list.
"""

import pandas as pd
import re

def extract_author_era(author_str):
    """Extract birth/death years and determine era from author string."""
    if pd.isna(author_str):
        return "Unknown"
    
    # Look for year patterns like "1850-1920" or "1850-"
    year_pattern = r'(\d{4})-(\d{4}|\?)?'
    match = re.search(year_pattern, str(author_str))
    
    if match:
        birth_year = int(match.group(1))
        
        # Determine era based on birth year
        if birth_year < 1500:
            return "Ancient/Medieval"
        elif birth_year < 1700:
            return "Renaissance/Early Modern"
        elif birth_year < 1800:
            return "18th Century"
        elif birth_year < 1850:
            return "Early 19th Century"
        elif birth_year < 1900:
            return "Late 19th Century"
        elif birth_year < 1950:
            return "Early 20th Century"
        else:
            return "Modern (1950+)"
    
    return "Unknown"

def extract_author_language(author_str):
    """Infer language from author name patterns (very approximate)."""
    if pd.isna(author_str):
        return "Unknown"
    
    author_lower = str(author_str).lower()
    
    # Very rough heuristics based on common patterns
    if any(x in author_lower for x in ['von ', 'van der', 'schmidt', 'müller', 'wagner']):
        return "German"
    elif any(x in author_lower for x in ['de ', 'du ', 'le ', 'la ']):
        return "French"
    elif any(x in author_lower for x in ['van ', 'de vries', 'van den']):
        return "Dutch"
    elif any(x in author_lower for x in ['ibn ', 'al-', 'muhammad']):
        return "Arabic"
    else:
        return "English/Other"

def main():
    """Add author metadata to book list."""
    print("📚 Loading book list...")
    # Load data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(BASE_DIR, 'Output/long_books_with_broad_genres.csv'))
    print(f"   Loaded {len(df)} books")
    
    print("\n🔍 Extracting author metadata...")
    df['AuthorEra'] = df['Author'].apply(extract_author_era)
    df['AuthorLanguage'] = df['Author'].apply(extract_author_language)
    
    print("\n📊 Author Era distribution:")
    print(df['AuthorEra'].value_counts())
    
    print("\n📊 Author Language distribution:")
    print(df['AuthorLanguage'].value_counts())
    
    print(f"\n💾 Saving updated book list...")
    # Save
    df.to_csv(os.path.join(BASE_DIR, 'Output/long_books_with_metadata.csv'), index=False)
    print("✅ Saved to Output/long_books_with_metadata.csv")

if __name__ == "__main__":
    main()
