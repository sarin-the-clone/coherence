#!/usr/bin/env python3
"""
refine_genres.py

Combines genre cleaning and refinement operations:
1. Fixes misclassifications in 'Other' category.
2. Splits 'Fiction' into 'General Fiction', 'Historical Fiction', and 'Romance'.
"""

import os
import pandas as pd

def fix_misclassifications(df):
    """Reclassify items currently in 'Other' to appropriate categories."""
    print("\n🔄 Fixing misclassifications in 'Other'...")
    
    changes = []
    other_mask = df['FinalGenre'] == 'Other'
    
    # Define mappings: Target Genre -> Keywords
    mappings = {
        'Science': [
            'Psychiatry', 'Psychology', 'Chemistry', 'Biochemistry',
            'Nature', 'Gardening', 'Animals', 'Technology', 'Biology',
            'Physics', 'Mathematics', 'Astronomy', 'Geology'
        ],
        'History': [
            'History', 'Medieval', 'Middle Ages', 'Early Modern', 'Classical Antiquity',
            'Slavery', 'United States', 'Warfare', 'Archaeology', 'Anthropology'
        ],
        'Fiction': [
            'British Literature', 'American Literature', 'Classics of Literature'
        ],
        'Children/YA': [
            "Children's Anthologies"
        ],
        'Humor': [
            'Humor' # Special case check for exact match or substring if needed
        ]
    }
    
    for target_genre, keywords in mappings.items():
        # Refresh mask
        other_mask = df['FinalGenre'] == 'Other'
        
        for keyword in keywords:
            if target_genre == 'Humor' and keyword == 'Humor':
                 mask = other_mask & (df['Genre'].str.lower() == 'humor')
            else:
                 mask = other_mask & df['Genre'].str.contains(keyword, case=False, na=False)
            
            count = mask.sum()
            if count > 0:
                df.loc[mask, 'FinalGenre'] = target_genre
                changes.append(f"  {keyword}: {count} books → {target_genre}")

    for change in changes:
        print(change)
    
    if not changes:
        print("  No misclassifications found to fix.")
        
    return df

def split_fiction(df):
    """Split Fiction/General Fiction category into 3 subcategories."""
    print("\n🔄 Splitting Fiction into subcategories...")
    
    changes = {
        'General Fiction': 0,
        'Historical Fiction': 0,
        'Romance': 0
    }
    
    # Create mask for Fiction or General Fiction category (to handle re-runs)
    fiction_mask = (df['FinalGenre'] == 'Fiction') | (df['FinalGenre'] == 'General Fiction')
    
    # 1. Romance
    romance_mask = fiction_mask & df['Genre'].str.contains('Romance', case=False, na=False)
    df.loc[romance_mask, 'FinalGenre'] = 'Romance'
    changes['Romance'] = romance_mask.sum()
    
    # 2. Historical Fiction
    # Exclude already assigned Romance
    historical_mask = fiction_mask & ~romance_mask & (
        df['Genre'].str.contains('Historical', case=False, na=False)
    )
    df.loc[historical_mask, 'FinalGenre'] = 'Historical Fiction'
    changes['Historical Fiction'] = historical_mask.sum()
    
    # 3. General Fiction (everything else that was Fiction)
    general_mask = fiction_mask & ~romance_mask & ~historical_mask
    df.loc[general_mask, 'FinalGenre'] = 'General Fiction'
    changes['General Fiction'] = general_mask.sum()
    
    for category, count in changes.items():
        print(f"  {category}: {count} books")
        
    return df

def main():
    print("📚 Loading book metadata...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, 'Output/long_books_with_metadata.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} books")
    
    # 1. Fix misclassifications
    df = fix_misclassifications(df)
    
    # 2. Split Fiction
    df = split_fiction(df)
    
    print("\n📋 Final Genre Distribution:")
    print(df['FinalGenre'].value_counts().sort_index())
    
    print(f"\n💾 Saving updated metadata...")
    df.to_csv(csv_path, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
