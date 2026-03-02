"""
add_title_metadata.py

Scans directory of metric JSON files and injects a clean, human-readable 'title'
field into each metric entry, derived from the filename.
"""
import os
import json
import glob

def clean_title_from_fname(fname):
    """
    Derive a clean human-readable title from the filename.
    Handles outliers (MobyDick etc) and modern format (Title___Author).
    """
    base = fname.replace("metrics_", "").replace(".json", "")
    
    # Famous outliers mapping
    SPECIAL_MAP = {
        "MobyDick": "Moby Dick",
        "LesMis": "Les Misérables", 
        "PridePrejudice": "Pride & Prejudice",
        "Sherlock": "Sherlock Holmes",
        "Frankenstein": "Frankenstein",
        "OriginSpecies": "Origin of Species",
        "Crusoe": "Robinson Crusoe",
        "WarPeace": "War and Peace",
        "Ulysses": "Ulysses",
        "MonteCristo": "Count of Monte Cristo",
        "DonQuixote": "Don Quixote",
        # Add others from process_new_modern_books.py NAME_MAP if needed
    }
    
    if base in SPECIAL_MAP:
        return SPECIAL_MAP[base]
    
    # Modern Books format: Title___Author
    if "___" in base:
        parts = base.split("___")
        title_part = parts[0]
        # Replace underscores with spaces
        return title_part.replace("_", " ").strip()
        
    # Standard format: 12345_Book_Title or just Book_Title
    # Typically process_new_modern_books saved them as "metrics_Book_Name.json" or "metrics_ID_Name.json"
    # If no triple underscore, try matching simple underscore replacement
    
    # Try removing numeric ID prefix if present (e.g. 12345_Title)
    parts = base.split('_', 1)
    if parts[0].isdigit() and len(parts) > 1:
        raw_title = parts[1]
    else:
        raw_title = base
        
    return raw_title.replace("_", " ").strip()

def update_metadata_in_place(directory):
    files = glob.glob(os.path.join(directory, "metrics_*.json"))
    print(f"Scanning {directory} ({len(files)} files)...")
    
    updated_count = 0
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Data is a list of metrics objects (one per chunk size)
            # We update all objects in the list
            modified = False
            fname = os.path.basename(fpath)
            clean_title = clean_title_from_fname(fname)
            
            for item in data:
                if item.get('title') != clean_title:
                    item['title'] = clean_title
                    modified = True
            
            if modified:
                with open(fpath, 'w') as f:
                    json.dump(data, f, indent=2)
                updated_count += 1
                
        except Exception as e:
            print(f"Error updating {fpath}: {e}")
            
    print(f"Updated {updated_count} files in {directory}.")

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Update both folders
    dirs_to_update = [
        os.path.join(SCRIPT_DIR, "Selected_Books"),
        os.path.join(SCRIPT_DIR, "Selected_Modern_Books")
    ]
    
    for d in dirs_to_update:
        if os.path.exists(d):
            update_metadata_in_place(d)
        else:
            print(f"Skipping missing directory: {d}")
