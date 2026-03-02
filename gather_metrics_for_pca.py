"""
gather_metrics_for_pca.py

Aggregates coherence metrics from valid experimental runs (generated texts) 
and selected natural reference books into a single JSON file for PCA projection.
Selects chunks at specific token landmarks (100k, 200k, etc.) to trace trajectories.
"""
import json
import sys
from pathlib import Path

# Add analysis dir to path to reuse loading logic
sys.path.append(str(Path(__file__).parent / "analysis"))

try:
    from plot_chunk_metrics import load_attempts, find_accepted_attempt
except ImportError:
    sys.path.append("analysis")
    from plot_chunk_metrics import load_attempts, find_accepted_attempt

def map_genre(meta, folder_name):
    # Map based on prompt filename (Experimental Design Source of Truth)
    prompt_file = meta.get("prompt_filename", "")
    if prompt_file:
        pf = prompt_file.lower()
        if "historical_library" in pf: return "Historical Fiction"
        if "romance_drama" in pf: return "Romance Drama"
        if "cyberpunk_grid" in pf: return "Cyberpunk"
        if "science_book" in pf: return "Science Book"
        if "philosophy_book" in pf: return "Philosophy"
        if "technical_reference" in pf: return "Technical Reference"
        if "biography" in pf: return "Biography"
        if "lighthouse_keeper" in pf: return "General Fiction"

    # Fallback to metadata/filename analysis
    raw = meta.get("broader_genre", "")
    spec = meta.get("specific_genre", "")
    fname = folder_name.lower()
    
    # Combined string for search
    s = (str(raw) + " " + str(spec) + " " + fname).lower()
    
    if "historical" in s: return "Historical Fiction"
    if "romance" in s: return "Romance"
    if "cyberpunk" in s or "sci-fi" in s or "fantasy" in s or "space" in s: return "Sci-Fi/Fantasy"
    if "science" in s: return "Science"
    if "philosophy" in s: return "Philosophy"
    if "reference" in s or "technical" in s: return "Reference"
    if "drama" in s: return "Poetry & Drama"
    if "thriller" in s or "mystery" in s: return "Mystery/Crime"
    if "adventure" in s: return "Adventure"
    
    if "biography" in s: return "Biography"
    
    return "Unknown"

def select_targets_from_list(valid_chunks, folder_name, genre, title, model_name, temp_level="Unknown"):
    """
    Select points corresponding to fixed targets [100k, 200k, 400k, 600k] + Max Length.
    """
    if not valid_chunks: return []
    
    # Filter chunks with known token count
    candidates = [c for c in valid_chunks if c.get('n_tokens') is not None]
    if not candidates: return []
    
    fixed_targets = [100000, 200000, 400000, 600000]
    
    # Find max chunk (Full Length)
    max_chunk = max(candidates, key=lambda x: x['n_tokens'])
    max_tokens = max_chunk['n_tokens']
    
    # Targets: Fixed + Max. deduplicated and sorted.
    # Only keep fixed targets that are <= max_tokens (to avoid looking for 600k in a 200k book)
    # But wait, if book is 500k, we want 100, 200, 400, 500.
    targets = sorted(list(set([t for t in fixed_targets if t < max_tokens] + [max_tokens])))
    
    selected_metrics_list = []
    selected_ids = set()

    for target in targets:
        # Find closest chunk
        closest = min(candidates, key=lambda x: abs(x['n_tokens'] - target))
        
        # Add if not already added
        c_id = closest.get('chunk_id') # chunk_id might be None for reference books if not present
        if c_id is not None:
             if c_id in selected_ids: continue
             selected_ids.add(c_id)
        else:
             # For reference books without chunk_ids, use n_tokens as identity proxy
             # But usually list is sorted by tokens.
             pass

        metrics = closest['metrics'].copy()
        
        # Ensure metadata is set
        metrics["file"] = folder_name
        metrics["genre"] = genre
        metrics["model"] = model_name
        metrics["title"] = title
        metrics["temperature_level"] = temp_level
        # Ensure chunk_id or similar is set?
        if c_id is not None: metrics["chunk_id"] = c_id
        
        selected_metrics_list.append(metrics)
        
    return selected_metrics_list

def gather_metrics(base_path, out_file):
    base = Path(base_path)
    all_metrics = []
    
    # Filter folders not starting with _ (except _chaos)
    folders = [p for p in base.glob("*") 
               if p.is_dir() and (not p.name.startswith("_") or p.name.startswith("_chaos"))
               and list(p.glob("chunk_*.txt"))]
    
    # Sort folders by name (timestamp)
    folders.sort(key=lambda p: p.name)

    print(f"Found {len(folders)} valid run folders.")

    for folder in folders:
        print(f"Processing {folder.name}...")
        try:
            attempts = load_attempts(folder)
            if not attempts:
                print(f"  No attempts found.")
                continue

            # Sort chunks
            def get_chunk_num(k):
                if isinstance(k, int): return k
                if isinstance(k, str) and '_' in k: return int(k.split('_')[1])
                try: return int(k)
                except: return 0

            sorted_chunks = sorted(attempts.keys(), key=get_chunk_num)
            
            # Load metadata
            meta = {}
            meta_path = folder / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as mf: meta = json.load(mf)
                except: pass
            
            # Check prompt validity
            prompt_file = meta.get("prompt_filename", "")
            valid_prompts = ["historical_library", "romance_drama", "cyberpunk_grid", 
                             "science_book", "philosophy_book", "technical_reference", "biography", "lighthouse_keeper", "Manual Entry"]
            
            is_valid_prompt = any(k in prompt_file for k in valid_prompts)
            is_chaos = folder.name.startswith("_chaos")
            
            if not is_valid_prompt and not is_chaos:
                print(f"  Skipping {folder.name} (invalid prompt: {prompt_file})")
                continue

            genre = map_genre(meta, folder.name)
            
            # Special Override: Check for "Unknown" in metadata specifically for chaos book
            if meta.get("specific_genre", "").lower() == "unknown" or meta.get("broader_genre", "").lower() == "unknown":
                genre = "Unknown"

            # Prepare candidates
            valid_chunks = []
            for chunk_id in sorted_chunks:
                attempt_id = find_accepted_attempt(chunk_id, attempts)
                if attempt_id is None: continue
                
                rec = attempts[chunk_id][attempt_id]
                metrics = rec.get("accumulated_metrics") or rec.get("chunk_metrics") or rec.get("metrics")
                if not metrics: continue
                
                n_tok = metrics.get('n_tokens') or metrics.get('tokens_count') or metrics.get('size')
                if n_tok is None: pass
                
                valid_chunks.append({
                    'chunk_id': chunk_id,
                    'n_tokens': n_tok,
                    'metrics': metrics
                })
            
            if not valid_chunks:
                 print(f"  No valid metrics for {folder.name}")
                 continue

            # Use Shared Helper
            model_name = "Mistral"
            if "model_config" in meta and "name" in meta["model_config"]:
                mn = meta["model_config"]["name"].lower()
                if "mistral" in mn: model_name = "Mistral"
                elif "qwen" in mn: model_name = "Qwen"
                elif "llama" in mn: model_name = "Llama"
                else: model_name = meta["model_config"]["name"]
            
            temp_level = "Unknown"
            temp_range = meta.get("temperature_range")
            if not temp_range and "model_config" in meta:
                temp_range = meta["model_config"].get("temperature_range")
                
            if temp_range and len(temp_range) > 0:
                if temp_range[0] < 1.0:
                    temp_level = "Low"
                else:
                    temp_level = "High"
            else:
                fname = folder.name.lower()
                if "low" in fname: 
                    temp_level = "Low"
                else:
                    temp_level = "High"
            
            selected = select_targets_from_list(valid_chunks, folder.name, genre, meta.get("title", folder.name), model_name, temp_level)

            # Filter incomplete books
            # Must have at least the relevant fixed targets (or be long enough to matter)
            # Relaxed to 2 points to ensure standard/short books are included if they have SOME trajectory.
            if len(selected) < 2:
                print(f"  Skipping {folder.name} (only {len(selected)} targets matched - short/unfinished)")
                continue

            all_metrics.extend(selected)
            print(f"  Selected {len(selected)} chunks for projection")
            
        except Exception as e:
            print(f"  Error processing {folder.name}: {e}")

    # ---------------------------------------------------------
    # Add Selected Natural Book Trajectories (Reference & Modern)
    # ---------------------------------------------------------
    base_dir = Path(__file__).parent
    
    nat_dirs = [
        base_dir / "natural_texts" / "Selected_Books",
        base_dir / "natural_texts" / "Selected_Modern_Books"
    ]
    
    for nat_dir in nat_dirs:
        if nat_dir.exists():
            print(f"\nScanning {nat_dir} for selected natural book trajectories...")
            for nat_file in nat_dir.glob("metrics_*.json"):
                try:
                    with open(nat_file) as nf:
                        nat_data = json.load(nf)

                    # Filter: Ignore books < 200k tokens
                    max_tokens = max((d.get("n_tokens", 0) for d in nat_data), default=0)
                    if max_tokens < 200000:
                        continue
                    
                    # Refine Metadata and Genre FIRST
                    # We need to pass genre to selector, usually first point has it
                    first_pt = nat_data[0]
                    g = first_pt.get("genre", "Unknown").replace("Natural ", "")
                    # Fix mappings to match Experimental/PCA categories
                    if "Sci-Fi Precursor" in g: g = "Sci-Fi/Fantasy"
                    if "Mystery" in g: g = "Mystery/Crime"
                    
                    # Use internal file/title if available to ensure matching works
                    # (e.g. "2600_War_and_Peace" vs "metrics_WarPeace")
                    internal_file = first_pt.get('file', nat_file.stem)
                    internal_title = first_pt.get('title', nat_file.stem)

                    # Prepare chunks format for helper
                    # nat_data is a list of metrics dicts
                    valid_chunks = []
                    for i, pt in enumerate(nat_data):
                        # Ensure n_tokens
                        if 'n_tokens' not in pt: continue
                        valid_chunks.append({
                            'chunk_id': i, # Synthetic ID
                            'n_tokens': pt['n_tokens'],
                            'metrics': pt
                        })
                    
                    # Use Shared Helper
                    selected = select_targets_from_list(valid_chunks, internal_file, g, internal_title, "Human", "Human")
                    
                    if selected:
                        all_metrics.extend(selected)
                        print(f"  Added {len(selected)} points from {nat_file.name}")
                        
                except Exception as e:
                    print(f"  Error loading {nat_file.name}: {e}")

    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Saved {len(all_metrics)} book metrics to {out_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    gather_metrics(base_dir / "generated_texts_og" / "prompts", 
                   base_dir / "generated_texts_og" / "experimental_metrics.json")
