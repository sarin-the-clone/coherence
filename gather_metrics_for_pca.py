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

def gather_metrics(base_path, out_file):
    base = Path(base_path)
    all_metrics = []
    
    # Filter folders not starting with _
    folders = [p for p in base.glob("*") 
               if p.is_dir() and not p.name.startswith("_") 
               and list(p.glob("chunk_*.txt"))]
    
    # Sort folders by name (timestamp) to ensure deterministic replicate ordering
    folders.sort(key=lambda p: p.name)

    print(f"Found {len(folders)} valid run folders.")

    for folder in folders:
        print(f"Processing {folder.name}...")
        try:
            attempts = load_attempts(folder)
            if not attempts:
                print(f"  No attempts found.")
                continue

            # Sort chunks to process in order
            def get_chunk_num(k):
                if isinstance(k, int): return k
                if isinstance(k, str) and '_' in k: return int(k.split('_')[1])
                try: return int(k)
                except: return 0

            sorted_chunks = sorted(attempts.keys(), key=get_chunk_num)
            
            # Load metadata once
            meta = {}
            meta_path = folder / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as mf:
                        meta = json.load(mf)
                except: pass
            
            # Filter by Valid Prompts (Experimental Design enforcement)
            prompt_file = meta.get("prompt_filename", "")
            valid_prompts = ["historical_library", "romance_drama", "cyberpunk_grid", 
                             "science_book", "philosophy_book", "technical_reference", "biography"]
            if not any(k in prompt_file for k in valid_prompts):
                print(f"  Skipping {folder.name} (invalid prompt: {prompt_file})")
                continue

            genre = map_genre(meta, folder.name)
            
            # Collect all valid chunks and their token counts
            valid_chunks = []
            for chunk_id in sorted_chunks:
                attempt_id = find_accepted_attempt(chunk_id, attempts)
                if attempt_id is None: continue
                
                rec = attempts[chunk_id][attempt_id]
                metrics = rec.get("accumulated_metrics") or rec.get("chunk_metrics") or rec.get("metrics")
                if not metrics: continue
                
                # Try to get token count
                n_tok = metrics.get('n_tokens') or metrics.get('tokens_count') or metrics.get('size')
                if n_tok is None:
                     # Try to parse from chunk_id if unavailable? 
                     # Usually accumulated metrics has it.
                     pass
                
                valid_chunks.append({
                    'chunk_id': chunk_id,
                    'n_tokens': n_tok,
                    'metrics': metrics
                })
            
            if not valid_chunks:
                 print(f"  No valid metrics for {folder.name}")
                 continue

            # Select chunks closest to target sizes
            targets = [100000, 200000, 400000, 600000, 1000000]
            selected_chunks = []
            selected_ids = set()

            for target in targets:
                # Find closest chunk
                # Filter chunks with known token count
                candidates = [c for c in valid_chunks if c['n_tokens'] is not None]
                if not candidates: continue
                
                closest = min(candidates, key=lambda x: abs(x['n_tokens'] - target))
                
                # Add if not already added (ensure distinct chunks per target unless stuck)
                if closest['chunk_id'] not in selected_ids:
                    selected_ids.add(closest['chunk_id'])
                    
                    metrics = closest['metrics'].copy()
                    metrics["file"] = folder.name
                    metrics["genre"] = genre
                    # Extract model name from metadata or default to Mistral
                    model_name = "Mistral"
                    if "model_config" in meta and "name" in meta["model_config"]:
                        mn = meta["model_config"]["name"].lower()
                        if "mistral" in mn: model_name = "Mistral"
                        elif "qwen" in mn: model_name = "Qwen"
                        elif "llama" in mn: model_name = "Llama"
                        else: model_name = meta["model_config"]["name"]
                    
                    metrics["model"] = model_name
                    metrics["chunk_id"] = closest['chunk_id']
                    metrics["title"] = meta.get("title", folder.name)
                    
                    selected_chunks.append((closest, metrics))
            
            # Filter incomplete books (must have all 5 target points)
            if len(selected_chunks) < len(targets):
                print(f"  Skipping {folder.name} (only {len(selected_chunks)}/{len(targets)} targets matched - unfinished)")
                continue

            for closest, metrics in selected_chunks:
                 all_metrics.append(metrics)
            
            print(f"  Selected {len(selected_chunks)} chunks for projection (Targets: {targets})")
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
                    
                    # Refine Metadata
                    refined_data = []
                    for pt in nat_data:
                        # Clean Genre
                        g = pt.get("genre", "Unknown")
                        g = g.replace("Natural ", "")
                        # Fix mappings to match Experimental/PCA categories
                        if "Sci-Fi Precursor" in g: g = "Sci-Fi/Fantasy"
                        if "Mystery" in g: g = "Mystery/Crime"
                        pt["genre"] = g
                        
                        # Clean File Name (for labeling)
                        f_name = pt.get("file", "")
                        if "Moby" in f_name: pt["file"] = "MD"
                        elif "Pride" in f_name: pt["file"] = "PP"
                        elif "Frankenstein" in f_name: pt["file"] = "Fr"
                        elif "Sherlock" in f_name: pt["file"] = "SH"
                        elif "Origin" in f_name: pt["file"] = "OS"
                        
                        # Enforce Human model type
                        pt["model"] = "Human"
                        
                        refined_data.append(pt)

                    all_metrics.extend(refined_data)
                    print(f"  Added {len(refined_data)} points from {nat_file.name}")
                except Exception as e:
                    print(f"  Error loading {nat_file.name}: {e}")

    with open(out_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"Saved {len(all_metrics)} book metrics to {out_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    gather_metrics(base_dir / "generated_texts_og" / "prompts", 
                   base_dir / "generated_texts_og" / "experimental_metrics.json")
