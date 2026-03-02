import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_chaotic_metrics():
    # Evaluate root path relative to the script location
    script_dir = Path(__file__).resolve().parent
    # Assuming script is in utils/, so root is one level up
    root_path = script_dir.parent
    
    metrics_path = root_path / "generated_texts_og" / "Output" / "chaotic_run_metrics.json"
    
    if not metrics_path.exists():
        logging.error(f"Metrics file not found: {metrics_path}")
        return

    logging.info(f"Loading metrics from {metrics_path}")
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Identify uniquely corrupted chunks based on master metrics (like sudden perplexity drops)
    # The normal start-up sequence has low perplexity, but after 100k tokens, perplexity stabilizes around ~1200.
    # An anomalous chunk dropping to < 800 mid-run signifies a corruption (like chunk 3696276).
    corrupted_tokens_str = set()
    if "perplexity" in data:
        for k_str, val in data["perplexity"].items():
            tok = int(k_str)
            if tok > 100000 and val is not None and val < 800:
                corrupted_tokens_str.add(k_str)
                logging.info(f"[Master Check] Flagged token {tok} as corrupted (perplexity={val})")

    metrics_fixed = 0

    for metric_name, value_map in data.items():
        # Sort keys to ensure sequence order
        tokens = sorted([int(k) for k in value_map.keys()])
        keys_str = [str(k) for k in tokens]
        
        # Build array for easier manipulation
        vals_arr = np.array([value_map[k] for k in keys_str], dtype=np.float64)
        
        has_changes = False
        
        for i, k_str in enumerate(keys_str):
            if k_str in corrupted_tokens_str:
                if not np.isnan(vals_arr[i]):
                    vals_arr[i] = np.nan
                    metrics_fixed += 1
                    has_changes = True
                        
        if has_changes:
            # Interpolate the NaNs we just created
            nan_mask = np.isnan(vals_arr)
            not_nan_mask = ~nan_mask
            if np.any(nan_mask) and np.any(not_nan_mask):
                val_interp = np.interp(np.array(tokens)[nan_mask], np.array(tokens)[not_nan_mask], vals_arr[not_nan_mask])
                vals_arr[nan_mask] = val_interp
                
            # Update the original map
            for i, k in enumerate(keys_str):
                # We can save back interpolated floats. If they were valid originally, they remain unchanged.
                # Only the new NaNs get overwritten with the interpolated float. 
                # (Actual NaNs were skipped in our drop check, but they are also interpolated here. 
                # If we want to strictly preserve real NaNs, we'd need another mask).
                # Actually, standard real NaNs (from failures) being smoothed out is also beneficial.
                if np.isnan(vals_arr[i]):
                    value_map[k] = None # Fallback if interpolation couldn't fix it (e.g., boundaries)
                else:
                    value_map[k] = float(vals_arr[i])

    if metrics_fixed > 0:
        logging.info(f"Fixed {metrics_fixed} anomalous drops.")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logging.info("Saved corrected metrics.")
    else:
        logging.info("No massive drops detected for fixing.")

if __name__ == "__main__":
    clean_chaotic_metrics()
