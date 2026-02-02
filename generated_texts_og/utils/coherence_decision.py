# ================================================================
# coherence_decision.py (v3)
#
# Decides whether an LLM-produced chunk is acceptable.
# Uses a mixture of:
#   - spectral beta, alpha
#   - H (DFA1 / DFA2 / DFA3 / DMA / WT Haar / WT db4)
#   - entropy
#   - LZ complexity
#   - MI decay rate gamma
#   - repetition checks
#   - minimal length check
#
# Returns: True / False
# ================================================================

import os
import sys
import re

import json

PARENT = os.path.dirname(os.path.dirname(__file__))  # coherence/
if PARENT not in sys.path:
    sys.path.append(PARENT)


# ---------------------------------------------------------------
# SOFT DECISION GATE CONFIG
# Updated based on distribution analysis (percentiles for skewed metrics)
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# SOFT DECISION GATE CONFIG
# Updated based on distribution analysis (percentiles for skewed metrics)
# ---------------------------------------------------------------

def load_adaptive_config():
    """Load adaptive thresholds from JSON config if available."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'adaptive_thresholds.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                # print(f"Loading adaptive thresholds from {config_path}")
                return json.load(f)
        except Exception as e:
            print(f"Error loading adaptive thresholds: {e}")
    
    print("Using hardcoded default thresholds (fallback)")
    return {
        "beta_min": 0.0866, "beta_max": 0.4867,
        "alpha_min": 0.0224, "alpha_max": 0.2153,
        "H_DMA_min": 0.5350, "H_DMA_max": 0.6810,
        "entropy_min": 8.3557, "entropy_max": 9.5585,
        "ppl_min": 309.70, "ppl_max": 755.93,
        "lz_min": 4.4294, "lz_max": 6.1216,
        "gamma_min": 0.0045, "gamma_max": 0.0353,
        "H_min": 0.5799, "H_max": 0.7031,
        "H_WT_db4_min": 0.5242, "H_WT_db4_max": 0.6732,
        "rep_penalty_threshold": 6,
        "gating_start_chunk": 10,
    }

_DEFAULT_CFG = load_adaptive_config()

def load_binned_config():
    """Load size-binned metrics for dynamic gating."""
    path = os.path.join(PARENT, '..', 'natural_texts', 'Output', 'analysis', 'size_binned_metrics.json')
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading binned metrics: {e}")
            return None
    return None

_BINNED_CFG = load_binned_config()

# Metrics actively used in gating decisions
GATED_METRICS = {"beta", "alpha", "H_DMA", "entropy_bits", "lz_complexity", "mi_decay_gamma", "perplexity"}


def get_dynamic_thresholds(n_tokens):
    """
    Get thresholds based on current token count using size bins (p2.5 - p97.5).
    """
    if _BINNED_CFG is None:
        return _DEFAULT_CFG.copy()

    # Determine Bin
    # 0: < 150k (Proxy: 100-150k bin)
    # 1: 150k - 300k
    # 2: 300k - 500k
    # 3: 500k - 800k
    # 4: > 800k
    # Determine which bin applies
    if n_tokens < 150000:
        bin_idx = 0  # 100-150k
    elif n_tokens < 300000:
        bin_idx = 1  # 150-300k
    elif n_tokens < 500000:
        bin_idx = 2  # 300-500k
    elif n_tokens < 800000:
        bin_idx = 3  # 500-800k
    else:
        bin_idx = 4  # >800k
        
    bin_data = _BINNED_CFG["bins"][bin_idx]["metrics"]
    
    # Map metric names to config keys
    mapping = {
        "beta": ("beta_min", "beta_max"),
        "alpha": ("alpha_min", "alpha_max"),
        "H_DMA": ("H_DMA_min", "H_DMA_max"),
        "entropy_bits": ("entropy_min", "entropy_max"),
        "perplexity": ("ppl_min", "ppl_max"),
        "lz_complexity": ("lz_min", "lz_max"),
        "mi_decay_gamma": ("gamma_min", "gamma_max"),
        "H_DFA1": ("H_min", "H_max"),
        "H_WT_db4": ("H_WT_db4_min", "H_WT_db4_max")
    }
    
    dyn = _DEFAULT_CFG.copy()
    
    for metric, keys in mapping.items():
        if metric in bin_data:
            stats = bin_data[metric]
            min_key, max_key = keys
            # Use p2.5 and p97.5 for wide checks
            dyn[min_key] = stats.get("p2_5", dyn.get(min_key))
            dyn[max_key] = stats.get("p97_5", dyn.get(max_key))
            # Also store std/mean for distance calc
            dyn[f"{metric}_mean"] = stats.get("mean", 0)
            dyn[f"{metric}_std"] = stats.get("std", 1)
            
    return dyn


# ---------------------------------------------------------------
def _repetition_detect(text, threshold):
    """
    Detect repeated n-gram patterns in the continuation.
    If too repetitive, disqualify.
    """
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < threshold * 2:
        return False

    # sliding window
    for i in range(len(tokens) - threshold * 2):
        if tokens[i : i + threshold] == tokens[i + threshold : i + threshold * 2]:
            return True
    return False

def check_hallucination_markers(text):
    """Check for common model failure patterns/hallucinations."""
    markers = ["(Story Cont", "(Story cont", "***", "Story Cont.", "Assistant:", "ASSISTANT:"]
    for m in markers:
        if m in text:
            # print(f"  ❌ Hallucination marker detected: '{m}'")
            return True
    return False

def check_paragraph_duplication(text):
    """Check if any substantial paragraph is repeated exactly."""
    # Split by newlines, filter short lines
    paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 100]
    seen = set()
    for p in paras:
        # Check existence
        if p in seen:
            # print(f"  ❌ Paragraph duplication detected: '{p[:30]}...'")
            return True
        seen.add(p)
    return False

def check_start_repetition(new_text, prev_text, n_chars=50):
    """
    Check if the new chunk starts with the same sequence as the previous chunk.
    Also checks for repetitive opening patterns like "As the..."
    """
    if not prev_text or len(prev_text) < n_chars or len(new_text) < n_chars:
        return False
    
    # Check for stuttering (repeating the end of previous context)
    if prev_text.strip().endswith(new_text[:n_chars].strip()):
        return True
    
    # Check if new text starts with the same first 3 words as previous chunk
    new_words = new_text.strip().split()[:3]
    prev_words = prev_text.strip().split()[:3]
    
    # If generic 3-word overlap (e.g. "The sun rose") AND longer text matches, bad.
    # But just 3 words can be valid. Let's stick to stutter check + internal checks.
    
    return False

def resolve_effective_config(metrics, cfg=None):
    """
    Resolve the final configuration by combining dynamic thresholds (based on size)
    and explicit overrides (from CLI/cfg).
    """
    # 1. Start with Dynamic Thresholds based on size
    n_tokens = metrics.get("n_tokens", 0)
    effective_cfg = get_dynamic_thresholds(n_tokens)
    
    # 2. Apply explicit CLI overrides from cfg
    # We only respect overrides for Entropy and PPL if they differ from static defaults
    # (Because the generator initializes heavily from static defaults)
    if cfg:
        # Check entropy override
        if cfg.get("entropy_min") != _DEFAULT_CFG.get("entropy_min"):
             effective_cfg["entropy_min"] = cfg["entropy_min"]
        
        # Check PPL override
        if cfg.get("ppl_min") != _DEFAULT_CFG.get("ppl_min"):
             effective_cfg["ppl_min"] = cfg["ppl_min"]
             
        # Copy other non-dynamic settings
        effective_cfg["gating_start_chunk"] = cfg.get("gating_start_chunk", 50)
        effective_cfg["rep_penalty_threshold"] = cfg.get("rep_penalty_threshold", 6)
        effective_cfg["min_chars"] = cfg.get("min_chars", 400)
        
    return effective_cfg


# ---------------------------------------------------------------
def is_chunk_acceptable(metrics, cfg=None, text=None, chunk_id=None, disable_gating=False):
    """
    Decide whether this chunk is acceptable.
    metrics: dict from CoherenceMetrics
    cfg: override thresholds (from generator)
    text: full candidate text
    chunk_id: current chunk index
    disable_gating: If True, skip metric checks (pass if sanity checks pass).
    """
    cfg = resolve_effective_config(metrics, cfg)

    # 1. length gate
    if text is not None:
        if len(text) < cfg["min_chars"]:
            return False
            
        # Reject hallucinations / loops
        if check_hallucination_markers(text):
            print("  ❌ Rejected: Hallucination markers detected.")
            return False

        if check_paragraph_duplication(text):
            print("  ❌ Rejected: Paragraph duplication detected.")
            return False

    # Decide on "late-stage" gates based on total token count
    # We defer strict gating until ~100k tokens because our "Bin 0" statistics
    # are derived from books with 100k-150k tokens. Books < 100k are too volatile.
    start_tokens_threshold = 100000
    
    is_late_stage = False
    n_tokens = metrics.get("n_tokens", 0)
    
    if n_tokens > start_tokens_threshold:
        is_late_stage = True
    elif chunk_id is not None:
        # Fallback if n_tokens is missing/zero but we have chunk index
        # Heuristic: 50 chunks * 2000 avg tokens = 100k
        start_chunk = cfg.get("gating_start_chunk", 50)
        if chunk_id > start_chunk:
             is_late_stage = True

    # 1.5 Disable Gating Check
    if disable_gating:
        # If disabled, we ONLY care about the sanity checks above (length, hallucination, duplication, stutter)
        # If we reached here, sanity checks passed.
        # We also need to check simple repetition (below) but metric gates are skipped.
        pass # Fall through to repetition checks, skipping metric blocks
    else:
        # Helper to check if we should gate a metric
        def check_gate(metric_key, val, min_key, max_key, condition=True):
            if not condition:
                return True
            if metric_key not in GATED_METRICS:
                return True
            low = cfg.get(min_key, -float('inf'))
            high = cfg.get(max_key, float('inf'))
            if low is None: low = -float('inf')
            if high is None: high = float('inf')
            return low <= val <= high

        # --- METRICS (Gated only after start_chunk) ---
        
        # 2. Beta (Spectral Exponent)
        if "beta" in metrics:
            if not check_gate("beta", metrics["beta"], "beta_min", "beta_max", condition=is_late_stage):
                return False
                
        # 3. Alpha (Spectral Slope)
        if "alpha" in metrics:
            if not check_gate("alpha", metrics["alpha"], "alpha_min", "alpha_max", condition=is_late_stage):
                return False

        # 4. Entropy
        if "entropy_bits" in metrics:
            if not check_gate("entropy_bits", metrics["entropy_bits"], "entropy_min", "entropy_max"):
                return False

        # 5. Perplexity
        if "perplexity" in metrics and metrics["perplexity"] is not None:
            if not check_gate("perplexity", metrics["perplexity"], "ppl_min", "ppl_max"):
                return False

        # --- CONDITIONALLY GATED (Extensive / Slow Convergence) ---
        
        # 6. Hurst (H_DMA)
        if "H_DMA" in metrics:
            if not check_gate("H_DMA", metrics["H_DMA"], "H_DMA_min", "H_DMA_max", condition=is_late_stage):
                return False

        # 7. Lz Complexity (Second-order growth)
        if "lz_complexity" in metrics:
            if not check_gate("lz_complexity", metrics["lz_complexity"], "lz_min", "lz_max", condition=is_late_stage):
                return False

        # 8. Gamma (MI Decay - needs long context)
        if "mi_decay_gamma" in metrics:
            if not check_gate("mi_decay_gamma", metrics["mi_decay_gamma"], "gamma_min", "gamma_max", condition=is_late_stage):
                return False


    # 7. repetition (n-gram)
    if text is not None:
        if _repetition_detect(text, cfg["rep_penalty_threshold"]):
            return False

    # 8. Start-of-chunk repetition check
    # Prevent chunks from starting exactly the same way as the previous context end
    # (This requires passing the previous context, but for now we can check if the 
    # start of the text looks like a generic "As the sun..." if it appears too often)
    # A better check is to ensure the chunk doesn't start with the same N chars as the previous chunk's start.
    # Since we don't have previous chunk here, we rely on the caller or just stricter n-gram.
    
    return True

def calculate_metric_distance(metrics, cfg=None, chunk_id=None, verbose=False, text=None):
    """
    Calculate a distance score for the metrics relative to the thresholds.
    Lower is better. 0.0 means all metrics are within bounds.
    Used for selecting the 'least bad' attempt when all fail.
    """
    # Resolve dynamic config
    cfg = resolve_effective_config(metrics, cfg)
    
    total_distance = 0.0
    
    # Decide on late-stage gates based on total token count
    start_tokens_threshold = 100000
    n_tokens = metrics.get("n_tokens", 0)
    
    is_late_stage = False
    if n_tokens > start_tokens_threshold:
        is_late_stage = True
    # Removed fallback 'elif chunk_id > 50' because chunks can be small (<1000 tokens), 
    # triggering gating prematurely at 40k tokens. 
    # We now trust n_tokens explicitly.

    def add_dist(metric_key, val, min_key, max_key, mean_key=None, std_key=None, condition=True):
        if not condition:
            return 0.0
        if metric_key not in GATED_METRICS:
            return 0.0
            
        low = cfg.get(min_key, -float('inf'))
        high = cfg.get(max_key, float('inf'))
        # Try to get std if available for robustness
        std = cfg.get(std_key, 1.0) if std_key else 1.0
        
        if low is None: low = -float('inf')
        if high is None: high = float('inf')
        
        dist = 0.0
        if val < low:
            raw_dist = low - val
            # Preferred: Z-score distance (how many sigmas away)
            if std and std > 0:
                dist = raw_dist / std
            else:
                # Fallback: Range normalization
                if high != float('inf') and low != -float('inf'):
                     width = high - low
                     dist = raw_dist / width if width > 0 else raw_dist
                else:
                    dist = raw_dist

        elif val > high:
            raw_dist = val - high
            if std and std > 0:
                dist = raw_dist / std
            else:
                 if high != float('inf') and low != -float('inf'):
                     width = high - low
                     dist = raw_dist / width if width > 0 else raw_dist
                 else:
                    dist = raw_dist
                    
        return dist

    
    
    # Debug print to diagnose early gating (Controlled verbosity)
    if verbose:
        print(f"DEBUG: Chunk {chunk_id}, Tokens: {n_tokens}, FullGate: {is_late_stage} (Threshold: {start_tokens_threshold})")

    if "beta" in metrics:
        total_distance += add_dist("beta", metrics["beta"], "beta_min", "beta_max", std_key="beta_std", condition=is_late_stage)
    
    if "alpha" in metrics:
        total_distance += add_dist("alpha", metrics["alpha"], "alpha_min", "alpha_max", std_key="alpha_std", condition=is_late_stage)
        
    if "entropy_bits" in metrics:
        # Informational metrics are valid early
        total_distance += add_dist("entropy_bits", metrics["entropy_bits"], "entropy_min", "entropy_max", std_key="entropy_bits_std")
        
    if "perplexity" in metrics and metrics["perplexity"] is not None:
        # Informational metrics are valid early
        total_distance += add_dist("perplexity", metrics["perplexity"], "ppl_min", "ppl_max", std_key=None)

    # --- PENALIZE REPETITION ---
    if text is not None:
        # If text has n-gram repetition, add severe penalty
        if _repetition_detect(text, cfg.get("rep_penalty_threshold", 0.5)):
             # Add a large penalty to make it worse than "slightly off metric" attempts
             total_distance += 100.0
        
    if "H_DMA" in metrics:
        total_distance += add_dist("H_DMA", metrics["H_DMA"], "H_DMA_min", "H_DMA_max", std_key="H_DMA_std", condition=is_late_stage)
        
    if "lz_complexity" in metrics:
        total_distance += add_dist("lz_complexity", metrics["lz_complexity"], "lz_min", "lz_max", std_key="lz_complexity_std", condition=is_late_stage)
        
    if "mi_decay_gamma" in metrics:
        total_distance += add_dist("mi_decay_gamma", metrics["mi_decay_gamma"], "gamma_min", "gamma_max", std_key="mi_decay_gamma_std", condition=is_late_stage)
        
    return total_distance
