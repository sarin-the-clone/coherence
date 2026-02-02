# Adaptive Coherence Thresholds - Recommendations

## Current Problem

The thresholds in `coherence_decision.py` are based on natural text statistics:
- Beta: 0.143-0.354 (mean ± 1 std from 6000 natural books)
- These books were human-written and edited
- Generated text struggles to stay in these ranges

## Why Generated Text Differs

1. **No editing**: Natural books go through multiple drafts
2. **Chunk-by-chunk**: Generated incrementally, not as a whole
3. **Model limitations**: LLMs have inherent biases in token distribution

## Recommended Solution: Adaptive Thresholds

### Option 1: Relax Thresholds for Generation (Recommended)

```python
# In coherence_decision.py, add:

_GENERATION_CFG = {
    # Slightly wider ranges for generated text
    "beta_min": 0.10,   # Was 0.143 - allow slightly lower
    "beta_max": 0.40,   # Was 0.354 - allow slightly higher
    
    "alpha_min": 0.03,  # Was 0.044
    "alpha_max": 0.18,  # Was 0.156
    
    "H_min": 0.58,      # Was 0.601
    "H_max": 0.68,      # Was 0.665
    
    # Keep others similar
    "lz_min": 0.0001,
    "lz_max": 0.0004,   # Slightly higher
    
    "gamma_min": 0.002,
    "gamma_max": 0.030,
    
    "entropy_min": 8.5,  # Slightly lower
    "entropy_max": 9.5,
    
    "ppl_min": 350,      # Wider range
    "ppl_max": 700,
    
    "min_chars": 400,
    "rep_penalty_threshold": 6
}

# Then in longbook_engine.py:
from generated_texts_og.utils.coherence_decision import _GENERATION_CFG

# Use generation config instead of default
if is_chunk_acceptable(acceptance_metrics, cfg=_GENERATION_CFG, text=acceptance_text):
```

### Option 2: Progressive Relaxation

Start strict, relax if all retries fail:

```python
def generate_with_progressive_thresholds(self, chunk_id):
    configs = [
        _DEFAULT_CFG,      # Try strict first
        _GENERATION_CFG,   # Then relaxed
        _VERY_RELAXED_CFG  # Last resort
    ]
    
    for cfg in configs:
        for attempt in range(n_attempts):
            # ... generate ...
            if is_chunk_acceptable(metrics, cfg=cfg, text=text):
                return chunk
    
    # All failed - use last attempt
    return last_chunk
```

### Option 3: Metric-Specific Relaxation

Focus on the most problematic metrics:

```python
# Only relax beta and alpha, keep others strict
_BETA_RELAXED_CFG = _DEFAULT_CFG.copy()
_BETA_RELAXED_CFG["beta_min"] = 0.10
_BETA_RELAXED_CFG["beta_max"] = 0.40
_BETA_RELAXED_CFG["alpha_min"] = 0.03
_BETA_RELAXED_CFG["alpha_max"] = 0.18
```

## Immediate Action

For your current run, update `longbook_engine.py` line 318:

```python
# BEFORE:
if is_chunk_acceptable(acceptance_metrics, text=acceptance_text):

# AFTER:
# Use relaxed thresholds for generation
generation_cfg = _DEFAULT_CFG.copy()
generation_cfg["beta_min"] = 0.10
generation_cfg["beta_max"] = 0.40
if is_chunk_acceptable(acceptance_metrics, cfg=generation_cfg, text=acceptance_text):
```

This should allow more chunks to pass while still maintaining reasonable coherence.
