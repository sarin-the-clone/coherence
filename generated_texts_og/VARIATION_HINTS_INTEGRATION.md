# Compact Variation Hints - Integration Guide

## Quick Integration (Minimal Changes)

### 1. Update Import in longbook_engine.py

```python
# Line 34 - Update this import
from generated_texts_og.utils.prompt_templates import (
    CHUNK_GENERATION_PROMPT,
    get_variation_hint  # ADD THIS
)
```

### 2. Modify _generate_one_chunk Method

In `longbook_engine.py`, around line 225, update the prompt building:

```python
# BEFORE (line 225-228):
base_prompt = CHUNK_GENERATION_PROMPT.format(
    memory_summary=memory_summary,
    previous_text=prev_text
)

# AFTER:
base_prompt = CHUNK_GENERATION_PROMPT.format(
    memory_summary=memory_summary,
    previous_text=prev_text,
    variation_hint=get_variation_hint(chunk_id)  # ADD THIS LINE
)
```

That's it! Only 2 small changes.

## How It Works

- Every chunk gets a different 1-line hint (or no hint every 7th chunk)
- Hints rotate through 7 variations:
  1. Change setting/scene type
  2. Adjust pacing
  3. Reference earlier events
  4. Vary sentence rhythm
  5. Shift emotional tone
  6. Add new obstacle
  7. (No hint - natural flow)

- **Prompt overhead**: Only ~10-15 tokens per chunk
- **No memory bloat**: Hints don't accumulate in memory system

## Expected Impact

- **Beta**: Should show more variation, less likely to settle at 0.05
- **Autocorrelation**: Better long-range structure from callbacks and pacing shifts
- **Readability**: More dynamic, less repetitive prose

## Optional: Adaptive Hints Based on Metrics

If you want to make hints responsive to current beta:

```python
def get_adaptive_variation_hint(chunk_id, current_beta, target_beta=0.25):
    \"\"\"Choose hint based on current coherence metrics.\"\"\"
    
    if current_beta < 0.15:  # Too predictable
        # Force more variety
        return VARIATION_HINTS[0]  # Change setting
    elif current_beta > 0.35:  # Too chaotic
        # Encourage continuity
        return VARIATION_HINTS[2]  # Reference earlier events
    else:
        # Normal rotation
        return get_variation_hint(chunk_id)
```

Then use in longbook_engine.py:

```python
# After computing accumulated_metrics (line 285)
current_beta = accumulated_metrics.get('beta', 0.25) if accumulated_metrics else 0.25

base_prompt = CHUNK_GENERATION_PROMPT.format(
    memory_summary=memory_summary,
    previous_text=prev_text,
    variation_hint=get_adaptive_variation_hint(chunk_id, current_beta)
)
```

## Testing

Run a generation and check debug outputs:
```bash
# Look for variation hints in prompt files
grep "NEW:" generated_texts_og/debug/*/chunk_*_prompt.txt

# Monitor beta evolution
python3 analysis/metrics_on_generated_texts.py
```
