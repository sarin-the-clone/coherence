# Prompt Sanitization Integration Guide

## Purpose

Add LLM-based sanitization to catch and fix:
- **Contamination**: JSON, code, meta-commentary leaking through
- **POV/tense errors**: Inconsistencies with story contract
- **Repetition**: Stuttering or duplicated text
- **Continuity errors**: Obvious mistakes

## Integration into longbook_engine.py

### Option 1: Sanitize All Chunks (Recommended)

Add after chunk generation, before acceptance check:

```python
# In longbook_engine.py, around line 260

from generated_texts_og.utils.prompt_sanitizer import sanitize_chunk, quick_contamination_check

# After generating chunk
new_text = extract_new_text(...)

# Quick check for obvious contamination
if quick_contamination_check(new_text):
    print(f"  Contamination detected, sanitizing...")
    new_text = sanitize_chunk(new_text, self.story_contract_compact, self.model)

# Then proceed with metrics and acceptance
toks = Tokenizer(kind="gpt4").encode(new_text)
```

### Option 2: Sanitize Only Failed Chunks

Only sanitize if chunk fails acceptance:

```python
# After all retries fail (line 323)

# Before fallback
if not is_chunk_acceptable(last_metrics, text=last_output):
    print(f"  All attempts failed, trying sanitization...")
    sanitized = sanitize_chunk(last_output, self.story_contract_compact, self.model)
    
    # Re-check metrics on sanitized version
    sanitized_toks = Tokenizer(kind="gpt4").encode(sanitized)
    sanitized_metrics = CoherenceMetrics(sanitized_toks).compute_all()
    
    if is_chunk_acceptable(sanitized_metrics, text=sanitized):
        print(f"  Sanitization successful!")
        self._update_memory(sanitized, sanitized_metrics, chunk_id)
        return sanitized

# Fallback to last attempt
self._update_memory(last_output, {}, chunk_id)
return last_output
```

### Option 3: Selective Sanitization

Only sanitize if specific issues detected:

```python
# After generating chunk
new_text = extract_new_text(...)

# Check for specific issues
needs_sanitization = (
    quick_contamination_check(new_text) or
    len(new_text) > 50000 or  # Suspiciously long
    new_text.count('\n\n') < 3  # No paragraph breaks
)

if needs_sanitization:
    new_text = sanitize_chunk(new_text, self.story_contract_compact, self.model)
```

## Performance Considerations

**Cost**: Each sanitization = 1 extra LLM call per chunk
- For 21 chunks: 21 extra calls
- With 5 attempts: Could be 105 calls if all fail

**Recommendation**: Use Option 1 (sanitize all) but with quick check first:
```python
if quick_contamination_check(new_text):
    new_text = sanitize_chunk(...)  # Only ~10-20% of chunks
```

## Testing

Test sanitization on a known contaminated chunk:

```python
from generated_texts_og.utils.prompt_sanitizer import sanitize_chunk
from generated_texts_og.models.model_interface import ModelFactory

# Load model
model = ModelFactory.create_model(model_config)

# Test text with contamination
contaminated = '''
Here is the next part of the story:

```json
{"character": "Bob"}
```

Bob walked down the street. As an AI, I will continue...
'''

# Sanitize
clean = sanitize_chunk(contaminated, "POV: Third-person | Tense: Past", model)
print(clean)
# Should output: "Bob walked down the street."
```

## Monitoring

Add logging to track sanitization:

```python
# In longbook_engine.py
sanitization_count = 0

if quick_contamination_check(new_text):
    sanitization_count += 1
    print(f"  [Sanitization {sanitization_count}/{chunk_id}]")
    new_text = sanitize_chunk(...)

# At end of generation
print(f"\nTotal sanitizations: {sanitization_count}/{n_chunks}")
```
