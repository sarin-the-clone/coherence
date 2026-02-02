# Story Contract Integration - Code Changes for longbook_engine.py

## Changes Needed

### 1. Add Import (top of file, around line 34)
```python
from generated_texts_og.utils.story_contract_parser import (
    parse_initial_prompt,
    format_contract_for_prompt
)
from generated_texts_og.utils.prompt_templates import (
    CHUNK_GENERATION_PROMPT,
    get_variation_hint
)
```

### 2. Update generate_book Method (around line 101-135)

Replace:
```python
def generate_book(self, initial_prompt, n_chunks):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desc = initial_prompt.strip()
    
    # ... metadata setup ...
    
    # Reset memories
    self.local_context.reset(desc)
    self.full_text = desc + "\n\n"
    final_text = desc + "\n\n"
```

With:
```python
def generate_book(self, initial_prompt, n_chunks):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Parse initial prompt into narrative seed and story contract
    narrative_seed, story_contract = parse_initial_prompt(initial_prompt.strip())
    
    # Format contract for compact injection
    self.story_contract_compact = format_contract_for_prompt(story_contract, compact=True)
    
    # Phase 1: Create timestamp-only folder (no title yet)
    self.serializer.init_temp_book_folder(timestamp)
    
    # Save initial metadata (include both parts)
    metadata = {
        "timestamp": timestamp,
        "model_config": self.model_config,
        "initial_prompt_full": initial_prompt.strip(),  # Keep full for reference
        "narrative_seed": narrative_seed,
        "story_contract": story_contract,
        "story_contract_compact": self.story_contract_compact,
        # ... rest of metadata ...
    }
    self.serializer.save_metadata(metadata)
    
    # Reset memories with ONLY narrative seed (not full prompt)
    self.local_context.reset(narrative_seed)
    self.full_text = narrative_seed + "\n\n"
    final_text = narrative_seed + "\n\n"
```

### 3. Update _generate_one_chunk Method (around line 225)

Replace:
```python
base_prompt = CHUNK_GENERATION_PROMPT.format(
    memory_summary=memory_summary,
    previous_text=prev_text
)
```

With:
```python
base_prompt = CHUNK_GENERATION_PROMPT.format(
    story_contract=self.story_contract_compact,  # ADD THIS
    memory_summary=memory_summary,
    previous_text=prev_text,
    variation_hint=get_variation_hint(chunk_id)  # ADD THIS
)
```

## Summary of Changes

**What this does**:
1. Splits initial prompt into narrative (summarized) and contract (persistent)
2. Stores contract in metadata for reference
3. Injects compact contract into EVERY chunk prompt
4. Adds variation hints to prevent repetition

**Result**:
- ✅ POV, tone, guardrails persist across all chunks
- ✅ No redundancy in memory system (only narrative gets summarized)
- ✅ Better coherence and consistency
- ✅ Variation hints prevent beta from dropping
