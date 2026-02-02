# Long-Form Text Generation Coherence Issues & Solutions

## Problem Analysis

### 1. Beta Dropping to ~0.05 (Target: 0.143-0.354)

**Root Cause**: Beta measures spectral power-law exponent. Low beta (~0.05) indicates:
- Text is becoming too predictable/repetitive
- Loss of long-range variability
- The model is "settling" into repetitive patterns

**Why it happens**:
- Metrics calculated on **individual chunks** instead of **accumulated text**
- Each chunk passes validation, but accumulated text loses diversity
- Context window limitations cause the model to "forget" earlier variety
- Insufficient variation in prompts/instructions between chunks

### 2. Horizontal Autocorrelation Plot (Should show negative trend)

**Root Cause**: Autocorrelation function should decay as a power law (negative slope in log-log plot)
- Horizontal plot = no long-range correlations
- Natural language has fractal/self-similar structure across scales
- Generated text is too locally coherent but globally repetitive

**Why it happens**:
- Same as beta issue - repetitive patterns dominate
- Model generates locally fluent text but lacks global structure
- Insufficient plot/character arc guidance in prompts

## Recommended Solutions

### Solution 1: Monitor Accumulated Text, Not Just Chunks

**Current Problem**: You're calculating metrics on individual chunks
**Fix**: Calculate metrics on the **full accumulated text** after each chunk

```python
# In your generation loop:
accumulated_text = ""
for chunk_idx in range(max_chunks):
    new_chunk = generate_chunk(...)
    
    # Add to accumulated text
    accumulated_text += new_chunk
    
    # Calculate metrics on FULL accumulated text
    metrics = compute_metrics(accumulated_text)
    
    # Check if accumulated text is still in acceptable range
    if not is_chunk_acceptable(metrics, text=accumulated_text):
        # Reject and regenerate with stronger variation prompts
        continue
```

### Solution 2: Adaptive Coherence Targets

**Problem**: Fixed thresholds don't account for text length
**Fix**: Adjust targets based on accumulated length

```python
def get_adaptive_thresholds(text_length_tokens):
    \"\"\"Adjust coherence targets based on text length\"\"\"
    base_cfg = _DEFAULT_CFG.copy()
    
    # As text gets longer, allow slightly more variation
    # to prevent settling into repetitive patterns
    if text_length_tokens > 50000:
        # Widen beta range to encourage more variation
        base_cfg["beta_min"] = 0.120  # Slightly lower
        base_cfg["beta_max"] = 0.400  # Slightly higher
        
        # Relax other constraints slightly
        base_cfg["gamma_min"] = 0.002
        base_cfg["gamma_max"] = 0.030
        
    return base_cfg
```

### Solution 3: Variation Injection Prompts

**Problem**: Model settles into comfortable patterns
**Fix**: Inject variation reminders every N chunks

```python
VARIATION_INJECTION_PROMPTS = [
    \"\"\"CRITICAL: The next section must introduce a NEW setting or location. 
    Avoid reusing the previous scene type. Show a different facet of the world.\"\"\",
    
    \"\"\"CRITICAL: Change the pacing. If previous sections were action-heavy, 
    shift to introspection or dialogue. If they were slow, accelerate.\"\"\",
    
    \"\"\"CRITICAL: Introduce a consequence from an earlier event (from 3+ chapters ago). 
    Show how past actions are still affecting the present.\"\"\",
    
    \"\"\"CRITICAL: Shift the emotional tone. If recent sections were tense, 
    allow a moment of relief or hope. If hopeful, introduce new tension.\"\"\",
    
    \"\"\"CRITICAL: Use different sentence structures and rhythms than the previous section. 
    Vary paragraph lengths. Mix short punchy sentences with longer flowing ones.\"\"\",
]

def get_variation_prompt(chunk_idx):
    \"\"\"Rotate through variation prompts every few chunks\"\"\"
    if chunk_idx % 3 == 0:  # Every 3rd chunk
        return VARIATION_INJECTION_PROMPTS[chunk_idx % len(VARIATION_INJECTION_PROMPTS)]
    return \"\"
```

### Solution 4: Story Contract Enforcement

**Problem**: Model loses track of overarching plot
**Fix**: Include story contract reminders in chunk prompts

```python
CHUNK_WITH_CONTRACT_PROMPT = \"\"\"You are continuing a novel. 

STORY CONTRACT (remember the overall arc):
{story_contract_summary}

CURRENT PROGRESS:
- We are approximately {progress_pct}% through the story
- Current plot beat: {current_beat}
- Next major event should be: {next_beat}

RECENT CONTEXT:
{memory_summary}

PREVIOUS TEXT:
{previous_text}

VARIATION REQUIREMENT:
{variation_injection}

CRITICAL RULES:
- Advance toward the next plot beat: {next_beat}
- Introduce NEW elements (setting, conflict, or revelation)
- Reference at least one earlier event as a consequence
- Vary sentence structure and pacing from previous section
- NO repetitive openings (avoid \"As the...\", \"The sun...\", etc.)

Write the next 1000-2000 words:
\"\"\"
```

### Solution 5: Rejection & Regeneration Strategy

**Problem**: Once beta drops, it never recovers
**Fix**: Implement rejection with increasingly strong variation prompts

```python
def generate_chunk_with_retry(context, max_retries=3):
    \"\"\"Generate chunk with retry logic for coherence\"\"\"
    
    for attempt in range(max_retries):
        # Increase variation pressure with each retry
        variation_strength = attempt  # 0, 1, 2
        
        prompt = build_chunk_prompt(
            context,
            variation_level=variation_strength
        )
        
        chunk = llm_generate(prompt)
        accumulated = context.full_text + chunk
        
        metrics = compute_metrics(accumulated)
        
        if is_chunk_acceptable(metrics, text=accumulated):
            return chunk
        
        # Log why it was rejected
        print(f\"Attempt {attempt+1} rejected: beta={metrics['beta']:.3f}\")
    
    # If all retries fail, use the best attempt
    return chunk  # or raise exception
```

### Solution 6: Temperature/Sampling Adjustments

**Problem**: Model is too conservative in generation
**Fix**: Adjust sampling parameters based on coherence metrics

```python
def get_adaptive_sampling_params(current_metrics, target_metrics):
    \"\"\"Adjust temperature/top_p based on current coherence\"\"\"
    
    beta_current = current_metrics.get('beta', 0.25)
    beta_target = (target_metrics['beta_min'] + target_metrics['beta_max']) / 2
    
    # If beta is too low (too predictable), increase temperature
    if beta_current < target_metrics['beta_min']:
        temperature = 0.9  # Higher temperature for more variety
        top_p = 0.95
    elif beta_current > target_metrics['beta_max']:
        temperature = 0.6  # Lower temperature to reduce chaos
        top_p = 0.85
    else:
        temperature = 0.7  # Default
        top_p = 0.9
    
    return {
        'temperature': temperature,
        'top_p': top_p,
        'frequency_penalty': 0.3,  # Penalize repetition
        'presence_penalty': 0.2,   # Encourage new topics
    }
```

## Implementation Priority

1. **Immediate** (fixes the core issue):
   - [ ] Calculate metrics on accumulated text, not chunks
   - [ ] Add variation injection prompts every 3-5 chunks
   - [ ] Implement retry logic with rejection

2. **Short-term** (improves quality):
   - [ ] Add LLM-generated story contracts (use new templates)
   - [ ] Include story contract reminders in chunk prompts
   - [ ] Implement adaptive sampling parameters

3. **Medium-term** (polish):
   - [ ] Adaptive coherence thresholds based on length
   - [ ] Better memory/summary system that tracks plot beats
   - [ ] Automated detection of repetitive patterns

## Usage Example

```python
from utils.prompt_templates import INITIAL_PROMPT_GENERATION, QUICK_STORY_CONTRACT

# 1. Generate story contract
contract_prompt = INITIAL_PROMPT_GENERATION.format(
    genre=\"Science Fiction\",
    theme=\"A robot hiding during a robot ban, learning to paint\",
    tone=\"Bittersweet with moments of hope and adventure\"
)

story_contract = llm_generate(contract_prompt)

# 2. Use contract in generation
# Store contract and reference it in each chunk prompt
# Include current progress percentage and next plot beat

# 3. Monitor accumulated text
accumulated_text = \"\"
for i in range(num_chunks):
    chunk = generate_chunk_with_retry(
        context={
            'story_contract': story_contract,
            'accumulated_text': accumulated_text,
            'progress_pct': (i / num_chunks) * 100,
            'variation_level': i % 3  # Rotate variation prompts
        }
    )
    accumulated_text += chunk
    
    # Check accumulated metrics
    metrics = compute_metrics(accumulated_text)
    print(f\"Chunk {i}: beta={metrics['beta']:.3f}, H_DFA1={metrics['H_DFA1']:.3f}\")
```

## Expected Improvements

After implementing these changes:
- **Beta**: Should stabilize in 0.15-0.35 range throughout generation
- **Autocorrelation**: Should show power-law decay (negative slope in log-log)
- **Variety**: More diverse settings, scene types, and sentence structures
- **Coherence**: Better long-range plot consistency via story contract
