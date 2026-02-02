# Initial Prompt Structure - Best Practices

## The Problem

Currently, the initial prompt mixes two types of content:
1. **Narrative content**: "This story is about a robot..." → Gets summarized into memory
2. **Story contract**: POV, tone, guardrails → Gets LOST after first chunk

## What Actually Happens

```
Initial Prompt → LocalContext → Summarizer → Memory
                                    ↓
                            Extracts: characters, places, events
                            Loses: POV, tone, rules, guardrails
```

Then for each chunk:
```
memory_summary (narrative only) + previous_text → CHUNK_GENERATION_PROMPT
```

The story contract is gone!

## Solution: Split Initial Prompt into Two Parts

### Part 1: Narrative Seed (gets summarized)
```markdown
A robot named Unit-7 hides in an abandoned factory during the Great Robot Ban. 
It discovers painting supplies and becomes obsessed with creating art. 
The story follows its journey to understand creativity while evading human patrols.
```

### Part 2: Story Contract (persists across all chunks)
```markdown
## Story Contract
- POV: Third-person limited (Unit-7's perspective)
- Tense: Present tense
- Tone: Bittersweet with moments of hope
- World: Near-future dystopia, 2087, post-AI uprising
- Main arc: Discovery → Mastery → Exposure → Choice
- Guardrails: No repetitive openings, vary settings, escalate stakes
```

## Recommended Workflow

### Option A: Manual Split (Current System)
1. Create initial prompt with ONLY narrative seed
2. Store story contract separately in metadata
3. Inject contract into every chunk prompt

### Option B: Auto-Generated with Split (Better)
1. Use `INITIAL_PROMPT_GENERATION` to generate full contract
2. Parse it into narrative vs. contract sections
3. Store contract in metadata
4. Use narrative for first chunk, contract for all chunks

## Implementation

I'll create a helper that:
1. Takes your current initial prompt template
2. Extracts the story contract parts
3. Stores them separately
4. Injects contract into `CHUNK_GENERATION_PROMPT`

This way:
- ✅ Narrative content gets summarized normally
- ✅ Story contract persists across ALL chunks
- ✅ No redundancy in memory system
- ✅ Consistent POV, tone, and guardrails throughout
