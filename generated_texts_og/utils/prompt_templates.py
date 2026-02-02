# Natural language prompt templates for longbook generation

# -------------------------------------------------------------------
# Chunk Generation
# -------------------------------------------------------------------

CHUNK_GENERATION_PROMPT = """You are a novelist continuing a story. Write at least {target_length} words.

STORY CONTRACT (maintain throughout):
{story_contract}

CONTEXT:
{memory_summary}

PREVIOUS TEXT (last section):
{previous_text}

CRITICAL RULES:
- Write ONLY story prose - no code, no JSON, no meta-commentary
- Do NOT include the original prompt or instructions
- Do NOT start with "assistant:" or any prefix
- Continue naturally from where the previous text ended
- Vary your sentence structures and openings
- Show, don't tell. **Avoid filtering words** (e.g., "she felt", "he saw", "it seemed"). Describe the sensation directly.

STYLE REQUIREMENTS (for natural text quality):
- **RICH VOCABULARY**: Use RICH, RARE, and PRECISE vocabulary. Avoid generic words (e.g., instead of "looked", use "gazed", "peered", "glared"). Do not repeat words within the same paragraph.
- **SYNTAX**: Use complex, varied sentence structures. Mix short punchy sentences with long, flowing ones containing subordinate clauses.
- **INFORMATION DENSITY**: Every sentence must add new information, texture, or character depth. Avoid filler sentences.
- **LONG-RANGE DEPENDENCE**: You MUST explicitly reference events, objects, or decisions from previous chapters (not just the recent ones). Weave in recurring motifs.
- **SENSORY DETAILS**: Incorporate specific sights, sounds, smells, textures, and tastes.
- **AVOID CLICHÉS**: Use fresh metaphors and similes.
- **NON REPETITIVE**: Do not copy-paste phrases or descriptions from the previous text. Constantly vary your phrasing, but do not be afraid to reference the same subjects (thematic cohesion).
- **THEMATIC CONTINUITY**: While varying your language, maintain the core atmosphere and recurring motifs. Repetition of key thematic words is acceptable for coherence.
{variation_hint}
Write the continuation now (prose only):
"""

# -------------------------------------------------------------------
# Compact Variation Hints (rotate to prevent repetition)
# -------------------------------------------------------------------
# These are SHORT injections (1-2 lines) to encourage variety
# without bloating the prompt. Rotate through them every few chunks.

VARIATION_HINTS = [
    "\n- NEW: Shift to a different setting or introduce a fresh scene type.",
    "\n- NEW: Change pacing - if recent text was fast, slow down; if slow, accelerate.",
    "\n- NEW: **CALLBACK**: Explicitly reference a consequence from an earlier event (try to reference something from at least 3 chapters ago).",
    "\n- NEW: Vary sentence rhythm - use short, jagged sentences to create tension.",
    "\n- NEW: Shift emotional tone drastically from the previous section.",
    "\n- NEW: Introduce a sudden, unexpected obstacle or complication.",
    "\n- NEW: **VOCABULARY**: Use archaic, scientific, or rare words appropriate to the context.",
    "\n- NEW: Employ complex, nested sentence structures with multiple subordinate clauses.",
    "\n- NEW: Focus on **MICRO-DETAILS**: Describe a small object or texture in extreme, microscopic detail.",
    "\n- NEW: Use a strange or surreal metaphor to describe a common feeling.",
    "\n- NEW: **MOTIF**: Reintroduce a recurring symbol or theme from the start of the book.",
    "\n- NEW: **Perspective Shift**: Focus intently on the internal physical sensations of the character.",
    "\n- NEW: **Unreliable Narration**: Write a passage where the character's perception might be flawed or distorted.",
    "",  # No hint - let model flow naturally
]

def get_variation_hint(chunk_idx):
    """Get a compact variation hint based on chunk index."""
    return VARIATION_HINTS[chunk_idx % len(VARIATION_HINTS)]

# -------------------------------------------------------------------
# Title Generation
# -------------------------------------------------------------------

TITLE_GENERATION_PROMPT = """
Based on the story summary below, suggest a compelling and market-appropriate title for this book.

STORY SUMMARY:
{summary}

GENRE: {genre}

INSTRUCTIONS:
- Provide ONLY the title.
- Do not add quotes or explanations.
- Make it catchy and relevant to the plot.

TITLE:
"""

# -------------------------------------------------------------------
# Genre Generation
# -------------------------------------------------------------------

BROADER_GENRE_PROMPT = """
Analyze the story summary below and determine the single best matching broader genre.

STORY SUMMARY:
{summary}

ALLOWED GENRES:
Fiction, Nonfiction, History, Drama, Poetry, Philosophy, Science,
Essays, Biography, Adventure, Fantasy, Mystery, Romance, Horror, Thriller.

INSTRUCTIONS:
- Choose exactly ONE genre from the list.
- Provide ONLY the genre name.

GENRE:
"""

SPECIFIC_GENRE_PROMPT = """
Analyze the story summary below and determine the most specific sub-genre.

STORY SUMMARY:
{summary}

BROADER GENRE: {broader}

INSTRUCTIONS:
- Provide ONLY the specific sub-genre name.
- Examples: "Cyberpunk", "Cozy Mystery", "Historical Fiction", "Space Opera", etc.
- Do not add quotes or explanations.

SPECIFIC GENRE:
"""

# -------------------------------------------------------------------
# Episodic Summary Generation
# -------------------------------------------------------------------

EPISODIC_SUMMARY_PROMPT = """You are analyzing a section of a story to create a high-level summary.

RECENT CHUNKS:
{chunks_text}

Create a JSON summary with the following structure:
{{
  "narrative_arc": "Brief description of the overall narrative arc of this episode",
  "major_transitions": ["List", "of", "key", "plot", "transitions"],
  "emotional_shift": "Description of how the emotional tone evolved",
  "unresolved_elements": ["List", "of", "plot", "threads", "left", "unresolved"]
}}

Return ONLY the JSON, no other text.
"""

# -------------------------------------------------------------------
# Initial Story Contract Generation (LLM-generated)
# -------------------------------------------------------------------

INITIAL_PROMPT_GENERATION = """You are a creative writing consultant helping to design a compelling long-form novel.

Given the following inputs, create a detailed story contract that will guide consistent, coherent long-form generation:

GENRE: {genre}
THEME/CONCEPT: {theme}
TONE: {tone}

Generate a complete story contract following this structure:

## Story Contract

### Core Premise
[2-3 sentences describing the central conflict and world]

### World/Genre/Tone
- Setting: [specific world details - time period, location, technology level, magic system if applicable]
- Genre blend: [primary genre + any secondary elements]
- Tone: [emotional register - e.g., "darkly humorous with moments of hope", "tense thriller with philosophical undertones"]

### POV and Tense
- Point of view: [first-person / third-person limited / third-person omniscient]
- Tense: [past / present]
- Voice characteristics: [e.g., "sardonic and observant", "lyrical and introspective", "direct and action-focused"]

### Main Throughline (4-5 concrete beats)
Provide specific plot beats that build toward a clear end-state:
1. [Opening situation and inciting incident]
2. [First major complication or discovery]
3. [Midpoint reversal or escalation]
4. [Crisis point where stakes are highest]
5. [Resolution approach and intended end-state]

### Subplots (3-4 intersecting arcs)
Each subplot should cross with the main arc at specific moments:
1. [Subplot A: description] - Intersects at: [when/how it changes the main arc]
2. [Subplot B: description] - Intersects at: [when/how it changes the main arc]
3. [Subplot C: description] - Intersects at: [when/how it changes the main arc]
4. [Subplot D: description] - Intersects at: [when/how it changes the main arc]

### Antagonistic Forces and Stakes
- Primary antagonist/force: [who/what and their motivation]
- Secondary obstacles: [list 2-3 other sources of conflict]
- Stakes progression: [how stakes escalate - no resets to status quo]
- What's at risk: [personal, relational, societal consequences]

### Character Arc Anchors
- Protagonist's starting state: [beliefs, skills, relationships]
- Core internal conflict: [what they must overcome within themselves]
- Evolution markers: [3-4 moments where character demonstrably changes]
- Ending state: [who they've become, what they've learned]

### Ending Goal
- What must be resolved: [specific plot and character resolutions required]
- Tone of resolution: [triumphant / bittersweet / tragic / ambiguous]
- Final image/moment: [the last scene or feeling]

## Progression Guardrails

### Avoid Repetition
- Each chapter/segment must change the situation: new obstacle or consequence; no loops or "back to normal"
- Reuse prior events as consequences; do not reintroduce the same discovery/conflict twice
- Shift settings and scene types periodically; avoid reusing identical openings (e.g., "As the sun rose...")
- Limit recap to one brief line if needed; prioritize forward motion and decisions
- Keep character goals evolving; when one goal is achieved/failed, replace it with a sharper one

### Pacing Targets
- Early intrigue: Hook within first 5% of text
- First act turn: Major shift by 25%
- Midpoint reversal: Significant change in direction or stakes at 50%
- Accelerating decisions: Faster pacing in final 25%
- Climax positioning: Peak tension in final 10-15%

### Texture Variation (to maintain distinctness)
Rotate these elements to keep scenes feeling fresh:
- **Settings**: [list 5-7 distinct locations/environments to cycle through]
- **Scene types**: [list varied scene structures: confrontation, discovery, travel, reflection, action, negotiation, etc.]
- **Imagery/motifs per section**:
  - First third: [specific sensory details, symbols, or recurring images]
  - Middle third: [different set of textures and motifs]
  - Final third: [culminating imagery that echoes but transforms earlier motifs]

### Dialogue/Narration Balance
- Dialogue-heavy scenes: [when - e.g., confrontations, revelations, character bonding]
- Narration-heavy scenes: [when - e.g., travel, introspection, world-building]
- Target ratio: [e.g., 40% dialogue / 60% narration, or adjust based on genre]

## Output Rules (Critical for Chunk Stitching)

- **Pure prose only** - no bullet lists, JSON, or meta-notes in the actual story
- **Do not restate** these instructions or the outline in the generated text
- **Heading consistency**: If using chapter headings, keep format simple and consistent (e.g., `Chapter X: Title`) or omit entirely; never change heading style mid-book
- **Continuity cues**: At the start of each new chunk, reference the last consequence, not a fresh exposition dump
- **No scaffolding**: Avoid blank templates like "Chapter Summary:" or repeated structural elements

## Next Chunk Reminder Template

When generating each new chunk, ensure it includes:
1. A new event (not a rehash)
2. A consequence of any earlier event (callback/continuity)
3. A change in goal or constraint (progression)
4. At least one fresh setting/texture detail (variety)
5. A forward hook (momentum)

Exclude:
- Recaps longer than two sentences
- Duplicated reveals or discoveries
- Copy/pasted openings from previous chunks
- Status resets that undo previous progress

---

Now generate the complete story contract based on the inputs provided above. Be specific, creative, and ensure all elements work together cohesively.
"""

# Simplified version for quick generation
QUICK_STORY_CONTRACT = """Create a story contract for a {genre} novel about {theme}.

Provide:
1. **Core premise** (2-3 sentences)
2. **POV/Tense**: [specify]
3. **Main plot beats** (4-5 specific events from start to end)
4. **3 subplots** that intersect with main arc
5. **Antagonistic forces** and how stakes escalate
6. **Ending goal** and tone
7. **5-7 distinct settings** to rotate through
8. **Imagery/motifs** for beginning, middle, and end sections

Be specific and concrete. Avoid generic placeholders.
"""

# -------------------------------------------------------------------
# Story Contract Refinement (for iterative improvement)
# -------------------------------------------------------------------

REFINE_STORY_CONTRACT = """Review this story contract and enhance it for better long-form coherence:

CURRENT CONTRACT:
{current_contract}

PROBLEMS OBSERVED:
{problems}

REFINEMENT GOALS:
- Make plot beats more specific and concrete
- Ensure subplots have clear intersection points
- Add more varied settings and scene types
- Strengthen the progression guardrails
- Ensure stakes genuinely escalate (no resets)

Provide the refined contract with the same structure but improved specificity and coherence.
"""
