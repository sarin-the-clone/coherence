"""
prompt_sanitizer.py

LLM-based sanitization of generated text chunks to remove contamination
and ensure they follow the story contract.
"""

SANITIZATION_PROMPT = """You are a text editor reviewing a generated story chunk for quality issues.

STORY CONTRACT:
{story_contract}

GENERATED TEXT:
{generated_text}

REVIEW CHECKLIST:
1. Remove any meta-commentary, JSON, code, or instructions that leaked through
2. Ensure consistent POV and tense as specified in contract
3. Remove any repetitive or stuttering text
4. Fix any obvious continuity errors
5. Ensure prose flows naturally

CRITICAL RULES:
- If text is clean, return it EXACTLY as-is
- Only make minimal necessary corrections
- Do NOT rewrite or expand the text
- Do NOT add new content
- Preserve the author's voice and style

Return ONLY the corrected text (pure prose, no explanations):
"""

def sanitize_chunk(generated_text: str, story_contract: str, model) -> str:
    """
    Use LLM to sanitize a generated chunk.
    
    Args:
        generated_text: The raw generated text
        story_contract: The story contract for reference
        model: The model instance to use for sanitization
        
    Returns:
        Sanitized text
    """
    prompt = SANITIZATION_PROMPT.format(
        story_contract=story_contract,
        generated_text=generated_text
    )
    
    # Use low temperature for conservative edits
    original_temp = model.temperature
    model.temperature = 0.3
    
    sanitized = model.generate(prompt, max_new_tokens=len(generated_text.split()) + 100)
    
    # Restore original temperature
    model.temperature = original_temp
    
    # Extract just the prose (remove any meta-commentary from sanitizer)
    sanitized = sanitized.strip()
    
    # If sanitizer added explanations, try to extract just the prose
    if sanitized.startswith("Here") or sanitized.startswith("The corrected"):
        # Try to find where the actual text starts
        lines = sanitized.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 100 and not line.startswith(('Here', 'The', 'I have', 'This')):
                sanitized = '\n'.join(lines[i:])
                break
    
    return sanitized


def quick_contamination_check(text: str) -> bool:
    """
    Quick heuristic check for obvious contamination.
    Returns True if contamination detected.
    """
    contamination_markers = [
        'assistant:',
        '```',
        '{',
        '}',
        '[INST]',
        '</s>',
        'JSON',
        'Here is',
        'I will',
        'Let me',
        'As an AI',
        'Chapter Summary:',
        'Outline:',
        'Plot:',
    ]
    
    text_lower = text.lower()
    for marker in contamination_markers:
        if marker.lower() in text_lower:
            return True
    
    return False
