# ================================================================
# chunk_tools.py (v4)
#
# Safe extraction of continuation text from an LLM output.
# Aggressively filters prompt contamination.
# ================================================================

import re


def extract_new_text(old_context, full_output, prompt=None):
    """
    Extracts only the new story text from LLM output.
    Aggressively filters:
    - The original prompt (if echoed)
    - Prompt echoes
    - Assistant/user prefixes
    - Code blocks
    - JSON/metadata
    - Meta-instructions
    """
    if not isinstance(old_context, str):
        old_context = str(old_context)
    if not isinstance(full_output, str):
        full_output = str(full_output)
    if prompt and not isinstance(prompt, str):
        prompt = str(prompt)

    # 0. Remove the exact prompt if it appears at the start (robust to whitespace)
    if prompt:
        # Normalize both for comparison
        norm_output = full_output.lstrip()
        norm_prompt = prompt.strip()
        
        # Check if output starts with prompt (ignoring leading whitespace)
        if norm_output.startswith(norm_prompt):
            # Find where the prompt ends in the original string
            # We use the length of the prompt + the difference in leading whitespace
            start_idx = len(full_output) - len(norm_output)
            full_output = full_output[start_idx + len(norm_prompt):]
    
    clean_text = full_output.strip()
    
    # 1. Remove common LLM prefixes
    prefixes_to_remove = [
        "assistant:",
        "Assistant:",
        "ASSISTANT:",
        "user:",
        "User:",
        "USER:",
        "Story:",
        "STORY:",
        "Continuation:",
        "CONTINUATION:",
    ]
    
    for prefix in prefixes_to_remove:
        if clean_text.lower().startswith(prefix.lower()):
            clean_text = clean_text[len(prefix):].strip()
    
    # 2. Remove code blocks (```...```)
    clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
    clean_text = re.sub(r'`[^`]+`', '', clean_text)  # inline code
    
    # 3. Remove JSON blocks (only if they look like actual JSON with key:value pairs)
    # More conservative: only remove if it contains quotes and colons (actual JSON structure)
    clean_text = re.sub(r'\{[^}]*"[^"]*":[^}]*\}', '', clean_text)
    
    # 4. Remove lines that look like prompts or instructions
    lines = clean_text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines initially (we'll clean up later)
        if not line_lower:
            filtered_lines.append(line)
            continue
            
        # Skip meta-text markers
        skip_patterns = [
            "=== end memory ===",
            "=== memory context ===",
            "=== story context ===",
            "=== previous text ===",
            "=== your task ===",
            "=== context ===",
            "now write",
            "[continuation",
            "[note:",
            "end of story",
            "story continuation",
            "begin writing",
            "write the next",
            "write the continuation",
            "critical guidelines:",
            "critical rules:",
            "**critical",
            "def ",  # Python function definitions
            "import ",  # Python imports
            "return ",  # Python returns
            "class ",  # Python classes
        ]
        
        if any(pattern in line_lower for pattern in skip_patterns):
            continue
            
        # Skip lines that are entirely markdown headers without content
        if re.match(r'^#+\s*$', line.strip()):
            continue
            
        # Filter out non-Latin characters but keep accents
        cleaned_line = "".join(c for c in line if ord(c) < 0x250 or 0x2000 <= ord(c) <= 0x206F)
        filtered_lines.append(cleaned_line)
    
    clean_text = '\n'.join(filtered_lines)
    
    # 5. Remove the previous context if it appears at the start
    if old_context:
        # Try to find where new content starts
        prev_end = old_context[-500:].strip() if len(old_context) > 500 else old_context.strip()
        if clean_text.startswith(prev_end):
            clean_text = clean_text[len(prev_end):].strip()
    
    # 6. Final cleanup: remove excessive whitespace
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # max 2 newlines
    clean_text = re.sub(r' {2,}', ' ', clean_text)  # max 1 space
    
    return clean_text.strip()
