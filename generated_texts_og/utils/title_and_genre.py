import re
from typing import Tuple, Dict, Any
from ..models.model_interface import BaseLLM
from ..memory.memory_summarizer import MemorySummarizer
from .prompt_templates import (
    TITLE_GENERATION_PROMPT,
    BROADER_GENRE_PROMPT,
    SPECIFIC_GENRE_PROMPT
)

def generate_title_and_genre(
    memory: Dict[str, Any],
    model: BaseLLM
) -> Tuple[str, str, str]:
    """
    Generates broader genre, specific genre, and title using the provided model.
    Uses robust extraction strategies to handle various model outputs.
    """
    
    # 1. Prepare Summary
    summary = MemorySummarizer.summarize_for_title(memory)
    
    # 2. Generate Broader Genre
    broader_prompt = BROADER_GENRE_PROMPT.format(summary=summary)
    raw_broader = model.generate(broader_prompt, max_new_tokens=32).strip()
    broader_genre = _extract_clean_value(raw_broader, "Fiction")
    
    # Step 2: Get specific sub-genre
    specific_prompt = SPECIFIC_GENRE_PROMPT.format(
        summary=summary,
        broader=broader_genre  # Fixed: was broader_genre
    )
    specific_raw = model.generate(specific_prompt, max_new_tokens=50).strip()
    specific_genre = _extract_clean_value(specific_raw, "General")
    
    # Step 3: Generate title
    title_prompt = TITLE_GENERATION_PROMPT.format(
        summary=summary,
        genre=f"{broader_genre} - {specific_genre}"
    )
    title_raw = model.generate(title_prompt, max_new_tokens=100).strip()
    title = _extract_clean_value(title_raw, "Untitled Story")
    
    return broader_genre, specific_genre, title

def _extract_clean_value(raw_text: str, default: str) -> str:
    """
    Robustly extracts a clean value from raw LLM output using multiple strategies.
    """
    if not raw_text:
        return default
        
    # Strategy 1: Remove quotes if the whole string is quoted
    if raw_text.startswith('"') and raw_text.endswith('"'):
        return raw_text[1:-1].strip()
    if raw_text.startswith("'") and raw_text.endswith("'"):
        return raw_text[1:-1].strip()
        
    # Strategy 2: Look for "Title: X" or "Genre: X" patterns
    match = re.search(r'(?:Title|Genre|Sub-genre):\s*(.+)', raw_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
        
    # Strategy 3: Take the first non-empty line
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    if lines:
        # If the first line is just a label like "Here is the title:", take the second
        if len(lines) > 1 and (lines[0].endswith(':') or "here is" in lines[0].lower()):
            return lines[1].strip()
        return lines[0]
        
    return default
