#!/usr/bin/env python3
"""
story_contract_parser.py

Parses initial prompts to separate narrative seed from persistent story contract.
"""

import re
from typing import Tuple, Dict


def parse_initial_prompt(initial_prompt: str) -> Tuple[str, str]:
    """
    Split initial prompt into narrative seed and story contract.
    
    Args:
        initial_prompt: Full initial prompt text
        
    Returns:
        (narrative_seed, story_contract)
    """
    
    # Find where the story contract begins
    # Look for common markers
    contract_markers = [
        "## Story contract",
        "## Story Contract",
        "## STORY CONTRACT",
        "Story contract:",
        "POV/tense:",
        "World/genre/tone:",
    ]
    
    contract_start = -1
    for marker in contract_markers:
        pos = initial_prompt.find(marker)
        if pos != -1:
            contract_start = pos
            break
    
    if contract_start == -1:
        # No contract found, treat entire prompt as narrative
        return initial_prompt.strip(), ""
    
    # Split
    narrative_seed = initial_prompt[:contract_start].strip()
    story_contract = initial_prompt[contract_start:].strip()
    
    # Clean up narrative seed (remove title markers if present)
    narrative_seed = re.sub(r'^#\s+.*?\n', '', narrative_seed)
    narrative_seed = narrative_seed.strip()
    
    return narrative_seed, story_contract


def extract_contract_essentials(story_contract: str) -> Dict[str, str]:
    """
    Extract key contract elements for easy access.
    
    Returns dict with: pov, tense, tone, world, guardrails
    """
    essentials = {
        'pov': '',
        'tense': '',
        'tone': '',
        'world': '',
        'guardrails': ''
    }
    
    # Extract POV - look for "Point of view:" pattern
    pov_match = re.search(r'Point of view:\s*([^\n]+)', story_contract, re.IGNORECASE)
    if pov_match:
        essentials['pov'] = pov_match.group(1).strip()
    
    # Extract Tense - look for "Tense:" pattern  
    tense_match = re.search(r'^-?\s*Tense:\s*([^\n]+)', story_contract, re.MULTILINE | re.IGNORECASE)
    if tense_match:
        essentials['tense'] = tense_match.group(1).strip()
    
    # Extract Tone - look for "Tone:" in World/Genre/Tone section
    tone_match = re.search(r'^-?\s*Tone:\s*([^\n]+)', story_contract, re.MULTILINE | re.IGNORECASE)
    if tone_match:
        essentials['tone'] = tone_match.group(1).strip()
    
    # Extract Setting from World/Genre/Tone section
    setting_match = re.search(r'^-?\s*Setting:\s*([^\n]+)', story_contract, re.MULTILINE | re.IGNORECASE)
    if setting_match:
        essentials['world'] = setting_match.group(1).strip()
    
    # Extract Core Premise as fallback for world
    if not essentials['world']:
        premise_match = re.search(r'### Core Premise\s*\n([^\n#]+)', story_contract, re.IGNORECASE)
        if premise_match:
            essentials['world'] = premise_match.group(1).strip()[:100] + "..."  # First 100 chars
    
    # Guardrails - just use standard text
    essentials['guardrails'] = "Vary settings/pacing, no repetitive openings, escalate stakes, forward motion"
    
    return essentials


def format_contract_for_prompt(story_contract: str, compact: bool = True) -> str:
    """
    Format story contract for injection into chunk prompts.
    
    Args:
        story_contract: Full contract text
        compact: If True, extract only essentials; if False, use full contract
        
    Returns:
        Formatted contract text for prompt injection
    """
    if compact:
        essentials = extract_contract_essentials(story_contract)
        
        parts = []
        if essentials['pov']:
            parts.append(f"POV: {essentials['pov']}")
        if essentials['tense']:
            parts.append(f"Tense: {essentials['tense']}")
        if essentials['tone']:
            parts.append(f"Tone: {essentials['tone']}")
        if essentials['world']:
            parts.append(f"World: {essentials['world']}")
        
        # Add key guardrails
        parts.append("Guardrails: Vary settings/pacing, no repetitive openings, escalate stakes")
        
        return " | ".join(parts)
    else:
        # Return full contract but clean it up
        return story_contract


if __name__ == "__main__":
    # Test with example
    example_prompt = """
You are writing a single, longform novel with consistent POV and tense.
This story is about a robot during the ban of all robots, hiding to learn painting.

## Story contract
- World/genre/tone: Near-future dystopia with bittersweet hope
- POV/tense: Third-person limited, present tense
- Main throughline: Discovery → Mastery → Exposure → Choice
- Guardrails: Vary settings, no repetitive openings, escalate stakes
"""
    
    narrative, contract = parse_initial_prompt(example_prompt)
    print("NARRATIVE SEED:")
    print(narrative)
    print("\n" + "="*60 + "\n")
    print("STORY CONTRACT:")
    print(contract)
    print("\n" + "="*60 + "\n")
    print("COMPACT FORMAT:")
    print(format_contract_for_prompt(contract, compact=True))
