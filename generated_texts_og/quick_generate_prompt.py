#!/usr/bin/env python3
"""
Quick script to generate a filled initial prompt using INITIAL_PROMPT_GENERATION
"""

import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from generated_texts_og.models.model_interface import ModelFactory
from generated_texts_og.utils.prompt_templates import INITIAL_PROMPT_GENERATION
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--genre", required=True)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--tone", default="balanced")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--output", default="generated_texts_og/prompts/filled_initial_prompt.md")
    args = parser.parse_args()
    
    # Load model
    with open("generated_texts_og/config/model_config.yaml") as f:
        config = yaml.safe_load(f)
    
    model_config = config["models"][args.model]
    print(f"Loading {args.model}...")
    model = ModelFactory.create_model(model_config)
    
    # Generate
    # Determine Identity & Format
    # Determine Identity & Format
    g_lower = args.genre.lower()
    if any(x in g_lower for x in ['science', 'philosophy', 'technical', 'non-fiction', 'textbook', 'biography']):
        if 'biography' in g_lower:
            identity_prompt = "You are an expert biographer and historian"
            format_type = "detailed biography"
        else:
            identity_prompt = "You are an expert domain consultant"
            format_type = "book or treatise"
    else:
        identity_prompt = "You are a creative writing consultant"
        format_type = "novel"

    prompt = INITIAL_PROMPT_GENERATION.format(
        identity_prompt=identity_prompt,
        format_type=format_type,
        genre=args.genre,
        theme=args.theme,
        tone=args.tone
    )
    
    print("Generating story contract...")
    model.temperature = 0.8
    contract = model.generate(prompt, max_new_tokens=2000)
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(contract)
    
    print(f"\n✓ Saved to: {args.output}")
    print(f"\nUse with:\n  python3 run_longbook.py --prompt \"$(cat {args.output})\" ...")

if __name__ == "__main__":
    main()
