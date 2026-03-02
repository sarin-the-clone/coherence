import sys
import os

# Ensure we can import from src/coherence
# This file is in src/coherence/check_sanitizer.py
# generated_texts_og is in src/coherence/generated_texts_og
# So we just need to add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generated_texts_og.utils.prompt_sanitizer import strip_chat_prefixes, quick_contamination_check

def test_strip_chat_prefixes():
    test_cases = [
        (
            "Here is the story continuation:\n\nThe sun rose over the horizon.",
            "The sun rose over the horizon."
        ),
        (
            "Sure, I can help with that!\n\nThe sun rose.",
            "The sun rose."
        ),
        (
            "Assistant: The sun rose.",
            "The sun rose."
        ),
        (
            "The sun rose.",
            "The sun rose."
        ),
        (
            "Here is a story about space:\n\nIt was dark.",
            "It was dark."
        ),
        (
             "I will write the story now.\n\nOnce upon a time...",
             "Once upon a time..."
        )
    ]

    print("Running Sanitizer Tests...")
    all_passed = True
    for original, expected in test_cases:
        try:
            cleaned = strip_chat_prefixes(original)
        except NameError:
             print("❌ strip_chat_prefixes not implemented yet.")
             return

        if cleaned.strip() != expected.strip():
            print(f"❌ Failed:\nOriginal: {original!r}\nExpected: {expected!r}\nGot:      {cleaned!r}")
            all_passed = False
        else:
            print(f"✅ Passed: {original[:20]}... -> {cleaned[:20]}...")
            
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed.")

if __name__ == "__main__":
    test_strip_chat_prefixes()
