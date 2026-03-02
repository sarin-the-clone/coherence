# ================================================================
# local_context.py (v3)
# Simple rolling window context manager
# ================================================================

import re

class LocalContext:
    """
    Maintains the rolling local context window for the LLM, used
    to keep continuity across chunks without exceeding context size.
    """

    def __init__(self, max_tokens=8000, fallback_tokens=3000):
        self.max_tokens = max_tokens
        self.fallback_tokens = fallback_tokens
        self.context = ""

    # ------------------------------------------------------------
    def reset(self, initial_text: str):
        """Initialize the context with the initial user prompt."""
        if not isinstance(initial_text, str):
            initial_text = str(initial_text)
        self.context = initial_text.strip()

    # ------------------------------------------------------------
    def update_context(self, new_chunk: str) -> str:
        """
        Append new chunk to rolling context and trim if needed.
        """
        if not isinstance(new_chunk, str):
            new_chunk = str(new_chunk)

        self.context = (self.context + "\n" + new_chunk).strip()

        # Hard trim by characters (approx 4 chars per token)
        # User requested simple multiplication here
        char_limit = self.max_tokens * 4
        if len(self.context) > char_limit:
            self.context = self.context[-char_limit:]

        return self.context

    # ------------------------------------------------------------
    def get_context(self) -> str:
        """Return the current context."""
        return self.context

    # ------------------------------------------------------------
    def get_minimal_context(self) -> str:
        """
        Return a trimmed fallback context, used when:
        - retry failures occur
        - memory injection becomes too large
        """
        return self.context[-self.fallback_tokens:]
