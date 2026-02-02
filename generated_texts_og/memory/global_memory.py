# ================================================================
# global_memory.py (v3)
#
# Maintains the global narrative summary (G3). This includes:
#   - aggregating episodic summaries
#   - aggregating chunk summaries
#   - incorporating semantic embeddings (mean vector)
#   - generating an LLM-ready summary prompt
#   - storing the final global summary text
#
# ================================================================

import json
import numpy as np


class GlobalMemory:
    """
    Stores the global-level summary text and provides utilities to
    build summary prompts for the LLM. The goal is to maintain an
    evolving high-level view of the book.
    """

    def __init__(self):
        self.summary_text = ""   # final G3 global summary (string)

    # ---------------------------------------------------------------
    def reset(self):
        self.summary_text = ""

    # ---------------------------------------------------------------
    def update(self, new_summary_text: str):
        """
        Replace the global summary with the LLM-produced text.
        """
        if not isinstance(new_summary_text, str):
            new_summary_text = str(new_summary_text)
        self.summary_text = new_summary_text.strip()

    # ---------------------------------------------------------------
    def get(self):
        """Return current global summary text."""
        return self.summary_text

    # ---------------------------------------------------------------
    def build_global_summary_prompt(self,
                                    recent_chunks,
                                    episodes,
                                    mean_semantic_vec):
        """
        Build the summary prompt for the LLM (G3). This is used
        by longbook_engine to request an updated global summary.

        Inputs:
          - recent_chunks: list of dict (S2 summaries)
          - episodes: list of dict (E2 summaries)
          - mean_semantic_vec: numpy array (64,)
        """

        # Convert everything to JSON for clarity
        data = {
            "recent_chunk_summaries": recent_chunks,
            "episode_summaries": episodes,
            "semantic_mean_vector": mean_semantic_vec.tolist(),
            "previous_global_summary": self.summary_text
        }

        json_blob = json.dumps(data, indent=2)

        prompt = (
            "Task: Update the global summary of a long-form narrative.\n"
            "You are given:\n"
            "  - The most recent chunk summaries (S2)\n"
            "  - Episodic summaries (E2)\n"
            "  - The previous global summary\n"
            "  - A semantic fusion vector (captures style and long-range topics)\n\n"
            "Output:\n"
            "  - A single clean paragraph summarizing the WHOLE story so far.\n\n"
            "DATA:\n"
            f"{json_blob}\n\n"
            "Now provide ONLY the updated global summary paragraph:\n"
        )

        return prompt

    # ---------------------------------------------------------------
    def to_serializable(self):
        """JSON-serializable export."""
        return {
            "global_summary": self.summary_text
        }

    # ---------------------------------------------------------------
    def dump_to_folder(self, folder_path, prefix="memory_global"):
        """
        Save global summary for ED64 debug.
        """
        import os
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, f"{prefix}.json"), "w") as f:
            json.dump(self.to_serializable(), f, indent=2)
