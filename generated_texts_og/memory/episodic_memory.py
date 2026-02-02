# ================================================================
# episodic_memory.py (v3)
#
# Stores episodic summaries generated every N chunks (E2).
# This keeps the summary logic modular and clean.
# ================================================================

class EpisodicMemory:
    """
    Stores intermediate summaries created every episodic interval.
    These summaries represent larger narrative blocks (chapters,
    arcs, scenes) rather than small chunk summaries.
    """

    def __init__(self):
        self.episodes = []

    # ---------------------------------------------------------------
    def reset(self):
        self.episodes.clear()

    # ---------------------------------------------------------------
    def append(self, summary_dict):
        """
        Add one episodic summary (E2). The summary_dict must be
        JSON-serializable and produced by the LLM.
        """
        self.episodes.append(summary_dict)

    # ---------------------------------------------------------------
    def get_all(self):
        """Return all episodic summaries."""
        return list(self.episodes)

    # ---------------------------------------------------------------
    def to_serializable(self):
        """Return JSON-serializable representation."""
        return list(self.episodes)
