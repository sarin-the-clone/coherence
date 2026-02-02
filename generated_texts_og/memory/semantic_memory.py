# ================================================================
# semantic_memory.py (v3)
# Maintains a bank of low-dimensional semantic vectors (64-dim)
# derived from the CompressionTransformer. Used for global memory.
# ================================================================

import torch
import numpy as np


class SemanticMemory:
    """
    Stores a rolling bank of semantic embeddings (64-dim vectors).

    Pipeline:
      - chunk text is embedded using CompressionTransformer
      - appended to semantic bank
      - J3 pruning keeps last N entries

    This provides a stable long-term signal of style, themes, and
    long-range structure beyond the LLM's local context.
    """

    def __init__(self, dim=None, max_entries=50):
        self.dim = dim  # Will be auto-detected from first vector if None
        self.max_entries = max_entries
        self.memory = []      # list of numpy arrays (dim,)

    # ---------------------------------------------------------------
    def append(self, vector: torch.Tensor):
        """
        Add a new vector to memory. Input is a 1D tensor.
        Auto-detects dimension from first vector.
        """
        if isinstance(vector, torch.Tensor):
            v = vector.detach().cpu().numpy().astype(float)
        else:
            v = np.array(vector, dtype=float)

        # Auto-detect dimension from first vector
        if self.dim is None:
            self.dim = v.shape[-1]
            print(f"SemanticMemory: Auto-detected dimension = {self.dim}")
        
        if v.shape[-1] != self.dim:
            raise ValueError(f"Semantic vector must have dim={self.dim}, got {v.shape}")

        self.memory.append(v)
        self._prune()

    # ---------------------------------------------------------------
    def _prune(self):
        """
        Keep only the last N semantic vectors (J3 pruning).
        """
        if len(self.memory) > self.max_entries:
            self.memory = self.memory[-self.max_entries:]

    # ---------------------------------------------------------------
    def get_all(self):
        """
        Return semantic memory as numpy array of shape (N, dim).
        """
        if len(self.memory) == 0:
            return np.zeros((0, self.dim), dtype=float)
        return np.stack(self.memory, axis=0)

    # ---------------------------------------------------------------
    def get_mean_embedding(self):
        """
        Returns a single fusion vector (mean embedding) used
        by the global memory summarizer.
        """
        if len(self.memory) == 0:
            return np.zeros((self.dim,), dtype=float)
        A = self.get_all()
        return A.mean(axis=0)

    # ---------------------------------------------------------------
    def to_serializable(self):
        """
        JSON-serializable representation.
        """
        return {
            "dim": self.dim,
            "n_entries": len(self.memory),
            "vectors": [v.tolist() for v in self.memory]
        }
