"""
SemanticCoherence
-----------------

Window scales (W1k–W40k) — defined in characters, tokeniser-agnostic.
All windows larger than SUB_WINDOW_CHARS (1 024 chars) are split into
sub-windows for hierarchical pooling; the referee model always sees
≤ 1 024-char chunks (≈ 256 GPT-4 tokens ≈ 320 Qwen2 tokens).

  W1k  —  1 000 chars  ≈   250 GPT-4 tok  — paragraph (1–2 paragraphs)
  W2k  —  2 048 chars  ≈   512 GPT-4 tok  — paragraph cluster / scene fragment (~3 paragraphs)
  W8k  —  8 000 chars  ≈ 2 000 GPT-4 tok  — scene (~1 300 words)
  W40k — 40 000 chars  ≈10 000 GPT-4 tok  — chapter (~6 700 words, 1–2 typical chapters)

Input : raw text string (encoding-agnostic; windows defined in characters)
Referee: gte-Qwen2-7B-instruct (default) — a model-agnostic referee that maps
         both natural books and LLM-generated text to the same semantic space.

Metrics (all prefixed sc_W{name}_ e.g. sc_W1k_, sc_W40k_):
  Scalar:
    n_windows              — number of windows at this scale
    off_diag_mean          — mean off-diagonal similarity (global semantic concentration)
    off_diag_std           — std of off-diagonal similarities
    novelty_mean           — mean novelty = 1 - max_sim_to_past (NaN-safe)
    novelty_std            — std of novelty
    novelty_final          — novelty at the last window (stagnation signal)
    mean_sim_to_past_mean  — mean of mean_sim_to_past across positions
    mean_sim_to_past_final — mean_sim_to_past at last window (convergence)
    sim_lag1_mean          — mean cosine sim between adjacent windows
    sim_lag1_std           — std of adjacent-window similarity
  Arrays (for plotting):
    novelty_trajectory     — np.ndarray (n_windows-1,) rolling novelty
    mean_sim_trajectory    — np.ndarray (n_windows-1,) rolling mean sim to past
    sim_matrix             — np.ndarray (n_windows, n_windows) cosine sim matrix

SemanticCoherenceMonitor (online mode):
  Processes one window at a time as text grows. Returns rolling sc_* metrics
  after each update(), maintaining embedding history entirely in host memory.
  Metric keys use sc_ prefix (no window suffix — one monitor instance per scale).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Window sizes in characters per scale W1k–W40k (tokeniser-agnostic).
# Keys are window indices 1–4 mapping to W1k, W2k, W8k, W40k respectively.
WINDOW_CHARS: Dict[int, int] = {
    1: 1_000,
    2: 2_048,
    3: 8_000,
    4: 40_000,
}

# Human-readable names for each window scale (used as output key prefix).
WINDOW_NAMES: Dict[int, str] = {
    1: "W1k",
    2: "W2k",
    3: "W8k",
    4: "W40k",
}

# Keep old names as aliases so existing callers don't break immediately.
LEVEL_WINDOW_CHARS = WINDOW_CHARS

# Sub-window size for hierarchical pooling (large levels).
# 1 024 chars ≈ 256 GPT-4 tokens ≈ 320 Qwen2 tokens — safely within any
# supported model's context window.
SUB_WINDOW_CHARS = 1_024

DEFAULT_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_windows(text: str, window_chars: int, stride_chars: int) -> List[str]:
    windows, start, n = [], 0, len(text)
    while start < n:
        end = min(start + window_chars, n)
        windows.append(text[start:end])
        if end == n:
            break
        start += stride_chars
    return windows


def _pool_last_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last-real-token pooling for decoder-only models."""
    seq_lens = attention_mask.sum(dim=1) - 1          # (B,)
    idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[idx, seq_lens]                       # (B, D)


def _pool_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)        # (B, D)


def _embed_batch(
    texts: List[str],
    tokenizer,
    model,
    device: str,
    max_length: int,
    pooling: str,
) -> torch.Tensor:
    """Embed a list of texts → L2-normalised tensor (N, D) on CPU."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        hidden = model(**encoded, use_cache=False).last_hidden_state  # (B, L, D)
    if pooling == "last_token":
        emb = _pool_last_token(hidden, encoded["attention_mask"])
    else:
        emb = _pool_mean(hidden, encoded["attention_mask"])
    return F.normalize(emb, dim=-1).cpu()              # (N, D)


def _embed_windows(
    windows: List[str],
    tokenizer,
    model,
    device: str,
    batch_size: int,
    max_length: int,
    pooling: str,
    hierarchical: bool,
) -> torch.Tensor:
    """
    Embed a list of (possibly large) windows.

    When hierarchical=True, windows longer than SUB_WINDOW_CHARS are split
    into sub-windows, each sub-window is embedded, and the window vector is
    the mean of the L2-normalised sub-window embeddings (then re-normalised).
    This preserves semantic content for windows that would otherwise be
    truncated to only their first few hundred tokens.
    """
    if not hierarchical:
        all_embs = []
        for i in range(0, len(windows), batch_size):
            all_embs.append(
                _embed_batch(windows[i : i + batch_size], tokenizer, model,
                             device, max_length, pooling)
            )
        return torch.cat(all_embs, dim=0)

    # Hierarchical: collect (window_idx, sub_window_text) pairs
    window_subs: List[List[str]] = []
    for w in windows:
        if len(w) <= SUB_WINDOW_CHARS:
            window_subs.append([w])
        else:
            window_subs.append(
                _split_windows(w, SUB_WINDOW_CHARS, SUB_WINDOW_CHARS)
            )

    # Flatten into one list for batched embedding
    flat_texts: List[str] = []
    offsets: List[int] = []        # start index of each window's subs in flat_texts
    for subs in window_subs:
        offsets.append(len(flat_texts))
        flat_texts.extend(subs)

    flat_embs = []
    for i in range(0, len(flat_texts), batch_size):
        flat_embs.append(
            _embed_batch(flat_texts[i : i + batch_size], tokenizer, model,
                         device, max_length, pooling)
        )
    flat_embs = torch.cat(flat_embs, dim=0)            # (total_subs, D)

    # Pool sub-window embeddings per window
    window_embs = []
    for wi, subs in enumerate(window_subs):
        start = offsets[wi]
        sub_vecs = flat_embs[start : start + len(subs)]  # (K, D)
        vec = sub_vecs.mean(dim=0)                        # (D,)
        vec = F.normalize(vec, dim=0)
        window_embs.append(vec)

    return torch.stack(window_embs, dim=0)              # (N, D)


def _load_model(model_name_or_path: str, device: str):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        dtype=torch.float16,
    ).to(device).eval()
    return tokenizer, model


def _sim_matrix_gpu(embeddings: torch.Tensor, device: str) -> np.ndarray:
    """Compute NxN cosine similarity on GPU, return as float32 CPU numpy array."""
    e = embeddings.to(device)
    sim = (e @ e.T).float().cpu().numpy()   # cast to float32 — float16 overflows np.std on large N
    return sim


def _window_metrics(sim: np.ndarray, window: int) -> dict:
    """Compute metrics from an NxN similarity matrix for window scale W{WINDOW_NAMES[window]}."""
    n = sim.shape[0]
    prefix = f"sc_{WINDOW_NAMES[window]}_"
    out: dict = {f"{prefix}n_windows": n}

    if n < 2:
        return out

    mask = ~np.eye(n, dtype=bool)
    off_diag = sim[mask]
    out[f"{prefix}off_diag_mean"] = float(np.mean(off_diag))
    out[f"{prefix}off_diag_std"]  = float(np.std(off_diag))

    novelty, mean_sim_to_past, sim_lag1 = [], [], []
    for i in range(1, n):
        past = sim[i, :i]
        novelty.append(float(1.0 - past.max()))
        mean_sim_to_past.append(float(past.mean()))
        sim_lag1.append(float(sim[i, i - 1]))

    nov = np.array(novelty)
    msp = np.array(mean_sim_to_past)
    lg1 = np.array(sim_lag1)

    out[f"{prefix}novelty_mean"]          = float(np.nanmean(nov))
    out[f"{prefix}novelty_std"]           = float(np.nanstd(nov))
    out[f"{prefix}novelty_final"]         = float(nov[-1])
    out[f"{prefix}mean_sim_to_past_mean"] = float(np.nanmean(msp))
    out[f"{prefix}mean_sim_to_past_final"]= float(msp[-1])
    out[f"{prefix}sim_lag1_mean"]         = float(np.nanmean(lg1))
    out[f"{prefix}sim_lag1_std"]          = float(np.nanstd(lg1))

    # Arrays for plotting
    out[f"{prefix}novelty_trajectory"]  = nov
    out[f"{prefix}mean_sim_trajectory"] = msp
    out[f"{prefix}sim_matrix"]          = sim

    return out


# keep old name as alias
_level_metrics = _window_metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SemanticCoherence:
    """
    Retrospective multi-scale semantic coherence metrics.

    Embed the full text at once across all requested levels, then compute
    the NxN similarity matrix and derived metrics per level.

    Parameters
    ----------
    text : str
        Raw text to analyse. Windows are character-based so any encoding is fine.
    model_name_or_path : str
        HuggingFace model ID or local path of the referee embedding model.
    levels : list of int, optional
        Subset of {2, 3, 4, 5} to compute. Defaults to all four.
    stride_factor : float
        Window stride = window_chars * stride_factor. 0.5 gives 50 % overlap.
    device : str, optional
        Torch device string. Defaults to 'cuda' if available, else 'cpu'.
    batch_size : int
        Number of (sub-)windows per forward pass through the referee.
    max_length : int
        Max token length passed to the referee tokeniser. 512 is safe for all
        levels when hierarchical=True; increase for non-hierarchical mode.
    pooling : str
        'last_token' (default, correct for decoder-only models such as Qwen2)
        or 'mean' (for encoder models).
    hierarchical : bool
        When True (default), windows larger than SUB_WINDOW_CHARS are split
        into sub-windows, each embedded, then averaged. Prevents truncation
        artefacts at larger window scales.
    tokenizer, model : optional
        Pre-loaded HuggingFace tokenizer / model. Pass these when processing
        multiple texts to avoid reloading the referee between calls.
    """

    LEVEL_WINDOW_CHARS = LEVEL_WINDOW_CHARS
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(
        self,
        text: str,
        model_name_or_path: str = DEFAULT_MODEL,
        levels: Optional[List[int]] = None,
        stride_factor: float = 1.0,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
        pooling: str = "last_token",
        hierarchical: bool = True,
        tokenizer=None,
        model=None,
    ):
        self.text = text
        self.model_name_or_path = model_name_or_path
        self.levels = levels if levels is not None else list(WINDOW_CHARS.keys())
        self.stride_factor = stride_factor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.hierarchical = hierarchical
        self.results: dict = {}

        self._tokenizer = tokenizer
        self._model = model

    def _ensure_model(self):
        if self._model is None:
            self._tokenizer, self._model = _load_model(
                self.model_name_or_path, self.device
            )

    def compute_window(self, window: int) -> dict:
        """Compute metrics for window scale W{window}. Updates and returns self.results."""
        self._ensure_model()
        window_chars = WINDOW_CHARS[window]
        stride_chars = max(1, int(window_chars * self.stride_factor))

        windows = _split_windows(self.text, window_chars, stride_chars)
        if not windows:
            return {}

        embeddings = _embed_windows(
            windows,
            self._tokenizer, self._model,
            self.device, self.batch_size, self.max_length,
            self.pooling, self.hierarchical,
        )

        sim = _sim_matrix_gpu(embeddings, self.device)
        metrics = _window_metrics(sim, window)
        self.results.update(metrics)
        return metrics

    # keep old name as alias
    compute_level = compute_window

    def compute_all(self, levels: Optional[List[int]] = None) -> dict:
        """Compute all requested window scales. Returns the full results dict."""
        for w in (levels if levels is not None else self.levels):
            self.compute_window(w)
        return self.results


class SemanticCoherenceMonitor:
    """
    Online / sequential semantic coherence monitoring.

    Process one window at a time as text grows (or as a reader progresses).
    After each update() call, rolling metrics are returned and appended to
    self.trajectory for later plotting.

    All returned metrics use the sc_* prefix (no level suffix — one monitor
    instance covers one level).

    Parameters
    ----------
    level : int
        Which scale level this monitor operates at (determines window_chars).
    model_name_or_path : str
        HuggingFace referee model ID or local path.
    loop_lag : int, optional
        Expected repetition period in windows. When set, sc_sim_at_loop_lag
        is included in each update's output.
    stride_factor : float
        Fraction of window_chars to advance between consecutive windows when
        calling update_from_text(). Ignored if you supply pre-split windows
        directly to update().
    device, batch_size, max_length, pooling, hierarchical, tokenizer, model :
        Same semantics as SemanticCoherence.
    """

    def __init__(
        self,
        level: int,
        model_name_or_path: str = DEFAULT_MODEL,
        loop_lag: Optional[int] = None,
        stride_factor: float = 1.0,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
        pooling: str = "last_token",
        hierarchical: bool = True,
        tokenizer=None,
        model=None,
    ):
        self.level = level
        self.model_name_or_path = model_name_or_path
        self.loop_lag = loop_lag
        self.stride_factor = stride_factor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.hierarchical = hierarchical
        self.trajectory: List[dict] = []

        self._tokenizer = tokenizer
        self._model = model
        self._embeddings: List[torch.Tensor] = []  # (D,) tensors on CPU

    def _ensure_model(self):
        if self._model is None:
            self._tokenizer, self._model = _load_model(
                self.model_name_or_path, self.device
            )

    def _embed_one(self, window_text: str) -> torch.Tensor:
        """Embed a single window, return L2-normalised (D,) tensor on CPU."""
        self._ensure_model()
        if self.hierarchical and len(window_text) > SUB_WINDOW_CHARS:
            subs = _split_windows(window_text, SUB_WINDOW_CHARS, SUB_WINDOW_CHARS)
            emb = _embed_windows(
                subs, self._tokenizer, self._model,
                self.device, self.batch_size, self.max_length,
                self.pooling, hierarchical=False,
            ).mean(dim=0)
            emb = F.normalize(emb, dim=0)
        else:
            emb = _embed_batch(
                [window_text], self._tokenizer, self._model,
                self.device, self.max_length, self.pooling,
            ).squeeze(0)
        return emb  # (D,) on CPU

    def update(self, window_text: str) -> dict:
        """
        Embed one new window and compute rolling similarity metrics against
        all previously seen windows.

        Returns a dict of sc_* metrics for this step and appends it to
        self.trajectory.
        """
        emb = self._embed_one(window_text)
        n_prev = len(self._embeddings)

        metrics: dict = {"sc_n_windows": n_prev + 1}

        if n_prev > 0:
            history = torch.stack(self._embeddings)                # (N, D) CPU
            sims = (history @ emb).numpy()                         # (N,)

            metrics["sc_novelty"]          = float(1.0 - sims.max())
            metrics["sc_max_sim_to_past"]  = float(sims.max())
            metrics["sc_mean_sim_to_past"] = float(sims.mean())
            metrics["sc_sim_lag1"]         = float(sims[-1])
            if n_prev >= 2:
                metrics["sc_sim_lag2"] = float(sims[-2])
            if n_prev >= 3:
                metrics["sc_sim_lag3"] = float(sims[-3])
            if self.loop_lag is not None and n_prev >= self.loop_lag:
                metrics["sc_sim_at_loop_lag"] = float(sims[-self.loop_lag])
        else:
            metrics["sc_novelty"]          = float("nan")
            metrics["sc_max_sim_to_past"]  = float("nan")
            metrics["sc_mean_sim_to_past"] = float("nan")
            metrics["sc_sim_lag1"]         = float("nan")

        self._embeddings.append(emb)
        self.trajectory.append(metrics)
        return metrics

    def update_from_text(self, text: str) -> List[dict]:
        """
        Convenience wrapper: split text into windows for this level's
        stride_factor and call update() for each new window.

        Only windows not yet processed are added (based on character position).
        Returns the list of metric dicts for the newly processed windows.
        """
        window_chars = WINDOW_CHARS[self.level]
        stride_chars = max(1, int(window_chars * self.stride_factor))
        all_windows = _split_windows(text, window_chars, stride_chars)
        new_results = []
        for i in range(len(self._embeddings), len(all_windows)):
            new_results.append(self.update(all_windows[i]))
        return new_results

    def novelty_trajectory(self) -> np.ndarray:
        """Return sc_novelty values across all processed windows."""
        return np.array([m.get("sc_novelty", float("nan")) for m in self.trajectory])

    def mean_sim_trajectory(self) -> np.ndarray:
        """Return sc_mean_sim_to_past values across all processed windows."""
        return np.array([m.get("sc_mean_sim_to_past", float("nan")) for m in self.trajectory])

    def sim_lag1_trajectory(self) -> np.ndarray:
        """Return sc_sim_lag1 (cosine sim to adjacent previous window) across all windows."""
        return np.array([m.get("sc_sim_lag1", float("nan")) for m in self.trajectory])

    def reset(self):
        """Clear all accumulated state."""
        self._embeddings.clear()
        self.trajectory.clear()
