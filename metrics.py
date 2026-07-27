"""
Unified Coherence Metrics
-------------------------

CoherenceMetrics
├── StructuralScaling [overall speed: moderate with GPU, slow on CPU]
│     ├── autocorrelation_decay (β)
│     ├── spectral_exponent (α)
│     ├── hurst_dfa1 (H_DFA1)
│     ├── hurst_dma (H_DMA)
│     ├── hurst_wdfa_db4 (H_WT_db4)
│     ├── fractal_dimension (D)
│     └── holder_exponent (optional)
├── InformationalStructure [overall speed: mixed; entropy GPU, LZC CPU, MI GPU/CPU]
│     ├── shannon_entropy (H₁)
│     ├── perplexity (PP)
│     ├── entropy_rate (optional)
│     ├── lz_complexity (LZC)
│     ├── mutual_information_decay (γ)
│     ├── conditional_entropy (optional)
│     └── cross_entropy / kl_divergence (optional)
└── TokenPredictability [overall speed: moderate with large GPU, slow on CPU]
      ├── token entropy H(token_t)
      ├── conditional entropy H(token_t | token_{t-k:t-1})
      └── redundancy / predictability gain from token context
Optional token-only naturalness families:
      ├── TokenDistributionScaling [overall speed: fast]: Zipf slope, Heaps exponent, type-token ratios
      ├── TokenRecurrence [overall speed: fast]: repeat-gap burstiness and recurrence distribution
      └── TokenContextDynamics [overall speed: very_slow]: chunk-local entropy, boundary entropy, drift

Core metric families:

1. Structural Scaling
   - Quantifies persistence and self-similarity in token correlations.
   - Core: β (autocorr), α (spectrum), H_DFA1/H_DMA/H_WT_db4, D

2. Informational Structure
   - Quantifies information organization, predictability, and coupling.
   - Core: Shannon entropy, Perplexity, LZC, Mutual Information decay

3. TokenPredictability
   - Language-agnostic token conditional entropy curve inspired by Shannon's
     finite-context entropy estimates.
   - Core: H0, H(token_t | previous k tokens), redundancy, entropy gain

Optional retrospective-only TokenNaturalness families:
   - TokenDistributionScaling (td_*): rank-frequency and vocabulary-growth laws.
   - TokenRecurrence (tr_*): bursty token reuse via repeat-gap statistics.
   - TokenContextDynamics (tc_*): F_k curves across chunks, boundaries, and time.

-----------------------------------
Automatically detects CUDA devices and accelerates the heavy parts:
    • autocorrelation_decay (β)
    • spectral_exponent (α)
    • hurst_dfa (H)
    • mutual_information_decay (γ)

Everything else (entropy, LZC, fractal_dimension) stays CPU.
"""

from __future__ import annotations
import math
import numpy as np
from scipy.stats import entropy
from collections import Counter
import torch
from numba import njit


METRIC_REFERENCES = {
    "TokenPredictability": [
        "Shannon CE. 1951. Prediction and Entropy of Printed English. Bell System Technical Journal 30(1):50-64. https://www.nokia.com/bell-labs/publications-and-media/publications/prediction-and-entropy-of-printed-english/",
        "Cover TM, Thomas JA. 2006. Elements of Information Theory. Wiley.",
        "Stanford CS information theory notes, Entropy of English. https://cs.stanford.edu/people/eroberts/courses/soco/projects/1999-00/information-theory/entropy_of_english_9.html",
    ],
    "TokenDistributionScaling": [
        "Zipf GK. 1949. Human Behavior and the Principle of Least Effort. Addison-Wesley. https://commons.wikimedia.org/wiki/File:George_Zipf_-_Human_Behavior_and_the_Principle_of_Least_Effort_(1949).pdf",
        "Heaps HS. 1978. Information Retrieval: Computational and Theoretical Aspects. Academic Press. https://openlibrary.org/books/OL4716764M/Information_retrieval_computational_and_theoretical_aspects",
    ],
    "TokenRecurrence": [
        "Goh KI, Barabasi AL. 2008. Burstiness and memory in complex systems. EPL 81:48002. https://arxiv.org/abs/physics/0610233",
        "Altmann EG, Pierrehumbert JB, Motter AE. 2009. Beyond Word Frequency: Bursts, Lulls, and Scaling in the Temporal Distributions of Words. PLoS ONE 4:e7678. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0007678",
    ],
    "TokenContextDynamics": [
        "Shannon CE. 1951. Prediction and Entropy of Printed English. Bell System Technical Journal 30(1):50-64. https://www.nokia.com/bell-labs/publications-and-media/publications/prediction-and-entropy-of-printed-english/",
        "Cover TM, Thomas JA. 2006. Elements of Information Theory. Wiley.",
    ],
}


METRIC_COMPUTE_COST = {
    # Existing live families
    "StructuralScaling": {
        "overall": "moderate_with_gpu_slow_on_cpu",
    "reason": "Autocorrelation, spectrum, DFA1, DEA, DMA, and db4 WDFA have GPU paths; CPU fallback is slow on long books.",
    },
    "InformationalStructure": {
        "overall": "mixed",
        "reason": "Shannon entropy and MI have GPU paths; LZC is CPU/sequential.",
    },

    # Retrospective token-only families
    "TokenPredictability": {
        "overall": "moderate_with_large_gpu_slow_on_cpu",
        "reason": "Conditional entropy uses n-gram row-unique counting; GPU path is practical on large VRAM, CPU path is slow.",
        "fast": ["tp_entropy_0"],
        "moderate": ["tp_cond_entropy_1", "tp_redundancy_1", "tp_entropy_gain_1"],
        "slow": [
            "tp_cond_entropy_2",
            "tp_cond_entropy_3",
            "tp_cond_entropy_4",
            "tp_redundancy_2",
            "tp_redundancy_3",
            "tp_redundancy_4",
            "tp_entropy_gain_2",
            "tp_entropy_gain_3",
            "tp_entropy_gain_4",
            "tp_entropy_slope",
            "tp_entropy_drop_4",
            "tp_context_efficiency_4",
        ],
    },
    "TokenDistributionScaling": {
        "overall": "fast",
        "reason": "Mostly one pass plus token count sorting; Heaps profile is one pass over tokens.",
        "fast": [
            "td_vocab_size",
            "td_type_token_ratio",
            "td_hapax_type_fraction",
            "td_hapax_token_fraction",
            "td_top1_token_fraction",
            "td_top10_token_fraction",
            "td_zipf_slope",
            "td_zipf_exponent",
            "td_zipf_r2",
            "td_heaps_exponent",
            "td_heaps_r2",
        ],
        "moderate": [],
        "slow": [],
    },
    "TokenRecurrence": {
        "overall": "fast",
        "reason": "Single pass over tokens plus simple repeat-gap summaries.",
        "fast": [
            "tr_repeat_rate",
            "tr_repeat_gap_mean",
            "tr_repeat_gap_mean_log2",
            "tr_repeat_gap_median_log2",
            "tr_repeat_gap_cv",
            "tr_repeat_gap_burstiness",
            "tr_repeat_gap_logbin_entropy",
        ],
        "moderate": [],
        "slow": [],
    },
    "TokenContextDynamics": {
        "overall": "very_slow",
        "reason": "Repeats conditional entropy over chunks, boundaries, and cumulative checkpoints.",
        "fast": ["tc_n_boundaries", "tc_boundary_window"],
        "moderate": [
            "tc_cross_boundary_F_0",
            "tc_cross_boundary_F_1",
            "tc_chunk_boundary_surprisal_jump",
        ],
        "slow": [
            "tc_F_0",
            "tc_F_1",
            "tc_F_2",
            "tc_F_3",
            "tc_F_4",
            "tc_redundancy_1",
            "tc_redundancy_2",
            "tc_redundancy_3",
            "tc_redundancy_4",
            "tc_context_gain",
            "tc_long_range_context_gain",
            "tc_cross_boundary_F_2",
            "tc_cross_boundary_F_3",
            "tc_cross_boundary_F_4",
            "tc_within_chunk_F_k_mean/std",
            "tc_entropy_rate_change_F_k_slope/r2",
            "tc_true_context_rank_*",
        ],
    },
}


METRIC_FAMILY_COMPUTE_SPEED = {
    "StructuralScaling": "moderate_with_gpu_slow_on_cpu",
    "InformationalStructure": "mixed_gpu_cpu",
    "TokenPredictability": "moderate_with_large_gpu_slow_on_cpu",
    "TokenDistributionScaling": "fast",
    "TokenRecurrence": "fast",
    "TokenContextDynamics": "very_slow",
}


CURATED_TOKEN_NATURALNESS_METRICS = (
    "tp_entropy_0",
    "tp_redundancy_1",
    "tp_entropy_slope",
    "tp_context_efficiency_4",
    "td_hapax_token_fraction",
    "td_zipf_exponent",
    "td_zipf_r2",
    "td_heaps_exponent",
    "td_heaps_r2",
    "tr_repeat_gap_burstiness",
    "tr_repeat_gap_cv",
    "tr_repeat_gap_logbin_entropy",
)


OPTIONAL_CHUNK_CONTEXT_METRICS = (
    "tc_chunk_boundary_surprisal_jump",
    "tc_entropy_rate_change_F_1_slope",
)


def filter_token_naturalness_metrics(metrics, include_chunk_context=False):
    """Keep the current curated retrospective token metrics from a metric dict."""
    allowed = set(CURATED_TOKEN_NATURALNESS_METRICS)
    if include_chunk_context:
        allowed.update(OPTIONAL_CHUNK_CONTEXT_METRICS)
    return {key: value for key, value in metrics.items() if key in allowed}


# ===============================================================
#  HELPERS
# ===============================================================
def _torch_polyfit_1d(x, y):
    """Simple linear fit y = a*x + b for 1D torch tensors."""
    X = torch.stack([x, torch.ones_like(x)], dim=1)
    coef = torch.linalg.lstsq(X, y.unsqueeze(1)).solution.squeeze()
    return coef[0].item(), coef[1].item()

def _hash_bins(x, B=4096):
    """
    Overflow-safe modular hash mapping token IDs to [0, B).
    Works on both CPU and GPU.
    """
    x = x.to(torch.int64)

    # Use a safe 61-bit modulus
    mod = 2**61 - 1

    # Reduce the large constant mod `mod` *before* making it a tensor
    const_val = 11400714819323198485 % mod

    # Create tensors on same device with safe range
    const = torch.tensor(const_val, dtype=torch.int64, device=x.device)
    mod_t = torch.tensor(mod, dtype=torch.int64, device=x.device)

    # Modular multiplication in safe 61-bit space
    h = (x.remainder(mod_t) * const.remainder(mod_t)) % mod_t
    return (h % B).to(torch.int64)

def _torch_histogram2d(a, b, bins=4096, device="cuda"):
    ai = _hash_bins(a, bins)
    bi = _hash_bins(b, bins)
    idx = ai * bins + bi
    hist = torch.bincount(idx, minlength=bins * bins)
    return hist.to(device).reshape(bins, bins).float()

def choose_tau_max(n_tokens, ratio=0.03, cap=50_000, floor=1_000):
    """Adaptive τ_max ~ ratio*N, clamped to [floor, cap]."""
    t = int(ratio * n_tokens)
    return max(floor, min(cap, t))

def make_tau_grid(n_tokens, tau_max=None, n_lags=48):
    """Log-spaced τ grid up to tau_max."""
    if tau_max is None:
        tau_max = choose_tau_max(n_tokens)
    taus = np.unique(np.ceil(np.logspace(0, np.log10(tau_max), n_lags)).astype(int))
    return taus[taus < n_tokens]  # safety

def mi_should_stop(mi_vals, eps=1e-4, k=5):
    # stop if last k MI values are below eps (or non-positive)
    if len(mi_vals) < k: return False
    tail = mi_vals[-k:]
    return all((m <= eps) for m in tail)

def _linear_fit_r2(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), r2

# ===============================================================
#  STRUCTURAL SCALING 
# ===============================================================
def autocorrelation_decay(tokens, max_lag=2000, device="cpu"):
    """ GPU-accelerated autocorrelation decay exponent β. """
    x = torch.tensor(tokens, dtype=torch.float32, device=device)
    x -= x.mean()
    n = len(x)
    f = torch.fft.rfft(x, n*2)
    acf = torch.fft.irfft(f * torch.conj(f))[:n]
    acf = acf / acf[0]

    # use up to first zero-crossing
    pos = (acf > 0).nonzero().flatten()
    if len(pos) < 5:
        return float('nan')
    last_pos = int(pos[-1].item())
    L = min(max_lag, last_pos)
    taus = torch.arange(1, L, device=device)
    y = acf[taus]
    # enforce monotone window (optional)
    valid = y > 0
    xlog = torch.log10(taus[valid])
    ylog = torch.log10(y[valid])
    slope, _ = _torch_polyfit_1d(xlog, ylog)
    return float(-slope)


def spectral_exponent(tokens, fs=1.0, device="cpu", frac_low=0.001, frac_high=0.1):
    """ GPU-accelerated spectral exponent α. """
    x = torch.tensor(tokens, dtype=torch.float32, device=device)
    x -= x.mean()
    n = len(x)
    f = torch.fft.rfftfreq(n, d=1/fs).to(device)
    Pxx = torch.abs(torch.fft.rfft(x))**2
    mask = (f > 0)
    f = f[mask]; Pxx = Pxx[mask]

    # pick a mid-low band: e.g., 0.1%..10% of Nyquist
    i0 = int(len(f) * frac_low)
    i1 = int(len(f) * frac_high)
    if i1 <= i0 + 3:  # fallback
        i0, i1 = 1, max(5, len(f)//10)

    fx = torch.log10(f[i0:i1])
    py = torch.log10(Pxx[i0:i1])
    slope, _ = _torch_polyfit_1d(fx, py)
    return float(-slope)


def hurst_dfa(tokens, min_scale=4, max_scale_ratio=0.25, n_scales=20, order=1, device="cuda"):
    """
    GPU-batched Detrended Fluctuation Analysis (DFA)
    ------------------------------------------------
    Computes the Hurst exponent H using vectorized least-squares fits per scale.

    Parameters
    ----------
    tokens : list[int] | np.ndarray | torch.Tensor
        Input signal (e.g., token IDs, normalized floats)
    min_scale : int
        Minimum window size
    max_scale_ratio : float
        Maximum scale as a fraction of signal length (e.g. 0.25 = N/4)
    n_scales : int
        Number of logarithmically spaced scales
    order : int
        Order of polynomial detrending (1=linear, 2=quadratic, etc.)
    device : str
        "cuda" or "cpu" (GPU recommended)

    Returns
    -------
    H : float
        Estimated Hurst exponent
    """
    # --- setup ---
    x = torch.as_tensor(tokens, dtype=torch.float32, device=device)
    x -= x.mean()
    y = torch.cumsum(x, dim=0)
    N = y.numel()

    # --- scales ---
    scales = torch.unique(
        torch.logspace(
            math.log10(min_scale),
            math.log10(N * max_scale_ratio),
            steps=n_scales,
            device=device,
        ).long()
    )

    F = []

    for s in scales:
        s = int(s.item())
        if s < order + 2: # Need enough points for fit
            continue
        n_segments = N // s
        if n_segments < 2:
            continue

        # --- segment the signal ---
        segs = y[: n_segments * s].view(n_segments, s)

        # --- build time vector and design matrix ---
        t = torch.arange(s, dtype=torch.float32, device=device)
        
        # Build design matrix [t^n, t^(n-1), ..., t^1, 1]
        # For order=1: [t, 1]
        # For order=2: [t^2, t, 1]
        cols = []
        for p in range(order, -1, -1):
            cols.append(t ** p)
        A = torch.stack(cols, dim=1)  # (s, order+1)
        
        # precompute pseudoinverse for small matrices
        pinvA = torch.pinverse(A)  # (order+1, s)
        # --- batched regression ---
        coeffs = segs @ pinvA.T  # (n_segments, order+1)
        trend = (A @ coeffs.T).T  # (n_segments, s)
        residuals = segs - trend
        rms = torch.sqrt(torch.mean(residuals ** 2, dim=1))
        F.append(torch.mean(rms))

        # free memory explicitly
        del segs, trend, residuals, rms
        torch.cuda.empty_cache()

    if len(F) < 3:
        return float("nan")

    # --- log-log fit ---
    F = torch.stack(F)
    scales = scales[: len(F)].float()
    log_s = torch.log10(scales)
    log_F = torch.log10(F)
    A = torch.stack([log_s, torch.ones_like(log_s)], dim=1)
    coef = torch.linalg.lstsq(A, log_F.unsqueeze(1)).solution.squeeze()
    H = coef[0].item()

    return H


def fractal_dimension(H, E=1):
    """D = E + 1 − H."""
    return E + 1 - H


# ===============================================================
#  INFORMATIONAL STRUCTURE  
# ===============================================================
def shannon_entropy(tokens, device="cpu"):
    """GPU-accelerated Shannon entropy & perplexity."""
    x = torch.tensor(tokens, dtype=torch.int64, device=device)
    counts = torch.bincount(x)
    counts = counts[counts > 0]
    probs = counts.float() / counts.sum()
    H = -(probs * torch.log2(probs)).sum()
    PP = torch.pow(2.0, H)
    return float(H.item()), float(PP.item())


def _entropy_bits_from_counts(counts):
    """Shannon entropy in bits from a vector of event counts."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts[counts > 0] / total
    return float(-(probs * np.log2(probs)).sum())


def _conditional_token_entropy(tokens, order):
    """
    Plug-in estimate of H(token_t | token_{t-order:t-1}) in bits.

    The implementation is exact for the observed token sequence and uses
    numpy row uniqueness instead of Python tuple counting to keep retrospective
    full-book computation tractable.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    arr = np.asarray(tokens, dtype=np.int64)
    if arr.size <= order:
        return float("nan")

    windows = np.lib.stride_tricks.sliding_window_view(arr, order + 1)
    _, joint_counts = np.unique(windows, axis=0, return_counts=True)
    _, context_counts = np.unique(windows[:, :-1], axis=0, return_counts=True)

    h_joint = _entropy_bits_from_counts(joint_counts)
    h_context = _entropy_bits_from_counts(context_counts)
    if not np.isfinite(h_joint) or not np.isfinite(h_context):
        return float("nan")

    return float(max(0.0, h_joint - h_context))


def _entropy_bits_from_torch_counts(counts):
    counts = counts.to(torch.float64)
    total = counts.sum()
    if total.item() <= 0:
        return float("nan")
    probs = counts[counts > 0] / total
    return float((-(probs * torch.log2(probs)).sum()).item())


def _conditional_token_entropy_torch(tokens, order, device="cuda"):
    """
    GPU plug-in estimate of H(token_t | token_{t-order:t-1}).

    This is intended for large-memory GPUs. torch.unique(dim=0) can allocate
    sizeable temporary workspaces for long books and high orders.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    x = torch.as_tensor(tokens, dtype=torch.int64, device=device)
    if x.numel() <= order:
        return float("nan")

    windows = x.unfold(0, order + 1, 1).contiguous()
    _, joint_counts = torch.unique(windows, dim=0, return_counts=True)
    _, context_counts = torch.unique(windows[:, :-1].contiguous(), dim=0, return_counts=True)

    h_joint = _entropy_bits_from_torch_counts(joint_counts)
    h_context = _entropy_bits_from_torch_counts(context_counts)
    if not np.isfinite(h_joint) or not np.isfinite(h_context):
        return float("nan")
    return float(max(0.0, h_joint - h_context))


def _conditional_token_entropy_at_positions(tokens, positions, order):
    """Conditional entropy restricted to selected target positions."""
    arr = np.asarray(tokens, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    if arr.size == 0 or positions.size == 0:
        return float("nan")

    positions = positions[(positions >= 0) & (positions < arr.size)]
    if order == 0:
        if positions.size == 0:
            return float("nan")
        _, counts = np.unique(arr[positions], return_counts=True)
        return _entropy_bits_from_counts(counts)

    if order < 0:
        raise ValueError("order must be >= 0")

    positions = positions[positions >= order]
    if positions.size == 0:
        return float("nan")

    contexts = np.empty((positions.size, order), dtype=np.int64)
    for j in range(order):
        contexts[:, j] = arr[positions - order + j]

    targets = arr[positions].reshape(-1, 1)
    joints = np.concatenate([contexts, targets], axis=1)
    _, joint_counts = np.unique(joints, axis=0, return_counts=True)
    _, context_counts = np.unique(contexts, axis=0, return_counts=True)

    h_joint = _entropy_bits_from_counts(joint_counts)
    h_context = _entropy_bits_from_counts(context_counts)
    if not np.isfinite(h_joint) or not np.isfinite(h_context):
        return float("nan")
    return float(max(0.0, h_joint - h_context))


def token_predictability_metrics(tokens, max_order=4, device="cpu", gpu_fallback=True):
    """
    TokenPredictability family.

    Estimates the finite-context entropy curve:
        H0 = H(token_t)
        Hk = H(token_t | token_{t-k:t-1})

    From this curve we derive redundancy and entropy gain. These are intended
    for retrospective analysis, not live gating, because high-order row
    counting over long books is CPU/RAM intensive compared with the existing
    live metrics.
    """
    use_gpu = str(device).startswith("cuda")
    arr = np.asarray(tokens, dtype=np.int64)
    metrics = {}

    if arr.size == 0:
        metrics["tp_entropy_0"] = float("nan")
        for k in range(1, max_order + 1):
            metrics[f"tp_cond_entropy_{k}"] = float("nan")
            metrics[f"tp_redundancy_{k}"] = float("nan")
            metrics[f"tp_entropy_gain_{k}"] = float("nan")
        metrics["tp_entropy_slope"] = float("nan")
        metrics[f"tp_entropy_drop_{max_order}"] = float("nan")
        metrics[f"tp_context_efficiency_{max_order}"] = float("nan")
        return metrics

    if use_gpu:
        try:
            h0, _ = shannon_entropy(arr, device=device)
        except Exception:
            if not gpu_fallback:
                raise
            _, unigram_counts = np.unique(arr, return_counts=True)
            h0 = _entropy_bits_from_counts(unigram_counts)
            use_gpu = False
    else:
        _, unigram_counts = np.unique(arr, return_counts=True)
        h0 = _entropy_bits_from_counts(unigram_counts)
    metrics["tp_entropy_0"] = h0

    entropies = [h0]
    orders = [0]

    for k in range(1, max_order + 1):
        if use_gpu:
            try:
                hk = _conditional_token_entropy_torch(arr, k, device=device)
            except torch.cuda.OutOfMemoryError:
                if not gpu_fallback:
                    raise
                torch.cuda.empty_cache()
                hk = _conditional_token_entropy(arr, k)
            except Exception:
                if not gpu_fallback:
                    raise
                hk = _conditional_token_entropy(arr, k)
        else:
            hk = _conditional_token_entropy(arr, k)
        metrics[f"tp_cond_entropy_{k}"] = hk

        if np.isfinite(h0) and h0 > 0 and np.isfinite(hk):
            metrics[f"tp_redundancy_{k}"] = float(1.0 - (hk / h0))
            metrics[f"tp_entropy_gain_{k}"] = float(h0 - hk)
            entropies.append(hk)
            orders.append(k)
        else:
            metrics[f"tp_redundancy_{k}"] = float("nan")
            metrics[f"tp_entropy_gain_{k}"] = float("nan")

    if len(orders) >= 2:
        metrics["tp_entropy_slope"] = float(np.polyfit(orders, entropies, 1)[0])
    else:
        metrics["tp_entropy_slope"] = float("nan")

    hmax = metrics.get(f"tp_cond_entropy_{max_order}", float("nan"))
    if np.isfinite(h0) and h0 > 0 and np.isfinite(hmax):
        metrics[f"tp_entropy_drop_{max_order}"] = float(h0 - hmax)
        metrics[f"tp_context_efficiency_{max_order}"] = float((h0 - hmax) / h0)
    else:
        metrics[f"tp_entropy_drop_{max_order}"] = float("nan")
        metrics[f"tp_context_efficiency_{max_order}"] = float("nan")

    return metrics


def _flatten_chunks(chunks):
    arrays = [np.asarray(chunk, dtype=np.int64) for chunk in chunks if len(chunk) > 0]
    if not arrays:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    lengths = np.asarray([arr.size for arr in arrays], dtype=np.int64)
    tokens = np.concatenate(arrays)
    boundaries = np.cumsum(lengths)[:-1]
    return tokens, boundaries


def _resolve_tokens_and_boundaries(tokens=None, chunks=None, boundaries=None):
    if chunks is not None:
        return _flatten_chunks(chunks)

    arr = np.asarray(tokens if tokens is not None else [], dtype=np.int64)
    if boundaries is None:
        return arr, np.array([], dtype=np.int64)

    boundaries = np.asarray(boundaries, dtype=np.int64)
    boundaries = boundaries[(boundaries > 0) & (boundaries < arr.size)]
    boundaries = np.unique(boundaries)
    return arr, boundaries


def _chunk_ranges(n_tokens, boundaries):
    boundaries = np.asarray(boundaries, dtype=np.int64)
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [n_tokens]])
    return list(zip(starts.astype(int), ends.astype(int)))


def _fk_curve(tokens, max_order=4):
    arr = np.asarray(tokens, dtype=np.int64)
    f_vals = {}
    if arr.size == 0:
        for k in range(max_order + 1):
            f_vals[k] = float("nan")
        return f_vals

    _, counts = np.unique(arr, return_counts=True)
    f_vals[0] = _entropy_bits_from_counts(counts)
    for k in range(1, max_order + 1):
        f_vals[k] = _conditional_token_entropy(arr, k)
    return f_vals


def token_context_rank_metrics(tokens, order=2):
    """
    Empirical true-token rank under an observed n-gram continuation table.

    For each context, continuations are ranked by empirical count. The metric
    answers whether the actual continuation usually belongs to the high-count
    continuation set for its local token context.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    metrics = {
        "tc_true_context_rank_order": int(order),
        "tc_true_context_rank_mean": float("nan"),
        "tc_true_context_rank_median": float("nan"),
        "tc_true_context_mrr": float("nan"),
        "tc_true_context_top1_rate": float("nan"),
        "tc_context_branching_mean": float("nan"),
    }

    if order < 1 or arr.size <= order:
        return metrics

    windows = np.lib.stride_tricks.sliding_window_view(arr, order + 1)
    unique_windows, counts = np.unique(windows, axis=0, return_counts=True)

    by_context = {}
    for row, count in zip(unique_windows, counts):
        context = tuple(int(x) for x in row[:-1])
        target = int(row[-1])
        by_context.setdefault(context, []).append((target, int(count)))

    rank_weights = []
    rank_sum = 0.0
    reciprocal_rank_sum = 0.0
    weighted_top1 = 0
    total = 0
    branching = []

    for continuations in by_context.values():
        counts_for_context = np.asarray([c for _, c in continuations], dtype=np.int64)
        branching.append(counts_for_context.size)
        max_count = int(counts_for_context.max())

        for _, count in continuations:
            rank = 1 + int(np.sum(counts_for_context > count))
            rank_weights.append((rank, count))
            rank_sum += rank * count
            reciprocal_rank_sum += (1.0 / rank) * count
            if count == max_count:
                weighted_top1 += count
            total += count

    if total == 0:
        return metrics

    rank_weights.sort(key=lambda item: item[0])
    cumulative = 0
    median_rank = float("nan")
    midpoint = total / 2
    for rank, weight in rank_weights:
        cumulative += weight
        if cumulative >= midpoint:
            median_rank = float(rank)
            break

    metrics["tc_true_context_rank_mean"] = float(rank_sum / total)
    metrics["tc_true_context_rank_median"] = median_rank
    metrics["tc_true_context_mrr"] = float(reciprocal_rank_sum / total)
    metrics["tc_true_context_top1_rate"] = float(weighted_top1 / total)
    metrics["tc_context_branching_mean"] = float(np.mean(branching)) if branching else float("nan")
    return metrics


def token_context_dynamics_metrics(
    tokens=None,
    chunks=None,
    boundaries=None,
    max_order=4,
    boundary_window=64,
    n_checkpoints=24,
    rank_order=2,
):
    """
    TokenContextDynamics family.

    Chunk-aware retrospective metrics:
      - F_k / redundancy curve on the full sequence.
      - within_chunk_F_k: mean/std of chunk-local entropy estimates.
      - cross_boundary_F_k: entropy on tokens immediately after chunk boundaries.
      - entropy_rate_change_over_run: slope of cumulative F_k over checkpoints.
      - empirical context rank and boundary surprisal jump.

    `boundaries` are token indices where chunks start, excluding 0.
    """
    arr, boundaries = _resolve_tokens_and_boundaries(tokens=tokens, chunks=chunks, boundaries=boundaries)
    metrics = {}

    f_vals = _fk_curve(arr, max_order=max_order)
    f0 = f_vals.get(0, float("nan"))

    for k in range(max_order + 1):
        fk = f_vals.get(k, float("nan"))
        metrics[f"tc_F_{k}"] = fk
        metrics[f"tc_redundancy_{k}"] = (
            float(1.0 - fk / f0)
            if k > 0 and np.isfinite(f0) and f0 > 0 and np.isfinite(fk)
            else (0.0 if k == 0 and np.isfinite(f0) else float("nan"))
        )

    fmax = f_vals.get(max_order, float("nan"))
    f1 = f_vals.get(1, float("nan"))
    metrics["tc_context_gain"] = (
        float(f0 - fmax) if np.isfinite(f0) and np.isfinite(fmax) else float("nan")
    )
    metrics["tc_long_range_context_gain"] = (
        float(f1 - fmax) if np.isfinite(f1) and np.isfinite(fmax) else float("nan")
    )

    chunk_ranges = _chunk_ranges(arr.size, boundaries) if boundaries.size else []
    for k in range(max_order + 1):
        local_vals = []
        for start, end in chunk_ranges:
            chunk = arr[start:end]
            if chunk.size <= k:
                continue
            local_vals.append(_fk_curve(chunk, max_order=k)[k])

        if local_vals:
            local_vals = np.asarray(local_vals, dtype=np.float64)
            metrics[f"tc_within_chunk_F_{k}_mean"] = float(np.nanmean(local_vals))
            metrics[f"tc_within_chunk_F_{k}_std"] = float(np.nanstd(local_vals))
        else:
            metrics[f"tc_within_chunk_F_{k}_mean"] = float("nan")
            metrics[f"tc_within_chunk_F_{k}_std"] = float("nan")

    if boundaries.size:
        boundary_positions = []
        for boundary in boundaries:
            end = min(arr.size, int(boundary) + int(boundary_window))
            boundary_positions.extend(range(int(boundary), end))
        boundary_positions = np.asarray(boundary_positions, dtype=np.int64)
    else:
        boundary_positions = np.array([], dtype=np.int64)

    for k in range(max_order + 1):
        metrics[f"tc_cross_boundary_F_{k}"] = _conditional_token_entropy_at_positions(
            arr, boundary_positions, k
        )

    boundary_fmax = metrics.get(f"tc_cross_boundary_F_{max_order}", float("nan"))
    within_fmax = metrics.get(f"tc_within_chunk_F_{max_order}_mean", float("nan"))
    metrics["tc_chunk_boundary_surprisal_jump"] = (
        float(boundary_fmax - within_fmax)
        if np.isfinite(boundary_fmax) and np.isfinite(within_fmax)
        else float("nan")
    )

    if arr.size:
        if boundaries.size:
            checkpoint_pool = np.concatenate([boundaries, [arr.size]])
        else:
            checkpoint_pool = np.unique(
                np.round(np.linspace(max(2, max_order + 1), arr.size, n_checkpoints)).astype(np.int64)
            )
        if checkpoint_pool.size > n_checkpoints:
            idx = np.unique(np.round(np.linspace(0, checkpoint_pool.size - 1, n_checkpoints)).astype(np.int64))
            checkpoint_pool = checkpoint_pool[idx]
        checkpoint_pool = checkpoint_pool[checkpoint_pool > max_order + 1]
    else:
        checkpoint_pool = np.array([], dtype=np.int64)

    x = checkpoint_pool / arr.size if arr.size and checkpoint_pool.size else np.array([])
    for k in range(max_order + 1):
        y = []
        for checkpoint in checkpoint_pool:
            y.append(_fk_curve(arr[:int(checkpoint)], max_order=k)[k])
        if len(y) >= 3:
            slope, _, r2 = _linear_fit_r2(x, y)
            metrics[f"tc_entropy_rate_change_F_{k}_slope"] = slope
            metrics[f"tc_entropy_rate_change_F_{k}_r2"] = r2
        else:
            metrics[f"tc_entropy_rate_change_F_{k}_slope"] = float("nan")
            metrics[f"tc_entropy_rate_change_F_{k}_r2"] = float("nan")

    metrics.update(token_context_rank_metrics(arr, order=rank_order))
    metrics["tc_boundary_window"] = int(boundary_window)
    metrics["tc_n_boundaries"] = int(boundaries.size)
    return metrics


def _heaps_profile(tokens, n_points=32, min_tokens=1_000):
    """Vocabulary size V(N) at log-spaced checkpoints."""
    arr = np.asarray(tokens, dtype=np.int64)
    n = arr.size
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    start = min(min_tokens, n)
    checkpoints = np.unique(
        np.round(np.logspace(np.log10(max(1, start)), np.log10(n), n_points)).astype(np.int64)
    )
    checkpoints = checkpoints[(checkpoints >= 1) & (checkpoints <= n)]
    if checkpoints.size == 0 or checkpoints[-1] != n:
        checkpoints = np.unique(np.append(checkpoints, n))

    seen = set()
    vocab_sizes = []
    checkpoint_idx = 0
    next_checkpoint = int(checkpoints[checkpoint_idx])

    for i, token in enumerate(arr, start=1):
        seen.add(int(token))
        while i == next_checkpoint:
            vocab_sizes.append(len(seen))
            checkpoint_idx += 1
            if checkpoint_idx >= checkpoints.size:
                return checkpoints, np.asarray(vocab_sizes, dtype=np.int64)
            next_checkpoint = int(checkpoints[checkpoint_idx])

    return checkpoints[:len(vocab_sizes)], np.asarray(vocab_sizes, dtype=np.int64)


def token_distribution_scaling_metrics(tokens, heaps_points=32, zipf_min_count=2):
    """
    TokenDistributionScaling family.

    Measures rank-frequency and vocabulary-growth structure with token IDs as
    symbols. These are token analogues of Zipf's law and Heaps' law, so they
    do not depend on words, English morphology, or semantic labels.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    metrics = {}
    n = arr.size

    if n == 0:
        for key in (
            "td_vocab_size",
            "td_type_token_ratio",
            "td_hapax_type_fraction",
            "td_hapax_token_fraction",
            "td_top1_token_fraction",
            "td_top10_token_fraction",
            "td_zipf_slope",
            "td_zipf_exponent",
            "td_zipf_r2",
            "td_heaps_exponent",
            "td_heaps_r2",
        ):
            metrics[key] = float("nan")
        return metrics

    _, counts = np.unique(arr, return_counts=True)
    counts = counts.astype(np.float64)
    counts_desc = np.sort(counts)[::-1]
    n_types = counts_desc.size

    metrics["td_vocab_size"] = int(n_types)
    metrics["td_type_token_ratio"] = float(n_types / n)
    metrics["td_hapax_type_fraction"] = float(np.sum(counts_desc == 1) / n_types)
    metrics["td_hapax_token_fraction"] = float(np.sum(counts_desc == 1) / n)
    metrics["td_top1_token_fraction"] = float(counts_desc[0] / n)
    metrics["td_top10_token_fraction"] = float(counts_desc[:10].sum() / n)

    ranks = np.arange(1, n_types + 1, dtype=np.float64)
    zipf_mask = counts_desc >= zipf_min_count
    if np.sum(zipf_mask) >= 3:
        slope, _, r2 = _linear_fit_r2(np.log10(ranks[zipf_mask]), np.log10(counts_desc[zipf_mask]))
        metrics["td_zipf_slope"] = slope
        metrics["td_zipf_exponent"] = float(-slope) if np.isfinite(slope) else float("nan")
        metrics["td_zipf_r2"] = r2
    else:
        metrics["td_zipf_slope"] = float("nan")
        metrics["td_zipf_exponent"] = float("nan")
        metrics["td_zipf_r2"] = float("nan")

    checkpoints, vocab_sizes = _heaps_profile(arr, n_points=heaps_points)
    heaps_valid = (checkpoints > 1) & (vocab_sizes > 1)
    if np.sum(heaps_valid) >= 3:
        slope, _, r2 = _linear_fit_r2(np.log10(checkpoints[heaps_valid]), np.log10(vocab_sizes[heaps_valid]))
        metrics["td_heaps_exponent"] = slope
        metrics["td_heaps_r2"] = r2
    else:
        metrics["td_heaps_exponent"] = float("nan")
        metrics["td_heaps_r2"] = float("nan")

    return metrics


def token_recurrence_metrics(tokens):
    """
    TokenRecurrence family.

    Tracks distances between repeated token IDs. This is a token-level version
    of inter-event-time burstiness: coherent/natural sequences tend to reuse
    symbols in clustered, non-Poisson ways, while resets or bag-like generation
    can alter the repeat-gap distribution.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    metrics = {}
    n = arr.size

    empty_keys = (
        "tr_repeat_rate",
        "tr_repeat_gap_mean",
        "tr_repeat_gap_mean_log2",
        "tr_repeat_gap_median_log2",
        "tr_repeat_gap_cv",
        "tr_repeat_gap_burstiness",
        "tr_repeat_gap_logbin_entropy",
    )

    if n == 0:
        for key in empty_keys:
            metrics[key] = float("nan")
        return metrics

    last_pos = {}
    gaps = []
    for i, token in enumerate(arr):
        token = int(token)
        previous = last_pos.get(token)
        if previous is not None:
            gaps.append(i - previous)
        last_pos[token] = i

    if not gaps:
        metrics["tr_repeat_rate"] = 0.0
        for key in empty_keys[1:]:
            metrics[key] = float("nan")
        return metrics

    gaps = np.asarray(gaps, dtype=np.float64)
    gap_mean = float(gaps.mean())
    gap_std = float(gaps.std())
    log_gaps = np.log2(gaps)
    log_bins = np.floor(log_gaps).astype(np.int64)
    _, logbin_counts = np.unique(log_bins, return_counts=True)

    metrics["tr_repeat_rate"] = float(gaps.size / n)
    metrics["tr_repeat_gap_mean"] = gap_mean
    metrics["tr_repeat_gap_mean_log2"] = float(log_gaps.mean())
    metrics["tr_repeat_gap_median_log2"] = float(np.median(log_gaps))
    metrics["tr_repeat_gap_cv"] = float(gap_std / gap_mean) if gap_mean > 0 else float("nan")
    metrics["tr_repeat_gap_burstiness"] = (
        float((gap_std - gap_mean) / (gap_std + gap_mean))
        if (gap_std + gap_mean) > 0
        else float("nan")
    )
    metrics["tr_repeat_gap_logbin_entropy"] = _entropy_bits_from_counts(logbin_counts)

    return metrics


@njit(fastmath=True)
def _lz_complexity_core(s, start_ptr=0, start_c=1):
    """
    Lempel-Ziv complexity (LZ76) using the Kaspar & Schuster (1987) algorithm.
    Optimized for incremental updates with correct 'l' pointer persistence.
    """
    n = len(s)
    if n == 0:
        return 0, 0
    
    l = 1
    c = 1
    
    # Resume Logic
    if start_ptr > 0:
        l = start_ptr
        c = start_c
        if l > n:
            return c, l

    i = 0
    k = 1
    k_max = 1
    
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k = k + 1
            if l + k > n:
                # End of string while inside a match -> Component incomplete.
                # Don't increment c yet.
                break
        else:
            if k > k_max:
                k_max = k
            i = i + 1
            if i == l:
                # Found new unique component
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c, l


def lz_complexity(tokens, lz_state=None):
    """
    Lempel–Ziv complexity (Numba CPU).
    
    Args:
        tokens: full list of tokens
        lz_state: dict with {'l_ptr': int, 'c': int} from previous step
    
    Returns:
        (normalized_complexity, debug_dict)
             debug_dict: {'c': raw_components, 'l_ptr': start_pointer}
    """
    arr = np.array(tokens, dtype=np.int32)
    n = len(arr)
    if n == 0:
        return 0.0, {'c': 0, 'l_ptr': 0}
        
    start_ptr = 0
    start_c = 1
    
    # Robustly handle different state formats if upgrading
    if lz_state:
        start_ptr = lz_state.get('l_ptr', lz_state.get('n', 0)) # Fallback to 'n' if l_ptr missing
        start_c = lz_state.get('c', 1)
        # Safety
        if start_ptr >= n:
             start_ptr = 0
             start_c = 1
             
    c, final_l = _lz_complexity_core(arr, start_ptr, start_c)
    
    # We display c + 1 (assuming the current hanging bit is a component for metric purposes)
    display_c = c + 1
    
    norm_c = float(display_c / (n / math.log2(n + 1)))
    return norm_c, {'c': c, 'l_ptr': final_l}


def mutual_information_decay(tokens, taus=None, tau_max=500, n_lags=48,
                             bins=4096, batch_size=32, device="cuda"):
    """
    Mutual Information decay exponent γ using batched GPU histogram estimation.

    Parameters
    ----------
    tokens : list[int] | np.ndarray | torch.Tensor
        Discrete sequence (token IDs or encoded text)
    taus : np.ndarray | None
        Specific τ values to compute (optional). If None, generated via log grid.
    tau_max : int
        Maximum τ (used only if taus=None)
    n_lags : int
        Number of lag points to sample (log-spaced if taus=None)
    bins : int
        Number of histogram bins (vocabulary coarsening)
    batch_size : int
        Number of τ values per GPU batch
    device : str
        'cuda' or 'cpu' (GPU recommended)

    Returns
    -------
    gamma : float
        Power-law decay exponent of MI(τ)
    tau_info : dict
        Metadata about τ grid used
    """
    with torch.no_grad():
        x = torch.as_tensor(tokens, dtype=torch.int64, device=device)

        # --------------- τ grid construction ---------------
        if taus is None:
            taus = make_tau_grid(len(x), tau_max=tau_max, n_lags=n_lags)
        else:
            taus = np.asarray(taus)

        MI_vals = []
        for t0 in range(0, len(taus), batch_size):
            batch = taus[t0:t0 + batch_size]
            for tau in batch:
                tau = int(tau)
                a = x[:-tau]
                b = x[tau:]
                H = _torch_histogram2d(a, b, bins=bins, device=device).float()
                Pxy = H / H.sum()
                Px = Pxy.sum(dim=1, keepdim=True)
                Py = Pxy.sum(dim=0, keepdim=True)
                mask = Pxy > 0
                mi = torch.sum(Pxy[mask] * torch.log2(Pxy[mask] / (Px @ Py)[mask]))
                MI_vals.append(mi.item())

            torch.cuda.empty_cache()

        MI = np.array(MI_vals)
        valid = MI > 0
        if valid.sum() < 3:
            return float("nan"), {"mi_tau_max": int(taus.max()), "mi_n_lags": len(taus)}

        slope, _ = np.polyfit(np.log10(taus[valid]), np.log10(MI[valid]), 1)
        gamma = float(-slope)
        
        tau_info = {"mi_tau_max": int(taus.max()), "mi_n_lags": len(taus)}
        return gamma, tau_info

def diffusion_entropy_analysis(tokens, min_win=1, max_win=None, n_scales=20, bins=100, device="cpu"):
    """
    Diffusion Entropy Analysis (DEA) to estimate scaling exponent δ.
    
    Logic:
      1. Construct diffusion trajectory: y(t) = cumsum(tokens).
      2. For window L: displacements D = y(t+L) - y(t).
      3. Compute entropy S(L) of the distribution of D.
      4. Fit S(L) = A + δ * ln(L).
    
    Returns:
      delta (float): Scaling exponent.
      debug (dict): Intermediate values for plotting.
    """
    # 1. Trajectory
    x = np.array(tokens, dtype=np.float64)
    # Center the tokens to avoid massive drift if mean != 0 (optional, but standard for DEA on fluctuations)
    x = x - np.mean(x)
    y = np.cumsum(x)
    N = len(y)
    
    if max_win is None:
        max_win = N // 4
    
    # 2. Scales (log-spaced)
    scales = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), n_scales).astype(int))
    scales = scales[scales > 0]
    scales = scales[scales < N]
    
    S_vals = []
    L_vals = []
    
    for L in scales:
        # Displacements
        # D[i] = y[i+L] - y[i]
        # We can do this efficiently with slicing
        D = y[L:] - y[:-L]
        
        # Histogram
        # We need to handle bin width carefully. 
        # If we use fixed number of bins, width w scales with range.
        # S_discrete = S_continuous - ln(w).
        # To recover S_continuous scaling, we must add ln(w).
        
        # Use numpy histogram
        counts, bin_edges = np.histogram(D, bins=bins, density=True)
        
        # Calculate bin width (assuming uniform bins)
        if len(bin_edges) > 1:
            w = bin_edges[1] - bin_edges[0]
        else:
            w = 1.0
            
        # Entropy (Shannon)
        # Filter zeros
        p = counts[counts > 0]
        # Normalize to sum to 1 just in case density=True didn't handle discrete perfectly
        p = p / p.sum()
        
        # S = - sum p ln p
        S_discrete = -np.sum(p * np.log(p))
        
        # Correct for bin width to estimate differential entropy
        S = S_discrete + np.log(w)
        
        S_vals.append(S)
        L_vals.append(L)
        
    # 3. Fit S(L) ~ delta * ln(L)
    if len(L_vals) < 3:
        return float('nan'), {}
        
    ln_L = np.log(L_vals)
    S_arr = np.array(S_vals)
    
    slope, intercept = np.polyfit(ln_L, S_arr, 1)
    delta = slope
    
    return delta, {"scales": L_vals, "entropy": S_vals}

def detrended_moving_average(tokens, min_win=2, max_win=None, n_scales=20, device="cpu"):
    """
    Detrending Moving Average (DMA) - Centered (CDMA) variant.
    
    Logic:
      1. Integrate: y(t) = cumsum(x - mean).
      2. For window L:
         a. Compute Moving Average y_tilde(t) of y(t) with window L.
         b. Detrend: eps(t) = y(t) - y_tilde(t).
         c. Fluctuation F(L) = sqrt( < eps(t)^2 > ).
      3. Fit F(L) ~ L^H.
      
    Returns:
      H (float): Hurst exponent from DMA.
      debug (dict): Intermediate values.
    """
    x = np.array(tokens, dtype=np.float64)
    x = x - np.mean(x)
    y = np.cumsum(x)
    N = len(y)
    
    if max_win is None:
        max_win = N // 4
        
    # Scales (log-spaced)
    # For CDMA, window must be odd usually, or handled carefully. 
    # We'll force odd windows for symmetry.
    scales = np.unique(np.logspace(np.log10(min_win), np.log10(max_win), n_scales).astype(int))
    scales = scales[scales % 2 == 0] + 1 # Make odd
    scales = scales[scales < N]
    scales = np.unique(scales) # Re-unique after +1
    
    F_vals = []
    L_vals = []
    
    for L in scales:
        # Moving Average (Centered)
        # We can use uniform_filter1d or convolution
        # uniform_filter1d handles boundaries by reflecting/constant, or we can just slice valid region.
        # For strict DMA, we usually discard boundaries where window doesn't fit.
        
        # Kernel
        w = np.ones(L) / L
        
        # Valid convolution: output size N - L + 1
        # This corresponds to the center of the window.
        # y_tilde starts at index L//2 of the original y.
        y_tilde = np.convolve(y, w, mode='valid')
        
        # Match y to y_tilde
        # y_tilde[0] corresponds to average of y[0:L]. Center is at index L//2.
        offset = L // 2
        y_segment = y[offset : offset + len(y_tilde)]
        
        # Detrend
        epsilon = y_segment - y_tilde
        
        # Fluctuation (RMS)
        F = np.sqrt(np.mean(epsilon**2))
        
        F_vals.append(F)
        L_vals.append(L)
        
    if len(L_vals) < 3:
        return float('nan'), {}
        
    # Fit log-log
    ln_L = np.log(L_vals)
    ln_F = np.log(F_vals)
    
    slope, intercept = np.polyfit(ln_L, ln_F, 1)
    H_dma = slope
    
    return H_dma, {"scales": L_vals, "fluctuation": F_vals}

def wavelet_dfa(tokens, wavelet_name='db4', device="cpu", crossover_scale=100):
    """
    CPU Wavelet Detrended Fluctuation Analysis (WDFA).

    This uses the same custom wavelet cascade as the established GPU path so
    CPU fallback values are numerically consistent with historical results.
    """
    return _wavelet_dfa_torch_impl(
        tokens,
        wavelet_name=wavelet_name,
        device=device,
        crossover_scale=crossover_scale,
    )


def _wavelet_dfa_torch_impl(tokens, wavelet_name='db4', device="cuda", crossover_scale=100):
    """Shared implementation for the established WDFA GPU path and CPU fallback."""
    x = torch.as_tensor(tokens, dtype=torch.float32, device=device)
    x = x - x.mean()
    y = torch.cumsum(x, dim=0)

    # Keep this filter configuration unchanged: it defines the historical metric.
    if wavelet_name == 'db4':
        dec_lo = torch.tensor(
            [0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604],
            device=device,
        )
        dec_hi = torch.tensor(
            [-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341],
            device=device,
        )
    else:
        dec_lo = torch.tensor([0.70710678, 0.70710678], device=device)
        dec_hi = torch.tensor([-0.70710678, 0.70710678], device=device)

    # PyTorch convolution is correlation, so reverse the filters first.
    dec_lo = torch.flip(dec_lo, [0]).view(1, 1, -1)
    dec_hi = torch.flip(dec_hi, [0]).view(1, 1, -1)
    approx = y.view(1, 1, -1)

    fluctuations = []
    scales = []
    max_levels = int(torch.log2(torch.tensor(len(tokens))).item()) - 3
    padding = dec_lo.shape[-1] - 1

    for level in range(max_levels):
        approx_padded = torch.nn.functional.pad(approx, (padding, padding), mode='reflect')
        detail = torch.nn.functional.conv1d(approx_padded, dec_hi, stride=2)
        approx = torch.nn.functional.conv1d(approx_padded, dec_lo, stride=2)

        fluctuation = torch.sqrt(torch.mean(detail**2))
        if not torch.isnan(fluctuation) and fluctuation > 0:
            fluctuations.append(fluctuation.item())
            scales.append(2 ** (level + 1))

    scales = np.array(scales)
    fluctuation_values = np.array(fluctuations)
    valid = fluctuation_values > 0
    if valid.sum() < 3:
        return float('nan'), float('nan'), float('nan'), {}

    ln_s = np.log2(scales[valid])
    ln_f = np.log2(fluctuation_values[valid])
    slope, _ = np.polyfit(ln_s, ln_f, 1)
    h_global = slope - 0.5

    short_mask = scales[valid] < crossover_scale
    long_mask = scales[valid] >= crossover_scale
    h_short = float('nan')
    h_long = float('nan')

    if short_mask.sum() >= 3:
        short_slope, _ = np.polyfit(ln_s[short_mask], ln_f[short_mask], 1)
        h_short = short_slope - 0.5

    if long_mask.sum() >= 3:
        long_slope, _ = np.polyfit(ln_s[long_mask], ln_f[long_mask], 1)
        h_long = long_slope - 0.5

    return h_global, h_short, h_long, {"scales": scales, "fluctuation": fluctuation_values}

    


# ===============================================================
#  MAIN CLASS
# ===============================================================
import time

def _dev_label(device: str) -> str:
    if device.startswith("cuda"):
        # show cuda:0 → Cuda0, or cuda → Cuda0
        if ':' in device:
            return f"Cuda{device.split(':')[1]}"
        return "Cuda0"
    return "Cpu"

class CoherenceMetrics:
    """Hybrid GPU/CPU Coherence Metrics with per-metric logging."""

    FAMILY_COMPUTE_SPEED = METRIC_FAMILY_COMPUTE_SPEED

    @classmethod
    def family_compute_speed(cls):
        """Overall compute speed/cost by metric family."""
        return dict(cls.FAMILY_COMPUTE_SPEED)

    # ------------------------------------------------------------------
    # 3. Wavelet DFA (PyTorch GPU)
    # ------------------------------------------------------------------
    @staticmethod
    def wavelet_dfa_torch(tokens, wavelet_name='db4', device="cuda", crossover_scale=100):
        """
        GPU-accelerated WDFA using generic DWT implementation.
        """
        return _wavelet_dfa_torch_impl(
            tokens,
            wavelet_name=wavelet_name,
            device=device,
            crossover_scale=crossover_scale,
        )

    # ------------------------------------------------------------------
    # 4. DMA (PyTorch GPU)
    # ------------------------------------------------------------------
    @staticmethod
    def dma_torch(tokens, min_win=2, max_win=None, n_scales=20, device="cuda"):
        """
        GPU-accelerated Detrended Moving Average.
        Uses conv1d for boxcar average.
        """
        x = torch.as_tensor(tokens, dtype=torch.float32, device=device)
        x = x - x.mean()
        y = torch.cumsum(x, dim=0)
        N = len(y)
        
        if max_win is None:
            max_win = N // 4
            
        # Log scales
        scales = torch.logspace(np.log10(min_win), np.log10(max_win), n_scales, device=device).long()
        scales = scales[scales % 2 == 0] + 1 # Make odd
        scales = torch.unique(scales)
        scales = scales[scales < N]
        
        F_vals = []
        L_vals = []
        
        y_view = y.view(1, 1, -1)
        
        for L in scales:
            L_int = int(L.item())
            # Kernel: Boxcar
            weight = torch.ones(1, 1, L_int, device=device) / L_int
            
            # Conv1d 'valid'
            # Start index: 0 corresponding to L//2 in original
            y_tilde = torch.nn.functional.conv1d(y_view, weight) 
            y_tilde = y_tilde.squeeze()
            
            # Match
            # y_tilde has length N - L + 1
            # Matches y[L//2 : L//2 + len]
            offset = L_int // 2
            length = y_tilde.shape[0]
            
            y_seg = y[offset : offset + length]
            
            eps = y_seg - y_tilde
            F = torch.sqrt(torch.mean(eps**2))
            
            F_vals.append(F.item())
            L_vals.append(L_int)
            
        if len(L_vals) < 3:
            return float('nan'), {}
            
        ln_L = np.log(L_vals)
        ln_F = np.log(F_vals)
        
        slope, _ = np.polyfit(ln_L, ln_F, 1)
        return slope, {"scales": L_vals, "fluctuation": F_vals}

    def __init__(self, tokens, device=None, chunk_boundaries=None):
        """
        Initialize with a list of tokens/integers.
        """
        self.tokens = tokens
        self.chunk_boundaries = chunk_boundaries
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n = len(tokens)
        self.results = {}

    # ------------------------- Structural -------------------------
    def compute_structural_scaling(self):

        # 1) β
        t0 = time.perf_counter()
        beta = autocorrelation_decay(self.tokens, device=self.device)
        t1 = time.perf_counter()


        # 2) α
        t0 = time.perf_counter()
        alpha = spectral_exponent(self.tokens, device=self.device)
        t1 = time.perf_counter()


        # 3) H (DFA1)
        t0 = time.perf_counter()
        H1 = hurst_dfa(self.tokens, device=self.device, order=1)
        t1 = time.perf_counter()


        # 4) D (derived from H1 for consistency, or we can output multiple Ds)
        # Let's keep D based on DFA1 for backward compatibility
        D = fractal_dimension(H1)


        # 5) DEA (delta)
        t0 = time.perf_counter()
        delta, _ = diffusion_entropy_analysis(self.tokens, device=self.device)
        t1 = time.perf_counter()

        
        # 6) DMA (H_DMA)
        t0 = time.perf_counter()
        try:
            H_dma, _ = self.dma_torch(self.tokens, device=self.device)
            dev_lbl_dma = _dev_label(self.device)
        except Exception as e:
            # print(f"DMA GPU failed: {e}")
            H_dma, _ = detrended_moving_average(self.tokens, device="cpu")
            dev_lbl_dma = "Cpu"
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
            print(f"   ⏱️  DMA took {t1 - t0:.2f}s on {dev_lbl_dma}")

        # 7) WDFA (Wavelet DFA, db4 only)
        t0 = time.perf_counter()
        try:
            H_db4_global, _, _, _ = self.wavelet_dfa_torch(self.tokens, wavelet_name='db4', device=self.device)
            dev_lbl_wdfa = _dev_label(self.device)
        except Exception as e:
            print(f"   ⚠️ WDFA GPU failed: {e}")
            H_db4_global, _, _, _ = wavelet_dfa(self.tokens, wavelet_name='db4', device="cpu")
            dev_lbl_wdfa = "Cpu"
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
             print(f"   ⏱️  WDFA took {t1 - t0:.2f}s on {dev_lbl_wdfa}")

        self.results.update({
            "beta": beta, 
            "alpha": alpha, 
            "H_DFA1": H1, 
            "H_DMA": H_dma,
            "H_WT_db4": H_db4_global,
            "D": D, 
            "delta": delta
        })
        return self.results

    # ------------------------- Informational -------------------------
    def compute_informational_structure(self):


        # 1) Entropy & Perplexity — try GPU first, fallback to CPU
        t0 = time.perf_counter()
        try:
            Hs, PP = shannon_entropy(self.tokens, device=self.device)  # GPU path
            dev_lbl = _dev_label(self.device)
        except Exception:
            Hs, PP = shannon_entropy(self.tokens, device="cpu")
            dev_lbl = "Cpu"
        t1 = time.perf_counter()

        # store both
        self.results.update({"entropy_bits": float(Hs), "perplexity": float(PP)})

        # 2) LZC — CPU (sequential)
        t0 = time.perf_counter()
        # lz_state is passed via compute_all -> compute_informational_structure? 
        # Actually I need to update compute_all to accept kwargs and pass them.
        # But CoherenceMetrics is initialized with tokens.
        # I will store lz_state in the instance if needed, or pass it here.
        # Let's check compute_all...
        
        # Hotfix: modifying the class to support a state injection method isn't clean unless I check `self`.
        # I'll rely on an optional attribute `self.lz_state` set by the caller before compute_all?
        # Or I add it to __init__.
        
        lz_st = getattr(self, "lz_state", None)
        LZC, lz_s_out = lz_complexity(self.tokens, lz_state=lz_st)  # Incremental if state present
        
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
             print(f"   ⏱️  LZC took {t1 - t0:.2f}s on Cpu")
        self.results.update({
            "lz_complexity": float(LZC),
            "lz_complexity_count": int(lz_s_out['c']), # For state tracking
            "lz_complexity_l": int(lz_s_out['l_ptr'])
        })

        # 3) MI decay — try GPU first, fallback to CPU
        t0 = time.perf_counter()
        n_tokens = len(self.tokens)
        taus = make_tau_grid(n_tokens, tau_max=choose_tau_max(n_tokens, ratio=0.3, cap=50_000), n_lags=48)

        try:
            gamma, tau_info = mutual_information_decay(self.tokens, taus=taus, device=self.device)  # your GPU impl
            dev_lbl = _dev_label(self.device)
        except torch.cuda.OutOfMemoryError:
            # graceful fallback on OOM
            gamma, tau_info = mutual_information_decay(self.tokens, taus=taus, device="cpu")
            dev_lbl = "Cpu"
        except Exception:
            # any other unexpected issue → CPU fallback
            gamma, tau_info = mutual_information_decay(self.tokens, taus=taus, device="cpu")
            dev_lbl = "Cpu"
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
             print(f"   ⏱️  MI took {t1 - t0:.2f}s on {dev_lbl}")

        self.results.update({"mi_decay_gamma": gamma, **tau_info})
        return self.results

    # ------------------------- TokenPredictability -------------------------
    def compute_token_predictability(self, max_order=4, device=None):
        t0 = time.perf_counter()
        metrics = token_predictability_metrics(self.tokens, max_order=max_order, device=device or self.device)
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
            print(f"   ⏱️  TokenPredictability took {t1 - t0:.2f}s on {_dev_label(str(device or self.device))}")
        self.results.update(metrics)
        return self.results

    # ------------------------- TokenDistributionScaling -------------------------
    def compute_token_distribution_scaling(self):
        t0 = time.perf_counter()
        metrics = token_distribution_scaling_metrics(self.tokens)
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
            print(f"   ⏱️  TokenDistributionScaling took {t1 - t0:.2f}s on Cpu")
        self.results.update(metrics)
        return self.results

    # ------------------------- TokenRecurrence -------------------------
    def compute_token_recurrence(self):
        t0 = time.perf_counter()
        metrics = token_recurrence_metrics(self.tokens)
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
            print(f"   ⏱️  TokenRecurrence took {t1 - t0:.2f}s on Cpu")
        self.results.update(metrics)
        return self.results

    # ------------------------- TokenNaturalness -------------------------
    def compute_token_naturalness(self, token_predictability_max_order=4):
        self.compute_token_predictability(max_order=token_predictability_max_order)
        self.compute_token_distribution_scaling()
        self.compute_token_recurrence()
        return self.results

    # ------------------------- TokenContextDynamics -------------------------
    def compute_token_context_dynamics(
        self,
        chunks=None,
        boundaries=None,
        max_order=4,
        boundary_window=64,
        n_checkpoints=24,
        rank_order=2,
    ):
        t0 = time.perf_counter()
        metrics = token_context_dynamics_metrics(
            tokens=self.tokens,
            chunks=chunks,
            boundaries=boundaries if boundaries is not None else self.chunk_boundaries,
            max_order=max_order,
            boundary_window=boundary_window,
            n_checkpoints=n_checkpoints,
            rank_order=rank_order,
        )
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
            print(f"   ⏱️  TokenContextDynamics took {t1 - t0:.2f}s on Cpu")
        self.results.update(metrics)
        return self.results

    # ------------------------- Combined -------------------------
    def compute_all(
        self,
        include_token_predictability=False,
        token_predictability_max_order=4,
        include_token_naturalness=False,
        include_token_context_dynamics=False,
        token_context_boundaries=None,
    ):
        self.compute_structural_scaling()
        self.compute_informational_structure()
        if include_token_naturalness:
            self.compute_token_naturalness(token_predictability_max_order=token_predictability_max_order)
        elif include_token_predictability:
            self.compute_token_predictability(max_order=token_predictability_max_order)
        if include_token_context_dynamics:
            self.compute_token_context_dynamics(
                boundaries=token_context_boundaries,
                max_order=token_predictability_max_order,
            )
        self.results["n_tokens"] = len(self.tokens)
        return self.results
