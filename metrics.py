"""
Unified Coherence Metrics
-------------------------

CoherenceMetrics
├── StructuralScaling
│     ├── autocorrelation_decay (β)
│     ├── spectral_exponent (α)
│     ├── hurst_dfa (H)
│     ├── fractal_dimension (D)
│     └── holder_exponent (optional)
└── InformationalStructure
      ├── shannon_entropy (H₁)
      ├── perplexity (PP)
      ├── entropy_rate (optional)
      ├── lz_complexity (LZC)
      ├── mutual_information_decay (γ)
      ├── conditional_entropy (optional)
      └── cross_entropy / kl_divergence (optional)

Two orthogonal metric families:

1. Structural Scaling
   - Quantifies persistence and self-similarity in token correlations.
   - Core: β (autocorr), α (spectrum), H (DFA), D (fractal dim)

2. Informational Structure
   - Quantifies information organization, predictability, and coupling.
   - Core: Shannon entropy, Perplexity, LZC, Mutual Information decay

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
    Wavelet Detrended Fluctuation Analysis (WDFA).
    
    Uses Discrete Wavelet Transform (DWT) to estimate H.
    The energy of detail coefficients at scale s (level j) scales as:
       E_j ~ s^(2H + 1)  (for integrated signal)
       RMS_j ~ s^(H + 0.5)
       
    Parameters
    ----------
    tokens : list/array
    wavelet_name : str
        Name of wavelet (e.g., 'db4', 'haar')
    crossover_scale : int
        Scale threshold to separate short vs long range scaling
        
    Returns
    -------
    H_global : float
        Hurst exponent from all scales
    H_short : float
        Hurst exponent from short scales (< crossover_scale)
    H_long : float
        Hurst exponent from long scales (>= crossover_scale)
    """
    try:
        import pywt
    except ImportError:
        print("PyWavelets not installed. Returning NaN for WDFA.")
        return float('nan'), float('nan'), float('nan'), {}

    x = np.array(tokens, dtype=np.float64)
    x = x - np.mean(x)
    y = np.cumsum(x) # Integrate
    
    # Decompose
    mode = 'symmetric'
    coeffs = pywt.wavedec(y, wavelet_name, mode=mode)
    
    details = coeffs[1:]
    n_levels = len(details)
    
    F_vals = []
    scales = []
    
    for i, cD in enumerate(details):
        level = n_levels - i
        scale = 2**level
        
        # Energy (RMS)
        F = np.sqrt(np.mean(cD**2))
        
        F_vals.append(F)
        scales.append(scale)
        
    scales = np.array(scales)
    F_vals = np.array(F_vals)
    
    # Filter valid
    valid = F_vals > 0
    if valid.sum() < 3:
        return float('nan'), float('nan'), float('nan'), {}
        
    ln_s = np.log2(scales[valid])
    ln_F = np.log2(F_vals[valid])
    
    # Global fit
    slope_global, _ = np.polyfit(ln_s, ln_F, 1)
    H_global = slope_global - 0.5
    
    # Short scales (< crossover_scale)
    short_mask = (scales[valid] < crossover_scale)
    if short_mask.sum() >= 2:
        slope_short, _ = np.polyfit(ln_s[short_mask], ln_F[short_mask], 1)
        H_short = slope_short - 0.5
    else:
        H_short = float('nan')
    
    # Long scales (>= crossover_scale)
    long_mask = (scales[valid] >= crossover_scale)
    if long_mask.sum() >= 2:
        slope_long, _ = np.polyfit(ln_s[long_mask], ln_F[long_mask], 1)
        H_long = slope_long - 0.5
    else:
        H_long = float('nan')
    
    return H_global, H_short, H_long, {"scales": scales, "fluctuation": F_vals}

    


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

    # ------------------------------------------------------------------
    # 3. Wavelet DFA (PyTorch GPU)
    # ------------------------------------------------------------------
    @staticmethod
    def wavelet_dfa_torch(tokens, wavelet_name='db4', device="cuda", crossover_scale=100):
        """
        GPU-accelerated WDFA using generic DWT implementation.
        """
        x = torch.as_tensor(tokens, dtype=torch.float32, device=device)
        x = x - x.mean()
        y = torch.cumsum(x, dim=0)
        
        # Get filters relative to 'db4' or similar
        # For simplicity, we hardcode db4 here or use a helper to get them
        # Low pass (dec_lo) and High pass (dec_hi) for decomposition
        # db4 coefficients
        if wavelet_name == 'db4':
             # Daubechies 4 decomposition filters
             dec_lo = torch.tensor([0.4829629131445341, 0.8365163037378079, 0.2241438680420134, -0.1294095225512604], device=device)
             dec_hi = torch.tensor([-0.1294095225512604, -0.2241438680420134, 0.8365163037378079, -0.4829629131445341], device=device)
        else:
             # Fallback to Haar if unknown
             dec_lo = torch.tensor([0.70710678, 0.70710678], device=device)
             dec_hi = torch.tensor([-0.70710678, 0.70710678], device=device)

        # Flip for convolution (pytorch conv is correlation)
        dec_lo = torch.flip(dec_lo, [0]).view(1, 1, -1)
        dec_hi = torch.flip(dec_hi, [0]).view(1, 1, -1)
        
        approx = y.view(1, 1, -1)
        
        fluctuations = []
        scales = []
        
        # Cascade
        curr_scale = 1
        max_levels = int(torch.log2(torch.tensor(len(tokens))).item()) - 3
        
        # We need padding matching pywt 'symmetric' ideally, but 'reflect' is close enough
        padding = dec_lo.shape[-1] - 1
        
        for level in range(max_levels):
             # Padding
             approx_padded = torch.nn.functional.pad(approx, (padding, padding), mode='reflect')
             
             # Detail (High Pass)
             detail = torch.nn.functional.conv1d(approx_padded, dec_hi, stride=2)
             
             # Approx (Low Pass) for next level
             approx = torch.nn.functional.conv1d(approx_padded, dec_lo, stride=2)
             
             # Calculate fluctuation for this scale
             scale = 2**(level+1)
             
             # RMS of details
             F = torch.sqrt(torch.mean(detail**2))
             
             if not torch.isnan(F) and F > 0:
                 fluctuations.append(F.item())
                 scales.append(scale)
                 
        scales = np.array(scales)
        F_vals = np.array(fluctuations)
        
        valid = F_vals > 0
        if valid.sum() < 3:
             return float('nan'), float('nan'), float('nan'), {}
             
        ln_s = np.log2(scales[valid])
        ln_F = np.log2(F_vals[valid])
        
        # Fit global
        slope, _ = np.polyfit(ln_s, ln_F, 1)
        H_global = slope - 0.5
        
        # Crossover
        mask_short = scales[valid] < crossover_scale
        mask_long = scales[valid] >= crossover_scale
        
        H_short = float('nan')
        H_long = float('nan')
        
        if mask_short.sum() >= 3:
            s_s, _ = np.polyfit(ln_s[mask_short], ln_F[mask_short], 1)
            H_short = s_s - 0.5
            
        if mask_long.sum() >= 3:
            s_l, _ = np.polyfit(ln_s[mask_long], ln_F[mask_long], 1)
            H_long = s_l - 0.5
            
        return H_global, H_short, H_long, {"scales": scales, "fluctuation": F_vals}

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

    def __init__(self, tokens, device=None):
        """
        Initialize with a list of tokens/integers.
        """
        self.tokens = tokens
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


        # 3) H (DFA) - Multi-order
        t0 = time.perf_counter()
        # DFA1 (Linear)
        H1 = hurst_dfa(self.tokens, device=self.device, order=1)
        # DFA2 (Quadratic)
        H2 = hurst_dfa(self.tokens, device=self.device, order=2)
        # DFA3 (Cubic)
        H3 = hurst_dfa(self.tokens, device=self.device, order=3)
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

        # 7) WDFA (Wavelet DFA)
        t0 = time.perf_counter()
        try:
            # Haar (1 vanishing moment, similar to DFA1)
            H_haar_global, H_haar_short, H_haar_long, _ = self.wavelet_dfa_torch(self.tokens, wavelet_name='haar', device=self.device)
            # db4 (4 vanishing moments, similar to DFA3/4)
            H_db4_global, H_db4_short, H_db4_long, _ = self.wavelet_dfa_torch(self.tokens, wavelet_name='db4', device=self.device)
            dev_lbl_wdfa = _dev_label(self.device)
        except Exception as e:
            print(f"   ⚠️ WDFA GPU failed: {e}")
            H_haar_global, H_haar_short, H_haar_long, _ = wavelet_dfa(self.tokens, wavelet_name='haar', device="cpu")
            H_db4_global, H_db4_short, H_db4_long, _ = wavelet_dfa(self.tokens, wavelet_name='db4', device="cpu")
            dev_lbl_wdfa = "Cpu"
        t1 = time.perf_counter()
        if t1 - t0 > 0.5:
             print(f"   ⏱️  WDFA took {t1 - t0:.2f}s on {dev_lbl_wdfa}")

        self.results.update({
            "beta": beta, 
            "alpha": alpha, 
            "H_DFA1": H1, 
            "H_DFA2": H2, 
            "H_DFA3": H3, 
            "H_DMA": H_dma,
            "H_WT_Haar": H_haar_global,
            "H_WT_Haar_short": H_haar_short,
            "H_WT_Haar_long": H_haar_long,
            "H_WT_db4": H_db4_global,
            "H_WT_db4_short": H_db4_short,
            "H_WT_db4_long": H_db4_long,
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

    # ------------------------- Combined -------------------------
    def compute_all(self):
        self.compute_structural_scaling()
        self.compute_informational_structure()
        self.results["n_tokens"] = len(self.tokens)
        return self.results

