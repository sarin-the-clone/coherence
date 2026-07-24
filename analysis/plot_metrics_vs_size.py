#!/usr/bin/env python3
"""
plot_metrics_vs_size.py

Analyzes and visualizes the relationship between coherence metrics and book size
in natural texts.
Generates two-column plots: Statistical metrics (left) vs Informational metrics (right),
including moving averages and curve fitting (Linear/Log-Quadratic).
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'natural_texts/Output/batch_results')
OUTPUT_DIR = os.path.join(BASE_DIR, 'natural_texts/Output/analysis')
SIZE_BINS_PATH = os.path.join(OUTPUT_DIR, 'size_binned_metrics.json')

# per-size-bin reference bands — these are the intervals the paper actually
# uses, and (unlike the fits) they extend across the >800k bin, covering the
# ~1M scale where the generated books sit.
BIN_PALETTE = ["#5B8FF9", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]

# x-axis display cap (tokens). Above this only 47/10068 books (0.5%) remain.
X_MAX_DISPLAY = None

# x-axis scale. Log spreads the dense 100k-300k region; linear shows true
# spacing but compresses 99.5% of books into the left edge.
X_LOG_SCALE = True

# Per-size-bin reference bands are off by default; flip to True to overlay the
# p2.5-p97.5 / p25-p75 / median blocks (the helper below is kept for that).
SHOW_SIZE_BIN_BANDS = False

# Highlight the "landmark" books for which we computed prefix trajectories
# (Frankenstein, Sherlock, ... War & Peace, Les Mis) as their full-book
# endpoints only. LABEL_LANDMARKS annotates each title.
LANDMARK_DIR = os.path.join(BASE_DIR, 'natural_texts/Selected_Books/metrics_with_token_naturalness')
LANDMARK_PREFIX_CSV = os.path.join(OUTPUT_DIR, '..', 'pca', 'generated_prefixes',
                                   'landmark_prefix_projection_scores.csv')
SHOW_LANDMARKS = True
LABEL_LANDMARKS = True          # draw the initials inside each marker
LANDMARK_COLOR = "#B3261E"      # accent (marker edge + text)
LANDMARK_MARKER = "o"           # circle: more room for centred initials than 'D'
LANDMARK_SIZE = 230             # marker area when labelled
FIT_COLOR = "#2E7D32"           # model-fit line colour (green)
FIT_XMAX = 1.0e6                # extend the fit line out to ~1M tokens

# metric-file stem -> compact initials (drawn inside the marker)
LANDMARK_INITIALS = {
    "Frankenstein": "F", "Sherlock": "SH", "PridePrejudice": "PP",
    "TracedTracked": "TT", "OriginSpecies": "OS", "Crusoe": "RC",
    "MobyDick": "MD", "Ulysses": "U", "DonQuixote": "DQ",
    "MonteCristo": "MC", "WarPeace": "WP", "LesMis": "LM",
}
LANDMARK_TITLES = {
    "F": "Frankenstein", "SH": "Sherlock Holmes", "PP": "Pride & Prejudice",
    "TT": "Traced & Tracked", "OS": "Origin of Species", "RC": "Robinson Crusoe",
    "MD": "Moby Dick", "U": "Ulysses", "DQ": "Don Quixote",
    "MC": "Count of Monte Cristo", "WP": "War and Peace", "LM": "Les Misérables",
}


def _prefix_landmark_files():
    """Metric-file basenames of the landmark books that have prefix metrics."""
    try:
        return set(pd.read_csv(LANDMARK_PREFIX_CSV)['landmark_metric_file'].unique())
    except Exception as exc:
        print(f"Landmark prefix list unavailable ({exc}); highlighting none.")
        return set()


def load_landmarks():
    """Full-book endpoint (title, n_tokens, metrics) for each prefix-landmark book."""
    keep = _prefix_landmark_files()
    rows = []
    for path in sorted(glob.glob(os.path.join(LANDMARK_DIR, 'metrics_*.json'))):
        if os.path.basename(path) not in keep:
            continue
        try:
            recs = json.load(open(path))
        except Exception:
            continue
        recs = [r for r in (recs if isinstance(recs, list) else [recs])
                if isinstance(r, dict) and r.get('n_tokens')]
        if recs:
            endpoint = max(recs, key=lambda r: r['n_tokens'])  # full-book endpoint
            stem = os.path.basename(path).replace('metrics_', '').replace('.json', '')
            endpoint['_initials'] = LANDMARK_INITIALS.get(stem, stem[:2].upper())
            rows.append(endpoint)
    return rows


def overlay_landmarks(ax, metric_key, landmarks):
    """Plot each landmark endpoint as a prominent marker with its initials inside.

    Labelled: white-filled circle with a thick coloured edge and dark initials
    (dark-on-light reads better than white-on-colour, and the circle leaves more
    centred room than a diamond). Unlabelled: small filled diamond.
    """
    first = True
    for r in landmarks:
        x, y = r.get('n_tokens'), r.get(metric_key)
        if x is None or y is None:
            continue
        if LABEL_LANDMARKS:
            # boxed initials: the box auto-sizes to the text, so the letters can
            # be large and readable (a fixed circle cramps 2-char initials).
            ax.text(x, y, r.get('_initials', ''), ha='center', va='center', zorder=9,
                    fontsize=8.5, fontweight='bold', color=LANDMARK_COLOR,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=LANDMARK_COLOR, linewidth=1.1))
        else:
            ax.scatter(x, y, s=55, marker='D', zorder=8, facecolors=LANDMARK_COLOR,
                       edgecolors='black', linewidths=0.9,
                       label='Landmark books' if first else None)
        first = False


def load_size_bins():
    """Natural per-size-bin descriptor statistics, or None if unavailable."""
    try:
        with open(SIZE_BINS_PATH) as f:
            return json.load(f).get('bins', [])
    except Exception as exc:
        print(f"Size-bin stats unavailable ({exc}); bands will be skipped.")
        return []


def overlay_size_bin_bands(ax, metric_key, bins):
    """Draw per-bin p2.5-p97.5 and p25-p75 bands with the bin median."""
    for idx, b in enumerate(bins):
        rng = b.get('range_tokens') or []
        stats = (b.get('metrics') or {}).get(metric_key)
        if len(rng) != 2 or not stats:
            continue
        lo, hi = float(rng[0]), float(rng[1])
        color = BIN_PALETTE[idx % len(BIN_PALETTE)]
        ax.fill_between([lo, hi], stats['p2_5'], stats['p97_5'],
                        color=color, alpha=0.09, linewidth=0, zorder=1)
        ax.fill_between([lo, hi], stats['p25'], stats['p75'],
                        color=color, alpha=0.20, linewidth=0, zorder=2)
        ax.hlines(stats['median'], lo, hi, color=color, linestyle='--',
                  linewidth=1.4, alpha=0.95, zorder=3)

def load_data():
    """Load all batch results."""
    print("Loading batch results...")
    batch_files = sorted(glob.glob(f'{BATCH_RESULTS_DIR}/batch_*.json'))
    
    if not batch_files:
        print("No batch files found!")
        return pd.DataFrame()
        
    all_results = []
    for f in batch_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                all_results.extend(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} books.")
    return df

def _fmt(v, sig=4):
    """Plain decimal, never scientific notation."""
    s = f"{v:.{sig}g}"
    if 'e' in s or 'E' in s:
        s = f"{v:.10f}".rstrip('0').rstrip('.')
    return s


def _signed(v, sig=4):
    """' + 0.0524' / ' - 0.0988' so terms read naturally."""
    return (" - " if v < 0 else " + ") + _fmt(abs(v), sig)


def linear_func(x, a, b):
    # Fit to log(x) usually for these plots as x axis is log
    return a * np.log(x) + b

def log_quadratic_func(x, a, b, c):
    # y = a*(log x)^2 + b*(log x) + c
    log_x = np.log(x)
    return a * log_x**2 + b * log_x + c

def plot_metrics_vs_size(df):
    """Plot Statistical and Informational metrics vs Size in two columns."""
    
    # Define metric groups with their fitting type
    # (Metric Key, Label, Fit Type)
    statistical_metrics = [
        ("beta", "Beta ($e^\\beta$)", "linear"),
        ("alpha", "Alpha ($\\alpha$)", "linear"),
        ("H_DFA1", "H_DFA1 (Hurst)", "linear"),
        ("H_WT_db4", "H_WT_db4 (Hurst)", "linear")
    ]
    
    informational_metrics = [
        ("entropy_bits", "Entropy (bits)", "quadratic"),
        ("perplexity", "Perplexity", "quadratic"),
        ("lz_complexity", "Lz Complexity", "quadratic"),
        ("mi_decay_gamma", "MI Decay Gamma ($\\gamma$)", "quadratic")
    ]

    # Collect all needed columns
    all_metrics = [(m[0], m[1]) for m in statistical_metrics + informational_metrics]
    required_cols = ['n_tokens'] + [m[0] for m in all_metrics]
    
    # Basic validation
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        return

    # Filter invalid and sort
    df_clean = df.dropna(subset=required_cols)
    df_clean = df_clean[df_clean['n_tokens'] > 0]
    df_sorted = df_clean.sort_values('n_tokens')
    
    print(f"Plotting {len(df_sorted)} points (all books)...")

    # Determine layout
    n_rows = max(len(statistical_metrics), len(informational_metrics))
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4 * n_rows), sharex=True)
    sns.set_style("whitegrid")
    
    # Helper to plot one metric
    size_bins = load_size_bins() if SHOW_SIZE_BIN_BANDS else []
    landmarks = load_landmarks() if SHOW_LANDMARKS else []

    def plot_one(ax, metric_key, label_text, fit_type):
        # Per-size-bin natural reference bands (behind everything)
        overlay_size_bin_bands(ax, metric_key, size_bins)

        # Scatter points (plain color)
        ax.scatter(df_sorted['n_tokens'], df_sorted[metric_key],
                   alpha=0.15, s=10, c='steelblue', edgecolors='none', label='Books')

        # Landmark books on top of everything
        overlay_landmarks(ax, metric_key, landmarks)
        
        # Calculate Moving Average (Black Dashed)
        # 200 books/point: the centered window ends at ~706k tokens (the 99th
        # percentile of book length), so the curve -- and the fit drawn over the
        # same support -- spans where 99% of the corpus lives. The sparse tail is
        # carried by the per-size-bin bands instead.
        window_size = 200
        roll_mean = df_sorted[metric_key].rolling(window=window_size, center=True).mean()
        
        # Curve Fitting (Red Solid)
        # We need valid x, y for fitting (drop NaNs from rolling)
        valid_indices = ~np.isnan(roll_mean)
        x_fit_data = df_sorted['n_tokens'][valid_indices]
        y_fit_data = roll_mean[valid_indices]
        
        if len(x_fit_data) > 100:
            try:
                if fit_type == 'quadratic':
                    popt, _ = curve_fit(log_quadratic_func, x_fit_data, y_fit_data)
                    # Plot fit line
                    # Draw the fit from where the moving average begins out to
                    # FIT_XMAX (~1M) -- a short extrapolation past the dense
                    # support, stopping short of the sparse multi-M tail.
                    x_line = np.linspace(x_fit_data.min(), FIT_XMAX, 500)
                    y_line = log_quadratic_func(x_line, *popt)
                    eq_label = (f"Fit: y = {_fmt(popt[0])}(ln x)²"
                                f"{_signed(popt[1])}(ln x){_signed(popt[2])}")
                else: # linear
                    popt, _ = curve_fit(linear_func, x_fit_data, y_fit_data)
                    # Draw the fit from where the moving average begins out to
                    # FIT_XMAX (~1M) -- a short extrapolation past the dense
                    # support, stopping short of the sparse multi-M tail.
                    x_line = np.linspace(x_fit_data.min(), FIT_XMAX, 500)
                    y_line = linear_func(x_line, *popt)
                    eq_label = f"Fit: y = {_fmt(popt[0])} ln(x){_signed(popt[1])}"
                
                # Plot fit line in RED
                ax.plot(x_line, y_line, color=FIT_COLOR, linestyle='-', linewidth=2, label='Model Fit')
                
                # Add equation to title or plot area
                # Adding it as text on the plot to be visible
                ax.text(0.05, 0.95, eq_label, transform=ax.transAxes, 
                        fontsize=9, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            except Exception as e:
                print(f"Fitting failed for {metric_key}: {e}")

        # Plot Rolling Mean above the fit for visibility
        ax.plot(df_sorted['n_tokens'], roll_mean, 
                color='black', linestyle='--', linewidth=2.5, alpha=0.9, label='Moving Avg', zorder=5)

        # Formatting
        ax.set_ylabel(label_text, fontsize=10)
        ax.set_title(f"{label_text} vs Size", fontsize=11, fontweight='bold')
        ax.set_xscale('log' if X_LOG_SCALE else 'linear')
        # Display range: X_MAX_DISPLAY caps the axis (only 47 books, 0.5%, lie
        # above 1M); set it to None to show the full corpus span out to 7.2M.
        x_hi = X_MAX_DISPLAY or df_sorted['n_tokens'].max() * 1.08
        ax.set_xlim(df_sorted['n_tokens'].min() * 0.92, x_hi)
        ax.grid(True, alpha=0.3)
        # per-panel legends removed; one common bottom legend is built below

    # Plot Column 1: Statistical
    for i in range(n_rows):
        ax = axes[i, 0]
        if i < len(statistical_metrics):
            key, label, fit_t = statistical_metrics[i]
            plot_one(ax, key, label, fit_t)
        else:
            ax.axis('off')

    # Plot Column 2: Informational
    for i in range(n_rows):
        ax = axes[i, 1]
        if i < len(informational_metrics):
            key, label, fit_t = informational_metrics[i]
            plot_one(ax, key, label, fit_t)
        else:
            ax.axis('off')

    # Set X label only on bottom active plots
    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel('Book Size (Tokens)' + (' - Log Scale' if X_LOG_SCALE else ''),
                          fontsize=11)

    # ── one common bottom legend: plot elements + the landmark initials key ──
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    from matplotlib.legend_handler import HandlerBase

    class BoxedInitialHandler(HandlerBase):
        """Render a legend key entry as a boxed initial, matching the panels."""
        def __init__(self, text, color, **kw):
            self._text, self._color = text, color
            super().__init__(**kw)

        def create_artists(self, legend, orig_handle, xdescent, ydescent,
                           width, height, fontsize, trans):
            t = Text(width / 2. - xdescent, height / 2. - ydescent, self._text,
                     ha='center', va='center', fontsize=8, fontweight='bold',
                     color=self._color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor=self._color, linewidth=1.1))
            t.set_transform(trans)
            return [t]

    element_handles = [
        Line2D([], [], marker='o', linestyle='none', markersize=6,
               markerfacecolor='steelblue', markeredgecolor='none', alpha=0.6, label='Books'),
        Line2D([], [], color='black', linestyle='--', linewidth=2, label='Moving average (200 books)'),
        Line2D([], [], color=FIT_COLOR, linestyle='-', linewidth=2, label='Model fit'),
        Patch(facecolor='white', edgecolor=LANDMARK_COLOR, linewidth=1.1,
              label='Landmark books'),
    ]
    # each key entry draws the boxed initials exactly as they appear in-panel
    key_handles, handler_map = [], {}
    if SHOW_LANDMARKS and LABEL_LANDMARKS:
        for ini, title in LANDMARK_TITLES.items():
            h = Line2D([], [], linestyle='none', marker='', label=title)
            key_handles.append(h)
            handler_map[h] = BoxedInitialHandler(ini, LANDMARK_COLOR)

    handles = element_handles + key_handles
    fig.legend(handles=handles, handler_map=handler_map,
               loc='lower center', ncol=8, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=2.0, handletextpad=0.6, columnspacing=1.4)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'metrics_vs_size_all.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

def main():
    df = load_data()
    if not df.empty:
        plot_metrics_vs_size(df)

if __name__ == "__main__":
    main()
