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
        ("H_DMA", "H_DMA (Hurst)", "linear"),
        ("H_WT_db4", "H_WT_db4 (Hurst)", "linear")
    ]
    
    informational_metrics = [
        ("entropy_bits", "Entropy (bits)", "linear"),
        ("perplexity", "Perplexity", "linear"),
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
    def plot_one(ax, metric_key, label_text, fit_type):
        # Scatter points (plain color)
        ax.scatter(df_sorted['n_tokens'], df_sorted[metric_key], 
                   alpha=0.15, s=10, c='steelblue', edgecolors='none', label='Books')
        
        # Calculate Moving Average (Black Dashed)
        window_size = 400
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
                    x_line = np.linspace(df_sorted['n_tokens'].min(), df_sorted['n_tokens'].max(), 500)
                    y_line = log_quadratic_func(x_line, *popt)
                    eq_label = f"Fit: y={popt[0]:.2e}(ln x)²+{popt[1]:.2e}(ln x)+{popt[2]:.2f}"
                else: # linear
                    popt, _ = curve_fit(linear_func, x_fit_data, y_fit_data)
                    x_line = np.linspace(df_sorted['n_tokens'].min(), df_sorted['n_tokens'].max(), 500)
                    y_line = linear_func(x_line, *popt)
                    eq_label = f"Fit: y={popt[0]:.2e} ln(x) + {popt[1]:.2f}"
                
                # Plot fit line in RED
                ax.plot(x_line, y_line, color='red', linestyle='-', linewidth=2, label='Model Fit')
                
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
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Legend on every plot to show what lines are
        ax.legend(loc='upper right', fontsize=8, frameon=True)

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
            ax.set_xlabel('Book Size (Tokens) - Log Scale', fontsize=11)

    plt.tight_layout()
    
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
