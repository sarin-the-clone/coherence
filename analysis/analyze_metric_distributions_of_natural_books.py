#!/usr/bin/env python3
"""
analyze_metric_distributions.py

Analyze the distribution of coherence metrics on natural texts to determine
appropriate threshold settings (symmetric vs skewed distributions).
"""

import json
import glob
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_natural_texts_data(natural_texts_dir):
    """Load all natural text metrics from batch files."""
    # Try batch_results subfolder first
    batch_pattern = os.path.join(natural_texts_dir, "Output", "batch_results", "batch_*.json")
    batch_files = sorted(glob.glob(batch_pattern))
    
    # Fallback to Output folder directly
    if not batch_files:
        batch_pattern = os.path.join(natural_texts_dir, "Output", "batch_*.json")
        batch_files = sorted(glob.glob(batch_pattern))
    
    print(f"Found {len(batch_files)} batch files")
    
    all_results = []
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
            all_results.extend(batch_data)
    
    return all_results

def analyze_distribution(values, metric_name):
    """Analyze distribution and return statistics."""
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaN
    
    # Basic stats
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    
    # Skewness and kurtosis
    skewness = stats.skew(values)
    kurtosis = stats.kurtosis(values)
    
    # Normality test (Shapiro-Wilk on sample if too large)
    if len(values) >= 3:
        if len(values) > 5000:
            sample = np.random.choice(values, 5000, replace=False)
            _, p_normal = stats.shapiro(sample)
        else:
            _, p_normal = stats.shapiro(values)
    else:
        p_normal = np.nan
    
    # Percentiles for robust thresholds
    p5, p95 = np.percentile(values, [5, 95])
    p2_5, p97_5 = np.percentile(values, [2.5, 97.5])
    
    return {
        'mean': mean,
        'median': median,
        'std': std,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'p_normal': p_normal,
        'p5': p5,
        'p95': p95,
        'p2_5': p2_5,
        'p97_5': p97_5,
        'min': np.min(values),
        'max': np.max(values),
        'n': len(values)
    }


def quantile(values, q):
    """Simple linear quantile for Python lists without numpy dependency."""
    if not values:
        return None
    values = [v for v in values if v is not None]
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def summarize(values):
    """Compact stats (mean/median/std/percentiles) for binned reporting."""
    if not values:
        return None
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    cleaned = arr.tolist()
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": quantile(cleaned, 0.25),
        "p75": quantile(cleaned, 0.75),
        "p2_5": quantile(cleaned, 0.025),
        "p97_5": quantile(cleaned, 0.975),
        "p5": quantile(cleaned, 0.05),
        "p95": quantile(cleaned, 0.95),
    }

def plot_distributions(data, output_dir):
    """Create distribution plots for all metrics."""
    metrics = ['beta', 'alpha', 'H_DFA1', 'H_DMA', 'H_WT_db4', 
               'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']
    
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Extract values
        values = [d.get(metric) for d in data if d.get(metric) is not None]
        values = np.array(values)
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            continue
        
        # Histogram
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black', density=True)
        
        # Stats
        stats_dict = analyze_distribution(values, metric)
        mean = stats_dict['mean']
        median = stats_dict['median']
        std = stats_dict['std']
        skew = stats_dict['skewness']
        
        # Add vertical lines
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.axvline(median, color='green', linestyle='--', linewidth=2, label=f'Median: {median:.4f}')
        ax.axvline(mean - std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(mean + std, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'±1 STD')
        
        # Title with skewness
        skew_type = "Symmetric" if abs(skew) < 0.5 else ("Right-skewed" if skew > 0 else "Left-skewed")
        ax.set_title(f'{metric}\nSkew: {skew:.2f} ({skew_type})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_distributions.png'), dpi=300, bbox_inches='tight')
    print(f"Saved distribution plots to: {os.path.join(output_dir, 'metric_distributions.png')}")
    plt.close()

def recommend_thresholds(stats_dict, metric_name):
    """Recommend thresholds based on distribution shape."""
    skew = stats_dict['skewness']
    
    if abs(skew) < 0.5:
        # Symmetric: use mean ± k*std
        method = "Mean ± STD (symmetric)"
        lower = stats_dict['mean'] - 2 * stats_dict['std']
        upper = stats_dict['mean'] + 2 * stats_dict['std']
    else:
        # Skewed: use percentiles
        method = "Percentiles (skewed)"
        lower = stats_dict['p2_5']
        upper = stats_dict['p97_5']
    
    return {
        'method': method,
        'lower': max(0, lower),  # Clamp to 0
        'upper': upper,
        'skew': skew
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--natural-texts-dir', default='natural_texts',
                       help='Directory containing natural texts batch files')
    parser.add_argument('--output', default='natural_texts/Output',
                       help='Output directory for plots and recommendations')
    args = parser.parse_args()
    
    print("Loading natural texts data...")
    data = load_natural_texts_data(args.natural_texts_dir)
    print(f"Loaded {len(data)} natural texts")
    
    print("\nAnalyzing distributions...")
    metrics = ['beta', 'alpha', 'H_DFA1', 'H_DMA', 'H_WT_db4', 
               'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']
    
    results = {}
    for metric in metrics:
        values = [d.get(metric) for d in data if d.get(metric) is not None]
        if len(values) > 0:
            results[metric] = analyze_distribution(values, metric)
    
    # Print analysis
    print("\n" + "="*100)
    print("DISTRIBUTION ANALYSIS")
    print("="*100)
    print(f"{'Metric':<20} {'Mean':<10} {'Median':<10} {'Std':<10} {'Skew':<10} {'Shape':<15} {'Normal?':<10}")
    print("-"*100)
    
    for metric, stats_dict in results.items():
        skew = stats_dict['skewness']
        if abs(skew) < 0.5:
            shape = "Symmetric"
        elif skew > 0:
            shape = "Right-skewed"
        else:
            shape = "Left-skewed"
        
        normal = "Yes" if stats_dict['p_normal'] > 0.05 else "No"
        
        print(f"{metric:<20} {stats_dict['mean']:<10.4f} {stats_dict['median']:<10.4f} "
              f"{stats_dict['std']:<10.4f} {skew:<10.2f} {shape:<15} {normal:<10}")
    
    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDED THRESHOLDS")
    print("="*100)
    print(f"{'Metric':<20} {'Method':<25} {'Lower':<12} {'Upper':<12} {'Current Lower':<15} {'Current Upper':<15}")
    print("-"*100)
    
    # Current thresholds (from coherence_decision.py)
    current = {
        'beta': (0.10, 0.461),
        'alpha': (0.0, 0.212),
        'H_DFA1': (0.569, 0.697),
        'entropy_bits': (8.308, 9.728),
        'perplexity': (304.2, 760.2),
        'lz_complexity': (0.0, 0.0004),
        'mi_decay_gamma': (0.0, 0.0346)
    }
    
    for metric, stats_dict in results.items():
        rec = recommend_thresholds(stats_dict, metric)
        curr = current.get(metric, (None, None))
        
        print(f"{metric:<20} {rec['method']:<25} {rec['lower']:<12.4f} {rec['upper']:<12.4f} "
              f"{str(curr[0]):<15} {str(curr[1]):<15}")
    
    # Create plots
    print("\nGenerating distribution plots...")
    plot_distributions(data, args.output)
    
    # Generate and save adaptive thresholds config
    print("\nGenerating adaptive thresholds config...")
    config = {}
    
    # Map metrics to config keys
    metric_map = {
        'beta': ('beta_min', 'beta_max'),
        'alpha': ('alpha_min', 'alpha_max'),
        'H_DFA1': ('H_min', 'H_max'),
        'entropy_bits': ('entropy_min', 'entropy_max'),
        'lz_complexity': ('lz_min', 'lz_max'),
        'mi_decay_gamma': ('gamma_min', 'gamma_max'),
        'perplexity': ('ppl_min', 'ppl_max'),
        # Reference only
        'H_DMA': ('H_DMA_min', 'H_DMA_max'),
        'H_WT_db4': ('H_WT_db4_min', 'H_WT_db4_max')
    }
    
    for metric, keys in metric_map.items():
        if metric in results:
            stats_dict = results[metric]
            rec = recommend_thresholds(stats_dict, metric)
            
            config[keys[0]] = rec['lower']
            config[keys[1]] = rec['upper']
            
            # Also save mean for reference (used by plot_chunk_metrics)
            config[f"{metric}_mean"] = stats_dict['mean']
            config[f"{metric}_std"] = stats_dict['std']
            
    # Add other fixed parameters
    config['min_chars'] = 400
    config['rep_penalty_threshold'] = 6
    config['gating_start_chunk'] = 10
    
    # Save to file
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'generated_texts_og', 'config')
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, 'adaptive_thresholds.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Saved adaptive thresholds to: {config_path}")

    # Size-binned statistics for overlaying on chunk plots
    print("\nComputing size-binned statistics for natural books...")
    n_tokens_all = [d.get("n_tokens") for d in data if d.get("n_tokens") is not None]
    if n_tokens_all:
        max_tokens = int(max(n_tokens_all))
        # Fixed bins aligned with user request (5 bins, top split 800k)
        bin_specs = [
            ("0-(100-150k)", 100_000, 150_000),
            ("1-(150-300k)", 150_000, 300_000),
            ("2-(300-500k)", 300_000, 500_000),
            ("3-(500-800k)", 500_000, 800_000),
            ("4-(>800k)",    800_000, max_tokens + 1),
        ]

        bins = []
        metrics = ['beta', 'alpha', 'H_DFA1', 'H_DMA', 'H_WT_db4',
                   'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']
        for name, lo, hi in bin_specs:
            bucket = {
                "name": name,
                "range_tokens": [lo, hi],
                "metrics": {},
                "count": 0,
            }
            bucket_data = [d for d in data if d.get("n_tokens", 0) >= 100_000 and lo <= d.get("n_tokens", 0) < hi]
            bucket["count"] = len(bucket_data)
            for m in metrics:
                vals = [d.get(m) for d in bucket_data if d.get(m) is not None]
                stats_small = summarize(vals)
                if stats_small:
                    bucket["metrics"][m] = stats_small
            bins.append(bucket)

        size_stats = {
            "edges": [lo for _, lo, _ in bin_specs] + [bin_specs[-1][2]],
            "metrics": metrics,
            "bins": bins,
        }

        size_out_dir = os.path.join(args.output, "analysis")
        os.makedirs(size_out_dir, exist_ok=True)
        size_out_path = os.path.join(size_out_dir, "size_binned_metrics.json")
        with open(size_out_path, "w") as f:
            json.dump(size_stats, f, indent=2)
        print(f"Saved size-binned stats to: {size_out_path}")
    else:
        print("No n_tokens found; skipping size-binned stats.")
    print("\nDone!")

if __name__ == "__main__":
    main()
