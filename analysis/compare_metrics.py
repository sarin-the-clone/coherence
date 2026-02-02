#!/usr/bin/env python3
import json
import pandas as pd
import numpy as np
import glob
import warnings
import os 

try:
    from scipy import stats
except ImportError:
    stats = None


def signif_marker(p, alpha=0.05):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return ""

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load generated books
gen_path = os.path.join(BASE_DIR, 'generated_texts_og/Output/gmetrics_all.json')
with open(gen_path) as f:
    gen_data = json.load(f)
gen_df = pd.DataFrame(gen_data)

# Load natural texts from batch results
batch_files = sorted(glob.glob(os.path.join(BASE_DIR, 'natural_texts/Output/batch_results/batch_*.json')))
all_results = []
for f in batch_files:
    with open(f) as file:
        all_results.extend(json.load(file))

nat_df = pd.DataFrame(all_results)

# Metrics to compare
metrics = ['beta', 'alpha', 'H_DFA1', 'H_DMA', 'H_WT_db4', 'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']

print('\n' + '='*150)
print(f'METRIC STATISTICS: Natural Texts (N={len(nat_df)}) vs Generated Texts (N={len(gen_df)})')
print('='*150)
print(f"{'Metric':<20} {'Nat Mean':<12} {'Gen Mean':<12} {'Diff':<12} {'% Diff':<12} {'Nat Std':<12} {'Gen Std':<12} {'p (Welch)':<12} {'p-var':<12} {'d (Cohen)':<12} {'Sig':<4}")
print('-'*150)

for metric in metrics:
    if metric in gen_df.columns and metric in nat_df.columns:
        nat_vals = nat_df[metric].dropna()
        gen_vals = gen_df[metric].dropna()

        nat_mean = nat_vals.mean()
        nat_std = nat_vals.std()
        gen_mean = gen_vals.mean()
        gen_std = gen_vals.std()
        diff = gen_mean - nat_mean
        pct_diff = (diff / nat_mean * 100) if nat_mean != 0 else 0

        # Welch's t-test and effect size
        if stats is not None and len(nat_vals) > 1 and len(gen_vals) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tstat, pval = stats.ttest_ind(nat_vals, gen_vals, equal_var=False)
                var_pval = stats.levene(nat_vals, gen_vals, center="median").pvalue
            # Cohen's d (pooled SD)
            pooled_var = ((len(nat_vals) - 1) * nat_vals.var(ddof=1) + (len(gen_vals) - 1) * gen_vals.var(ddof=1))
            pooled_var /= (len(nat_vals) + len(gen_vals) - 2) if (len(nat_vals) + len(gen_vals) - 2) > 0 else np.nan
            pooled_sd = np.sqrt(pooled_var)
            cohen_d = diff / pooled_sd if pooled_sd and not np.isnan(pooled_sd) else np.nan
        else:
            pval = np.nan
            var_pval = np.nan
            cohen_d = np.nan

        p_text = f"{pval:.3g}" if not np.isnan(pval) else "n/a"
        var_text = f"{var_pval:.3g}" if not np.isnan(var_pval) else "n/a"
        d_text = f"{cohen_d:+.2f}" if not np.isnan(cohen_d) else "n/a"
        sig = signif_marker(pval)

        print(f'{metric:<20} {nat_mean:<12.4f} {gen_mean:<12.4f}  {diff:<+12.4f} {pct_diff:<+11.1f} {nat_std:<12.4f} {gen_std:<12.4f} {p_text:<12} {var_text:<12} {d_text:<12} {sig:<4}')
    else:
        print(f'{metric:<20} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"N/A":<12} {"n/a":<12} {"n/a":<12} {"n/a":<12} {"":<4}')
print('='*150)

print('\n% Diff = 100*Diff/Nat Mean')
if stats is None:
    print("Note: scipy not installed; p-values and effect sizes reported as n/a.")
else:
    print("Sig: * p<0.05, ** p<0.01, *** p<0.001. p-var uses Levene (med). Welch t-test for means.")
