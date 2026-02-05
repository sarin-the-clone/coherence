#!/usr/bin/env python3
"""
pca_on_batches.py

Run PCA analysis on accumulated batch results at intervals.
Saves PCA plots to Output/pca_snapshots/snapshot_XXXX/
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'Output/batch_results')
PCA_SNAPSHOTS_DIR = os.path.join(BASE_DIR, 'Output/pca_snapshots')
SNAPSHOT_INTERVAL = 837  # Run PCA every 2000 books
MAX_SNAPSHOTS = 10

# Set style
sns.set_style("whitegrid")

def load_all_batch_results():
    """Load and combine all batch result files, then merge with metadata."""
    batch_files = sorted(glob.glob(f'{BATCH_RESULTS_DIR}/batch_*.json'))
    
    all_results = []
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            results = json.load(f)
            all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    
    # Load metadata to get AuthorEra
    metadata_file = os.path.join(BASE_DIR, 'Output/long_books_with_metadata.csv')
    if os.path.exists(metadata_file):
        metadata = pd.read_csv(metadata_file)
        # Merge on book_id to get AuthorEra
        df = df.merge(metadata[['ID', 'AuthorEra']], 
                     left_on='book_id', right_on='ID', how='left')
        df = df.drop('ID', axis=1)
        # Fill missing AuthorEra with 'Unknown'
        df['AuthorEra'] = df['AuthorEra'].fillna('Unknown')
    else:
        # If metadata file doesn't exist, set all to Unknown
        df['AuthorEra'] = 'Unknown'
    
    return df

def should_run_pca(n_books):
    """Determine if we should run PCA based on book count."""
    snapshot_num = n_books // SNAPSHOT_INTERVAL
    return snapshot_num <= MAX_SNAPSHOTS and (n_books % SNAPSHOT_INTERVAL == 0 or n_books % SNAPSHOT_INTERVAL < 100)

def run_pca_snapshot(df, snapshot_num):
    """Run PCA analysis and save plots for a snapshot."""
    print(f"\n{'='*80}")
    print(f"PCA SNAPSHOT {snapshot_num} - {len(df)} books")
    print(f"{'='*80}")
    
    # Create snapshot directory
    snapshot_dir = f'{PCA_SNAPSHOTS_DIR}/snapshot_{snapshot_num:04d}_{len(df):05d}_books'
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Select quantitative variables
    quant_vars = [
        'beta', 'alpha', 'H_DFA1', 'H_DFA3', 'H_DMA', 'delta',
        'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma'
    ]
    quant_vars = [v for v in quant_vars if v in df.columns]
    
    # Extract data
    X = df[quant_vars].values
    authors = df['author'].values
    broad_genres = df['broad_genre'].values
    author_eras = df.get('AuthorEra', pd.Series(['Unknown'] * len(df))).values
    titles = df['title'].values
    n_tokens = df['n_tokens'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    # Selective normalization
    log_tokens = np.log10(n_tokens)
    median_log_tokens = np.median(log_tokens)
    normalization_factor = log_tokens / median_log_tokens
    
    X_normalized = X.copy()
    informational_metrics_indices = [quant_vars.index(m) for m in 
                                     ['entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma'] 
                                     if m in quant_vars]
    
    for idx in informational_metrics_indices:
        X_normalized[:, idx] = X[:, idx] / normalization_factor
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normalized)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Scree plot
    fig, ax = plt.subplots(figsize=(10, 6))
    variance_ratio = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(variance_ratio)
    
    ax.bar(range(1, len(variance_ratio)+1), variance_ratio, alpha=0.6, label='Individual')
    ax.plot(range(1, len(variance_ratio)+1), cumulative_variance, 'ro-', label='Cumulative')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title(f'PCA Scree Plot - Snapshot {snapshot_num} ({len(df)} books)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{snapshot_dir}/scree_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Biplot by broad genre with convex hulls
    fig, ax = plt.subplots(figsize=(16, 12))
    
    unique_genres = np.unique(broad_genres)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_genres)))  # Better color scheme
    genre_to_color = dict(zip(unique_genres, colors))
    
    # Plot books by genre
    size_factor = np.log10(n_tokens) / np.log10(np.median(n_tokens))
    for genre in unique_genres:
        mask = broad_genres == genre
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[genre_to_color[genre]], label=genre,
                  s=size_factor[mask] * 50, alpha=0.2,
                  edgecolors='k', linewidth=0.3)
    
    # Add median markers for each genre (stars)
    for genre in unique_genres:
        mask = broad_genres == genre
        if np.sum(mask) > 0:
            median_pc1 = np.median(X_pca[mask, 0])
            median_pc2 = np.median(X_pca[mask, 1])
            ax.scatter(median_pc1, median_pc2,
                      c=[genre_to_color[genre]],
                      s=200, marker='*', edgecolors=genre_to_color[genre],
                      linewidth=1.5, zorder=10, alpha=1.0)
    
    # Add convex hulls for each genre
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon
    
    for genre in unique_genres:
        mask = broad_genres == genre
        points = X_pca[mask, :2]
        if len(points) >= 5:  # Require at least 5 points for hull
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                patch = Polygon(hull_points, fill=False,
                              edgecolor=genre_to_color[genre],
                              linewidth=3, linestyle='--', alpha=0.8)
                ax.add_patch(patch)
                
                # Add label at the rightmost point of the hull
                rightmost_idx = np.argmin(hull_points[:, 1])
                label_x = hull_points[rightmost_idx, 0]
                label_y = hull_points[rightmost_idx, 1]
                ax.text(label_x, label_y, genre,
                       fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', 
                                facecolor='white', 
                                edgecolor=genre_to_color[genre],
                                linewidth=2, alpha=0.9),
                       ha='left', va='center')
            except:
                pass
    
    # Plot variable loadings with color coding
    loading_scale = 4
    
    # Define metric types
    structural_metrics = ['beta', 'alpha', 'H_DFA1', 'H_DFA3', 'H_DMA', 'delta']
    informational_metrics = ['entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']
    
    for i, var in enumerate(quant_vars):
        ax.arrow(0, 0,
                pca.components_[0, i] * loading_scale,
                pca.components_[1, i] * loading_scale,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
        
        # Color code by metric type
        if var in structural_metrics:
            label_color = 'lightblue'
            edge_color = 'blue'
        elif var in informational_metrics:
            label_color = 'lightgreen'
            edge_color = 'green'
        else:
            label_color = 'yellow'
            edge_color = 'black'
        
        ax.text(pca.components_[0, i] * loading_scale * 1.1,
               pca.components_[1, i] * loading_scale * 1.1,
               var, fontsize=8, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=label_color,
                        edgecolor=edge_color, linewidth=1.5, alpha=0.7))
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'PCA Biplot by Genre - Snapshot {snapshot_num} ({len(df)} books, {len(unique_genres)} genres)\nBlue=Structural, Green=Informational, Dashed lines=Hulls, Stars=Medians')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{snapshot_dir}/biplot_by_genre.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create statistics table
    fig_stats, ax_stats = plt.subplots(figsize=(14, max(6, len(unique_genres) * 0.4)))
    ax_stats.axis('tight')
    ax_stats.axis('off')
    
    def fmt_val(v):
        if abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.2f}"
    
    cell_text = []
    row_labels = []
    for genre in unique_genres:
        mask = broad_genres == genre
        n_books = np.sum(mask)
        row_labels.append(f"{genre} ({n_books})")
        row_data = []
        for i, var in enumerate(quant_vars):
            var_mean = np.mean(X[mask, i])
            var_std = np.std(X[mask, i])
            row_data.append(f"{fmt_val(var_mean)} ± {fmt_val(var_std)}")
        cell_text.append(row_data)
    
    # Add Overall row
    row_labels.append(f"Overall ({len(df)})")
    row_data = []
    for i, var in enumerate(quant_vars):
        var_mean = np.mean(X[:, i])
        var_std = np.std(X[:, i])
        row_data.append(f"{fmt_val(var_mean)} ± {fmt_val(var_std)}")
    cell_text.append(row_data)
    
    table = ax_stats.table(cellText=cell_text,
                          rowLabels=row_labels,
                          colLabels=quant_vars,
                          loc='center',
                          cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    ax_stats.set_title(f'PCA Statistics by Genre - Snapshot {snapshot_num} ({len(df)} books)', 
                       fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{snapshot_dir}/stats_by_genre.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ============================================================================
    # Author Era Biplot
    # ============================================================================
    
    fig_era, ax_era = plt.subplots(figsize=(16, 12))
    
    unique_eras = np.unique(author_eras)
    colors_era = plt.cm.tab10(np.linspace(0, 1, len(unique_eras)))  # Better color scheme
    era_to_color = dict(zip(unique_eras, colors_era))
    
    # Plot books by era
    for era in unique_eras:
        mask = author_eras == era
        ax_era.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=[era_to_color[era]], label=era,
                      s=size_factor[mask] * 50, alpha=0.2,
                      edgecolors='k', linewidth=0.3)
    
    # Add median markers for each era (stars)
    for era in unique_eras:
        mask = author_eras == era
        if np.sum(mask) > 0:
            median_pc1 = np.median(X_pca[mask, 0])
            median_pc2 = np.median(X_pca[mask, 1])
            ax_era.scatter(median_pc1, median_pc2,
                          c=[era_to_color[era]],
                          s=200, marker='*', edgecolors=era_to_color[era],
                          linewidth=1.5, zorder=10, alpha=1.0)
    
    # Add convex hulls for each era
    for era in unique_eras:
        mask = author_eras == era
        points = X_pca[mask, :2]
        if len(points) >= 5:  # Require at least 5 points for hull
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                patch = Polygon(hull_points, fill=False,
                              edgecolor=era_to_color[era],
                              linewidth=3, linestyle='--', alpha=0.8)
                ax_era.add_patch(patch)
                
                # Add label at the rightmost point of the hull
                rightmost_idx = np.argmin(hull_points[:, 1])
                label_x = hull_points[rightmost_idx, 0]
                label_y = hull_points[rightmost_idx, 1]
                ax_era.text(label_x, label_y, era,
                           fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.4',
                                    facecolor='white',
                                    edgecolor=era_to_color[era],
                                    linewidth=2, alpha=0.9),
                           ha='left', va='center')
            except:
                pass
    
    # Plot variable loadings with color coding (same as genre biplot)
    for i, var in enumerate(quant_vars):
        ax_era.arrow(0, 0,
                    pca.components_[0, i] * loading_scale,
                    pca.components_[1, i] * loading_scale,
                    head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
        
        if var in structural_metrics:
            label_color = 'lightblue'
            edge_color = 'blue'
        elif var in informational_metrics:
            label_color = 'lightgreen'
            edge_color = 'green'
        else:
            label_color = 'yellow'
            edge_color = 'black'
        
        ax_era.text(pca.components_[0, i] * loading_scale * 1.1,
                   pca.components_[1, i] * loading_scale * 1.1,
                   var, fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=label_color,
                            edgecolor=edge_color, linewidth=1.5, alpha=0.7))
    
    ax_era.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax_era.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    
    ax_era.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax_era.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax_era.set_title(f'PCA Biplot by Author Era - Snapshot {snapshot_num} ({len(df)} books)\nBlue=Structural, Green=Informational, Dashed lines=Hulls, Stars=Medians')
    ax_era.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax_era.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{snapshot_dir}/biplot_by_era.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Author Era Statistics Table
    fig_stats_era, ax_stats_era = plt.subplots(figsize=(14, max(6, len(unique_eras) * 0.4)))
    ax_stats_era.axis('tight')
    ax_stats_era.axis('off')
    
    cell_text_era = []
    row_labels_era = []
    for era in unique_eras:
        mask = author_eras == era
        n_books = np.sum(mask)
        row_labels_era.append(f"{era} ({n_books})")
        row_data = []
        for i, var in enumerate(quant_vars):
            var_mean = np.mean(X[mask, i])
            var_std = np.std(X[mask, i])
            row_data.append(f"{fmt_val(var_mean)} ± {fmt_val(var_std)}")
        cell_text_era.append(row_data)
    
    # Add Overall row
    row_labels_era.append(f"Overall ({len(df)})")
    row_data = []
    for i, var in enumerate(quant_vars):
        var_mean = np.mean(X[:, i])
        var_std = np.std(X[:, i])
        row_data.append(f"{fmt_val(var_mean)} ± {fmt_val(var_std)}")
    cell_text_era.append(row_data)
    
    table_era = ax_stats_era.table(cellText=cell_text_era,
                                   rowLabels=row_labels_era,
                                   colLabels=quant_vars,
                                   loc='center',
                                   cellLoc='center')
    table_era.auto_set_font_size(False)
    table_era.set_fontsize(9)
    table_era.scale(1, 1.5)
    
    ax_stats_era.set_title(f'PCA Statistics by Author Era - Snapshot {snapshot_num} ({len(df)} books)',
                          fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f'{snapshot_dir}/stats_by_era.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Snapshot {snapshot_num} saved to {snapshot_dir}")
    print(f"   PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"   PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
    print(f"   Cumulative: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}%")

def main():
    """Main function to check and run PCA snapshots."""
    print("📊 Checking for PCA snapshot opportunities...")
    
    # Load all results
    df = load_all_batch_results()
    n_books = len(df)
    
    print(f"   Total books processed: {n_books}")
    
    if n_books == 0:
        print("   No books processed yet.")
        return
    
    # Determine which snapshots to create
    existing_snapshots = glob.glob(f'{PCA_SNAPSHOTS_DIR}/snapshot_*')
    n_existing = len(existing_snapshots)
    
    print(f"   Existing snapshots: {n_existing}")
    
    # Create snapshots at intervals
    for i in range(1, MAX_SNAPSHOTS + 1):
        target_books = i * SNAPSHOT_INTERVAL
        if n_books >= target_books:
            snapshot_dir = f'{PCA_SNAPSHOTS_DIR}/snapshot_{i:04d}_{target_books:05d}_books'
            if not os.path.exists(snapshot_dir):
                # Create snapshot with first N books
                df_subset = df.head(target_books)
                run_pca_snapshot(df_subset, i)
    
    # Also create a snapshot for current total if it's not at an exact interval
    if n_books % SNAPSHOT_INTERVAL != 0 and n_books > (n_existing * SNAPSHOT_INTERVAL):
        snapshot_num = (n_books // SNAPSHOT_INTERVAL) + 1
        if snapshot_num <= MAX_SNAPSHOTS:
            run_pca_snapshot(df, snapshot_num)

if __name__ == "__main__":
    main()
