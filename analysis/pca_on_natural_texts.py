#!/usr/bin/env python3
"""
pca_on_natural_texts.py

Create PCA density plots with KDE contours.
Can generate snapshots at intervals or just the final plot.
"""

import argparse

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_RESULTS_DIR = os.path.join(BASE_DIR, 'natural_texts/Output/batch_results')
METADATA_FILE = os.path.join(BASE_DIR, 'natural_texts/Output/long_books_with_metadata.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'natural_texts/Output/pca/density')
SNAPSHOT_INTERVAL = 1000  # Create snapshot every N books

# Metrics to use
quant_vars = [
    'beta', 'alpha', 'H_DFA1', 'H_DFA3', 'D', 'delta', 'H_DMA',
    'entropy_bits', 'lz_complexity', 'mi_decay_gamma'
]

# Genre descriptions for plots
GENRE_DESCRIPTIONS = {
    'Fiction': 'Novels, adventure, mystery, short stories',
    'Sci-Fi/Fantasy': 'Science fiction, fantasy, mythology',
    'Historical Fiction': 'Historical novels',
    'Romance': 'Romance novels',
    'Poetry & Drama': 'Poetry, plays',
    'Science & Philosophy': 'Scientific & philosophical works',
    'Reference': 'Dictionaries, guides',
    'Non-Fiction': 'History, biography, religion, essays',
    'Unknown': 'Uncategorized works'
}

def assign_size_category(n_tokens):
    """Assign size category based on token count."""
    if n_tokens < 150000:
        return '0-(100-150k)'
    elif n_tokens < 300000:
        return '1-(150-300k)'
    elif n_tokens < 500000:
        return '2-(300-500k)'
    elif n_tokens < 800000:
        return '3-(500-800k)'
    else:
        return '4-(>800k)'

def simplify_genre(g):
    if not isinstance(g, str): return 'Unknown'
    g = g.strip()
    
    mapping = {
        'General Fiction': 'Fiction',
        'Adventure': 'Fiction', 
        'Mystery/Crime': 'Fiction',
        'Short Stories': 'Fiction',
        'Children/YA': 'Fiction',
        'Humor': 'Fiction',
        
        'Sci-Fi/Fantasy': 'Sci-Fi/Fantasy',
        'Mythology': 'Sci-Fi/Fantasy',
        
        'Historical Fiction': 'Historical Fiction',
        
        'Romance': 'Romance',
        
        'Poetry & Drama': 'Poetry & Drama',
        
        'Science': 'Science & Philosophy',
        'Philosophy': 'Science & Philosophy',
        
        'Reference': 'Reference',
        
        'Biography': 'Non-Fiction',
        'History': 'Non-Fiction',
        'Essays & Letters': 'Non-Fiction',
        'Religion': 'Non-Fiction',
        'Travel': 'Non-Fiction',
        'Other': 'Non-Fiction'
    }
    
    return mapping.get(g, 'Unknown')


def load_all_batch_results():
    """Load all batch results and merge with metadata."""
    batch_files = sorted(glob.glob(f'{BATCH_RESULTS_DIR}/batch_*.json'))
    all_results = []
    for f in batch_files:
        with open(f) as file:
            all_results.extend(json.load(file))
    
    df = pd.DataFrame(all_results)
    
    # Load metadata and merge
    metadata = pd.read_csv(METADATA_FILE)
    df = df.merge(metadata[['ID', 'FinalGenre', 'AuthorEra']], 
                  left_on='book_id', right_on='ID', how='left')
    
    # Simplify genres
    df['FinalGenre'] = df['FinalGenre'].apply(simplify_genre)
    
    # Add size category
    df['SizeCategory'] = df['n_tokens'].apply(assign_size_category)
    
    # Prefix eras for sorting
    era_mapping = {
        'Ancient/Medieval': '0-Ancient/Medieval',
        'Renaissance/Early Modern': '1-Renaissance/Early Modern',
        '18th Century': '2-18th Century',
        'Early 19th Century': '3-Early 19th Century',
        'Late 19th Century': '4-Late 19th Century',
        'Early 20th Century': '5-Early 20th Century',
        'Modern (1950+)': '6-Modern (1950+)',
        'Unknown': '7-Unknown'
    }
    df['AuthorEra'] = df['AuthorEra'].map(lambda x: era_mapping.get(x, x))
    
    return df

def apply_pca(df, n_components=2):
    """Apply PCA with selective normalization.
    
    Args:
        df: DataFrame with metrics
        n_components: Number of principal components to compute
    """
    # Separate structural and informational metrics
    structural = ['beta', 'alpha', 'H_DFA1', 'H_DFA3', 'D', 'delta', 'H_DMA']
    informational = ['entropy_bits', 'lz_complexity', 'mi_decay_gamma']
    
    # Normalize informational metrics by log(n_tokens)
    df_norm = df.copy()
    log_n = np.log(df_norm['n_tokens'])
    for col in informational:
        if col in df_norm.columns:
            df_norm[col] = df_norm[col] / log_n
    
    # Drop rows with NaN values in metrics
    df_clean = df_norm.dropna(subset=quant_vars)
    
    # Standardize all metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[quant_vars])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, df_clean

def plot_density_contour(ax, points, color, kde_threshold=8, two_level=True):
    """Plot density contour for a single category.
    
    Args:
        two_level: If True, use two discrete shading levels (50% and 90%).
                   If False, use gradient with multiple contour lines.
    """
    if len(points) < kde_threshold:
        return
    
    try:
        # Create KDE
        kde = gaussian_kde(points.T)
        
        # Create grid for contour
        x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
        y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kde(positions).T, xx.shape)
        
        contour_color = to_rgba(color)
        
        if two_level:
            # Single contour line at 50% - no fill
            # Calculate actual percentile level from density values
            density_flat = density.flatten()
            level_50 = np.percentile(density_flat[density_flat > 0], 50)  # 50th percentile (median)
            
            # Plot 50% contour line only (no fill)
            ax.contour(xx, yy, density, levels=[level_50],
                      colors=[contour_color],
                      linewidths=2.5, alpha=0.8)
        else:
            # Gradient with multiple contour lines
            level_max = density.max()
            level_95 = level_max * 0.05
            n_levels = 10
            levels = np.linspace(level_95, level_max, n_levels)
            
            colors_with_alpha = []
            for alpha_val in np.linspace(0.0, 0.5, n_levels-1):
                rgba = list(contour_color)
                rgba[3] = alpha_val
                colors_with_alpha.append(rgba)
            
            # Add gradient fill first
            ax.contourf(xx, yy, density, levels=levels, colors=colors_with_alpha)
            
            # Plot multiple contour lines on top
            ax.contour(xx, yy, density, levels=levels,
                      colors=[contour_color],
                      linewidths=1.5, alpha=0.6)
    except:
        pass

def plot_individual_points(ax, X_pca, categories, category_to_color, size_factor):
    """Plot individual data points."""
    unique_categories = np.unique(categories)
    for cat in unique_categories:
        mask = categories == cat
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=[category_to_color[cat]],
                  s=size_factor[mask] * 30, alpha=0.1,
                  edgecolors='none')

def plot_median_markers(ax, X_pca, categories, category_to_color, category_to_marker, category_to_size):
    """Plot median markers with hierarchical markers for each category."""
    unique_categories = np.unique(categories)
    for cat in unique_categories:
        mask = categories == cat
        if np.sum(mask) > 0:
            median_pc1 = np.median(X_pca[mask, 0])
            median_pc2 = np.median(X_pca[mask, 1])
            ax.scatter(median_pc1, median_pc2,
                      c=[category_to_color[cat]],
                      s=category_to_size[cat], marker=category_to_marker[cat], 
                      edgecolors='black',
                      linewidth=1.5, zorder=10, alpha=1.0)

def set_axis_limits(ax, X_pca, margin=0.8):
    """Set axis limits to 5th-95th percentile with margin."""
    pc1_5, pc1_95 = np.percentile(X_pca[:, 0], [5, 95])
    pc2_5, pc2_95 = np.percentile(X_pca[:, 1], [5, 95])
    ax.set_xlim(pc1_5 - margin * (pc1_95 - pc1_5), pc1_95 + margin * (pc1_95 - pc1_5))
    ax.set_ylim(pc2_5 - margin * (pc2_95 - pc2_5), pc2_95 + margin * (pc2_95 - pc2_5))

def add_genre_labels(fig, ax, unique_genres, broad_genres, genre_to_color, genre_to_marker, genre_to_size, genre_to_cluster):
    """Add category labels using matplotlib legend, sorted by cluster."""
    # Sort genres by cluster for better visual organization
    sorted_genres = sorted(unique_genres, key=lambda g: (genre_to_cluster[g], g))
    
    # Create legend handles with descriptions
    handles = []
    for genre in sorted_genres:
        n_samples = np.sum(broad_genres == genre)
        desc = GENRE_DESCRIPTIONS.get(genre, '')
        label = f'{genre} ({n_samples})' if not desc else f'{genre} ({n_samples})\n  {desc}'
        handle = plt.Line2D([0], [0], marker=genre_to_marker[genre], color='w',
                           markerfacecolor=genre_to_color[genre], 
                           markeredgecolor='black', markeredgewidth=1.5,
                           markersize=12, label=label)
        handles.append(handle)
    
    # Add legend outside plot area
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=9, frameon=True, fancybox=True, shadow=True)

def add_era_labels(fig, ax, unique_eras, author_eras, era_to_color, era_to_marker, era_to_size, era_to_cluster):
    """Add category labels using matplotlib legend, sorted by cluster."""
    # Sort eras by cluster for better visual organization
    sorted_eras = sorted(unique_eras, key=lambda e: (era_to_cluster[e], e))
    
    # Create legend handles (no descriptions for eras - they're self-explanatory)
    handles = []
    for era in sorted_eras:
        n_samples = np.sum(author_eras == era)
        handle = plt.Line2D([0], [0], marker=era_to_marker[era], color='w',
                           markerfacecolor=era_to_color[era], 
                           markeredgecolor='black', markeredgewidth=1.5,
                           markersize=12, label=f'{era} ({n_samples})')
        handles.append(handle)
    
    # Add legend outside plot area
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=10, frameon=True, fancybox=True, shadow=True)

def add_size_labels(fig, ax, unique_sizes, size_categories, size_to_color, size_to_marker, size_to_size, size_to_cluster):
    """Add size category labels using matplotlib legend, sorted by size."""
    # Define size order (sorted by prefix)
    size_order = [
         '0-(100-150k)', '1-(150-300k)', 
         '2-(300-500k)', '3-(500-1M)', 
         '4-(>1M)'
    ]
    
    # Sort sizes by defined order
    sorted_sizes = [s for s in size_order if s in unique_sizes]
    
    # Create legend handles
    handles = []
    for size_cat in sorted_sizes:
        n_samples = np.sum(size_categories == size_cat)
        handle = plt.Line2D([0], [0], marker=size_to_marker[size_cat], color='w',
                           markerfacecolor=size_to_color[size_cat], 
                           markeredgecolor='black', markeredgewidth=1.5,
                           markersize=12, label=f'{size_cat} ({n_samples})')
        handles.append(handle)
    
    # Add legend outside plot area
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
             fontsize=10, frameon=True, fancybox=True, shadow=True)

def plot_variable_loadings(ax, pca, loading_scale=4, pc_indices=(0, 1)):
    """Plot variable loadings as arrows with color coding.
    
    Args:
        pc_indices: Tuple of (pc_x, pc_y) indices for which components to plot
    """
    structural = ['beta', 'alpha', 'H_DFA1', 'H_DFA3', 'D', 'delta', 'H_DMA']
    informational = ['entropy_bits', 'lz_complexity', 'mi_decay_gamma']
    
    pc_x, pc_y = pc_indices
    
    for i, var in enumerate(quant_vars):
        # Color code: blue for structural, red for informational
        if var in structural:
            arrow_color = 'steelblue'
            edge_color = 'darkblue'
        else:
            arrow_color = 'coral'
            edge_color = 'darkred'
        
        ax.arrow(0, 0,
                pca.components_[pc_x, i] * loading_scale,
                pca.components_[pc_y, i] * loading_scale,
                head_width=0.15, head_length=0.15,
                fc=arrow_color, ec=edge_color, alpha=0.6, linewidth=1.5)
        ax.text(pca.components_[pc_x, i] * loading_scale * 1.15,
               pca.components_[pc_y, i] * loading_scale * 1.15,
               var, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=edge_color, alpha=0.7, linewidth=1.5))

def create_individual_category_plots(df, X_pca, pca, categories, category_to_color, 
                                     category_to_marker, category_to_size, category_to_cluster,
                                     unique_categories, output_path, title_prefix, pc_indices=(0, 1)):
    """Create a figure with individual subplots for each category showing detailed density.
    
    Args:
        pc_indices: Tuple of (pc_x, pc_y) indices to plot (default: (0, 1) for PC1 vs PC2)
    """
    pc_x, pc_y = pc_indices
    # Sort categories by cluster for better visual organization
    sorted_categories = sorted(unique_categories, key=lambda cat: (category_to_cluster[cat], cat))
    
    n_categories = len(sorted_categories)
    n_cols = 4
    # Summary takes 2x2 (4 positions), so we need space for summary + all categories
    n_rows = ((n_categories + 3) // n_cols) + 1  # +1 row for 2x2 summary
    
    fig = plt.figure(figsize=(20, 5*n_rows))
    
    # Calculate global axis limits for consistency across all subplots
    pc1_5, pc1_95 = np.percentile(X_pca[:, pc_x], [5, 95])
    pc2_5, pc2_95 = np.percentile(X_pca[:, pc_y], [5, 95])
    margin = 0.3
    xlim = (pc1_5 - margin * (pc1_95 - pc1_5), pc1_95 + margin * (pc1_95 - pc1_5))
    ylim = (pc2_5 - margin * (pc2_95 - pc2_5), pc2_95 + margin * (pc2_95 - pc2_5))
    
    # First subplot: Summary with all category medians and loadings (2x2 grid)
    ax_summary = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=2, rowspan=2, fig=fig)
    
    # Plot median stars for all categories with hierarchical markers
    # Plot global density contour and points (all samples)
    points_all = X_pca[:, [pc_x, pc_y]]
    if len(points_all) >= 8:
        plot_density_contour(ax_summary, points_all, (0.5, 0.5, 0.5, 1.0), 
                           kde_threshold=8, two_level=False)
    
    # Plot all individual points faintly
    ax_summary.scatter(points_all[:, 0], points_all[:, 1],
                     c='gray', s=10, alpha=0.05, 
                     edgecolors='none', zorder=1)

    # Plot median stars for all categories
    for cat in sorted_categories:
        mask = categories == cat
        if np.sum(mask) > 0:
            median_pcx = np.median(X_pca[mask, pc_x])
            median_pcy = np.median(X_pca[mask, pc_y])
            ax_summary.scatter(median_pcx, median_pcy,
                             c=[category_to_color[cat]], 
                             s=category_to_size[cat], 
                             marker=category_to_marker[cat],
                             edgecolors='black', linewidth=1.5, zorder=10)
    
    # Plot variable loadings
    plot_variable_loadings(ax_summary, pca, loading_scale=4, pc_indices=pc_indices)
    
    ax_summary.set_title('Category Medians & Variable Loadings', fontsize=14, weight='bold')
    ax_summary.set_xlabel(f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)', fontsize=11)
    ax_summary.set_ylabel(f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)', fontsize=11)
    ax_summary.set_xlim(xlim)
    ax_summary.set_ylim(ylim)
    ax_summary.grid(True, alpha=0.3)
    ax_summary.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax_summary.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Individual category subplots
    # Calculate positions for remaining subplots (after 2x2 summary)
    subplot_positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            # Skip the 2x2 summary area (rows 0-1, cols 0-1)
            if row < 2 and col < 2:
                continue
            subplot_positions.append((row, col))
    
    for idx, cat in enumerate(sorted_categories):
        if idx >= len(subplot_positions):
            break
        row, col = subplot_positions[idx]
        ax = plt.subplot2grid((n_rows, n_cols), (row, col), fig=fig)
        
        mask = categories == cat
        points = X_pca[mask, :][:, [pc_x, pc_y]]  # Select only the specified PC components
        n_points = len(points)
        
        if n_points >= 8:
            # Plot with gradient and multiple contour lines
            plot_density_contour(ax, points, category_to_color[cat], 
                               kde_threshold=8, two_level=False)
        
        # Always plot individual points (even if < 8)
        if n_points > 0:
            ax.scatter(points[:, 0], points[:, 1],
                      c=[category_to_color[cat]], s=10, alpha=0.05,
                      edgecolors='none')
            
            # Always plot median with hierarchical marker (even if < 8)
            median_pc1 = np.median(points[:, 0])
            median_pc2 = np.median(points[:, 1])
            ax.scatter(median_pc1, median_pc2,
                      c=[category_to_color[cat]], 
                      s=category_to_size[cat], 
                      marker=category_to_marker[cat],
                      edgecolors='black', linewidth=1.5, zorder=10)
        
        ax.set_title(f'{cat} (n={n_points})', fontsize=12, weight='bold')
        # Add genre description as subtitle if available
        desc = GENRE_DESCRIPTIONS.get(cat, '')
        if desc:
            ax.text(0.5, 0.98, desc, transform=ax.transAxes, 
                   fontsize=9, ha='center', va='top', style='italic', color='gray')
        ax.set_xlabel(f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)', fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    fig.suptitle(title_prefix, fontsize=16, weight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_hierarchical_colors(X_pca, categories, unique_categories, base_colormap, pc_indices=(0, 1)):
    """Create hierarchical color scheme based on median proximity.
    
    Groups categories into clusters of ~3 based on median distances,
    assigns same base color to each cluster with different shades and markers.
    Uses HSL color space for more perceptually distinct colors.
    
    Args:
        X_pca: PCA-transformed data (n_samples, n_components)
        categories: Category labels for each sample
        unique_categories: Unique category names
        base_colormap: Matplotlib colormap for base colors
        pc_indices: Tuple of (pc_x, pc_y) indices to use for clustering (default: (0, 1) for PC1 vs PC2)
    
    Returns:
        category_to_color, category_to_marker, category_to_size, category_to_cluster
    """
    import colorsys
    
    pc_x, pc_y = pc_indices
    
    # Calculate medians for each category using specified PC components
    medians = []
    for cat in unique_categories:
        mask = categories == cat
        if np.sum(mask) > 0:
            median_pcx = np.median(X_pca[mask, pc_x])
            median_pcy = np.median(X_pca[mask, pc_y])
            medians.append([median_pcx, median_pcy])
        else:
            medians.append([0, 0])
    
    medians = np.array(medians)
    
    # Normalize medians so both PC dimensions contribute equally to clustering
    # (otherwise the PC with higher variance dominates)
    medians_normalized = (medians - medians.mean(axis=0)) / (medians.std(axis=0) + 1e-10)
    
    # Cluster medians into groups of ~3
    n_categories = len(unique_categories)
    n_clusters = max(1, (n_categories + 2) // 3)  # Aim for groups of 3
    
    
    # Hierarchical clustering on normalized coordinates
    distances = pdist(medians_normalized, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Use tab20 for distinct base colors (better than algorithmic generation)
    cmap = plt.cm.tab20
    # Sample colors from the colormap
    base_colors = [cmap(i / max(1, n_clusters - 1) if n_clusters > 1 else 0) for i in range(n_clusters)]
    # Use tab10 for simplified genres (9 categories)
    if n_clusters <= 10:
        cmap = plt.cm.tab10
        base_colors = [cmap(i) for i in range(n_clusters)]
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
        base_colors = [cmap(i) for i in range(n_clusters)]
    else:
        # Fallback for many clusters
        base_colors = [plt.cm.nipy_spectral(i / n_clusters) for i in range(n_clusters)]
    
    # Define shades and markers with stronger variation
    # Extended marker set to handle clusters with more than 3 categories
    lightness_values = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]  # 6 lightness levels
    saturation_values = [0.95, 0.85, 0.75, 0.90, 0.80, 0.70]  # 6 saturation levels
    markers = ['o', '*', 'D', 's', '^', 'v', 'p', 'h', 'X']  # 9 distinct markers
    marker_sizes = [200, 300, 250, 220, 240, 240, 260, 240, 220]  # Corresponding sizes
    
    # Assign colors, markers, and metadata
    category_to_color = {}
    category_to_marker = {}
    category_to_size = {}
    category_to_cluster = {}  # Track cluster assignment for sorting
    
    for cluster_id in range(1, n_clusters + 1):
        cluster_cats = [unique_categories[i] for i in range(n_categories) 
                       if cluster_labels[i] == cluster_id]
        base_rgb = base_colors[cluster_id - 1][:3] # Get RGB part
        
        for idx, cat in enumerate(cluster_cats):
            # Use modulo to cycle through available options
            # lightness_idx = idx % len(lightness_values) # Skip lightness variation to keep colors distinct
            saturation_idx = idx % len(saturation_values)
            marker_idx = idx % len(markers)
            
            # Convert base RGB to HLS to adjust saturation/lightness if needed
            h, l, s = colorsys.rgb_to_hls(*base_rgb)
            
            # Slight variation for items in same cluster, but keep close to base
            # Vary saturation slightly
            new_s = max(0.2, min(1.0, s * saturation_values[saturation_idx]))
            # Keep lightness relatively constant to avoid washing out
            new_l = l 
            
            # Convert back to RGB
            rgb = colorsys.hls_to_rgb(h, new_l, new_s)
            
            category_to_color[cat] = rgb + (1.0,)
            category_to_marker[cat] = markers[marker_idx]
            category_to_size[cat] = marker_sizes[marker_idx]
            category_to_cluster[cat] = cluster_id
    
    return category_to_color, category_to_marker, category_to_size, category_to_cluster

def create_multipanel_biplot(df, X_pca, pca, categories, category_col,
                              colormap, title, output_path, label_func, manual_style=None):
    """Create a 2x2 subplot biplot showing PC1vsPC2, PC2vsPC3, PC3vsPC4, and legend/stats.
    
    Args:
        df: DataFrame with book data
        X_pca: PCA-transformed data
        pca: Fitted PCA object
        categories: Category labels for each sample
        category_col: Name of category column
        colormap: Matplotlib colormap
        title: Overall title for the figure
        output_path: Where to save the plot
        label_func: Function to add labels (used only in stats panel)
        manual_style: Optional tuple (cat_to_color, cat_to_marker, cat_to_size, cat_to_cluster)
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Setup hierarchical colors or use manual style
    unique_categories = np.unique(categories)
    
    if manual_style:
        category_to_color, category_to_marker, category_to_size, category_to_cluster = manual_style
    else:
        category_to_color, category_to_marker, category_to_size, category_to_cluster = create_hierarchical_colors(
            X_pca, categories, unique_categories, colormap, pc_indices=(0, 1)
        )
    
    # PC combinations to plot
    pc_pairs = [(0, 1), (1, 2), (2, 3)]
    
    # Plot first 3 subplots with different PC combinations
    for idx, (pc_x, pc_y) in enumerate(pc_pairs):
        ax = axes[idx]
        
        # Create temporary 2D array for this PC pair
        X_pca_pair = X_pca[:, [pc_x, pc_y]]
        
        # Plot individual points
        size_factor = np.log(df['n_tokens']) / np.log(df['n_tokens']).max()
        plot_individual_points(ax, X_pca_pair, categories, category_to_color, size_factor)
        
        # Plot median markers
        plot_median_markers(ax, X_pca_pair, categories, category_to_color, category_to_marker, category_to_size)
        
        # Set axis limits
        set_axis_limits(ax, X_pca_pair)
        
        # Plot variable loadings
        plot_variable_loadings(ax, pca, loading_scale=4, pc_indices=(pc_x, pc_y))
        
        # Formatting
        ax.set_xlabel(f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)', fontsize=12, weight='bold')
        ax.set_ylabel(f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)', fontsize=12, weight='bold')
        ax.set_title(f'PC{pc_x+1} vs PC{pc_y+1}', fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # 4th subplot: Legend and statistics
    ax_stats = axes[3]
    ax_stats.axis('off')
    
    # Add title for stats panel
    ax_stats.text(0.5, 0.98, 'Category Statistics', 
                  ha='center', va='top', fontsize=16, weight='bold',
                  transform=ax_stats.transAxes)
    
    # Sort categories by cluster for better organization
    sorted_categories = sorted(unique_categories, key=lambda c: (category_to_cluster[c], c))
    
    # Find max category name length for alignment
    max_cat_len = max(len(cat) for cat in sorted_categories)
    
    # Create legend handles with integrated statistics
    handles = []
    
    for cat in sorted_categories:
        mask = categories == cat
        n_samples = np.sum(mask)
        
        # Compute means of key metrics for this category
        subset = df[mask]
        mean_beta = subset['beta'].mean()
        mean_alpha = subset['alpha'].mean()
        mean_h = subset['H_DFA1'].mean()
        mean_s = subset['entropy_bits'].mean()
        mean_p = subset['perplexity'].mean()
        mean_l = subset['lz_complexity'].mean()
        mean_g = subset['mi_decay_gamma'].mean()
        
        # Create legend handle with statistics in label (category name, n, and mean metrics)
        # Pad category name to max length for alignment
        # Scale Lz by 1e3 and Gamma by 1e2 for readability
        mean_l_scaled = mean_l * 1000
        mean_g_scaled = mean_g * 100
        
        # Create legend handle with statistics in label (category name, n, and mean metrics)
        # Pad category name to max length for alignment
        # β=beta, α=alpha, H=H_DFA1, S=entropy, Ppl=perplexity, Lz=LZ(1e-3), γ=MI Decay(1e-2)
        label = (f'{cat:<{max_cat_len}} n={n_samples:<4} '
                 f'β={mean_beta:.2f} α={mean_alpha:.2f} H={mean_h:.2f} '
                 f'S={mean_s:.1f} Ppl={mean_p:.0f} Lz(1e-3)={mean_l_scaled:.2f} γ(1e-2)={mean_g_scaled:.2f}')
        handle = plt.Line2D([0], [0], marker=category_to_marker[cat], color='w',
                           markerfacecolor=category_to_color[cat], 
                           markeredgecolor='black', markeredgewidth=1.5,
                           markersize=10, label=label)
        handles.append(handle)
    
    # Add legend with integrated statistics (monospace font for alignment)
    # Reduced left padding (bbox_to_anchor x from 0.05 to -0.05)
    legend = ax_stats.legend(handles=handles, loc='upper left', 
                            bbox_to_anchor=(-0.05, 1.0),
                            fontsize=10, frameon=True, fancybox=True, shadow=True,
                            ncol=1, handletextpad=0.5, labelspacing=0.3,
                            prop={'family': 'monospace'})
    
    # Add overall statistics
    total_books = len(df)
    n_categories = len(unique_categories)
    total_variance = pca.explained_variance_ratio_[:4].sum() * 100
    
    overall_stats = f'''
Overall Statistics:
  Total books: {total_books:,}
  Categories: {n_categories}
  PC1-PC4 variance: {total_variance:.1f}%
    '''
    
    ax_stats.text(0.05, 0.15, overall_stats.strip(), 
                  ha='left', va='top', fontsize=11, weight='bold',
                  transform=ax_stats.transAxes,
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle(title, fontsize=18, weight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return unique_categories, category_to_color, category_to_marker, category_to_size, category_to_cluster

def create_main_biplot(df, X_pca, pca, categories, category_col, 
                        colormap, title, output_path, label_func, pc_indices=(0, 1)):
    """Generic function to create a main PCA biplot (without density contours).
    
    Args:
        pc_indices: Tuple of (pc_x, pc_y) indices to plot (default: (0, 1) for PC1 vs PC2)
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    pc_x, pc_y = pc_indices
    
    # Setup hierarchical colors based on median proximity
    unique_categories = np.unique(categories)
    category_to_color, category_to_marker, category_to_size, category_to_cluster = create_hierarchical_colors(
        X_pca, categories, unique_categories, colormap, pc_indices=pc_indices
    )
    
    # Plot density contours - DISABLED for main biplots (contours only in individual subplots)
    
    # Plot individual points
    size_factor = np.log(df['n_tokens']) / np.log(df['n_tokens']).max()
    plot_individual_points(ax, X_pca[:, [pc_x, pc_y]], categories, category_to_color, size_factor)
    
    # Plot median markers
    plot_median_markers(ax, X_pca[:, [pc_x, pc_y]], categories, category_to_color, category_to_marker, category_to_size)
    
    # Set axis limits to focus on data region
    set_axis_limits(ax, X_pca[:, [pc_x, pc_y]])
    
    # Add labels with marker and cluster information
    label_func(fig, ax, unique_categories, categories, category_to_color, category_to_marker, category_to_size, category_to_cluster)
    
    # Plot variable loadings
    plot_variable_loadings(ax, pca, loading_scale=4, pc_indices=pc_indices)
    
    # Formatting
    ax.set_xlabel(f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return unique_categories, category_to_color, category_to_marker, category_to_size, category_to_cluster

def run_pca_plots(df, snapshot_num, n_books_label):
    """Run PCA and generate plots for a specific snapshot."""
    print(f"\n================================================================================")
    print(f"PCA PLOTS {n_books_label}")
    print(f"================================================================================")
    
    # Apply PCA (compute 4 components for multipanel plots)
    X_pca, pca, df_clean = apply_pca(df, n_components=4)
    
    snapshot_dir = f'{OUTPUT_DIR}/snapshot_{snapshot_num:04d}_{len(df_clean):05d}_books'
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # 1. Multi-Panel Genre Biplot (PC1vsPC2, PC2vsPC3, PC3vsPC4 + stats)
    unique_genres, genre_to_color, genre_to_marker, genre_to_size, genre_to_cluster = create_multipanel_biplot(
        df_clean, X_pca, pca, df_clean['FinalGenre'].values, 'FinalGenre', 
        plt.cm.tab20, f'PCA Analysis by Genre ({n_books_label})', 
        f'{snapshot_dir}/pca_multipanel_genre.jpg', add_genre_labels
    )
    
    # 2. Multi-Panel Era Biplot (PC1vsPC2, PC2vsPC3, PC3vsPC4 + stats)
    unique_eras, era_to_color, era_to_marker, era_to_size, era_to_cluster = create_multipanel_biplot(
        df_clean, X_pca, pca, df_clean['AuthorEra'].values, 'AuthorEra', 
        plt.cm.tab10, f'PCA Analysis by Author Era ({n_books_label})', 
        f'{snapshot_dir}/pca_multipanel_era.jpg', add_era_labels
    )
    

    
    # 3. Individual Genre Plots
    create_individual_category_plots(
        df_clean, X_pca, pca, 
        categories=df_clean['FinalGenre'].values,
        category_to_color=genre_to_color,
        category_to_marker=genre_to_marker,
        category_to_size=genre_to_size,
        category_to_cluster=genre_to_cluster,
        unique_categories=unique_genres,
        output_path=f'{snapshot_dir}/individual_genres.jpg',
        title_prefix=f'Individual Genre Density Distributions ({n_books_label})'
    )
    

    
    # 4. Individual Era Plots
    create_individual_category_plots(
        df_clean, X_pca, pca,
        categories=df_clean['AuthorEra'].values,
        category_to_color=era_to_color,
        category_to_marker=era_to_marker,
        category_to_size=era_to_size,
        category_to_cluster=era_to_cluster,
        unique_categories=unique_eras,
        output_path=f'{snapshot_dir}/individual_eras.jpg',
        title_prefix=f'Individual Era Density Distributions ({n_books_label})'
    )
    
    # 5. Multi-Panel Size Category Biplot with 3-color scheme for 6 categories
    
    # Define manual style for Size
    size_cats = [
        '0-(100-150k)', '1-(150-300k)', 
        '2-(300-500k)', '3-(500-800k)', 
        '4-(>800k)'
    ]
    
    # Colors (RGBA)
    c1 = (0.12, 0.47, 0.71, 1.0) # Blue (Short)
    c2 = (0.17, 0.63, 0.17, 1.0) # Green (Medium)
    c3 = (0.84, 0.15, 0.16, 1.0) # Red (Long)
    c4 = (0.58, 0.40, 0.74, 1.0) # Purple (Very Long)
    c5 = (1.00, 0.50, 0.05, 1.0) # Orange (Epic)
    
    size_to_color = {
        size_cats[0]: c1, 
        size_cats[1]: c2,
        size_cats[2]: c3, 
        size_cats[3]: c4,
        size_cats[4]: c5
    }
    
    size_to_marker = {
        size_cats[0]: 'o', 
        size_cats[1]: 'D', 
        size_cats[2]: '^', 
        size_cats[3]: 's', 
        size_cats[4]: '*'
    }
    size_to_size = {c: 200 for c in size_cats}
    size_to_cluster = {
        size_cats[0]: 1, 
        size_cats[1]: 2,
        size_cats[2]: 3, 
        size_cats[3]: 4,
        size_cats[4]: 5
    }
    
    manual_size_style = (size_to_color, size_to_marker, size_to_size, size_to_cluster)

    unique_sizes, size_to_color, size_to_marker, size_to_size, size_to_cluster = create_multipanel_biplot(
        df_clean, X_pca, pca, df_clean['SizeCategory'].values, 'SizeCategory', 
        plt.cm.viridis, f'PCA Analysis by Book Size ({n_books_label})', 
        f'{snapshot_dir}/pca_multipanel_size.jpg', 
        add_size_labels, manual_style=manual_size_style
    )
    
    # 6. Individual Size Category Plots
    create_individual_category_plots(
        df_clean, X_pca, pca,
        categories=df_clean['SizeCategory'].values,
        category_to_color=size_to_color,
        category_to_marker=size_to_marker,
        category_to_size=size_to_size,
        category_to_cluster=size_to_cluster,
        unique_categories=unique_sizes,
        output_path=f'{snapshot_dir}/individual_sizes.jpg',
        title_prefix=f'Individual Size Category Density Distributions ({n_books_label})'
    )
    
    print(f"✅ PCA plots saved to {snapshot_dir}")
    print(f"   PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"   PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='Generate PCA plots for natural texts')
    parser.add_argument('--final-only', action='store_true',
                        help='Only generate final plot for all books (skip intermediate snapshots)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: auto-generated)')
    args = parser.parse_args()
    
    print("📊 Creating PCA plots...")
    
    # Load all results
    df = load_all_batch_results()
    n_books = len(df)
    print(f"   Total books processed: {n_books}")
    
    if args.final_only:
        # Just create final plot with all books
        print("\n" + "="*80)
        print(f"PCA PLOT - ALL {n_books} BOOKS")
        print("="*80)
        run_pca_plots(df, 1, f"{n_books} books")
    else:
        # Determine which snapshots to create
        snapshots_to_create = []
        for i in range(1, (n_books // SNAPSHOT_INTERVAL) + 2):
            snapshot_books = min(i * SNAPSHOT_INTERVAL, n_books)
            
            # Only add if it's a new snapshot or the last one
            if snapshot_books > 0 and (i == 1 or snapshot_books > snapshots_to_create[-1][1] if snapshots_to_create else True):
                snapshots_to_create.append((i, snapshot_books))
        
        # Create snapshots
        for snapshot_num, snapshot_books in snapshots_to_create:
            df_subset = df.iloc[:snapshot_books]
            run_pca_plots(df_subset, snapshot_num, f"{snapshot_books} books")

if __name__ == "__main__":
    main()
