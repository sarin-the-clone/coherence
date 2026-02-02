#!/usr/bin/env python3
"""
project_on_natural_books_PCs.py

Projects a generated book onto the PCA space derived from natural texts.
Shows where AI-generated literature sits relative to real books.

Usage:
    python3 project_on_natural_books_PCs.py --book path/to/book.txt
    python3 project_on_natural_books_PCs.py --metrics path/to/metrics.json
"""

import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob
import warnings

try:
    from scipy import stats
except ImportError:
    stats = None

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer
from metrics import CoherenceMetrics


def load_natural_texts_data(natural_texts_dir):
    """Load the natural texts metrics and compute PCA."""
    # Load from batch results to get all books
    batch_files = sorted(glob.glob(os.path.join(natural_texts_dir, "Output", "batch_results", "batch_*.json")))
    
    if not batch_files:
        # Fallback to coherence_results.json if no batch files
        print("No batch files found, falling back to coherence_results.json")
        results_path = os.path.join(natural_texts_dir, "Output", "coherence_results.json")
        with open(results_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            records = []
            for book_id, metrics in data.items():
                record = metrics.copy()
                record['book_id'] = book_id
                records.append(record)
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame(data)
    else:
        print(f"Loading natural texts from {len(batch_files)} batch files...")
        all_results = []
        for f in batch_files:
            with open(f) as file:
                all_results.extend(json.load(file))
        df = pd.DataFrame(all_results)

    metadata_path = os.path.join(natural_texts_dir, "Output", "long_books_with_metadata.csv")
    
    # Load metadata and merge to get genre information
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        # Extract numeric ID from book_id (e.g., "twain_74" -> 74)
        # Handle both string IDs with underscores and pure numeric IDs if present
        df['ID_num'] = df['book_id'].astype(str).str.extract(r'_(\d+)$')[0].astype(float)
        
        # If extraction failed (all NaNs), try using book_id directly if it looks numeric
        if df['ID_num'].isna().all():
             df['ID_num'] = pd.to_numeric(df['book_id'], errors='coerce')

        df = df.merge(metadata[['ID', 'FinalGenre', 'AuthorEra']], 
                     left_on='ID_num', right_on='ID', how='left')
        print(f"Loaded metadata with {len(metadata)} books")
    else:
        print(f"Warning: Metadata file not found at {metadata_path}")
        df['FinalGenre'] = 'Unknown'
    
    # Map metric names (natural texts uses different names)
    metric_mapping = {
        'beta': 'beta',
        'alpha': 'alpha',
        'H_WT_db4': 'H_WT_db4',
        'H_WT_db4_long': 'H_WT_db4_long',
        'H_DFA': 'H_DFA1',  # Natural texts uses H_DFA1
        'H_DMA': 'H_DMA',
        'H_WDFA': 'H_WT_Haar',  # Approximation
        'H_DEA': 'D',  # Approximation
        'entropy': 'entropy_bits',
        'perplexity': 'perplexity',
        'lz_complexity': 'lz_complexity',
        'mi_decay': 'mi_decay_gamma'
    }
    
    # Select metric columns that exist
    metric_cols = []
    for target_name, source_name in metric_mapping.items():
        if source_name in df.columns:
            metric_cols.append(source_name)
        else:
            # Only warn if it's a critical metric we expect
            pass
            
    # Filter to only include rows with all metrics
    df_clean = df.dropna(subset=metric_cols)
    
    X = df_clean[metric_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA (compute 4 components)
    pca = PCA(n_components=4)
    pca_coords = pca.fit_transform(X_scaled)
    
    # Add PCA coordinates to dataframe
    df_clean = df_clean.copy()
    df_clean['PC1'] = pca_coords[:, 0]
    df_clean['PC2'] = pca_coords[:, 1]
    df_clean['PC3'] = pca_coords[:, 2]
    df_clean['PC4'] = pca_coords[:, 3]
    
    # Return both the column names and the mapping
    return df_clean, pca, scaler, metric_cols, metric_mapping


def compute_book_metrics(book_path):
    """Compute coherence metrics for a book."""
    with open(book_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize
    tokenizer = Tokenizer(kind="gpt4")
    tokens = tokenizer.encode(text)
    
    print(f"Book: {os.path.basename(book_path)}")
    print(f"Tokens: {len(tokens)}")
    
    # Compute metrics
    cm = CoherenceMetrics(tokens)
    metrics = cm.compute_all()
    
    return metrics


def project_book(metrics, pca, scaler, metric_cols):
    """Project book metrics onto PCA space."""
    # Extract metric values in the same order
    values = [metrics.get(col, np.nan) for col in metric_cols]
    
    # Check for missing metrics
    if any(np.isnan(values)):
        missing = [metric_cols[i] for i, v in enumerate(values) if np.isnan(v)]
        print(f"Warning: Missing metrics: {missing}")
        return None
    
    # Reshape to (1, n_features)
    X = np.array(values).reshape(1, -1)
    
    # Standardize using the same scaler
    X_scaled = scaler.transform(X)
    
    # Project onto PCA
    coords = pca.transform(X_scaled)
    
    return coords[0]  # Return (PC1, PC2, PC3)


def get_genre_color_map():
    """Get a color map for genres matching the natural_texts pca_plots.py style."""
    # Use tab20 colormap for consistency with pca_plots.py
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    genre_colors = {
        'General Fiction': colors[0],
        'Historical Fiction': colors[1],
        'Romance': colors[2],
        'Mystery/Crime': colors[3],
        'Sci-Fi/Fantasy': colors[4],
        'Short Stories': colors[5],
        'Poetry & Drama': colors[6],
        'Children/YA': colors[7],
        'Adventure': colors[8],
        'Biography': colors[9],
        'History': colors[10],
        'Travel': colors[11],
        'Science': colors[12],
        'Philosophy': colors[13],
        'Religion': colors[14],
        'Mythology': colors[15],
        'Reference': colors[16],
        'Essays & Letters': colors[17],
        'Humor': colors[18],
        'Other': colors[19],
        'Unknown': (0.5, 0.5, 0.5, 1.0)  # Gray for unknown
    }
    return genre_colors


def plot_variable_loadings(ax, pca, metric_cols, pc_indices=(0, 1), scale=4):
    """Plot variable loadings as arrows."""
    pc_x, pc_y = pc_indices
    
    # Define structural vs informational metrics for coloring
    structural = ['beta', 'alpha', 'H_DFA1', 'H_DFA3', 'D', 'delta', 'H_DMA', 'H_WT_db4', 'H_WT_db4_long']
    
    for i, var in enumerate(metric_cols):
        # Color code: blue for structural, red for informational
        if var in structural:
            arrow_color = 'steelblue'
            edge_color = 'darkblue'
        else:
            arrow_color = 'coral'
            edge_color = 'darkred'
        
        x_loading = pca.components_[pc_x, i] * scale
        y_loading = pca.components_[pc_y, i] * scale
        
        ax.arrow(0, 0, x_loading, y_loading,
                head_width=0.15, head_length=0.15,
                fc=arrow_color, ec=edge_color, alpha=0.6, linewidth=1.5)
        
        ax.text(x_loading * 1.15, y_loading * 1.15,
               var, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=edge_color, alpha=0.7, linewidth=1.5))


def plot_subplot(ax, df_natural, projected_books, pca, metric_cols, pc_indices, genre_colors, model_colors):
    """Helper to plot a single subplot."""
    pc_x, pc_y = pc_indices
    pc_x_name = f'PC{pc_x+1}'
    pc_y_name = f'PC{pc_y+1}'
    
    # Plot natural texts with increased transparency
    if 'FinalGenre' in df_natural.columns:
        genres = sorted(df_natural['FinalGenre'].unique())
        for genre in genres:
            mask = df_natural['FinalGenre'] == genre
            genre_data = df_natural.loc[mask]
            color = genre_colors.get(genre, (0.5, 0.5, 0.5, 1.0))
            
            # Plot individual points with high transparency
            ax.scatter(
                genre_data[pc_x_name],
                genre_data[pc_y_name],
                c=[color],
                alpha=0.15,
                s=30,
                edgecolors='none'
            )
            
            # Plot median as a star
            if len(genre_data) > 0:
                median_x = genre_data[pc_x_name].median()
                median_y = genre_data[pc_y_name].median()
                ax.scatter(
                    median_x,
                    median_y,
                    c=[color],
                    marker='*',
                    s=200,
                    edgecolors='black',
                    linewidths=1,
                    zorder=5,
                    alpha=0.9
                )
    else:
        ax.scatter(
            df_natural[pc_x_name],
            df_natural[pc_y_name],
            c='lightgray',
            alpha=0.15,
            s=30
        )
    
    # Plot generated books as diamonds
    for book_name, coords, model_name in projected_books:
        color = model_colors.get(model_name, 'red')
        ax.scatter(
            coords[pc_x],
            coords[pc_y],
            c=[color],
            marker='D',  # Diamond marker
            s=200,
            edgecolors='black',
            linewidths=1.5,
            zorder=10
        )
        # Add text label for generated book
        ax.text(coords[pc_x], coords[pc_y], f" {book_name[:15]}...", 
               fontsize=8, fontweight='bold', color='darkred', zorder=11)

    # Plot variable loadings
    plot_variable_loadings(ax, pca, metric_cols, pc_indices)
    
    # Formatting
    ax.set_xlabel(f'{pc_x_name} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)', fontsize=10)
    ax.set_ylabel(f'{pc_y_name} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)


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


def compute_comparison_stats(df_natural, df_generated, metric_mapping):
    """Compute comparison statistics between natural and generated texts."""
    stats_data = []
    
    # Reverse mapping to get original metric names if needed, but here we use the mapped names in df_natural
    # df_natural has columns like 'beta', 'alpha', etc.
    # df_generated has columns like 'beta', 'alpha', etc.
    
    metrics = ['beta', 'alpha', 'H_DFA1', 'H_DMA', 'H_WT_db4', 'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']
    
    for metric in metrics:
        # Map metric name if necessary (df_natural uses mapped names)
        nat_col = metric
        gen_col = metric
        
        if nat_col in df_natural.columns and gen_col in df_generated.columns:
            nat_vals = df_natural[nat_col].dropna()
            gen_vals = df_generated[gen_col].dropna()
            
            if len(nat_vals) == 0 or len(gen_vals) == 0:
                continue

            nat_mean = nat_vals.mean()
            nat_std = nat_vals.std()
            gen_mean = gen_vals.mean()
            gen_std = gen_vals.std()
            diff = gen_mean - nat_mean
            pct_diff = (diff / nat_mean * 100) if nat_mean != 0 else 0

            # Welch's t-test
            pval = np.nan
            if stats is not None and len(nat_vals) > 1 and len(gen_vals) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tstat, pval = stats.ttest_ind(nat_vals, gen_vals, equal_var=False)
            
            sig = signif_marker(pval)
            
            stats_data.append({
                'Metric': metric,
                'Nat Mean': f"{nat_mean:.3f}",
                'Gen Mean': f"{gen_mean:.3f}",
                'Mean Diff': f"{diff:+.3f}",
                '% Diff': f"{pct_diff:+.1f}%",
                'Nat Std': f"{nat_std:.3f}",
                'Gen Std': f"{gen_std:.3f}",
                'Mean Diff Sig': sig,
            })
            
    return pd.DataFrame(stats_data)


def plot_projection(df_natural, projected_books, pca, metric_cols, output_path, model_colors, comparison_stats):
    """Plot 4-panel projection: PC1v2, PC2v3, PC3v4, and Comparison Stats."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes = axes.flatten()
    
    genre_colors = get_genre_color_map()
    
    # 1. PC1 vs PC2
    plot_subplot(axes[0], df_natural, projected_books, pca, metric_cols, (0, 1), genre_colors, model_colors)
    axes[0].set_title('PC1 vs PC2', fontsize=14, fontweight='bold')
    
    # 2. PC2 vs PC3
    plot_subplot(axes[1], df_natural, projected_books, pca, metric_cols, (1, 2), genre_colors, model_colors)
    axes[1].set_title('PC2 vs PC3', fontsize=14, fontweight='bold')
    
    # 3. PC3 vs PC4
    plot_subplot(axes[2], df_natural, projected_books, pca, metric_cols, (2, 3), genre_colors, model_colors)
    axes[2].set_title('PC3 vs PC4', fontsize=14, fontweight='bold')
    
    # 4. Comparison Stats & Legend (Bottom Right)
    ax_stats = axes[3]
    ax_stats.axis('off')
    
    # Add Comparison Statistics Table
    if comparison_stats is not None and not comparison_stats.empty:
        table_data = [comparison_stats.columns.tolist()] + comparison_stats.values.tolist()
        table = ax_stats.table(cellText=table_data, loc='top', cellLoc='center', bbox=[0.05, 0.4, 0.9, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax_stats.text(0.5, 0.95, "Statistical Comparison (Natural vs Generated)", 
                     ha='center', va='bottom', fontsize=14, fontweight='bold', transform=ax_stats.transAxes)
    
    # Add Legend below table
    handles = []
    
    # Natural Text Genres with counts (showing median as star)
    if 'FinalGenre' in df_natural.columns:
        genres = sorted(df_natural['FinalGenre'].unique())
        for genre in genres:
            color = genre_colors.get(genre, (0.5, 0.5, 0.5, 1.0))
            count = np.sum(df_natural['FinalGenre'] == genre)
            handle = plt.Line2D([0], [0], marker='*', color='w',
                               markerfacecolor=color, markeredgecolor='black',
                               markersize=12, linewidth=1,
                               label=f'{genre} ({count})')
            handles.append(handle)
    
    # Generated Book Models (diamonds)
    for model, color in model_colors.items():
        handle = plt.Line2D([0], [0], marker='D', color='w',
                           markerfacecolor=color, markeredgecolor='black',
                           markersize=10, label=f'Model: {model}')
        handles.append(handle)
        
    # Loadings Legend
    struct_handle = plt.Line2D([0], [0], color='steelblue', linewidth=2, label='Structural Metrics')
    info_handle = plt.Line2D([0], [0], color='coral', linewidth=2, label='Informational Metrics')
    handles.extend([struct_handle, info_handle])
    
    # Add legend to the bottom area
    ax_stats.legend(handles=handles, loc='lower center', fontsize=10, ncol=2, 
                   title="Legend (Generated Models & Metrics)", title_fontsize=12, bbox_to_anchor=(0.5, 0.0))
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def shorten_label(filename):
    """Shorten filename to characters between LONGBOOK__ and the next __."""
    if "LONGBOOK__" in filename:
        try:
            parts = filename.split("LONGBOOK__")[1].split("__")
            if len(parts) > 1:
                return parts[0]
        except IndexError:
            pass
    return filename[:20] # Fallback


def extract_model_name(filename):
    """Extract model name from filename."""
    if "LONGBOOK__" in filename:
        try:
            parts = filename.split("LONGBOOK__")[1].split("__")
            if len(parts) > 0:
                return parts[0]
        except IndexError:
            pass
    return "Unknown Model"


def main():
    parser = argparse.ArgumentParser(description='Project generated book onto natural texts PCA space')
    parser.add_argument('--book', type=str, help='Path to generated book .txt file')
    parser.add_argument('--metrics', type=str, help='Path to pre-computed metrics .json file')
    parser.add_argument('--natural-texts-dir', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'natural_texts'),
                       help='Path to natural_texts directory')
    parser.add_argument('--output', '-o', type=str, help='Output plot path (optional)')
    parser.add_argument('--components', type=str, default='1,2',
                       help='Comma-separated list of 2 components to plot (e.g. "1,2" or "2,3"). Default: "1,2"')
    
    args = parser.parse_args()
    
    # Parse components
    try:
        pc_indices = tuple(int(x.strip()) - 1 for x in args.components.split(','))
        if len(pc_indices) != 2:
            raise ValueError
    except:
        parser.error("Components must be two comma-separated integers (e.g. '1,2')")
    
    # Default metrics file if nothing provided
    if not args.book and not args.metrics:
        default_metrics = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'generated_texts_og', 'Output', 'gmetrics_all.json')
        if os.path.exists(default_metrics):
            print(f"No input provided. Defaulting to: {default_metrics}")
            args.metrics = default_metrics
        else:
            parser.error("Must provide either --book or --metrics, and default metrics file not found.")
    
    # Load natural texts PCA
    print("Loading natural texts data and fitting PCA...")
    df_natural, pca, scaler, metric_cols, metric_mapping = load_natural_texts_data(args.natural_texts_dir)
    print(f"Loaded {len(df_natural)} natural texts")
    print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}, PC3={pca.explained_variance_ratio_[2]:.2%}")
    
    # Get book metrics
    books_to_plot = []
    generated_metrics_list = []
    
    if args.book:
        print(f"\nComputing metrics for: {args.book}")
        metrics = compute_book_metrics(args.book)
        book_name = os.path.basename(args.book).replace('.txt', '')
        books_to_plot.append((book_name, metrics))
        generated_metrics_list.append(metrics)
    else:
        print(f"\nLoading metrics from: {args.metrics}")
        with open(args.metrics, 'r') as f:
            data = json.load(f)
            
        # Handle both single dict and list of dicts
        if isinstance(data, list):
            print(f"Found {len(data)} books in metrics file")
            for item in data:
                book_name = item.get('file', 'Unknown').replace('.txt', '')
                books_to_plot.append((book_name, item))
                generated_metrics_list.append(item)
        else:
            book_name = os.path.basename(args.metrics).replace('.json', '')
            books_to_plot.append((book_name, data))
            generated_metrics_list.append(data)
    
    # Create DataFrame for generated metrics for comparison
    df_generated = pd.DataFrame(generated_metrics_list)
    
    # Compute comparison stats
    print("\nComputing comparison statistics...")
    comparison_stats = compute_comparison_stats(df_natural, df_generated, metric_mapping)
    
    # Assign colors to models
    unique_models = sorted(list(set([extract_model_name(b[0]) for b in books_to_plot])))
    # Use a distinct colormap for models
    model_cmap = plt.get_cmap("tab10") 
    model_colors = {model: model_cmap(i / max(1, len(unique_models))) for i, model in enumerate(unique_models)}
    
    # Project all books onto PCA
    print(f"\nProjecting {len(books_to_plot)} book(s) onto PCA space...")
    projected_books = []
    
    for book_name, metrics in books_to_plot:
        coords = project_book(metrics, pca, scaler, metric_cols)
        
        if coords is None:
            print(f"Warning: Could not project '{book_name}' (missing metrics)")
            continue
        
        # Shorten label
        short_label = shorten_label(book_name)
        model_name = extract_model_name(book_name)
        
        print(f"  {short_label} ({model_name}): PC1={coords[0]:.3f}, PC2={coords[1]:.3f}, PC3={coords[2]:.3f}, PC4={coords[3]:.3f}")
        projected_books.append((short_label, coords, model_name))
        
        # Find nearest natural text for this book (using all 4 components)
        dist_sq = 0
        for i in range(4):
            dist_sq += (df_natural[f'PC{i+1}'] - coords[i])**2
        distances = np.sqrt(dist_sq)
        
        nearest_idx = distances.idxmin()
        nearest = df_natural.loc[nearest_idx]
        
        print(f"    Nearest: {nearest.get('book_id', 'Unknown')} (distance: {distances[nearest_idx]:.3f})")
    
    if not projected_books:
        print("Error: No books could be projected")
        return 1
    
    # Determine output path
    output_path = args.output
    if not output_path:
        if args.book:
            output_path = args.book.replace('.txt', '_pca_projection.jpg')
        elif args.metrics:
            output_path = args.metrics.replace('.json', '_pca_projection.jpg')
        else:
            output_path = 'pca_projection.jpg'
            
    # Plot
    print("\nGenerating plot...")
    plot_projection(df_natural, projected_books, pca, metric_cols, output_path, model_colors, comparison_stats)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
