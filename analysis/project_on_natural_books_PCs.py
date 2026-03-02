#!/usr/bin/env python3
"""
project_on_natural_books_PCs.py

Projects generated books and selected reference texts onto the PCA space derived from natural texts.
Generates comprehensive genre-based grid density plots to visualize 
AI-generated trajectories relative to the landscape of human literature.
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
    from scipy.stats import gaussian_kde
except ImportError:
    stats = None
    gaussian_kde = None

from matplotlib.colors import to_rgba
import matplotlib.lines as mlines

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer
from metrics import CoherenceMetrics

# Genre descriptions for plots
GENRE_DESCRIPTIONS = {
    'Fiction': 'General fiction, short stories, humor, historical fiction',
    'Adventure/Mystery/Sci-Fi': 'Adventure, mystery, thriller, crime, sci-fi, fantasy',
    'Romance, Poetry & Drama': 'Romance, poetry, plays',
    'Science & Philosophy': 'Scientific & philosophical works',
    'Reference': 'Dictionaries, guides',
    'Non-Fiction': 'History, biography, religion, essays',
    'Unknown': 'Uncategorized works'
}

# Centrally managed name map for selected books
NAME_MAP_CLEAN = {
    'HeartWorld': 'Heart of the World',
    'CalebWilliams': 'Caleb Williams',
    'Marriage': 'Marriage',
    'WormOuroboros': 'The Worm Ouroboros',
    'Aspasia': 'Aspasia',
    'SecretDoctrine': 'The Secret Doctrine',
    'NarHist': 'Narrative & Crit. Hist.',
    'TracedTracked': 'Traced and Tracked',
    'JargonFile': 'The Jargon File',
    'MobyDick': 'Moby Dick',
    'Ulysses': 'Ulysses',
    'Crusoe': 'Robinson Crusoe',
    'Deronda': 'Daniel Deronda',
    'Shirley': 'Shirley',
    'LesMis': 'Les Misérables',
    'PridePrejudice': 'Pride and Prejudice',
    'OriginSpecies': 'On the Origin of Species',
    'WarPeace': 'War and Peace',
    'MonteCristo': 'Count of Monte Cristo',
    'DonQuixote': 'Don Quixote',
    'Frankenstein': 'Frankenstein',
    'Sherlock': 'Sherlock Holmes',
    'IndianWigwam': 'Indian in his Wigwam',
    'TopWorld': 'Top of the World',
    'TrentsCase': "Trent's Last Case",
    'GlimpsesMoon': 'The Glimpses of the Moon',
    'LookingBackward': 'Looking Backward',
    'OrganismWhole': 'The Organism as a Whole',
    'SophyKravonia': 'Sophy of Kravonia',
    'LureMask': 'The Lure of the Mask',
    'SunMaid': 'The Sun Maid',
    '34916': 'The Chainbearer',
    '51649': 'The Mysteries of London',
    '63912': 'American Encyclopedia'
}

def simplify_genre(g, title=''):
    if not isinstance(g, str): return 'Unknown'
    g = g.strip()
    
    # Title-based overrides for specific reference books
    t = title.lower()
    if 'origin of species' in t: return 'Science & Philosophy'
    if 'nietzsche' in t: return 'Science & Philosophy'
    if 'secretdoctrine' in t: return 'Science & Philosophy'
    if 'gone with the wind' in t: return 'Romance, Poetry & Drama'
    if 'les mis' in t or 'misérables' in t: return 'Fiction'
    if 'war and peace' in t: return 'Fiction'
    if 'crusoe' in t: return 'Adventure/Mystery/Sci-Fi'
    if 'roosevelt tome' in t: return 'Non-Fiction'
    if 'ice and fire tome' in t: return 'Adventure/Mystery/Sci-Fi'
    
    mapping = {
        'General Fiction': 'Fiction',
        'Fiction': 'Fiction',
        'Adventure': 'Adventure/Mystery/Sci-Fi',
        'Mystery/Crime': 'Adventure/Mystery/Sci-Fi',
        'Short Stories': 'Fiction',
        'Children/YA': 'Fiction',
        'Humor': 'Fiction',
        
        'Sci-Fi/Fantasy': 'Adventure/Mystery/Sci-Fi',
        'Mythology': 'Adventure/Mystery/Sci-Fi',
        'Cyberpunk': 'Adventure/Mystery/Sci-Fi',
        
        'Historical Fiction': 'Fiction',
        
        'Romance': 'Romance, Poetry & Drama',
        'Romance Drama': 'Romance, Poetry & Drama',
        
        'Poetry & Drama': 'Romance, Poetry & Drama',
        
        'Science': 'Science & Philosophy',
        'Science Book': 'Science & Philosophy',
        'Philosophy': 'Science & Philosophy',
        
        'Reference': 'Science & Philosophy',
        'Technical Reference': 'Science & Philosophy',
        
        'Biography': 'Non-Fiction',
        'Non-Fiction': 'Non-Fiction',
        'History': 'Non-Fiction',
        'Essays & Letters': 'Non-Fiction',
        'Religion': 'Non-Fiction',
        'Travel': 'Non-Fiction',
        'Other': 'Non-Fiction'
    }
    
    return mapping.get(g, 'Unknown')

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

def load_natural_texts_data(natural_texts_dir):
    """Load the natural texts metrics and compute PCA."""
    batch_files = sorted(glob.glob(os.path.join(natural_texts_dir, "Output", "batch_results", "batch_*.json")))
    
    if not batch_files:
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
    
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        df['ID_num'] = df['book_id'].astype(str).str.extract(r'_(\d+)$')[0].astype(float)
        if df['ID_num'].isna().all():
             df['ID_num'] = pd.to_numeric(df['book_id'], errors='coerce')

        df = df.merge(metadata[['ID', 'Title', 'FinalGenre', 'AuthorEra', 'Genre']], 
                     left_on='ID_num', right_on='ID', how='left')
        df.rename(columns={'Genre': 'SpecificGenre'}, inplace=True)
        df['FinalGenre'] = df.apply(lambda x: simplify_genre(x['FinalGenre'], str(x.get('Title', ''))), axis=1)
        
        if 'n_tokens' in df.columns:
            df['SizeCategory'] = df['n_tokens'].apply(assign_size_category)

        # Prefix eras for proper chronological sorting
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
        if 'AuthorEra' in df.columns:
            df['AuthorEra'] = df['AuthorEra'].map(lambda x: era_mapping.get(x, x))

        print(f"Loaded metadata with {len(metadata)} books")
    else:
        print(f"Warning: Metadata file not found at {metadata_path}")
        df['FinalGenre'] = 'Unknown'
        df['SizeCategory'] = 'Unknown'
    
    metric_mapping = {
        'beta': 'beta',
        'alpha': 'alpha',
        'H_WT_db4': 'H_WT_db4',
        'H_WT_db4_long': 'H_WT_db4_long',
        'H_DFA': 'H_DFA1',
        'H_DMA': 'H_DMA',
        'H_WDFA': 'H_WT_Haar',
        'H_DEA': 'D',
        'entropy': 'entropy_bits',
        'perplexity': 'perplexity',
        'lz_complexity': 'lz_complexity',
        'mi_decay': 'mi_decay_gamma'
    }
    
    metric_cols = []
    for target_name, source_name in metric_mapping.items():
        if source_name in df.columns:
            metric_cols.append(source_name)
            
    df_clean = df.dropna(subset=metric_cols)
    X = df_clean[metric_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=4)
    pca_coords = pca.fit_transform(X_scaled)
    
    df_clean = df_clean.copy()
    df_clean['PC1'] = pca_coords[:, 0]
    df_clean['PC2'] = pca_coords[:, 1]
    df_clean['PC3'] = pca_coords[:, 2]
    df_clean['PC4'] = pca_coords[:, 3]
    
    return df_clean, pca, scaler, metric_cols, metric_mapping

def project_book(metrics, pca, scaler, metric_cols):
    """Project book metrics onto PCA space."""
    values = [metrics.get(col, np.nan) for col in metric_cols]
    if any(np.isnan(values)):
        missing = [metric_cols[i] for i, v in enumerate(values) if np.isnan(v)]
        print(f"Warning: Missing metrics: {missing}")
        return None
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)
    coords = pca.transform(X_scaled)
    return coords[0]

def get_genre_color_map():
    """Get a color map for genres matching the simplified schema."""
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    genre_colors = {
        'Fiction': colors[4],            
        'Adventure/Mystery/Sci-Fi': colors[0], 
        'Romance, Poetry & Drama': colors[6], 
        'Science & Philosophy': colors[5], 
        'Non-Fiction': colors[2],        
        'Unknown': (0.5, 0.5, 0.5, 1.0)
    }
    return genre_colors

def plot_variable_loadings(ax, pca, metric_cols, pc_indices=(0, 1), scale=4):
    """Plot variable loadings as arrows."""
    pc_x, pc_y = pc_indices
    # Define statistical vs informational metrics
    informational = ['entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma', 'mi_tau_max', 'mi_n_lags']
    
    label_map = {
        'entropy_bits': 'Entropy',
        'perplexity': 'Perplexity',
        'lz_complexity': 'Lz Complexity',
        'mi_decay_gamma': 'MI Decay',
        'H_DMA': 'H_DMA',
        'H_WT_db4': 'H_WT_db4',
        'alpha': 'Alpha',
        'beta': 'Beta',
        'D': 'D'
    }
    
    # Simple label collision management
    placed_positions = []
    
    for i, var in enumerate(metric_cols):
        # Only filter scaling metrics starting with H_, but KEEP H_DMA and H_WT_db4
        if var.startswith('H_') and var not in ['H_DMA', 'H_WT_db4']:
            continue
            
        if var in informational:
            arrow_color = 'coral'
            edge_color = 'coral'
        else:
            arrow_color = 'teal' 
            edge_color = 'teal'
        
        x_loading = pca.components_[pc_x, i] * scale
        y_loading = pca.components_[pc_y, i] * scale
        
        ax.arrow(0, 0, x_loading, y_loading,
                head_width=0.2, head_length=0.2,
                fc=arrow_color, ec=edge_color, alpha=0.7, linewidth=3.0, zorder=30)
        
        # Position label very close to arrowhead (1.04x length)
        mag = np.sqrt(x_loading**2 + y_loading**2)
        lx, ly = x_loading * 1.04, y_loading * 1.04
        if var == 'H_DMA':
             lx, ly = x_loading * 1.02, y_loading * 1.02
        elif mag < 0.8: # Very short
             lx, ly = x_loading * 1.4, y_loading * 1.4
        elif mag < 1.5: # Short
             lx, ly = x_loading * 1.15, y_loading * 1.15
             
        # Collision avoidance Logic: Iterative relaxation with moderate repulsion
        for _ in range(12): 
            for px, py in placed_positions:
                dist = np.sqrt((lx-px)**2 + (ly-py)**2)
                rad = 0.4 if var == 'H_DMA' else 0.6
                if dist < rad: 
                    # Push away from the existing label
                    angle = np.arctan2(ly-py, lx-px)
                    force = (rad - dist) * 0.4
                    lx += np.cos(angle) * force
                    ly += np.sin(angle) * force
        
        placed_positions.append((lx, ly))
        
        disp_name = label_map.get(var, var)
        ax.text(lx, ly, disp_name, fontsize=11, ha='center', va='center', fontweight='bold',
               color=edge_color, zorder=31,
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor=edge_color, alpha=0.9, linewidth=1.5))


        ax.text(lx, ly, disp_name, fontsize=11, ha='center', va='center', fontweight='bold',
               color=edge_color, zorder=31,
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                        edgecolor=edge_color, alpha=0.9, linewidth=1.5))



def get_arrow_char(x, y):
    """Return unicode arrow based on vector direction."""
    angle = np.arctan2(y, x) * 180 / np.pi
    if angle < 0: angle += 360
    # Map to 8 directions
    # 0=E, 45=NE, 90=N, 135=NW, 180=W, 225=SW, 270=S, 315=SE
    idx = int((angle + 22.5) / 45) % 8
    arrows = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
    return arrows[idx]

def plot_composite_arrows(ax, pca, metric_cols, pc_indices=(0, 1), scale=5, plot_labels=False):
    """Plot composite arrows and return their direction characters."""
    pc_x, pc_y = pc_indices
    
    stat_metrics = ['alpha', 'beta', 'H_DFA1', 'H_DMA', 'H_WT_Haar', 'H_WT_db4', 'H_WT_db4_long']
    info_metrics = ['entropy_bits', 'perplexity', 'lz_complexity'] 
    #info_metrics = ['entropy_bits', 'perplexity', 'lz_complexity', 'mi_tau_max', 'mi_n_lags'] 
    
    groups = {
        'Stat.': {'metrics': stat_metrics, 'color': 'teal'},
        'Info.': {'metrics': info_metrics, 'color': 'coral'}
    }
    
    arrow_chars = {}
    
    for label, config in groups.items():
        indices = [i for i, col in enumerate(metric_cols) if col in config['metrics']]
        if not indices: 
            arrow_chars[config['color']] = '?'
            continue
        
        mean_x = np.mean(pca.components_[pc_x, indices])
        mean_y = np.mean(pca.components_[pc_y, indices])
        
        # Determine Unicode Arrow
        arrow_char = get_arrow_char(mean_x, mean_y)
        arrow_chars[config['color']] = arrow_char
        
        x_loading = mean_x * scale
        y_loading = mean_y * scale
        
        color = config['color']
        
        # Plot Arrow
        ax.arrow(0, 0, x_loading, y_loading,
                head_width=0.4, head_length=0.4,
                fc=color, ec=color, alpha=0.6, linewidth=2.5, zorder=10) # zorder 10 (below trajectories=20, above scores=1)
                
        if plot_labels:
            lx, ly = x_loading * 1.15, y_loading * 1.15
            ax.text(lx, ly, label, fontsize=12, ha='center', va='center', fontweight='bold',
                color=color, zorder=36,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=color, alpha=0.8, linewidth=1.5))
                            
    return arrow_chars['teal'], arrow_chars['coral']
def compute_comparison_stats(df_natural, df_generated, metric_mapping):
    """Compute comparison statistics between natural and generated texts."""
    stats_data = []
    metrics = ['beta', 'alpha', 'H_DFA1', 'H_DMA', 'H_WT_db4', 'entropy_bits', 'perplexity', 'lz_complexity', 'mi_decay_gamma']
    
    for metric in metrics:
        if metric in df_natural.columns and metric in df_generated.columns:
            nat_vals = df_natural[metric].dropna()
            gen_vals = df_generated[metric].dropna()
            
            if len(nat_vals) == 0 or len(gen_vals) == 0: continue

            nat_mean, nat_std = nat_vals.mean(), nat_vals.std()
            gen_mean, gen_std = gen_vals.mean(), gen_vals.std()
            diff = gen_mean - nat_mean
            pct_diff = (diff / nat_mean * 100) if nat_mean != 0 else 0

            pval = np.nan
            if stats is not None and len(nat_vals) > 1 and len(gen_vals) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tstat, pval = stats.ttest_ind(nat_vals, gen_vals, equal_var=False)
            
            sig = ""
            if not np.isnan(pval):
                if pval < 0.001: sig = "***"
                elif pval < 0.01: sig = "**"
                elif pval < 0.05: sig = "*"
            
            stats_data.append({
                'Metric': metric,
                'Nat Mean': f"{nat_mean:.3f}",
                'Gen Mean': f"{gen_mean:.3f}",
                'Mean Diff': f"{diff:+.3f}",
                '% Diff': f"{pct_diff:+.1f}%",
                'Nat Std': f"{nat_std:.3f}",
                'Gen Std': f"{gen_std:.3f}",
                'Sig': sig,
            })
    return pd.DataFrame(stats_data)



def plot_density_contour(ax, points, color, kde_threshold=8, two_level=False):
    if len(points) < kde_threshold or gaussian_kde is None: return
    try:
        kde = gaussian_kde(points.T)
        x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
        y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kde(positions).T, xx.shape)
        contour_color = to_rgba(color)
        if two_level:
            density_flat = density.flatten()
            level_50 = np.percentile(density_flat[density_flat > 0], 50)
            ax.contour(xx, yy, density, levels=[level_50], colors=[contour_color], linewidths=2.5, alpha=0.8, linestyles=':')
            level_max = density.max()
            level_95 = level_max * 0.05
            levels = np.linspace(level_95, level_max, 10)
            ax.contourf(xx, yy, density, levels=levels, colors=[list(contour_color)[:3] + [a] for a in np.linspace(0, 0.5, 9)])
            ax.contour(xx, yy, density, levels=levels, colors=[contour_color], linewidths=1.5, alpha=0.6, linestyles=':')
        else:
            level_max = density.max()
            level_95 = level_max * 0.05
            levels = np.linspace(level_95, level_max, 10)
            ax.contourf(xx, yy, density, levels=levels, colors=[list(contour_color)[:3] + [a] for a in np.linspace(0, 0.5, 9)])
            ax.contour(xx, yy, density, levels=levels, colors=[contour_color], linewidths=1.5, alpha=0.6, linestyles=':')
    except: pass

def plot_projection(df_natural, projected_books, pca, metric_cols, output_path, genre_colors, comp_stats):
    unique_sizes = sorted(df_natural['SizeCategory'].unique())
    cmap = plt.cm.viridis
    size_colors = {cat: cmap(i/max(1, len(unique_sizes)-1)) for i, cat in enumerate(unique_sizes)}
    n_rows = ((len(unique_sizes) + 2) // 3) + 1
    fig = plt.figure(figsize=(24, 7*n_rows))
    pc1_vals, pc2_vals = df_natural['PC1'].values, df_natural['PC2'].values
    p1_5, p1_95 = np.percentile(pc1_vals, [5, 95])
    p2_5, p2_95 = np.percentile(pc2_vals, [5, 95])
    margin = 0.3
    xlim = (p1_5 - margin * (p1_95 - p1_5), p1_95 + margin * (p1_95 - p1_5))
    ylim = (p2_5 - margin * (p2_95 - p2_5), p2_95 + margin * (p2_95 - p2_5))
    
    ax_summary = plt.subplot2grid((n_rows, 3), (0, 0), colspan=2, rowspan=2, fig=fig)
    pts_all = df_natural[['PC1', 'PC2']].values
    if len(pts_all) >= 8: plot_density_contour(ax_summary, pts_all, (0.5, 0.5, 0.5, 1.0))
    ax_summary.scatter(pts_all[:, 0], pts_all[:, 1], c='gray', s=10, alpha=0.05, edgecolors='none', zorder=1)
    
    median_coords = []
    for cat in unique_sizes:
        mask = df_natural['SizeCategory'] == cat
        subset = df_natural[mask]
        if len(subset) > 0:
            mx, my = subset['PC1'].median(), subset['PC2'].median()
            ax_summary.scatter(mx, my, c=[size_colors.get(cat, 'gray')], s=300, marker='*', edgecolors='black', zorder=10, label=cat)
            if 'Unknown' not in cat: median_coords.append((mx, my))
    if median_coords:
        mxs, mys = zip(*median_coords)
        ax_summary.plot(mxs, mys, color='black', linestyle=':', linewidth=1.5, zorder=9, alpha=0.8)
    
    from collections import defaultdict
    book_grp = defaultdict(list)
    for name, coords, genre, n_tokens, model_type, is_outlier, temp_level in projected_books:
        book_grp[name].append((n_tokens, coords, genre, model_type, is_outlier, temp_level))
    
    human_handles = []
    for name, pts in book_grp.items():
        pts.sort(key=lambda x: x[0] or 0)
        is_h, is_o, genre = (pts[0][3] == 'Human'), pts[0][4], pts[0][2]
        color, m, ec, lc = genre_colors.get(genre, 'red'), ('^' if is_o else ('D' if is_h else 'o')), ('red' if is_o else ('blue' if is_h else 'black')), ('red' if is_o else ('blue' if is_h else 'black'))
        ax_summary.plot([p[1][0] for p in pts], [p[1][1] for p in pts], color=color, linestyle='-', linewidth=2, alpha=0.7)
        tot_b = pts[-1][0] if is_h else None
        for i, (nt, coord, _, _, _, _) in enumerate(pts):
            ntv = nt or 0
            s_b = (250 * (ntv/tot_b)) if (is_h and tot_b) else ((50 + ((min(ntv, 1500000)/100000.0)**2)*15)/2.0)
            ax_summary.scatter(coord[0], coord[1], c=[color], marker=m, s=s_b, edgecolors=ec, linewidths=0.5, zorder=20)
            if i == len(pts)-1:
                if is_h:
                    fn = NAME_MAP_CLEAN.get(name, name)
                    human_handles.append(mlines.Line2D([],[], color=color, marker=m, linestyle='-', label=fn, markeredgecolor=ec))
                    ax_summary.text(coord[0]+0.15, coord[1], f"{fn} ({int(ntv/1000)}k)", fontsize=8, fontweight='bold', color=lc, ha='left', va='center', zorder=26, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=lc, alpha=0.8, linewidth=1.0))
                    ax_summary.text(coord[0], coord[1], name.split()[-1], fontsize=8, fontweight='bold', color='white', ha='center', va='center', zorder=25)

    plot_variable_loadings(ax_summary, pca, metric_cols, (0, 1))
    ax_summary.set_xlim(xlim); ax_summary.set_ylim(ylim); ax_summary.grid(True, alpha=0.3)
    ax_summary.set_title('Summary: Size Medians & All Trajectories', fontsize=14, weight='bold')
    
    subplot_positions = [(r, c) for r in range(n_rows) for c in range(3) if not (r < 2 and c < 2)]
    for idx, cat in enumerate(unique_sizes):
        if idx >= len(subplot_positions): break
        row, col = subplot_positions[idx]
        ax = plt.subplot2grid((n_rows, 3), (row, col), fig=fig)
        mask = df_natural['SizeCategory'] == cat
        subset = df_natural[mask]
        pts_nat = subset[['PC1', 'PC2']].values
        if len(pts_nat) >= 8: plot_density_contour(ax, pts_nat, size_colors.get(cat, 'gray'))
        ax.scatter(pts_nat[:, 0], pts_nat[:, 1], c=[size_colors.get(cat, 'gray')], s=10, alpha=0.08, edgecolors='none')
        if len(pts_nat) > 0: ax.scatter(np.median(pts_nat[:, 0]), np.median(pts_nat[:, 1]), c=[size_colors.get(cat,'gray')], s=300, marker='*', edgecolors='black', zorder=10)

        for b_name, b_pts in book_grp.items():
            for nt, coord, genre, mtype, is_o, t_level in b_pts:
                if assign_size_category(nt or 0) == cat:
                    m, ec = ('^' if is_o else ('D' if mtype == 'Human' else 'o')), ('red' if is_o else ('blue' if mtype == 'Human' else 'black'))
                    ax.scatter(coord[0], coord[1], c=[genre_colors.get(genre, 'red')], marker=m, s=50+(min(nt or 0, 1500000)/1000000.0)*350, edgecolors=ec, linewidth=0.5, zorder=20)
        ax.set_title(f'Size: {cat}', fontsize=12, weight='bold')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f"{len(subset)} natural books", transform=ax.transAxes, fontsize=9, fontstyle='italic', color='gray', va='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, linewidth=0))

    if human_handles:
        uh = {h.get_label(): h for h in human_handles}
        ax_summary.legend(handles=[v for k,v in sorted(uh.items())], loc='lower center', fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.15))
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close()

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
    parser.add_argument('--find-centrals', action='store_true', help='Find the most central natural books for each genre')
    args = parser.parse_args()
    
    try:
        pc_indices = tuple(int(x.strip()) - 1 for x in args.components.split(','))
        if len(pc_indices) != 2: raise ValueError
    except: parser.error("Components must be two comma-separated integers (e.g. '1,2')")
    
    if not args.book and not args.metrics:
        default_metrics = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generated_texts_og', 'experimental_metrics.json')
        if os.path.exists(default_metrics): args.metrics = default_metrics
        else:
            # Fallback to gmetrics_all.json if experimental_metrics is missing
            fallback = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'generated_texts_og', 'Output', 'gmetrics_all.json')
            if os.path.exists(fallback): args.metrics = fallback
    
    print("Loading natural texts data and fitting PCA...")
    df_natural, pca, scaler, metric_cols, metric_mapping = load_natural_texts_data(args.natural_texts_dir)
    print(f"Loaded {len(df_natural)} natural texts")
    
    if args.find_centrals:
        print("\n=== FINDING CENTRAL BOOKS PER GENRE ===")
        print("(Calculating distance to genre median in current PCA space)")
        unique_genres = [g for g in df_natural['FinalGenre'].unique() if g != 'Unknown']
        for genre in sorted(unique_genres):
            subset = df_natural[df_natural['FinalGenre'] == genre].copy()
            if len(subset) < 5: continue
            
            med_x = subset['PC1'].median()
            med_y = subset['PC2'].median()
            
            subset['dist'] = np.sqrt((subset['PC1'] - med_x)**2 + (subset['PC2'] - med_y)**2)
            
            print(f"\nGenre: {genre}")
            print(f"  Median (Star): ({med_x:.3f}, {med_y:.3f})")
            
            # Sort by distance
            top_centrals = subset.sort_values('dist').head(10)
            print(f"  Top 10 candidates (Sorted by distance):")
            for _, row in top_centrals.iterrows():
                print(f"    - ID: {int(row['ID']) if not pd.isna(row['ID']) else row['book_id']}, Dist: {row['dist']:.3f}, Title: {str(row.get('Title', 'Unknown'))[:50]}")
        return 0
    
    books_to_plot, generated_metrics_list = [], []
    if args.book:
        metrics = compute_book_metrics(args.book)
        books_to_plot.append((os.path.basename(args.book).replace('.txt', ''), metrics))
        generated_metrics_list.append(metrics)
    
    if args.metrics:
        with open(args.metrics, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                books_to_plot.append((item.get('file', 'Unknown').replace('.txt', ''), item))
                generated_metrics_list.append(item)
        elif isinstance(data, dict):
            books_to_plot.append((os.path.basename(args.metrics).replace('.json', ''), data))
            generated_metrics_list.append(data)
            
    # Autoload Reference Books removed to avoid duplicates (now handled by gather_metrics)
    # ref_dirs = ...
    
    df_gen = pd.DataFrame(generated_metrics_list)
    print("\nComputing comparison statistics...")
    comp_stats = compute_comparison_stats(df_natural, df_gen, metric_mapping)
    
    print(f"\nProjecting {len(books_to_plot)} book(s)...")
    projected_books = []
    genre_counters, file_rep_map = {}, {}
    SHORT_CODES = {'Historical Fiction': 'Hist', 'Romance Drama': 'Rom', 'Cyberpunk': 'Cyber', 'Science Book': 'Sci', 'Philosophy': 'Phil', 'Technical Reference': 'Tech', 'Adventure': 'Adv', 'Mystery/Crime': 'Myst', 'Sci-Fi/Fantasy': 'SFF', 'Fantasy': 'Fan', 'Romance': 'Rom', 'Poetry & Drama': 'Drama'}
    
    for b_name, metrics in books_to_plot:
        coords = project_book(metrics, pca, scaler, metric_cols)
        if coords is None: continue
        raw_g = metrics.get('genre', 'Unknown')
        model_type = metrics.get('model', 'Mistral')
        
        # Fallback: Extract from filename if following LONGBOOK convention
        if raw_g == 'Unknown' and 'file' in metrics:
            fname = metrics['file']
            if 'LONGBOOK__' in fname:
                parts = fname.split('__')
                if len(parts) >= 3:
                    # Model (index 1), Genre (index 2)
                    raw_model = parts[1]
                    raw_g_from_file = parts[2]
                    
                    raw_g = raw_g_from_file.replace('_', ' ').title()
                    
                    if 'mistral' in raw_model.lower(): model_type = 'Mistral'
                    elif 'llama' in raw_model.lower(): model_type = 'Llama'
                    elif 'qwen' in raw_model.lower(): model_type = 'Qwen'
        
        # Use Title from metrics if available
        full_title = metrics.get('title', b_name)
        if len(full_title) > 25:
            short_l = full_title
        else:
            short_l = full_title
        
        n_toks = metrics.get('n_tokens') or metrics.get('tokens_count')
        temp_level = metrics.get('temperature_level', 'High')
        final_g = simplify_genre(raw_g, full_title)
        projected_books.append((short_l, coords, final_g, n_toks, model_type, metrics.get('is_outlier', False), temp_level))
    
    out_p = args.output or (args.book.replace('.txt', '_pca.jpg') if args.book else args.metrics.replace('.json', '_pca.jpg'))
    print("\nGenerating plots...")
    plot_genre_breakdown_combined(df_natural, projected_books, pca, get_genre_color_map(), metric_cols, out_p.replace('.jpg', '_genre_grid_combined.jpg'))
    return 0

def plot_genre_breakdown_combined(df_natural, projected_books, pca, genre_colors, metric_cols, output_path):
    """Merged Figure: Grid of genres showing Natural Density/Dots + Generated Trajectories + Reference Books (3x3)."""
    from collections import defaultdict
    from matplotlib import colors
    import matplotlib.lines as mlines
    
    print("Generating Figure Combined (Grid 3x3)...")
    
    # MANUAL SELECTION OF REFERENCE BOOKS
    # Edit this list to select which reference books to display.
    # Use partial matches (e.g. 'Ice_and_Fire', 'War_and_Peace').
    # MANUALLY SELECT WHICH REFERENCE BOOKS APPEAR IN WHICH GENRE GRID
    GENRE_TO_REFS = {
        'Fiction': ['War and Peace', 'Les Misérables', 'Fall of Giants'],
        'Adventure/Mystery/Sci-Fi': ['A Song of Ice and Fire Tome1', 'name of the rose', 'Robinson Crusoe'],
        'Non-Fiction': ['Pillars of the earth', 'Malcolm X', 'Roosevelt Tome1'],
        'Romance, Poetry & Drama': ['Gone with the Wind', 'LadyIsle', 'Marriage'],
        'Science & Philosophy': ['Origin of Species', 'Nietzsche', 'SecretDoctrine']
    }
    
    unique_genres = sorted([g for g in df_natural['FinalGenre'].unique() if g != 'Unknown'])
    if len(unique_genres) > 8: unique_genres = unique_genres[:8]

    fig = plt.figure(figsize=(18, 18))
    pc1_nat, pc2_nat = df_natural['PC1'], df_natural['PC2']

    ax_ov = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
    colors_ov = 'gray' # Monochromatic overall plot
    ax_ov.scatter(df_natural['PC1'], df_natural['PC2'], c=colors_ov, s=15, alpha=0.15, edgecolors='none', zorder=1)
    plot_density_contour(ax_ov, df_natural[['PC1', 'PC2']].values, 'gray')

    # Large PC Labels
    ax_ov.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=16, fontweight='bold', labelpad=10)
    ax_ov.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=16, fontweight='bold', labelpad=10)


    ov_x_min, ov_x_max = np.percentile(pc1_nat, 3), np.percentile(pc1_nat, 97)
    ov_y_min, ov_y_max = np.percentile(pc2_nat, 3), np.percentile(pc2_nat, 97)
    
    # Store initial limits
    xlims = [ov_x_min - (ov_x_max-ov_x_min)*0.15, ov_x_max + (ov_x_max-ov_x_min)*0.15]
    ylims = [ov_y_min - (ov_y_max-ov_y_min)*0.15, ov_y_max + (ov_y_max-ov_y_min)*0.30]

    # Dynamically expand limits if chaotic book is present (to ensure it is visible)
    # This requires looking for the chaotic book BEFORE setting limits?
    # Or we can just set them later. 
    # But density lines are plotted based on data, not limits.
    # Let's set initial limits here and potentially update them later if chaos book is plotted.
    
    ax_ov.set_xlim(xlims)
    ax_ov.set_ylim(ylims)
    
    # Updated Main Title and Subtitle
    # Updated Main Title and Subtitle
    ax_ov.set_title("(A) Overall : Scores density distribution of natural books in PC1xPC2\n", 
                   fontweight='bold', fontsize=16)
    # Add subtitle as separate text for styling
    ax_ov.text(0.5, 1.01, "Contour lines are at 5% percentile steps", transform=ax_ov.transAxes,
               fontsize=12, color='gray', fontstyle='italic', ha='center', va='bottom')
    
    ax_ov.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=10, fontweight='normal')
    ax_ov.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=10, fontweight='normal')
    ax_ov.grid(True, alpha=0.3)


    # Plot Variable Loadings on Overall (Blue for STAT, Coral for INFO)
    plot_variable_loadings(ax_ov, pca, metric_cols, scale=5)
    
    # --- Plot Size & Era Medians on Overall ---
    summary_handles = []
    
    # Size Category Mean (Stars)
    if 'SizeCategory' in df_natural.columns:
        size_cats = sorted([s for s in df_natural['SizeCategory'].unique() if isinstance(s, str) and s != 'Unknown'])
        size_colors = plt.cm.spring(np.linspace(0.1, 0.9, len(size_cats)))
        size_colors = size_colors[::-1]
        size_pts = []
        for i, sc in enumerate(size_cats):
            sub = df_natural[df_natural['SizeCategory'] == sc]
            if len(sub) >= 20:
                mx, my = sub['PC1'].mean(), sub['PC2'].mean()
                size_pts.append((mx, my))
                c = size_colors[i]
                ax_ov.scatter(mx, my, marker='*', s=800, color=c, edgecolor='white', linewidth=1.5, zorder=110, alpha=1.0)
                
                # Format label for legend: "Mean natural book within 100-150k tokens"
                range_str = sc.split('(')[1].replace(')', '')
                summary_handles.append(mlines.Line2D([], [], color=c, marker='*', linestyle='None', 
                                                   label=f"{range_str}", markersize=14, markeredgecolor='white'))
        if len(size_pts) > 1:
            xs, ys = zip(*size_pts)
            ax_ov.plot(xs, ys, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=100)
    
    # Legend for overall logic
    if summary_handles:
        ax_ov.legend(handles=summary_handles, loc='upper left', fontsize=10, framealpha=0.95, 
                    ncol=1, title="Mean score of natural books within size bins:", 
                    bbox_to_anchor=(0.02, 0.98))

    # Pre-process Reference Books logic
    book_groups = defaultdict(list)
    for name, coords, genre, n_tokens, model_type, is_outlier, temp_level in projected_books:
        book_groups[name].append((n_tokens, coords, genre, model_type, is_outlier, temp_level))

    # --- Overall Plot: Add Generated Book Trajectories ---
    # PLOT SPECIFIC CHAOTIC BOOK TRAJECTORY HERE
    '''
    chaos_book_name = "_chaos4m_20251230_012534_the_irresistible_storm_thriller"
    # Try to find this book in book_groups
    chaos_entry = None
    for bname, entry in book_groups.items():
        # Clean both for comparison
        clean_key = bname.lower().replace('_', ' ')
        clean_target = "irresistible storm"
        if clean_target in clean_key:
             # Use the first match that is likely the one
             chaos_entry = entry
             # Use the first match that is likely the one
             chaos_entry = entry
             break
    
    # Debug: List all available keys in book_groups
    # print("DEBUG: Keys in book_groups:", list(book_groups.keys()))
    if not chaos_entry:
        print("DEBUG: Chaotic book not found. Trying approximate search...")
        for key in book_groups:
             if "_chaos" in key.lower() or "irresistible" in key.lower():
                 print(f"DEBUG: Found candidate: {key}")
                 chaos_entry = book_groups[key]
                 break
    
    if chaos_entry:
         print(f"Found chaotic book for Overall plot: {chaos_book_name}")
         points = chaos_entry
         points.sort(key=lambda x: x[0] if x[0] is not None else 0)
         
         xs, ys = [p[1][0] for p in points], [p[1][1] for p in points]
         
         # Dynamically expand limits if needed
         if xs and ys:
             c_x_min, c_x_max = min(xs), max(xs)
             c_y_min, c_y_max = min(ys), max(ys)
             
             # Expand with padding
             new_xlim = [min(xlims[0], c_x_min - 0.5), max(xlims[1], c_x_max + 0.5)]
             new_ylim = [min(ylims[0], c_y_min - 0.5), max(ylims[1], c_y_max + 0.5)]
             
             ax_ov.set_xlim(new_xlim)
             ax_ov.set_ylim(new_ylim)
             print(f"DEBUG: Expanded Overall limits to include Chaotic Book: X {new_xlim}, Y {new_ylim}")

         # Plot Trajectory
         # Color it distinctly, e.g., 'purple' or 'magenta' or even standard red
         c_color = 'magenta'
         
         ax_ov.plot(xs, ys, color='white', alpha=0.6, linewidth=4.0, zorder=40)
         ax_ov.plot(xs, ys, color=c_color, alpha=0.9, linewidth=2.0, zorder=41)
         
         # Nodes
         for i, (nt, coord, _, _, _, _) in enumerate(points):
             is_last = (i == len(points)-1)
             s_val = 300 if is_last else max(40, (nt/1000000)*350) if nt else 80
             
             ax_ov.scatter([coord[0]], [coord[1]], color=c_color, s=s_val, alpha=0.9, edgecolors='black', linewidth=1.0, zorder=42)
             
         # Label it
         if len(xs) > 0:
             ax_ov.text(xs[-1], ys[-1], "Chaotic Book", color=c_color, fontsize=10, fontweight='bold', 
                        ha='left', va='bottom', zorder=50, 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=c_color))
    else:
         print(f"Warning: Chaotic book '{chaos_book_name}' not found in loaded metrics.")
    '''

    # --- 2. Genre-Specific Subplots (Remaining 5 slots) ---
    genre_positions = [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    for idx, genre in enumerate(unique_genres):
        if idx >= len(genre_positions): break
        row, col = genre_positions[idx]
        ax = plt.subplot2grid((3, 3), (row, col))
        color = genre_colors.get(genre, 'gray')
        rgb = colors.to_rgb(color)
        dark_c = [max(0, val * 0.5) for val in rgb]
        mask = df_natural['FinalGenre'] == genre
        nat_pts = df_natural[mask][['PC1', 'PC2']].values
        avg_toks = df_natural[mask]['n_tokens'].mean() if 'n_tokens' in df_natural.columns else 0
        
        # Plot Natural Density
        if len(nat_pts) > 0:
            if len(nat_pts) > 5: plot_density_contour(ax, nat_pts, color)
            ax.scatter(nat_pts[:,0], nat_pts[:,1], c=[color], s=15, alpha=0.15, edgecolors='none', zorder=1)
            # Removed median marker as requested
        
        # Legend Handles
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
        legend_handles = []
        
        # Shared Generated Book Handles
        target_steps = [100000, 200000, 400000, 600000, 1000000]
        closest_step = min(target_steps, key=lambda x: abs(x - avg_toks))
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', markerfacecolor=color, markeredgecolor='black', markersize=10, label='Generated books (high temperature)'))
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', markerfacecolor=color, markeredgecolor='black', markersize=10, alpha=0.15, label='Generated books (low temperature)'))
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', markerfacecolor='red', markeredgecolor='black', markersize=10, label=f'Partial generations ({int(closest_step/1000)}k)'))
        
        # --- Plot Generated Books (Model != Human) ---
        g_books = [p for p in book_groups.items() if p[1][0][2] == genre and p[1][0][3] != 'Human']
        # Sort generated books?
        g_books.sort(key=lambda x: x[0])
        
        # Counters for independent 1,2,3 numbering for High/Low runs
        low_counter = 0
        high_counter = 0
        
        for g_idx, (name, points) in enumerate(g_books):
             points.sort(key=lambda x: x[0] if x[0] is not None else 0)
             xs, ys = [p[1][0] for p in points], [p[1][1] for p in points]
             
             is_low_temp = (len(points) > 0 and points[0][5] == 'Low')
             line_alpha = 0.15 if is_low_temp else 1.0
             bg_alpha = 0.05 if is_low_temp else 0.5
             node_alpha = 0.10 if is_low_temp else 0.8
             term_alpha = 0.20 if is_low_temp else 1.0
             
             # Draw trajectory line: high-contrast background glow + foreground line (equalized to 2.5)
             ax.plot(xs, ys, color='white', alpha=bg_alpha, linewidth=5.0, zorder=19) 
             ax.plot(xs, ys, color='black', alpha=line_alpha, linewidth=2.5, zorder=20) 
             # Find node closest to category average length
             closest_idx = -1
             if len(points) > 0:
                 diffs = [abs((p[0] or 0) - avg_toks) for p in points]
                 closest_idx = np.argmin(diffs)

             # Intermediate nodes: matched to reference pentagon scaling
             if len(xs) > 1:
                 for i in range(len(xs)-1):
                     is_last = (i == len(points)-1)
                     p = points[i]
                     n_color = 'red' if i == closest_idx else color
                     #s_val = max(50, (p[0] / 1000000) * 450) if p[0] else 100
                     s_val = 500 if is_last else max(20, (p[0]/1000000)*250) if p[0] else 40
                     
                     ax.scatter([xs[i]], [ys[i]], color=n_color, s=s_val, alpha=node_alpha, edgecolors='black', linewidth=1.0, zorder=21)
             
             # Terminal node: Large and emphasized (matching pentagon terminal size)
             if xs:
                 token_size = 500 # Default terminal size
                 if points[-1][0]:
                      token_size = max(450, (points[-1][0] / 1000000) * 250)
                 
                 is_term_closest = (closest_idx == len(points)-1)
                 term_color = 'red' if is_term_closest else color
                 
                 ax.scatter([xs[-1]], [ys[-1]], color=term_color, s=token_size, alpha=term_alpha, edgecolors='black', zorder=22)
                 # Label number inside terminal point
                 if is_low_temp:
                     low_counter += 1
                     g_num = str(low_counter)
                 else:
                     high_counter += 1
                     g_num = str(high_counter)
                     
                 ax.text(xs[-1], ys[-1], g_num, color='white', fontsize=15, fontweight='bold', ha='center', va='center', zorder=23,
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor=term_color, edgecolor='black', alpha=term_alpha, linewidth=1.0))


        # --- Plot Reference Books (Model == Human) ---
        # Find Reference Books for this genre
        target_refs = GENRE_TO_REFS.get(genre, [])
        r_books = []
        for name, entry in book_groups.items():
            if entry[0][3] == 'Human': # Only Humans
                # Robust matching: handle underscores in filenames vs spaces in target_refs
                clean_name = name.lower().replace('_', ' ')
                if any(sel.lower() in clean_name for sel in target_refs):
                    r_books.append((name, entry))
        
        if r_books:
            print(f"Genre {genre}: Matched {len(r_books)} reference books: {[b[0] for b in r_books]}")
        # Filter by MANUAL_REF_SELECTION if not empty
        # Priority sort
        # r_books.sort(...) # Skipping auto-sort
        # Dynamic Limit: Keep all 'Ice and Fire' if present, otherwise top 3
        r_books.sort(key=lambda x: x[0])
        
        for rep_idx, (name, points) in enumerate(r_books):
             import string
             letter = string.ascii_lowercase[rep_idx] # a, b, c
             symbol = 'p' # Pentagon as requested
             points.sort(key=lambda x: x[0] or 0)
             xs, ys = [p[1][0] for p in points], [p[1][1] for p in points]
             
             # Reference Styles: Solid line, Heavy marker
             ax.plot(xs, ys, color=color, linestyle='-', linewidth=2.5, alpha=0.9, zorder=15)
             
             total_toks = points[-1][0] if len(points) else 0
             for i, (nt, coord, _, _, _, _) in enumerate(points):
                is_last = (i == len(points)-1)
                s_val = 600 if is_last else max(50, (nt/1000000)*500) if nt else 100
                
                #ax.scatter(coord[0], coord[1], c=[color], marker=symbol, s=s_val, edgecolors='white', linewidths=0.5, zorder=20)
                ax.scatter(coord[0], coord[1], c=[color], marker=symbol, s=s_val, edgecolors='white', zorder=20)
                
                if is_last:
                     ax.text(coord[0], coord[1], letter, color='white', fontsize=16, fontweight='bold', ha='center', va='center', zorder=25)
                     
                     disp_name = NAME_MAP_CLEAN.get(name, name)
                     if '_' in disp_name and ' ' not in disp_name: disp_name = disp_name.replace('_', ' ').title()
                     ref_label = f"{letter}: {disp_name} ({int(total_toks/1000)}k)"
                     legend_handles.append(mlines.Line2D([], [], color='white', marker='p', markerfacecolor=color, markeredgecolor='none', markersize=10, label=ref_label))

        # Axis limits based on natural points + reference books + generated books
        all_xs = [] # Only trajectories determine expansion beyond contours
        all_ys = []
        # Add generated & reference points
        for name, points in g_books + r_books:
            all_xs.extend([p[1][0] for p in points])
            all_ys.extend([p[1][1] for p in points])
            
        lx_min, lx_max = np.percentile(nat_pts[:,0], 2), np.percentile(nat_pts[:,0], 98)
        ly_min, ly_max = np.percentile(nat_pts[:,1], 2), np.percentile(nat_pts[:,1], 98)
        
        if all_xs:
            lx_min, lx_max = min(lx_min, min(all_xs)), max(lx_max, max(all_xs))
            ly_min, ly_max = min(ly_min, min(all_ys)), max(ly_max, max(all_ys))
            
        # Pad limits
        pad_factor = 0.35 if ('Adventure' in genre or 'Fiction' == genre) else 0.25
        ax.set_xlim(lx_min - (lx_max-lx_min)*pad_factor, lx_max + (lx_max-lx_min)*pad_factor)
        ax.set_ylim(ly_min - (ly_max-ly_min)*pad_factor, ly_max + (ly_max-ly_min)*pad_factor)
        
        # Letter Prefix (B, C, D, E, F) - Overall is A
        import string
        plot_letter = string.ascii_uppercase[idx+1]
        
        # Override for specific subplots
        if plot_letter == 'C':
            ax.set_xlim(right=5.0)
            ax.set_ylim(-6.0, 5.0)
        elif plot_letter == 'B':
            ax.set_xlim(-4.50, 6.0)
            ax.set_ylim(-6.0, 4.0)
        elif plot_letter == 'D':
            ax.set_xlim(-6.0, 8.0)
            ax.set_ylim(-6.0, 5.0)
        elif plot_letter == 'E':
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(-6.0, 5.0)
        elif plot_letter == 'F':
            ax.set_xlim(-6.0, 4.0)
            ax.set_ylim(-7.0, 4.0)
            
        ax.set_title(f"({plot_letter}) {genre}", fontweight='bold', fontsize=14, color=color); ax.grid(True, alpha=0.3)
        
        # Plot Composite Arrows (Stat/Info) in each subplot
        blue_arrow, coral_arrow = plot_composite_arrows(ax, pca, metric_cols, scale=5, plot_labels=False)

        # Build Subplot Description Legend
        subset = df_natural[mask]
        num_books = len(subset)
        if 'SpecificGenre' in subset.columns:
            top_subs = subset['SpecificGenre'].value_counts().head(3).index.tolist()
            subgenre_str = ", ".join([str(s).replace('/', ' ') for s in top_subs])
        else:
            subgenre_str = genre
            
        # Line 1: Subgenres (Gray)
        ax.text(0.04, 0.99, subgenre_str, transform=ax.transAxes, fontsize=8, fontstyle='italic', color='gray', va='top')
        
        # Line 2: Book Count (Gray) 
        ax.text(0.04, 0.96, f"{num_books} books", transform=ax.transAxes, fontsize=9, fontstyle='italic', color='gray', va='top')
        
        # Line 3: Avg Len (Red) - Restore red color
        ax.text(0.04, 0.93, f"Mean length: {int(avg_toks/1000)}k", transform=ax.transAxes, fontsize=9, fontstyle='italic', color='crimson', va='top')
        
        # Line 4: Coherence Arrows 
        ax.text(0.04, 0.90, f"Coherence directions:", transform=ax.transAxes, fontsize=9, fontstyle='italic', color='gray', va='top')
        ax.text(0.07, 0.87, f"{blue_arrow} statistical, ", transform=ax.transAxes, fontsize=9, fontstyle='italic', fontweight='normal', color='teal', va='top')
        ax.text(0.07, 0.84, f"{coral_arrow} informational ", transform=ax.transAxes, fontsize=9, fontstyle='italic', fontweight='normal', color='coral', va='top')
        
        if legend_handles:
            # Custom legend placement: right for Adventure, left for others
            loc = 'lower right' if ('Adventure' in genre or plot_letter in ['D', 'E', 'F']) else 'lower left'
            ax.legend(handles=legend_handles, loc=loc, fontsize=9, framealpha=0.9, facecolor='white', edgecolor='gray')

        #desc_txt = GENRE_DESCRIPTIONS.get(genre, 'Genre description')
        #ax.text(0.04, 0.91, desc_txt, transform=ax.transAxes, fontsize=9, fontstyle='italic', color='gray', va='top')

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Combined Figure to {output_path}')

def compute_book_metrics(book_path):
    with open(book_path, 'r', encoding='utf-8') as f: text = f.read()
    tokens = Tokenizer(kind="gpt4").encode(text)
    return CoherenceMetrics(tokens).compute_all()

if __name__ == '__main__':
    sys.exit(main())
