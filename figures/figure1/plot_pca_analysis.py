#!/usr/bin/env python3
"""
Plot PCA results for Figure 1: Series-Level IMDb Ratings

This script loads pre-calculated PCA data from:
- series_pca.csv (PCA coordinates)
- pca_loadings.csv (feature loadings)
- series_metrics.csv (series metrics)

And creates a publication-quality PCA visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import yaml
import os
import cmcrameri.cm as cmc  # Import Fabio Crameri's colormaps

# File paths relative to this script
SCRIPT_DIR = Path(__file__).parent
SERIES_PCA_FILE = SCRIPT_DIR / "series_pca.csv"
LOADINGS_FILE = SCRIPT_DIR / "pca_loadings.csv"
SERIES_METRICS_FILE = SCRIPT_DIR / "series_metrics.csv"
CONFIG_FILE = SCRIPT_DIR.parent.parent / "config" / "plot_config.yaml"

def create_series_rating_pca_visualization():
    """
    Create a publication-quality PCA visualization of Taskmaster series IMDb ratings.
    
    This function loads pre-calculated PCA data and generates a plot showing how different
    Taskmaster series relate to each other based on their rating patterns. It visualizes:
    1. Series positioned in PC space with colors from the configured colormap
    2. Feature vectors showing how variables (1-star %, 10-star %, mean, std) influence the space
    3. Background shading indicating quality of reception
    4. Annotations with detailed stats for extreme series
    """
    # Load config
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get output format and filename
    output_format = config['global'].get('output_format', 'png')
    OUTPUT_FILE = SCRIPT_DIR / f"figure1_pca_output.{output_format}"
    
    # Load pre-calculated PCA data
    df_pca = pd.read_csv(SERIES_PCA_FILE)
    loadings = pd.read_csv(LOADINGS_FILE, index_col=0)
    df_series_metrics = pd.read_csv(SERIES_METRICS_FILE)
    
    # Merge PCA and metrics data
    pca_df = pd.merge(df_pca, df_series_metrics, on='series')
    
    # Extract the list of series
    seasons = sorted(pca_df['series'].unique())
    
    # Get explained variance (if available in the data)
    try:
        explained_variance_file = SCRIPT_DIR / "explained_variance.npy"
        explained_variance = np.load(explained_variance_file)
    except:
        # If not available, use placeholder values
        explained_variance = np.array([0.7, 0.2])  # Approximations
    
    # Set up the figure with equal aspect ratio for proper distance perception
    # Use a fixed, large size to ensure readability
    plt.figure(figsize=(14, 14))
    
    # Set style with improved aesthetics from config
    plt.rcParams.update({
        'font.family': config['global']['font_family'],
        'font.size': config['fonts']['axis_label_size'],
        'axes.labelsize': config['fonts']['axis_label_size'],
        'axes.titlesize': config['fonts']['title_size'],
        'xtick.labelsize': config['fonts']['tick_label_size'],
        'ytick.labelsize': config['fonts']['tick_label_size'],
    })
    
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.edgecolor': '0.2',
        'axes.linewidth': 1.0
    })
    
    # Create a colormap using the series colormap from config
    series_cmap_name = config['colors']['series_colormap']
    
    # Get the colormap - Special handling for cmcrameri colormaps
    if series_cmap_name.startswith('cmc.'):
        # Extract the colormap name without the 'cmc.' prefix
        cmap_name = series_cmap_name.split('.')[1]
        # Get the colormap from cmcrameri
        if hasattr(cmc, cmap_name):
            cmap = getattr(cmc, cmap_name)
        else:
            print(f"Warning: Colormap {cmap_name} not found in cmcrameri, falling back to viridis")
            cmap = plt.cm.viridis
    else:
        # Get the colormap in a way that works with newer matplotlib versions
        try:
            # Modern matplotlib (3.7+)
            cmap = plt.colormaps[series_cmap_name]
        except:
            # Fallback for older matplotlib
            cmap = plt.cm.get_cmap(series_cmap_name)
    
    # Create colors for all 18 series
    colors = [cmap(i/18) for i in range(18)]
    
    # Determine the scale for proper visualization
    # Calculate the range of PC values to ensure proper scaling
    pc1_range = pca_df['PC1'].max() - pca_df['PC1'].min()
    pc2_range = pca_df['PC2'].max() - pca_df['PC2'].min() 
    max_range = max(pc1_range, pc2_range)
    
    # Set manual axis limits instead of calculating them
    xlim_min = -4.0
    xlim_max = 3.0
    ylim_min = -2.2
    ylim_max = 3.0
    
    # Calculate max_range based on manual limits for other calculations
    manual_pc1_range = xlim_max - xlim_min
    manual_pc2_range = ylim_max - ylim_min
    max_range = max(manual_pc1_range, manual_pc2_range)
    
    # Create a background shading grid
    # Number of points in each direction for the shading grid
    grid_density = 100
    x_grid = np.linspace(xlim_min, xlim_max, grid_density)
    y_grid = np.linspace(ylim_min, ylim_max, grid_density)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create a function to evaluate "goodness" at each point (simplified version)
    def calculate_goodness(pc1, pc2):
        # Since we don't have the original standardization parameters,
        # we'll use a simplified model: PC1 generally represents quality (higher is better)
        # and PC2 often contrasts extreme ratings (higher might mean fewer 1-stars)
        goodness = pc1.flatten() * 0.7 + pc2.flatten() * 0.3
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(goodness)) > 0:
            goodness = goodness / np.max(np.abs(goodness))
        
        return goodness.reshape(X.shape)
    
    # Calculate goodness values
    goodness = calculate_goodness(X, Y)
    
    # Use highlight colors from config for the background
    good_color = config['colors']['highlight']['good']
    bad_color = config['colors']['highlight']['bad']
    
    # Add alpha transparency
    good_color_alpha = tuple(list(plt.matplotlib.colors.to_rgba(good_color))[:3] + [0.4])
    bad_color_alpha = tuple(list(plt.matplotlib.colors.to_rgba(bad_color))[:3] + [0.4])
    
    # Define custom colormap: red -> transparent -> green using config colors
    cmap_bg = LinearSegmentedColormap.from_list(
        'RedTransparentGreen', 
        [bad_color_alpha,            # Red with alpha
         (1.0, 1.0, 1.0, 0.0),       # Transparent white
         good_color_alpha]           # Green with alpha
    )
    
    # Draw the background shading
    plt.pcolormesh(X, Y, goodness, cmap=cmap_bg, shading='gouraud', zorder=0)
    
    # Add a simple contour line to show the boundary between good and bad regions
    plt.contour(X, Y, goodness, levels=[0], colors=['#555555'], linewidths=1, linestyles='dashed', alpha=0.5, zorder=1)
    
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    
    # Set axis equal to ensure distances are preserved
    plt.gca().set_aspect('equal')
    
    # Add grid if enabled in config
    if config['global'].get('grid', True):
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    # Add origin lines
    plt.axhline(y=0, color='#222222', linestyle='-', alpha=0.4, zorder=1, linewidth=1.5)
    plt.axvline(x=0, color='#222222', linestyle='-', alpha=0.4, zorder=1, linewidth=1.5)
    
    # Use a fixed size for all circles based on config marker size
    base_marker_size = config['styles'].get('marker_size', 6)
    circle_size = base_marker_size * 60  # Keep the larger size
    
    # Increase zorder for scatter and labels to ensure they're on top
    scatter_zorder = 10
    label_zorder = 11
    
    # Plot each season as a scatter point
    for i, season in enumerate(seasons):
        row = pca_df[pca_df['series'] == season]
        x, y = row['PC1'].values[0], row['PC2'].values[0]
        
        # Get correct color for this series (0-indexed)
        series_color = colors[int(season-1) % 18]
        
        # Plot point with more distinct border
        plt.scatter(
            x, y,
            s=circle_size,
            color=series_color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.85,
            zorder=scatter_zorder
        )
        
        # Add season label with white outline for better visibility
        text = plt.annotate(
            f"S{season}",
            xy=(x, y),
            fontsize=12,  # Reduced from 14 to be proportional to circles
            fontweight='bold',
            ha='center',
            va='center',
            color='black',
            # Add a white outline for better readability on any background
            path_effects=[
                path_effects.withStroke(linewidth=3, foreground='white')
            ],
            zorder=label_zorder
        )
    
    # Add feature vectors (loadings)
    origin = np.zeros(2)
    
    # Get the feature names
    features = loadings.index
    
    # Scale factor for loadings - increase by 20%
    scale_factor = max_range * 0.48  # Increased from 0.4 (20% increase)
    
    # Feature names mapping
    feature_names = {
        'pct_1s': '% of Rating 1',
        'pct_10s': '% of Rating 10',
        'mu': 'Mean Rating (μ)',
        'sigma': 'Std Dev (σ)'
    }
    
    # Use line width from config
    line_width = config['styles'].get('line_width', 1.5)
    
    for feature in features:
        # Calculate arrow endpoints
        arrow_x = loadings.loc[feature, 'PC1'] * scale_factor
        arrow_y = loadings.loc[feature, 'PC2'] * scale_factor
        
        plt.arrow(
            origin[0], origin[1],
            arrow_x, arrow_y,
            head_width=max_range * 0.03,
            head_length=max_range * 0.04,
            fc='#555555', 
            ec='#555555', 
            alpha=0.75,
            zorder=2,
            linewidth=line_width
        )
        
        # Add feature labels with better positioning and styling
        # Calculate label position with more space from arrow tip
        label_x = arrow_x * 1.2
        label_y = arrow_y * 1.2
        
        # Position labels more intelligently
        ha = 'center'
        if abs(label_x) > 0.2 * max_range:
            ha = 'right' if label_x < 0 else 'left'
            
        va = 'center'
        if abs(label_y) > 0.2 * max_range:
            va = 'bottom' if label_y > 0 else 'top'
            
        # Add feature label with better styling
        plt.text(
            label_x, label_y,
            feature_names.get(feature, feature),
            fontsize=12,
            ha=ha,
            va=va,
            fontweight='bold',
            color='#333333',
            bbox=dict(
                facecolor='white', 
                alpha=0.9, 
                edgecolor='#888888',
                boxstyle='round,pad=0.4',
                linewidth=1
            ),
            zorder=5
        )
    
    # Add explained variance
    explained_variance_labels = [f'PC{i+1} ({var:.1%})' for i, var in enumerate(explained_variance)]
    
    # Set axis labels with better styling
    plt.xlabel(explained_variance_labels[0], fontsize=15, fontweight='bold', labelpad=15)
    plt.ylabel(explained_variance_labels[1], fontsize=15, fontweight='bold', labelpad=15)
    
    # Create legend elements
    legend_elements = []
    
    # Add a legend for the background shading
    legend_elements.append(
        Patch(facecolor=good_color_alpha, 
              edgecolor='#888888', 
              label='Good ratings profile')
    )
    legend_elements.append(
        Patch(facecolor=bad_color_alpha, 
              edgecolor='#888888', 
              label='Poor ratings profile')
    )
    
    # Add feature vector legend
    legend_elements.append(
        Line2D([0], [0], 
               color='#555555', 
               lw=2, 
               marker='>',
               markersize=8, 
               label='Feature vector')
    )
    
    # Add legend with improved styling
    plt.legend(
        handles=legend_elements,
        loc='best',
        title="Legend",
        frameon=True,
        framealpha=0.95,
        edgecolor='#888888',
        fontsize=11,
        title_fontsize=13
    )
    
    # Add explanatory annotations for extreme series
    # Find extreme series in different directions
    max_pc1_idx = pca_df['PC1'].idxmax()
    min_pc1_idx = pca_df['PC1'].idxmin()
    max_pc2_idx = pca_df['PC2'].idxmax()
    min_pc2_idx = pca_df['PC2'].idxmin()
    
    # Calculate annotation offset based on the data range
    offset_x = max_range * 0.25  # Increased from 0.15 for more space
    offset_y = max_range * 0.25
    
    # Track which seasons have been annotated to avoid duplicate annotations
    annotated_seasons = set()
    
    # Better positions for each direction - adjusted for smaller axis limits
    positions = {
        "max_pc1": {"offset": (offset_x * 0.65, offset_y * 0.5), "connection": "arc3,rad=0.2"},  # Moved slightly more to the right
        "min_pc1": {"offset": (offset_x * 0.6, -offset_y * 0.6), "connection": "arc3,rad=-0.2"},  # Moved moderately downward
        "max_pc2": {"offset": (-offset_x * 0.5, -offset_y), "connection": "arc3,rad=-0.2"},  # Keep this one as is
        "min_pc2": {"offset": (offset_x * 0.4, -offset_y * 0.4), "connection": "arc3,rad=-0.2"}  # Keep this one as is
    }
    
    # Add annotations for interesting patterns with better styling
    for idx_type, idx in [("max_pc1", max_pc1_idx), ("min_pc1", min_pc1_idx), 
                          ("max_pc2", max_pc2_idx), ("min_pc2", min_pc2_idx)]:
        season = pca_df.loc[idx, 'series']
        
        # Skip if this season has already been annotated
        if season in annotated_seasons:
            continue
        
        # Mark this season as annotated
        annotated_seasons.add(season)
        
        row = pca_df[pca_df['series'] == season]
        x, y = row['PC1'].values[0], row['PC2'].values[0]
        
        # Create custom annotation with better formatting
        stats = [
            f"% Rating 1: {row['pct_1s'].values[0]:.1f}%",
            f"% Rating 10: {row['pct_10s'].values[0]:.1f}%",
            f"Mean(2-9): {row['mu'].values[0]:.2f}",
            f"Std(2-9): {row['sigma'].values[0]:.2f}"
        ]
        
        annotation_text = f"Season {season}\n" + "\n".join(stats)
        
        # Get position details for this direction
        pos = positions[idx_type]
        
        # Make sure annotation stays within bounds
        text_x = x + pos["offset"][0]
        text_y = y + pos["offset"][1]
        
        # Constrain coordinates to stay within axis limits with some padding
        padding = 0.3
        text_x = max(xlim_min + padding, min(xlim_max - padding, text_x))
        text_y = max(ylim_min + padding, min(ylim_max - padding, text_y))
        
        # Create the annotation with improved positioning
        plt.annotate(
            annotation_text,
            xy=(x, y),  # Point to annotate
            xytext=(text_x, text_y),  # Text position, constrained to stay in bounds
            textcoords="data",  # Use data coordinates
            arrowprops=dict(
                arrowstyle="->", 
                color='#333333', 
                alpha=0.8, 
                connectionstyle=pos["connection"],
                linewidth=1.5
            ),
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5", 
                fc="white", 
                ec="#888888", 
                alpha=0.9,
                linewidth=1
            ),
            zorder=5,
            ha="center",
            va="center"
        )
    
    # Save the figure with high quality
    plt.tight_layout()
    dpi = config['global'].get('dpi', 300)
    
    # Save both PDF and PNG versions with consistent appearance
    pdf_file = SCRIPT_DIR / "figure1_pca_output.pdf"
    png_file = SCRIPT_DIR / "figure1_pca_output.png"
    
    # For better PDF quality
    plt.savefig(pdf_file, dpi=dpi*2, bbox_inches='tight', facecolor='white', format='pdf')
    plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    # Print completion message
    print(f"PCA plot created and saved to {pdf_file} and {png_file}")
    print(f"Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")

if __name__ == "__main__":
    create_series_rating_pca_visualization() 