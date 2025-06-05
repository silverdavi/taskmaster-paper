#!/usr/bin/env python3
"""
Create a ridgeline plot of IMDb rating distributions for Taskmaster series
using pre-calculated Gaussian fits.

This script loads data from:
- series_metrics.csv - Contains the mu and sigma for each series
- or directly from series_gaussians.csv if available

And creates a publication-quality ridgeline plot with:
- Decomposed rating distributions (Gaussian for ratings 2-9)
- Delta spikes for 1-star and 10-star ratings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import os
import yaml
import matplotlib as mpl
import cmcrameri.cm as cmc  # Import Fabio Crameri's colormaps

# File paths relative to this script
SCRIPT_DIR = Path(__file__).parent
SERIES_METRICS_FILE = SCRIPT_DIR / "series_metrics.csv"
GAUSSIANS_FILE = SCRIPT_DIR / "series_gaussians.csv"  # Optional: if exists
OUTPUT_FILE = SCRIPT_DIR / "figure1_ridge_output.png"
CONFIG_FILE = SCRIPT_DIR.parent.parent / "config" / "plot_config.yaml"

def create_series_rating_distribution_plot():
    """
    Create a ridgeline plot showing the distribution of IMDb ratings for each Taskmaster series.
    The plot shows fitted Gaussian curves for ratings 2-9, with special markers for 1-star and 10-star ratings.
    """
    # Load config
    with open(CONFIG_FILE, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get the series colormap from config
    series_cmap_name = config['colors']['series_colormap']
    
    # Load pre-calculated data
    try:
        # First try to load gaussian parameters if available
        if os.path.exists(GAUSSIANS_FILE):
            print(f"Loading Gaussian parameters from {GAUSSIANS_FILE}")
            df_gaussians = pd.read_csv(GAUSSIANS_FILE)
            has_gaussians = True
        else:
            # Fallback to series metrics which should have mu and sigma
            print(f"Loading series metrics from {SERIES_METRICS_FILE}")
            df_series = pd.read_csv(SERIES_METRICS_FILE)
            
            # Debug: Check if IMDb ratings are present
            if 'imdb_rating' in df_series.columns:
                print(f"IMDb ratings found in the data. Example: Series 1 = {df_series.loc[df_series['series'] == 1, 'imdb_rating'].values[0]}")
            else:
                print("WARNING: No IMDb ratings column found in the data")
            
            has_gaussians = 'mu' in df_series.columns and 'sigma' in df_series.columns
            
            if not has_gaussians:
                raise ValueError("Required Gaussian parameters (mu, sigma) not found in series_metrics.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare data for plotting
    if 'gaussians' not in locals() or not has_gaussians:
        # Create data structure from series metrics
        df_gaussians = df_series[['series', 'mu', 'sigma', 'pct_1s', 'pct_10s', 'total_votes', 'imdb_rating']]
    
    # Sort series in descending order for better visualization
    df_gaussians = df_gaussians.sort_values('series', ascending=False)
    
    # Create a custom colormap with sequential but distinct colors
    n_series = len(df_gaussians)
    
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
        # Use the colormap from config
        try:
            # Modern matplotlib (3.7+)
            cmap = plt.colormaps[series_cmap_name]
        except:
            # Fallback for older matplotlib
            cmap = plt.cm.get_cmap(series_cmap_name)
    
    # Override with RdYlGn colormap for intuitive color mapping
    # Higher μ (better) = Green, Lower μ (worse) = Red
    try:
        cmap = plt.colormaps['RdYlGn']
    except:
        cmap = plt.cm.get_cmap('RdYlGn')
    
    print(f"Using colormap: RdYlGn (Green=High μ, Red=Low μ)")
    print(f"μ (Gaussian mean) range: {df_gaussians['mu'].min():.3f} to {df_gaussians['mu'].max():.3f}")
    
    # Create colors based on mean rating (μ) instead of series number
    min_mu = df_gaussians['mu'].min()
    max_mu = df_gaussians['mu'].max()
    mu_range = max_mu - min_mu
    
    # Create a list to store colors for each series
    palette = []
    for _, row in df_gaussians.iterrows():
        # Normalize the mean rating to [0,1] range
        normalized_mu = (row['mu'] - min_mu) / mu_range
        # Get color from colormap
        color = cmap(normalized_mu)
        palette.append(color)
    
    # Create the figure with larger size for better detail
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Set style
    plt.rcParams.update({
        'font.family': config['global']['font_family'],
        'font.size': config['fonts']['axis_label_size'],
        'axes.labelsize': config['fonts']['axis_label_size'],
        'axes.titlesize': config['fonts']['title_size'],
        'xtick.labelsize': config['fonts']['tick_label_size'],
        'ytick.labelsize': config['fonts']['tick_label_size'],
    })
    
    # Calculate plot parameters
    x_min, x_max = 1, 10  # Rating range
    y_step = 1.0  # Distance between distributions
    
    # Set up the plot
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    y_max = y_step * (n_series) 
    ax.set_ylim(-1, y_max)
    
    # Create a range of x values for plotting the Gaussian curves
    x = np.linspace(1, 10, 500)
    
    # Calculate scaling factor for each series
    scaling_factors = []
    for i, (_, row) in enumerate(df_gaussians.iterrows()):
        # For a normal distribution, PDF maximum height is 1/(sigma*sqrt(2*pi))
        pdf_max = 1 / (row['sigma'] * np.sqrt(2 * np.pi))
        scaling_factors.append(pdf_max)
    
    # Calculate proper scaling to fit the plot
    if scaling_factors:
        # Scale all distributions to have a reasonable height
        target_height = 0.8  # Height in plot units
        scale = target_height / max(scaling_factors)
    else:
        scale = 1.0
    
    # Calculate normalized sizes for scatter points
    # Get min/max percentages for normalization
    min_pct_1 = df_gaussians['pct_1s'].min()
    max_pct_1 = df_gaussians['pct_1s'].max()
    min_pct_10 = df_gaussians['pct_10s'].min()
    max_pct_10 = df_gaussians['pct_10s'].max()
    
    # Define size range
    min_size = 20
    max_size = 200
    
    # After loading data, find min and max IMDb ratings for scaling
    min_imdb = None
    max_imdb = None
    if 'imdb_rating' in df_gaussians.columns:
        min_imdb = df_gaussians['imdb_rating'].min()
        max_imdb = df_gaussians['imdb_rating'].max()
        imdb_range = max_imdb - min_imdb
        print(f"IMDb rating range: {min_imdb:.2f} - {max_imdb:.2f}, range: {imdb_range:.2f}")
    
    # Plot each distribution
    for i, (_, row) in enumerate(df_gaussians.iterrows()):
        series = row['series']
        mu = row['mu']
        sigma = row['sigma']
        pct_1s = row['pct_1s'] / 100  # Convert from percentage to proportion
        pct_10s = row['pct_10s'] / 100
        total_votes = row['total_votes'] if 'total_votes' in row else 0
        
        # Get IMDb rating if available
        try:
            imdb_rating = row['imdb_rating']
            print(f"Found IMDb rating for Series {series}: {imdb_rating}")
        except:
            imdb_rating = None
            print(f"No IMDb rating for Series {series}")
        
        # Base y position for this series
        y_base = i * y_step
        
        # Plot the Gaussian component (ratings 2-9)
        # Calculate PDF values for the normal distribution
        pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
        
        # Scale the PDF to a reasonable height
        pdf_scaled = pdf * scale
        
        # Adjust for 1-star and 10-star proportions
        # The Gaussian should represent only ratings 2-9, so scale accordingly
        remaining_proportion = 1.0 - (pct_1s + pct_10s)
        if remaining_proportion > 0:
            pdf_scaled = pdf_scaled * remaining_proportion
        
        # Plot the distribution
        ax.fill_between(x, y_base, y_base + pdf_scaled, alpha=0.8, color=palette[i])
        ax.plot(x, y_base + pdf_scaled, lw=2, color='black')
        
        # Add delta spikes for 1-star and 10-star ratings
        # Calculate height proportional to percentage
        spike_height = max(pdf_scaled) * 1.2  # Make spikes slightly higher than curve peak
        
        # Calculate normalized scatter sizes
        size_1s = min_size
        if max_pct_1 > min_pct_1:
            normalized_pct_1 = (row['pct_1s'] - min_pct_1) / (max_pct_1 - min_pct_1)
            size_1s = min_size + normalized_pct_1 * (max_size - min_size)
        
        size_10s = min_size
        if max_pct_10 > min_pct_10:
            normalized_pct_10 = (row['pct_10s'] - min_pct_10) / (max_pct_10 - min_pct_10)
            size_10s = min_size + normalized_pct_10 * (max_size - min_size)
        
        # 1-star spike
        if pct_1s > 0:
            # Draw the spike
            ax.plot([1, 1], [y_base, y_base + spike_height * pct_1s], 
                   color='red', lw=2.5, solid_capstyle='round')
            # Add a dot at the top with size proportional to percentage
            ax.scatter(1, y_base + spike_height * pct_1s, color='red', s=size_1s, zorder=5, 
                      edgecolor='black', linewidth=1)
        
        # 10-star spike
        if pct_10s > 0:
            # Draw the spike
            ax.plot([10, 10], [y_base, y_base + spike_height * pct_10s], 
                   color='green', lw=2.5, solid_capstyle='round')
            # Add a dot at the top with size proportional to percentage
            ax.scatter(10, y_base + spike_height * pct_10s, color='green', s=size_10s, zorder=5,
                      edgecolor='black', linewidth=1)
        
        # Add mean label on the curve - only show "μ=" on the first one
        if remaining_proportion > 0.05:  # Only label if enough area under curve
            # Determine if this is the first visible series
            is_first = i == (len(df_gaussians) - 1)  # Since we're plotting in reverse order
            
            # Create label text with or without μ prefix
            mu_label = f"μ={mu:.2f}" if is_first else f"{mu:.2f}"
            
            ax.text(mu, y_base + pdf_scaled.max() / 2, 
                   mu_label, ha='center', va='center', 
                   fontweight='bold', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))
        
        # Add IMDb rating with more prominent styling
        if imdb_rating is not None and not pd.isna(imdb_rating):
            print(f"Adding IMDb rating label for Series {series}: {imdb_rating}")
            
            # Determine if this is the first visible series
            is_first = i == (len(df_gaussians) - 1)  # Since we're plotting in reverse order
            
            # Create label text with or without IMDb prefix
            imdb_label = f"IMDb={imdb_rating:.1f}" if is_first else f"{imdb_rating:.2f}"
            
            # Position at x=4 plus a scaling factor based on the IMDb rating
            if min_imdb is not None and max_imdb is not None:
                # Scale position based on where this rating falls in the range
                # Use a smaller scaling factor (0.5) to keep positions reasonable
                position_offset = 0.5 * (imdb_rating - min_imdb) / (max_imdb - min_imdb)
                imdb_x = 4.0 + position_offset
            else:
                imdb_x = 4.0
            
            # Use a different y-position to avoid overlap with μ
            imdb_y = y_base + pdf_scaled.max() * 0.75
            
            # Create label with IMDb yellow styling
            ax.text(imdb_x, imdb_y, 
                   imdb_label, ha='center', va='center', 
                   fontweight='bold', fontsize=10,
                   color='black',  # Black text on yellow background
                   bbox=dict(
                       facecolor='#F5C518',  # IMDb yellow
                       alpha=0.9, 
                       boxstyle='round,pad=0.3',
                       edgecolor='black',
                       linewidth=1
                   ),
                   zorder=10)  # Ensure it's on top
        else:
            print(f"Warning: No valid IMDb rating available for Series {series}")
        
        # Add series label and vote count
        ax.text(x_min - 0.3, y_base, f"S{int(series)}", 
               ha='right', va='center', fontsize=12, fontweight='bold')
        
        # Add vote count on the left
        if total_votes > 0:
            ax.text(x_min - 1.0, y_base, f"n={int(total_votes):,}", 
                   ha='right', va='center', fontsize=10, color='#555555')
    
    # Set up the axes
    ax.set_xlabel('IMDb Rating', fontsize=14, fontweight='bold', labelpad=10)
    
    # Format x-axis to show only integer ratings
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(range(1, 11), fontsize=12)
    
    # Hide y-axis ticks and labels
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add grid for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.7, zorder=-1)
    
    # Remove the bounding box for the axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Adjust layout and save - fix the tight layout warning
    plt.subplots_adjust(top=0.95, right=0.95, left=0.05, bottom=0.05)
    
    # Save both PDF and PNG versions
    dpi = config['global'].get('dpi', 300)
    pdf_file = SCRIPT_DIR / "fig1a.pdf"
    png_file = SCRIPT_DIR / "fig1a.png"
    
    plt.savefig(pdf_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    print(f"Ridgeline plot created and saved to {pdf_file} and {png_file}")

if __name__ == "__main__":
    create_series_rating_distribution_plot() 