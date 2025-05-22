#!/usr/bin/env python3
"""
Create a ridgeline plot for Taskmaster ratings using seaborn's overlapping density curves
with improved smoothing for discrete data.
"""

import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Configuration
CSV_FILE = "taskmaster_histograms_corrected_fixed.csv"

def extract_season_histograms():
    """
    Extract normalized histograms for each season without expanding to individual votes.
    This allows us to apply custom interpolation for better smoothing.
    """
    # Initialize dictionary to store counts for each rating by season
    season_ratings = {}
    
    # Read CSV data
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            season = int(row[0])
            
            if season not in season_ratings:
                # Initialize array with 10 bins (ratings 1-10)
                season_ratings[season] = np.zeros(10)
            
            # Process histogram data
            # Remember: hist1 = rating 10, hist2 = rating 9, etc.
            for i in range(10):
                rating_idx = 9 - i  # Convert to 0-indexed (rating 1 = index 0)
                votes_count = int(row[7 + i*2])  # hist_votes columns
                season_ratings[season][rating_idx] += votes_count
    
    # Normalize each season's histogram
    season_normalized = {}
    season_totals = {}
    for season, histogram in season_ratings.items():
        total = np.sum(histogram)
        season_totals[season] = total
        if total > 0:
            season_normalized[season] = histogram / total
        else:
            season_normalized[season] = histogram
    
    return season_normalized, season_totals

def interpolate_smooth_histogram(histogram, points=100):
    """
    Create a smoothed version of a histogram using cubic interpolation.
    Centers bin values at integers (1-10) and ensures smooth transitions between them.
    """
    # Original bin centers at integers 1-10
    bin_centers = np.arange(1, 11)
    
    # Add extra points at ends for better interpolation
    extended_x = np.concatenate(([0], bin_centers, [11]))
    extended_y = np.concatenate(([0], histogram, [0]))
    
    # Cubic spline interpolation between points
    f = interp1d(extended_x, extended_y, kind='cubic', bounds_error=False, fill_value=0)
    
    # Create smooth x and y arrays
    x_smooth = np.linspace(0.5, 10.5, points)
    y_smooth = f(x_smooth)
    
    # Ensure no negative values
    y_smooth = np.maximum(y_smooth, 0)
    
    return x_smooth, y_smooth

def create_smoothed_dataframe():
    """
    Create a DataFrame with smoothed histogram data for plotting with seaborn FacetGrid.
    """
    season_histograms, season_totals = extract_season_histograms()
    seasons = sorted(season_histograms.keys())
    
    # Calculate season means for annotation
    season_means = {}
    bin_centers = np.arange(1, 11)
    for season, histogram in season_histograms.items():
        season_means[season] = np.sum(bin_centers * histogram)
    
    # Create a DataFrame for easy plotting with FacetGrid
    facet_data = []
    for season in seasons:
        # Get smooth interpolated histogram
        x_smooth, y_smooth = interpolate_smooth_histogram(season_histograms[season], points=200)
        
        # Add to dataframe
        for x, y in zip(x_smooth, y_smooth):
            facet_data.append({
                'season': f"S{season}",
                'rating': x,
                'density': y,
                'mean': season_means[season],
                'total_votes': season_totals[season]
            })
    
    return pd.DataFrame(facet_data)

def plot_seaborn_ridgeline():
    """Generate a seaborn-based ridgeline plot for Taskmaster ratings with improved smoothing."""
    # Set seaborn theme
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Get the smoothed data
    df = create_smoothed_dataframe()
    
    # Get unique seasons in correct order
    seasons = sorted(df['season'].unique(), key=lambda x: int(x[1:]))
    
    # Prepare season statistics for labels
    season_stats = df.drop_duplicates('season')[['season', 'mean', 'total_votes']]
    season_stats_dict = {row['season']: {'mean': row['mean'], 'count': row['total_votes']} 
                        for _, row in season_stats.iterrows()}
    
    # Create a colormap with viridis for all seasons
    n_seasons = len(seasons)
    palette = sns.color_palette("viridis", n_seasons)
    
    # Initialize the FacetGrid object
    g = sns.FacetGrid(
        df, 
        row="season",
        hue="season",
        aspect=18,  # Make the plot wide and short
        height=0.75,  # Height of each facet
        palette=palette,
        row_order=seasons  # Ensure consistent season ordering
    )
    
    # Draw the density curves using our pre-smoothed data
    def draw_line(data, **kwargs):
        ax = plt.gca()
        data = data.sort_values('rating')
        plt.fill_between(data['rating'], 0, data['density'], alpha=0.7, **kwargs)
        plt.plot(data['rating'], data['density'], color='white', lw=1, alpha=0.9)
        return ax
    
    g.map_dataframe(draw_line)
    
    # Add horizontal reference line at y=0
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
    
    # Add labels with mean and count
    for ax, season_label in zip(g.axes.flat, g.row_names):
        # Get color from palette
        row_idx = g.row_names.index(season_label)
        color = palette[row_idx]
        
        # Get season stats
        stats = season_stats_dict[season_label]
        
        # Add text annotation
        ax.text(
            0.05, 0.5,
            f"{season_label}  (μ={stats['mean']:.2f}, n={int(stats['count'])})",
            fontweight="bold",
            color=color,
            ha="left", 
            va="center", 
            transform=ax.transAxes,
            fontsize=12
        )
    
    # Adjust the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.7)  # More negative value for more overlap
    
    # Set the figure title
    plt.suptitle('Taskmaster Rating Distributions by Season\n(Improved Smoothing)', 
                y=0.98, fontsize=16, fontweight='bold')
    
    # Add rating range annotation at the bottom
    plt.figtext(
        0.5, 0.01,
        "Rating (1-10)",
        ha="center", 
        fontsize=12, 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )
    
    # Clean up axes
    g.set_titles("")
    g.set(xlim=(0.5, 10.5))  # Set x-axis limits from 0.5 to 10.5
    g.set(xticks=range(1, 11))  # Set x-ticks to 1-10
    g.set(yticks=[])
    g.set(ylabel="")
    g.despine(bottom=True, left=True)
    
    # Save the figure
    plt.savefig('seaborn_ridgeline_improved.png', dpi=300, bbox_inches='tight')
    print("Improved seaborn ridgeline plot created successfully!")

if __name__ == "__main__":
    plot_seaborn_ridgeline() 