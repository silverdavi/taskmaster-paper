#!/usr/bin/env python3
"""
Create a ridgeline plot for Taskmaster ratings showing the decomposition of ratings
into three components:
1. A Gaussian distribution (central ratings)
2. Delta function at rating 1 (strongly negative ratings)
3. Delta function at rating 10 (strongly positive ratings)
"""

import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

# Configuration
CSV_FILE = "taskmaster_histograms_corrected_fixed.csv"

def extract_season_histograms():
    """
    Extract normalized histograms for each season without expanding to individual votes.
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

def gaussian_function(x, amplitude, mean, stddev):
    """Gaussian function for curve fitting"""
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def decompose_histogram(histogram, ratings=np.arange(1, 11)):
    """
    Decompose a histogram into three components:
    1. Gaussian distribution (middle ratings)
    2. Delta function at rating 1
    3. Delta function at rating 10
    """
    # Initial values: extract the delta components first
    delta_1_value = histogram[0]  # Rating 1
    delta_10_value = histogram[9]  # Rating 10
    
    # Create a copy of the histogram with reduced delta components for fitting
    hist_for_fitting = histogram.copy()
    hist_for_fitting[0] = 0  # Remove delta at 1
    hist_for_fitting[9] = 0  # Remove delta at 10
    
    # Initial guess for Gaussian parameters
    # Use weighted average for mean and standard deviation
    non_zero_idx = np.where(hist_for_fitting > 0)[0]
    if len(non_zero_idx) > 2:  # Need at least 3 points for a meaningful fit
        weights = hist_for_fitting[non_zero_idx]
        initial_mean = np.sum((non_zero_idx + 1) * weights) / np.sum(weights)
        initial_amplitude = np.max(hist_for_fitting)
        initial_stddev = 2.0  # Reasonable starting point
        
        try:
            # Fit Gaussian to the middle part (excluding delta components)
            popt, _ = curve_fit(
                gaussian_function, 
                ratings, 
                hist_for_fitting,
                p0=[initial_amplitude, initial_mean, initial_stddev],
                bounds=([0, 1, 0.1], [1, 10, 5])
            )
            amplitude, mean, stddev = popt
            
            # Generate Gaussian component
            gaussian_component = gaussian_function(ratings, amplitude, mean, stddev)
            
            # Ensure the Gaussian component doesn't exceed the original histogram
            gaussian_component = np.minimum(gaussian_component, histogram)
            
            # Adjust delta components by subtracting any Gaussian contribution
            delta_1_adj = max(0, histogram[0] - gaussian_component[0])
            delta_10_adj = max(0, histogram[9] - gaussian_component[9])
            
            return {
                'gaussian': gaussian_component,
                'delta_1': delta_1_adj,
                'delta_10': delta_10_adj,
                'params': {
                    'amplitude': amplitude,
                    'mean': mean,
                    'stddev': stddev
                }
            }
        except:
            # If curve fitting fails, return a simple decomposition
            pass
    
    # Fallback if fitting fails: just use the raw values
    return {
        'gaussian': hist_for_fitting,
        'delta_1': delta_1_value,
        'delta_10': delta_10_value,
        'params': {
            'amplitude': 0,
            'mean': 0,
            'stddev': 0
        }
    }

def create_decomposed_dataframe():
    """
    Create a DataFrame with decomposed histogram data for plotting.
    """
    season_histograms, season_totals = extract_season_histograms()
    seasons = sorted(season_histograms.keys())
    ratings = np.arange(1, 11)
    
    # Calculate season means for annotation
    season_means = {}
    for season, histogram in season_histograms.items():
        season_means[season] = np.sum(ratings * histogram)
    
    # Create a DataFrame for plotting
    facet_data = []
    
    for season in seasons:
        # Decompose the histogram
        decomposition = decompose_histogram(season_histograms[season])
        
        # Add Gaussian component
        for rating, density in zip(ratings, decomposition['gaussian']):
            if density > 0:
                facet_data.append({
                    'season': f"S{season}",
                    'rating': rating,
                    'density': density,
                    'component': 'Gaussian',
                    'mean': season_means[season],
                    'total_votes': season_totals[season]
                })
        
        # Add delta at rating 1
        if decomposition['delta_1'] > 0:
            facet_data.append({
                'season': f"S{season}",
                'rating': 1,
                'density': decomposition['delta_1'],
                'component': 'Delta(1)',
                'mean': season_means[season],
                'total_votes': season_totals[season]
            })
        
        # Add delta at rating 10
        if decomposition['delta_10'] > 0:
            facet_data.append({
                'season': f"S{season}",
                'rating': 10,
                'density': decomposition['delta_10'],
                'component': 'Delta(10)',
                'mean': season_means[season],
                'total_votes': season_totals[season]
            })
    
    return pd.DataFrame(facet_data)

def plot_decomposed_ridgeline():
    """Generate a ridgeline plot showing histogram decomposition into three components."""
    # Set seaborn theme
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Get the decomposed data
    df = create_decomposed_dataframe()
    
    # Get unique seasons in correct order
    seasons = sorted(df['season'].unique(), key=lambda x: int(x[1:]))
    
    # Prepare season statistics for labels
    season_stats = df.drop_duplicates('season')[['season', 'mean', 'total_votes']]
    season_stats_dict = {row['season']: {'mean': row['mean'], 'count': row['total_votes']} 
                        for _, row in season_stats.iterrows()}
    
    # Create a colormap for seasons
    n_seasons = len(seasons)
    palette = sns.color_palette("viridis", n_seasons)
    
    # Component color map
    component_colors = {
        'Gaussian': 'royalblue',
        'Delta(1)': 'firebrick',
        'Delta(10)': 'forestgreen'
    }
    
    # Set figure size to be taller to avoid ridge overlap
    plt.figure(figsize=(18, n_seasons * 0.8))
    
    # Initialize the FacetGrid object
    g = sns.FacetGrid(
        df, 
        row="season",
        hue="component",
        aspect=18,
        height=0.8,  # Slightly increase height
        palette=component_colors,
        row_order=seasons,
        hue_order=['Gaussian', 'Delta(1)', 'Delta(10)']
    )
    
    # Draw the decomposed components
    def draw_components(data, **kwargs):
        ax = plt.gca()
        component = data['component'].iloc[0]
        
        if component == 'Gaussian':
            # Draw Gaussian as a filled area
            ratings = sorted(data['rating'])
            densities = [data.loc[data['rating'] == r, 'density'].iloc[0] if len(data.loc[data['rating'] == r]) > 0 else 0 
                         for r in range(1, 11)]
            
            # Interpolate for smoother curve
            x_smooth = np.linspace(0.5, 10.5, 100)
            from scipy.interpolate import interp1d
            f = interp1d(np.arange(1, 11), densities, kind='cubic', bounds_error=False, fill_value=0)
            y_smooth = f(x_smooth)
            y_smooth = np.maximum(y_smooth, 0)  # Ensure no negative values
            
            plt.fill_between(x_smooth, 0, y_smooth, alpha=0.7, **kwargs)
            plt.plot(x_smooth, y_smooth, color='white', lw=1, alpha=0.9)
        
        else:  # Delta components
            # Draw deltas as vertical bars with scaled height
            for _, row in data.iterrows():
                # Scale down the delta components to be more proportional
                scale_factor = 0.5 if component == 'Delta(10)' else 0.7
                plt.bar(row['rating'], row['density'] * scale_factor, width=0.6, alpha=0.85, **kwargs)
                
                # Add a thin white line on top for better visibility
                plt.plot([row['rating']-0.3, row['rating']+0.3], 
                         [row['density'] * scale_factor, row['density'] * scale_factor], 
                         color='white', lw=1, alpha=0.9)
        
        return ax
    
    # Map the drawing function to each row and component
    g.map_dataframe(draw_components)
    
    # Add horizontal reference line at y=0
    g.refline(y=0, linewidth=1, linestyle="-", color=None, clip_on=False)
    
    # Add labels with mean and count
    for ax, season_label in zip(g.axes.flat, g.row_names):
        # Get season color from palette
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
    
    # Add a legend
    g.figure.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in component_colors.values()],
        labels=component_colors.keys(),
        loc='upper right',
        title='Components',
        frameon=True
    )
    
    # Adjust the subplots to avoid overlap
    g.figure.subplots_adjust(hspace=-0.5)  # Less negative value for less overlap
    
    # Set the figure title
    plt.suptitle('Taskmaster Rating Distributions by Season\n(Decomposed into Gaussian + Delta Functions)', 
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
    plt.savefig('seaborn_ridgeline_decomposed.png', dpi=300, bbox_inches='tight')
    print("Decomposed ridgeline plot created successfully!")

if __name__ == "__main__":
    plot_decomposed_ridgeline() 