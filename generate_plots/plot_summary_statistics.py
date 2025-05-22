#!/usr/bin/env python3
"""
Create a visualization showing four key statistics for each Taskmaster season:
1. Percentage of 1 ratings
2. Percentage of 10 ratings
3. Mean of ratings 2-9
4. Standard deviation of ratings 2-9
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Configuration
CSV_FILE = "taskmaster_histograms_corrected_fixed.csv"

def extract_season_statistics():
    """
    Extract the four key statistics for each season:
    - Percentage of 1 ratings
    - Percentage of 10 ratings
    - Mean of ratings 2-9
    - Standard deviation of ratings 2-9
    """
    # Initialize dictionaries to store statistics by season
    season_stats = {}
    all_season_data = {}
    
    # Read CSV data
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            season = int(row[0])
            
            if season not in all_season_data:
                all_season_data[season] = {
                    'votes': np.zeros(10),
                    'episodes': 0,
                    'total_votes': 0
                }
            
            # Count episodes
            all_season_data[season]['episodes'] += 1
            
            # Process histogram data
            # Remember: hist1_votes = rating 10, hist2_votes = rating 9, etc.
            for i in range(10):
                rating_idx = 9 - i  # Convert to 0-indexed (rating 1 = index 0)
                votes_count = int(row[7 + i*2])  # hist_votes columns
                all_season_data[season]['votes'][rating_idx] += votes_count
                all_season_data[season]['total_votes'] += votes_count
    
    # Calculate statistics for each season
    for season, data in all_season_data.items():
        # Convert to percentages
        votes_pct = data['votes'] / data['total_votes'] * 100
        
        # Extract ratings 2-9 (indices 1-8)
        middle_ratings = np.arange(2, 10)
        middle_votes = votes_pct[1:9]
        
        # Calculate statistics
        percent_1s = votes_pct[0]
        percent_10s = votes_pct[9]
        
        # Calculate weighted mean and std for middle ratings
        weights = data['votes'][1:9]
        if np.sum(weights) > 0:
            mean_2_9 = np.average(middle_ratings, weights=weights)
            # Weighted std calculation
            variance = np.average((middle_ratings - mean_2_9)**2, weights=weights)
            std_2_9 = np.sqrt(variance)
        else:
            mean_2_9 = 0
            std_2_9 = 0
        
        # Store calculated statistics
        season_stats[season] = {
            'percent_1s': percent_1s,
            'percent_10s': percent_10s,
            'mean_2_9': mean_2_9,
            'std_2_9': std_2_9,
            'episodes': data['episodes'],
            'total_votes': data['total_votes'],
            'distribution': votes_pct,
            'middle_distribution': middle_votes
        }
    
    return season_stats

def plot_season_statistics():
    """
    Create a combination plot with:
    1. Line plot for percentages of 1s
    2. Line plot for percentages of 10s
    3. Box plot for ratings 2-9 with mean line
    """
    # Get season statistics
    season_stats = extract_season_statistics()
    
    # Prepare data for plotting
    seasons = sorted(season_stats.keys())
    
    # Extract statistics into lists for plotting
    percent_1s = [season_stats[s]['percent_1s'] for s in seasons]
    percent_10s = [season_stats[s]['percent_10s'] for s in seasons]
    mean_2_9 = [season_stats[s]['mean_2_9'] for s in seasons]
    std_2_9 = [season_stats[s]['std_2_9'] for s in seasons]
    total_votes = [season_stats[s]['total_votes'] for s in seasons]
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Plot percentage of 1s and 10s as lines
    line_1s = ax1.plot(seasons, percent_1s, 'o-', color='firebrick', linewidth=2, markersize=8, label='% Rating 1')
    line_10s = ax1.plot(seasons, percent_10s, 'o-', color='forestgreen', linewidth=2, markersize=8, label='% Rating 10')
    
    # Calculate lower and upper bounds for the middle ratings (2-9)
    lower_bound = [mean - std for mean, std in zip(mean_2_9, std_2_9)]
    upper_bound = [mean + std for mean, std in zip(mean_2_9, std_2_9)]
    
    # Plot mean of ratings 2-9 as a line
    line_mean = ax1.plot(seasons, mean_2_9, 'o-', color='royalblue', linewidth=2, markersize=8, label='Mean of Ratings 2-9')
    
    # Plot std range as a shaded area
    ax1.fill_between(seasons, lower_bound, upper_bound, color='royalblue', alpha=0.2, label='±1 Std Dev')
    
    # Customize the plot
    ax1.set_xlabel('Season', fontsize=14)
    ax1.set_ylabel('Percentage / Rating Value', fontsize=14)
    ax1.set_title('Taskmaster Rating Statistics by Season', fontsize=18, pad=20)
    ax1.set_xticks(seasons)
    ax1.set_xticklabels([f'S{s}\n({season_stats[s]["episodes"]} ep)' for s in seasons], fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add marker size based on total votes
    min_size = 50
    max_size = 200
    normalized_sizes = [min_size + (max_size - min_size) * (votes / max(total_votes)) for votes in total_votes]
    
    # Add scatter points with size proportional to number of votes
    scatter = ax1.scatter(seasons, mean_2_9, s=normalized_sizes, color='royalblue', alpha=0.5, zorder=5)
    
    # Create legend elements
    legend_elements = [
        line_1s[0],
        line_10s[0],
        line_mean[0],
        Patch(facecolor='royalblue', alpha=0.2, label='±1 Std Dev'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', 
                  markersize=12, alpha=0.5, label='Vote Count')
    ]
    
    # Add legend
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=12, 
              title="Statistics", title_fontsize=13)
    
    # Add a second y-axis for annotations
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    
    # Add vote count annotations
    for i, s in enumerate(seasons):
        ax2.annotate(f"{total_votes[i]:,} votes", 
                    xy=(s, percent_1s[i]), 
                    xytext=(5, 10),
                    textcoords='offset points',
                    fontsize=9,
                    color='dimgray')
    
    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits with some padding
    max_pct = max(max(percent_1s), max(percent_10s)) * 1.1
    ax1.set_ylim(0, max(10, max_pct))
    
    # Add annotations for notable seasons
    max_1s_idx = percent_1s.index(max(percent_1s))
    max_10s_idx = percent_10s.index(max(percent_10s))
    
    ax1.annotate(f"Highest % of 1s: {percent_1s[max_1s_idx]:.1f}%",
                xy=(seasons[max_1s_idx], percent_1s[max_1s_idx]),
                xytext=(seasons[max_1s_idx] + 0.2, percent_1s[max_1s_idx] + 2),
                arrowprops=dict(arrowstyle="->", color='firebrick', alpha=0.7),
                color='firebrick',
                fontsize=10)
    
    ax1.annotate(f"Highest % of 10s: {percent_10s[max_10s_idx]:.1f}%",
                xy=(seasons[max_10s_idx], percent_10s[max_10s_idx]),
                xytext=(seasons[max_10s_idx] - 0.2, percent_10s[max_10s_idx] + 2),
                arrowprops=dict(arrowstyle="->", color='forestgreen', alpha=0.7),
                color='forestgreen',
                fontsize=10)
    
    # Add secondary panel below showing distribution of middle ratings
    plt.subplots_adjust(bottom=0.3)
    ax3 = fig.add_axes([0.1, 0.05, 0.8, 0.2])
    
    # Create a heatmap of the ratings distribution for middle ratings
    middle_data = np.array([season_stats[s]['middle_distribution'] for s in seasons])
    sns.heatmap(middle_data, ax=ax3, cmap='YlGnBu', cbar_kws={'label': 'Percentage'})
    
    # Set labels for the heatmap
    ax3.set_xlabel('Rating Value', fontsize=12)
    ax3.set_ylabel('Season', fontsize=12)
    ax3.set_title('Distribution of Middle Ratings (2-9)', fontsize=14)
    ax3.set_yticks(np.arange(len(seasons)) + 0.5)
    ax3.set_yticklabels([f'S{s}' for s in seasons])
    ax3.set_xticks(np.arange(8) + 0.5)
    ax3.set_xticklabels(range(2, 10))
    
    # Save the figure
    plt.savefig('taskmaster_season_statistics.png', dpi=300, bbox_inches='tight')
    print("Season statistics visualization created successfully!")

if __name__ == "__main__":
    plot_season_statistics() 