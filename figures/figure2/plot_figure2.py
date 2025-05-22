#!/usr/bin/env python3
"""
Plot Figure 2: Episode Rating Trajectories by Contestant Ranking Patterns

This script creates a violin plot visualization showing episode rating distributions
for different contestant ranking patterns (123, 213, 231, 312) and positions (First, Middle, Last).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import os
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# Set up paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
CONFIG_FILE = ROOT_DIR / "config" / "plot_config.yaml"

# Input files
PATTERN_EPISODES_FILE = SCRIPT_DIR / "episode_patterns.csv"
SERIES_PATTERNS_FILE = SCRIPT_DIR / "series_patterns.csv"

# Output files - will be determined based on config


def load_config():
    """Load configuration from the config file."""
    try:
        with open(CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def load_data():
    """Load the processed data files needed for plotting."""
    try:
        # Load episode patterns data
        episode_df = pd.read_csv(PATTERN_EPISODES_FILE)
        print(f"Loaded {len(episode_df)} episodes with pattern data")
        
        # Load series patterns data
        series_df = pd.read_csv(SERIES_PATTERNS_FILE)
        print(f"Loaded {len(series_df)} series patterns")
        
        return episode_df, series_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def create_violin_plot(episode_df, series_df, config):
    """
    Create a violin plot visualization of episode ratings by pattern and position.
    """
    # Get output format from config
    output_format = config['global'].get('output_format', 'png')
    
    # Get DPI setting
    dpi = config['global'].get('dpi', 300)
    
    # Define color palette for positions (First, Middle, Last)
    position_colors = ['#f4d03f', '#e67e22', '#c0392b']  # Mustard yellow, Orange, Deep red
    
    # Filter to only include common patterns (with at least 2 series)
    pattern_counts = series_df['pattern'].value_counts()
    common_patterns = pattern_counts[pattern_counts >= 2].index.tolist()
    
    # If fewer than 2 patterns have multiple series, include all patterns
    if len(common_patterns) < 2:
        common_patterns = pattern_counts.index.tolist()
    
    # Filter data to include only common patterns
    filtered_episodes = episode_df[episode_df['pattern'].isin(common_patterns)]
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Update matplotlib settings based on config
    plt.rcParams.update({
        'font.family': config['global'].get('font_family', 'Arial'),
        'font.size': config['fonts'].get('axis_label_size', 12),
        'axes.labelsize': config['fonts'].get('axis_label_size', 12),
        'axes.titlesize': config['fonts'].get('title_size', 16),
        'xtick.labelsize': config['fonts'].get('tick_label_size', 10),
        'ytick.labelsize': config['fonts'].get('tick_label_size', 10),
    })
    
    # Create the violin plot
    ax = sns.violinplot(
        data=filtered_episodes,
        x='pattern',
        y='imdb_rating',
        hue='position',
        palette=position_colors,
        split=False,
        inner='quartile',
        cut=0,
        linewidth=1.5
    )
    
    # Set plot labels and title
    plt.xlabel('Contestant Ranking Pattern', fontsize=14, fontweight='bold')
    plt.ylabel('IMDb Rating', fontsize=14, fontweight='bold')
    
    # Customize legend
    legend_elements = [
        mpatches.Patch(facecolor=position_colors[0], edgecolor='black', label='First Third'),
        mpatches.Patch(facecolor=position_colors[1], edgecolor='black', label='Middle Third'),
        mpatches.Patch(facecolor=position_colors[2], edgecolor='black', label='Last Third')
    ]
    plt.legend(handles=legend_elements, title='Episode Position', fontsize=12)
    
    # Add pattern descriptions as annotations
    pattern_descriptions = {
        '123': 'Rising: Contestants start\nlow, improve steadily',
        '213': 'J-shaped: Drop\nthen rebound',
        '231': 'Middle improvement\nthen decline',
        '312': 'Improving\nsignificantly at end',
        '132': 'Rise then\nslight improvement',
        '321': 'Consistently\nimproving'
    }
    
    # Add pattern descriptions above the violins
    for i, pattern in enumerate(ax.get_xticklabels()):
        pattern_text = pattern.get_text()
        if pattern_text in pattern_descriptions:
            ax.text(
                i, 
                9.5,  # Position above the highest violin
                pattern_descriptions[pattern_text],
                ha='center', 
                va='center',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3', edgecolor='gray')
            )
    
    # Add series numbers below each pattern
    for i, pattern in enumerate(sorted(filtered_episodes['pattern'].unique())):
        # Get series with this pattern
        series_with_pattern = series_df[series_df['pattern'] == pattern]['series'].tolist()
        series_text = f"Series: {', '.join(map(str, sorted(series_with_pattern)))}"
        
        ax.text(
            i, 
            6.2,  # Position below the violins
            series_text,
            ha='center', 
            va='center',
            fontsize=9,
            style='italic',
            color='#555555'
        )
    
    # Set y-axis limits to ensure all annotations are visible
    plt.ylim(6.0, 9.7)
    
    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_file_pdf = SCRIPT_DIR / f"figure2_output.pdf"
    output_file_png = SCRIPT_DIR / f"figure2_output.png"
    
    plt.savefig(output_file_pdf, dpi=dpi, bbox_inches='tight')
    plt.savefig(output_file_png, dpi=dpi, bbox_inches='tight')
    
    print(f"Saved figure to {output_file_pdf} and {output_file_png}")
    
    return output_file_pdf, output_file_png


def main():
    """Main plotting function."""
    print("Creating Figure 2: Episode Rating Trajectories by Contestant Ranking Patterns")
    
    # Load config
    config = load_config()
    
    # Load data
    episode_df, series_df = load_data()
    
    if episode_df is None or series_df is None:
        print("Data loading failed. Cannot create plot.")
        return
    
    # Create violin plot
    output_files = create_violin_plot(episode_df, series_df, config)
    
    print(f"Figure 2 plotting complete! Output files: {output_files}")


if __name__ == "__main__":
    main() 