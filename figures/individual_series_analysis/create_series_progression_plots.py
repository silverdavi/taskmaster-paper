#!/usr/bin/env python3
"""
Create Series Deep Dive Plots

This script creates visualizations for series deep dives showing:
1. Top plot: Ranking progression with episode boundaries
2. Bottom plots: Cumulative scores with seaborn styling

Author: Taskmaster Analysis Team
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent.parent / "config"))
from plot_utils import apply_plot_style, get_series_colors

def load_series_data(series_num):
    """Load processed series data from JSON file"""
    try:
        with open(f'series_{series_num}_data.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Data file for series {series_num} not found. Run data processing first.")
        return None

def create_ranking_plot(ax, data):
    """
    Create ranking progression plot (top subplot)
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Series data dictionary
    """
    contestants = data['contestants']
    task_positions = data['task_positions']
    episode_boundaries = data['episode_boundaries']
    rankings = data['rankings']
    
    # Get colors for contestants
    colors = sns.color_palette("husl", len(contestants))
    
    # Plot ranking progression for each contestant
    for i, contestant in enumerate(contestants):
        contestant_rankings = rankings[contestant]
        
        # Plot line with markers (beads)
        ax.plot(task_positions, contestant_rankings, 
                color=colors[i], linewidth=2.5, alpha=0.8,
                marker='o', markersize=8, markerfacecolor=colors[i], 
                markeredgecolor='white', markeredgewidth=1.5,
                label=contestant, zorder=3)
    
    # Add episode boundaries with shading
    for i in range(len(episode_boundaries) - 1):
        start = episode_boundaries[i]
        end = episode_boundaries[i + 1]
        
        # Alternate shading for episodes
        if i % 2 == 0:
            ax.axvspan(start - 0.5, end - 0.5, alpha=0.1, color='gray', zorder=1)
        
        # Add episode boundary lines
        if i > 0:  # Don't add line at the very beginning
            ax.axvline(start - 0.5, color='black', linestyle='--', alpha=0.5, zorder=2)
    
    # Customize axes
    ax.set_xlabel('Task Number', fontweight='bold')
    ax.set_ylabel('Ranking Position', fontweight='bold')
    
    # Invert y-axis so rank 1 is at the top
    ax.invert_yaxis()
    ax.set_ylim(len(contestants) + 0.5, 0.5)
    
    # Set integer ticks for rankings
    ax.set_yticks(range(1, len(contestants) + 1))
    
    # No legend on this plot - will be shown on cumulative plot
    
    # Add episode labels
    episode_centers = []
    for i in range(len(episode_boundaries) - 1):
        start = episode_boundaries[i]
        end = episode_boundaries[i + 1]
        center = (start + end - 1) / 2
        episode_centers.append(center)
    
    # Add episode numbers at the top
    for i, center in enumerate(episode_centers):
        ax.text(center, 0.3, f'Ep {i+1}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold', alpha=0.7)

def create_cumulative_plot(ax, data):
    """
    Create cumulative score plot (bottom subplot)
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : dict
        Series data dictionary
    """
    contestants = data['contestants']
    task_positions = data['task_positions']
    cumulative_scores = data['cumulative_scores']
    
    # Get colors for contestants (same as ranking plot)
    colors = sns.color_palette("husl", len(contestants))
    
    # Line plot with seaborn style
    for i, contestant in enumerate(contestants):
        scores = cumulative_scores[contestant]
        ax.plot(task_positions, scores, 
                color=colors[i], linewidth=2.5, alpha=0.8,
                marker='o', markersize=6, markerfacecolor=colors[i],
                markeredgecolor='white', markeredgewidth=1,
                label=contestant)
    
    ax.set_xlabel('Task Number', fontweight='bold')
    ax.set_ylabel('Cumulative Score', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)

def create_series_deep_dive_plot(series_num, save_path=None):
    """
    Create the complete series deep dive plot
    
    Parameters:
    -----------
    series_num : int
        Series number to plot
    save_path : str, optional
        Path to save the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    # Load data
    data = load_series_data(series_num)
    if data is None:
        return None
    
    # Apply plot style
    config = apply_plot_style()
    
    # Create figure with 2 rows layout
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid: top plot and bottom plot
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Top plot (ranking progression)
    ax_top = fig.add_subplot(gs[0])
    
    # Bottom plot (cumulative scores)
    ax_bottom = fig.add_subplot(gs[1])
    
    # Create plots
    create_ranking_plot(ax_top, data)
    create_cumulative_plot(ax_bottom, data)
    
    # Apply styling to all axes
    for ax in [ax_top, ax_bottom]:
        ax.tick_params(labelsize=config['fonts']['tick_label_size'])
    
    # Add overall title
    fig.suptitle(f'Taskmaster Series {data["series"]} Deep Dive', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=config['global']['dpi'], 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to: {save_path}")
        
        # Also save as PDF
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, dpi=config['global']['dpi'], 
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Plot saved to: {pdf_path}")
    
    return fig

def main():
    """Main function to create series deep dive plots"""
    print("=== Creating Series Deep Dive Plots ===")
    
    # Check which series data files are available
    available_series = []
    for i in range(1, 19):  # Check series 1-18
        if Path(f'series_{i}_data.json').exists():
            available_series.append(i)
    
    print(f"Available series data: {available_series}")
    
    if not available_series:
        print("No series data files found. Run data processing first.")
        return
    
    # Create plots for available series
    for series_num in available_series:
        print(f"\nCreating plot for Series {series_num}...")
        
        save_path = f's3_fig_{series_num}.png'
        fig = create_series_deep_dive_plot(series_num, save_path)
        
        if fig is not None:
            plt.close(fig)  # Close to free memory
    
    print(f"\n=== Plotting Complete ===")
    print(f"Created plots for {len(available_series)} series")

if __name__ == "__main__":
    main() 