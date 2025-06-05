#!/usr/bin/env python3
"""
Plot Figure 6: Scoring Pattern Analysis

This script creates a scatter plot visualization showing the geometry of 
Taskmaster scoring patterns, where:
- Each point represents a possible score distribution for 5 contestants
- X-axis: Mean score
- Y-axis: Variance (higher = more varied scores)  
- Color: Skew (asymmetry of distribution)
- Size: Frequency of use in actual data (black circles)

Input: figure6_scoring_patterns.csv
Output: figure6.png, figure6.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from pathlib import Path

# Configuration
INPUT_FILE = Path("figure6_scoring_patterns.csv")
OUTPUT_DIR = Path(".")
FIGURE_NAME = "fig8"

def load_configuration():
    """Load plotting configuration."""
    # Set consistent styling
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Color scheme from crameri
    return {
        'colormap': 'coolwarm',
        'figure_size': (14, 12),
        'dpi': 300
    }

def load_processed_data():
    """Load the processed scoring patterns data."""
    try:
        data = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(data)} scoring patterns")
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        print("Please run process_data.py first to generate the processed data.")
        raise

def add_jitter(df, x_col, y_col, amount=0.01):
    """
    Add small random jitter to x and y coordinates to separate overlapping points.
    Only adds jitter to points with identical coordinates.
    
    Args:
        df: DataFrame containing the data
        x_col: Name of x-coordinate column
        y_col: Name of y-coordinate column
        amount: Maximum jitter amount
        
    Returns:
        DataFrame with jittered x and y coordinates
    """
    from collections import Counter
    
    # Create a copy of the DataFrame
    df_jittered = df.copy()
    
    # Find duplicate coordinates
    coords = df_jittered[[x_col, y_col]].values
    coord_counts = Counter(map(tuple, coords))
    
    # Only add jitter to points with the same coordinates
    for i, row in df_jittered.iterrows():
        coord = (row[x_col], row[y_col])
        if coord_counts[coord] > 1:
            # Add small random jitter
            df_jittered.at[i, x_col] += np.random.uniform(-amount, amount)
            df_jittered.at[i, y_col] += np.random.uniform(-amount, amount)
    
    return df_jittered

def create_scoring_visualization(data, config):
    """
    Create the main scoring patterns visualization.
    
    Args:
        data: DataFrame with scoring pattern data
        config: Configuration dictionary
        
    Returns:
        Figure and axis objects
    """
    # Create figure
    fig, ax = plt.subplots(figsize=config['figure_size'])
    
    # Plot all possible histograms as background
    scatter = ax.scatter(
        data['mean'],            # X-axis: Mean score
        data['variance'],        # Y-axis: Variance
        c=data['skew'],          # Color: Skew
        s=50,                    # Size: Fixed for all possible histograms
        alpha=0.9,               # Transparency
        cmap=config['colormap'], # Colormap
        edgecolors='none',       # No edge color for background points
        zorder=1                 # Base layer
    )
    
    # Overlay actual histograms with black circles
    actual_df = data[data['is_used']].copy()
    if not actual_df.empty:
        # Add jitter to prevent overlapping circles
        actual_df = add_jitter(actual_df, 'mean', 'variance', amount=0.02)
        
        # Scale frequencies for better visibility (sqrt scaling)
        sizes = actual_df['sqrt_frequency'] * 50
        
        # Plot actual histograms
        ax.scatter(
            actual_df['mean'],          # X-axis: Mean score
            actual_df['variance'],      # Y-axis: Variance
            s=sizes,                    # Size: Based on frequency
            facecolors='none',          # No fill
            edgecolors='black',         # Black edge
            linewidths=1.5,             # Edge width
            alpha=0.8,                  # Transparency
            zorder=2                    # Top layer
        )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Skew')
    
    # Add size legend
    if not actual_df.empty:
        # Select a few size examples for legend
        freq_examples = [1, 10, 50]
        max_freq = actual_df['frequency'].max()
        freq_examples = [f for f in freq_examples if f <= max_freq]
        
        if max_freq > 50:
            freq_examples.append(int(max_freq))
        
        legend_elements = []
        for freq in freq_examples:
            size = np.sqrt(freq) * 50
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                      markeredgecolor='black', markersize=np.sqrt(size/np.pi),
                      label=f'{freq} task{"s" if freq != 1 else ""}')
            )
        
        if legend_elements:
            ax.legend(handles=legend_elements, title="Frequency", loc="upper right")
    
    # Add annotations for key patterns
    add_key_pattern_annotations(ax, data)
    
    # Customize appearance
    ax.set_xlabel('Mean Score')
    ax.set_ylabel('Variance')

    return fig, ax, actual_df

def add_key_pattern_annotations(ax, data):
    """Add annotations for key scoring patterns showing both score set and histogram."""
    
    # Define key patterns to annotate
    key_patterns = [
        # Extreme low: all zeros
        {'sorted_set': '{0, 0, 0, 0, 0}', 'histogram': '[5, 0, 0, 0, 0, 0]'},
        # High variance: mostly zeros with one max
        {'sorted_set': '{0, 0, 0, 0, 5}', 'histogram': '[4, 0, 0, 0, 0, 1]'},
        # Middle pattern: perfect spread
        {'sorted_set': '{1, 2, 3, 4, 5}', 'histogram': '[0, 1, 1, 1, 1, 1]'},
        # Extreme high: all fives
        {'sorted_set': '{5, 5, 5, 5, 5}', 'histogram': '[0, 0, 0, 0, 0, 5]'}
    ]
    
    for pattern_info in key_patterns:
        # Find this pattern in the data
        pattern_data = data[data['sorted_set'] == pattern_info['sorted_set']]
        
        if not pattern_data.empty:
            row = pattern_data.iloc[0]
            x, y = row['mean'], row['variance']
            
            # Create annotation text with both representations
            annotation_text = f"{pattern_info['sorted_set']}\n{pattern_info['histogram']}"
            
            # Position annotation based on location
            if x < 1.5:  # Left side (low scores)
                xytext = (x + 0.3, y + 0.3)
                ha = 'left'
            elif x > 4.0:  # Right side (high scores)
                xytext = (x - 0.3, y + 0.3)
                ha = 'right'
            else:  # Middle
                xytext = (x, y + 0.5)
                ha = 'center'
            
            # Add annotation
            ax.annotate(
                annotation_text,
                xy=(x, y),
                xytext=xytext,
                ha=ha,
                va='bottom',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=1.5),
                zorder=3
            )

def save_figure(fig, output_dir, figure_name, config):
    """Save figure in multiple formats."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in multiple formats
    for ext in ['png', 'pdf']:
        output_path = output_dir / f"{figure_name}.{ext}"
        fig.savefig(output_path, dpi=config['dpi'], bbox_inches='tight')
        print(f"Saved figure to {output_path}")

def print_summary_statistics(data):
    """Print summary statistics about the scoring patterns."""
    total_patterns = len(data)
    used_patterns = data['is_used'].sum()
    used_percentage = used_patterns / total_patterns * 100
    
    print(f"\n=== Figure 8: Scoring Patterns Summary ===")
    print(f"Total possible patterns: {total_patterns}")
    print(f"Patterns used in show: {used_patterns} ({used_percentage:.1f}%)")
    print(f"Total task instances: {data['frequency'].sum()}")
    
    # Most frequent patterns
    print(f"\nTop 5 most frequent patterns:")
    used_data = data[data['is_used']]
    top_patterns = used_data.nlargest(5, 'frequency')
    for _, row in top_patterns.iterrows():
        print(f"  {row['sorted_set']}: {row['frequency']} times (mean={row['mean']:.2f}, var={row['variance']:.2f})")

def main():
    """Main function to create Figure 8."""
    print("Creating Figure 8: Scoring Pattern Analysis...")
    
    # Load configuration and data
    config = load_configuration()
    data = load_processed_data()
    
    # Create visualization
    fig, ax, actual_df = create_scoring_visualization(data, config)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    save_figure(fig, OUTPUT_DIR, FIGURE_NAME, config)
    
    # Close figure to free memory
    plt.close(fig)
    
    # Print summary statistics
    print_summary_statistics(data)
    
    print(f"\nFigure 8 complete! Check {OUTPUT_DIR} for output files.")

if __name__ == "__main__":
    main() 