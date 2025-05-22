#!/usr/bin/env python3
"""
Visualize all possible 5-person Taskmaster scoring histograms.

This script creates a scatter plot where:
- Each point represents a possible score histogram
- X-axis: Mean score
- Y-axis: Variance (higher = more varied scores)
- Color: Skew (asymmetry of distribution)
- Size of circle for frequency in real data.

The visualization shows all possible histograms and highlights which ones
were actually used in the dataset and how frequently.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import seaborn as sns
from collections import Counter
from scipy.stats import skew
from itertools import combinations_with_replacement
import os
from pathlib import Path

# Set styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create results directory
results_dir = Path("results/histogram_analysis")
results_dir.mkdir(parents=True, exist_ok=True)

# Set maximum score (0 to MAX_SCORE)
MAX_SCORE = 6

def load_detailed_scores():
    """Load detailed contestant-level scores data."""
    try:
        # Try to load scores from main data file
        scores_data = pd.read_csv("data/scores.csv")
        print(f"Loaded scores data: {len(scores_data)} score entries")
        return scores_data
    except FileNotFoundError:
        print("Error: Could not find scores data file.")
        return None

def create_histogram_from_scores(scores):
    """
    Convert a list of scores into a histogram.
    
    Args:
        scores: List of scores (0-MAX_SCORE)
        
    Returns:
        A tuple representing counts of scores 0-MAX_SCORE
    """
    # Initialize histogram with zeros
    histogram = [0] * (MAX_SCORE + 1)  # For scores 0-MAX_SCORE
    
    # Count occurrences of each score
    for score in scores:
        if 0 <= score <= MAX_SCORE:  # Only consider scores 0-MAX_SCORE
            histogram[score] += 1
            
    return tuple(histogram)

def histogram_to_sorted_set(histogram):
    """
    Convert a histogram to a sorted set representation.
    
    Args:
        histogram: Tuple of score counts [count0, count1, count2, ...]
        
    Returns:
        String representation of sorted scores like "{1,2,3,4,5}"
    """
    # Expand histogram into individual scores
    scores = []
    for score, count in enumerate(histogram):
        scores.extend([score] * count)
    
    # Return as a sorted set string
    return "{" + ", ".join(map(str, sorted(scores))) + "}"

def calculate_histogram_stats(histogram):
    """
    Calculate statistics for a histogram.
    
    Args:
        histogram: Tuple of score counts [count0, count1, count2, ...]
        
    Returns:
        Dict with mean, variance, skew
    """
    # Expand histogram into a list of scores
    scores = []
    for i, count in enumerate(histogram):
        scores.extend([i] * count)
    
    # Calculate statistics
    mean = np.mean(scores)
    variance = np.var(scores)
    skew_value = skew(scores) if len(set(scores)) > 1 else 0
    
    return {
        'mean': mean,
        'variance': variance,
        'skew': skew_value
    }

def generate_all_possible_histograms():
    """
    Generate all possible histograms for 5 contestants with scores 0-MAX_SCORE.
    
    Returns:
        List of all possible histograms as tuples
    """
    # Generate all combinations of 5 positions (with replacement) from possible scores
    all_combinations = combinations_with_replacement(range(MAX_SCORE + 1), 5)
    
    # Convert each combination to a histogram
    all_histograms = []
    histogram_set = set()
    
    for combo in all_combinations:
        # Create histogram from this combination
        hist = [0] * (MAX_SCORE + 1)
        for score in combo:
            hist[score] += 1
        
        # Add as tuple (immutable)
        hist_tuple = tuple(hist)
        if hist_tuple not in histogram_set:
            histogram_set.add(hist_tuple)
            all_histograms.append(hist_tuple)
    
    return all_histograms

def get_actual_histogram_frequencies():
    """
    Get frequencies of histograms in the actual dataset.
    
    Returns:
        Dict mapping histogram tuples to their frequency counts
    """
    contestant_scores = load_detailed_scores()
    
    if contestant_scores is None:
        return {}
    
    # Group scores by task and create histograms
    histograms = {}
    
    # Identify task keys (combination of series, episode, task_id)
    tasks = contestant_scores.groupby(['series', 'episode', 'task_id'])
    
    # Process each task
    for (series, episode, task_id), task_data in tasks:
        # Extract scores for this task
        try:
            task_scores = task_data['total_score'].astype(int).values
            
            # Skip tasks with scores > MAX_SCORE
            if any(score > MAX_SCORE for score in task_scores):
                continue
                
        except:
            # Skip if scores can't be converted to integers
            continue
        
        # Skip if not exactly 5 contestants
        if len(task_scores) != 5:
            continue
            
        # Create histogram
        histogram = create_histogram_from_scores(task_scores)
        
        # Store with task key
        task_key = f"{series}_{episode}_{task_id}"
        histograms[task_key] = histogram
    
    # Count frequencies
    histogram_counts = Counter(histograms.values())
    
    return histogram_counts

def format_histogram(hist):
    """Format histogram with consistent spacing."""
    return "[" + ", ".join(map(str, hist)) + "]"

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

def select_points_to_label(df, n_points=10):
    """
    Select diverse points to label, focusing on extremes and interesting patterns.
    
    Args:
        df: DataFrame with histogram data
        n_points: Number of points to select
        
    Returns:
        DataFrame with selected points
    """
    # Ensure we have at least some points
    if len(df) <= n_points:
        return df
        
    selected_points = pd.DataFrame()
    
    # Always include most frequent pattern if available
    most_frequent_idx = df['frequency'].idxmax()
    selected_points = pd.concat([selected_points, df.loc[[most_frequent_idx]]])
    
    # Get the remaining points, excluding most frequent
    remaining = df.drop(most_frequent_idx)
    
    # Select extreme points:
    # 1. Lowest mean score with frequency > 1
    if not remaining[remaining['frequency'] > 1].empty:
        lowest_mean_idx = remaining[remaining['frequency'] > 1]['mean'].idxmin()
        selected_points = pd.concat([selected_points, df.loc[[lowest_mean_idx]]])
        remaining = remaining.drop(lowest_mean_idx)
    
    # 2. Highest mean score with frequency > 1
    if not remaining[remaining['frequency'] > 1].empty:
        highest_mean_idx = remaining[remaining['frequency'] > 1]['mean'].idxmax()
        selected_points = pd.concat([selected_points, df.loc[[highest_mean_idx]]])
        remaining = remaining.drop(highest_mean_idx)
    
    # 3. Highest variance with frequency > 1
    if not remaining[remaining['frequency'] > 1].empty:
        highest_var_idx = remaining[remaining['frequency'] > 1]['variance'].idxmax()
        selected_points = pd.concat([selected_points, df.loc[[highest_var_idx]]])
        remaining = remaining.drop(highest_var_idx)
    
    # 4. Lowest variance with frequency > 1
    if not remaining[remaining['frequency'] > 1].empty:
        lowest_var_idx = remaining[remaining['frequency'] > 1]['variance'].idxmin()
        selected_points = pd.concat([selected_points, df.loc[[lowest_var_idx]]])
        remaining = remaining.drop(lowest_var_idx)
    
    # 5. Most positively skewed with frequency > 1
    if not remaining[remaining['frequency'] > 1].empty:
        pos_skew_idx = remaining[remaining['frequency'] > 1]['skew'].idxmax()
        selected_points = pd.concat([selected_points, df.loc[[pos_skew_idx]]])
        remaining = remaining.drop(pos_skew_idx)
    
    # 6. Most negatively skewed with frequency > 1
    if not remaining[remaining['frequency'] > 1].empty:
        neg_skew_idx = remaining[remaining['frequency'] > 1]['skew'].idxmin()
        selected_points = pd.concat([selected_points, df.loc[[neg_skew_idx]]])
        remaining = remaining.drop(neg_skew_idx)
    
    # Check if we need more points
    if len(selected_points) < n_points and not remaining.empty:
        # Get some distant points based on extremeness
        remaining['extremeness'] = (
            ((remaining['mean'] - df['mean'].mean()) / df['mean'].std())**2 +
            ((remaining['variance'] - df['variance'].mean()) / df['variance'].std())**2 +
            ((remaining['skew'] - df['skew'].mean()) / df['skew'].std())**2
        )
        
        # Get additional extreme points
        n_additional = min(n_points - len(selected_points), len(remaining))
        extreme_points_idx = remaining['extremeness'].nlargest(n_additional).index
        selected_points = pd.concat([selected_points, df.loc[extreme_points_idx]])
    
    return selected_points

def visualize_histograms():
    """Create visualization of all possible histograms with actual usage."""
    # Generate all possible histograms
    all_histograms = generate_all_possible_histograms()
    print(f"Generated {len(all_histograms)} possible histograms for scores 0-{MAX_SCORE}")
    
    # Get actual frequencies
    actual_frequencies = get_actual_histogram_frequencies()
    print(f"Found {len(actual_frequencies)} histograms in the actual data")
    
    # Calculate statistics for each histogram
    histogram_stats = []
    for hist in all_histograms:
        stats = calculate_histogram_stats(hist)
        stats['histogram'] = hist
        stats['frequency'] = actual_frequencies.get(hist, 0)
        stats['sorted_set'] = histogram_to_sorted_set(hist)
        stats['formatted_hist'] = format_histogram(hist)
        histogram_stats.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(histogram_stats)
    
    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot all possible histograms
    scatter = ax.scatter(
        stats_df['mean'],            # X-axis: Mean score
        stats_df['variance'],        # Y-axis: Variance
        c=stats_df['skew'],          # Color: Skew
        s=50,                        # Size: Fixed for all possible histograms
        alpha=0.9,                   # Transparency
        cmap='coolwarm',               # Colormap: plasma (original)
        edgecolors='none',           # No edge color for all possible histograms
        zorder=1                     # Base layer
    )
    
    # Overlay actual histograms with black circles
    actual_df = stats_df[stats_df['frequency'] > 0].copy()
    if not actual_df.empty:
        # Add jitter to prevent overlapping circles
        actual_df = add_jitter(actual_df, 'mean', 'variance', amount=0.02)
        
        # Scale frequencies for better visibility (sqrt scaling)
        sizes = np.sqrt(actual_df['frequency']) * 50
        
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
        # Select a few size examples
        freq_examples = [1, 10, 50, 350]
        legend_elements = []
        
        for freq in freq_examples:
            if freq <= actual_df['frequency'].max():
                size = np.sqrt(freq) * 50
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                          markeredgecolor='black', markersize=np.sqrt(size/np.pi),
                          label=f'{freq} tasks')
                )
        
        if legend_elements:
            ax.legend(handles=legend_elements, title="Frequency", loc="upper right")
    
    # Customize appearance
    ax.set_xlabel('Mean Score')
    ax.set_ylabel('Variance')
    plt.title(f'Geometry of Taskmaster Scoring Patterns (Scores 0-{MAX_SCORE})', fontsize=16)
    
    # Add explanatory note
    plt.figtext(0.5, 0.01, 
                "Each point represents a possible score distribution for 5 contestants.\n" +
                "Color shows skew. Black circles indicate patterns that occurred in the show (size = frequency).\n" +
                "Points without circles represent valid score distributions that were never used.",
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save the figure
    plt.savefig(results_dir / f'taskmaster_scoring_geometry_0-{MAX_SCORE}.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / f'taskmaster_scoring_geometry_0-{MAX_SCORE}.pdf', bbox_inches='tight')
    
    # Create a version with histogram labels including sorted sets
    if not actual_df.empty:
        # Select interesting points to label
        points_to_label = select_points_to_label(actual_df, n_points=8)
        
        for point in points_to_label.itertuples():
            # Add annotation with both sorted set and histogram
            ax.annotate(
                f"{point.sorted_set}\n{point.formatted_hist}",
                xy=(point.mean, point.variance),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
            )
        
        # Save labeled version in both formats
        plt.savefig(results_dir / f'taskmaster_scoring_geometry_0-{MAX_SCORE}_labeled.png', dpi=300, bbox_inches='tight')
        plt.savefig(results_dir / f'taskmaster_scoring_geometry_0-{MAX_SCORE}_labeled.pdf', bbox_inches='tight')
    
    plt.close()
    
    # Calculate additional statistics
    patterns_with_min_2_occurrences = sum(1 for freq in actual_frequencies.values() if freq >= 2)
    percentage_min_2 = patterns_with_min_2_occurrences / len(all_histograms) * 100
    
    # Print summary of usage coverage
    used_percentage = len(actual_frequencies) / len(all_histograms) * 100
    print(f"The show used {len(actual_frequencies)} out of {len(all_histograms)} possible patterns ({used_percentage:.1f}%)")
    print(f"Patterns used at least twice: {patterns_with_min_2_occurrences} ({percentage_min_2:.1f}%)")
    
    # Print top patterns
    print("\nTop 10 most frequent patterns:")
    top_patterns = sorted(actual_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    for pattern, count in top_patterns:
        stats = calculate_histogram_stats(pattern)
        sorted_set = histogram_to_sorted_set(pattern)
        formatted_hist = format_histogram(pattern)
        print(f"{sorted_set} {formatted_hist}: {count} times (mean={stats['mean']:.2f}, var={stats['variance']:.2f}, skew={stats['skew']:.2f})")

if __name__ == "__main__":
    visualize_histograms() 