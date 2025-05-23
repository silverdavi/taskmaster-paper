#!/usr/bin/env python3
"""
Data processing for Figure 6: Scoring Pattern Analysis

This script:
1. Generates all possible scoring histograms for 5 contestants (scores 0-5)
2. Extracts actual scoring patterns from the dataset
3. Calculates statistical properties (mean, variance, skew) for each pattern
4. Saves processed data for visualization

Output: data/processed/figure6_scoring_patterns.csv
"""

import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import skew
from itertools import combinations_with_replacement
import os
from pathlib import Path

# Configuration
MAX_SCORE = 5  # Maximum score in Taskmaster (0-5)
OUTPUT_DIR = Path(".")
OUTPUT_FILE = OUTPUT_DIR / "figure6_scoring_patterns.csv"

def load_scores_data():
    """Load the scores dataset."""
    try:
        scores_data = pd.read_csv("../../data/raw/scores.csv")
        print(f"Loaded scores data: {len(scores_data)} score entries")
        return scores_data
    except FileNotFoundError:
        print("Error: Could not find ../../data/raw/scores.csv")
        raise

def create_histogram_from_scores(scores):
    """
    Convert a list of scores into a histogram.
    
    Args:
        scores: List of scores (0-MAX_SCORE)
        
    Returns:
        A tuple representing counts of scores 0-MAX_SCORE
    """
    histogram = [0] * (MAX_SCORE + 1)  # For scores 0-MAX_SCORE
    
    for score in scores:
        if 0 <= score <= MAX_SCORE:
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
    scores = []
    for score, count in enumerate(histogram):
        scores.extend([score] * count)
    
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

def get_actual_histogram_frequencies(scores_data):
    """
    Get frequencies of histograms in the actual dataset.
    
    Args:
        scores_data: DataFrame with scores data
        
    Returns:
        Dict mapping histogram tuples to their frequency counts
    """
    histograms = {}
    
    # Group scores by task and create histograms
    tasks = scores_data.groupby(['series', 'episode', 'task_id'])
    
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
    """Format histogram as a string."""
    return "[" + ", ".join(map(str, hist)) + "]"

def process_scoring_data():
    """
    Main processing function that generates all data needed for Figure 6.
    
    Returns:
        DataFrame with all histogram data and statistics
    """
    print("Processing scoring pattern data for Figure 6...")
    
    # Load scores data
    scores_data = load_scores_data()
    
    # Generate all possible histograms
    all_histograms = generate_all_possible_histograms()
    print(f"Generated {len(all_histograms)} possible histograms for scores 0-{MAX_SCORE}")
    
    # Get actual frequencies
    actual_frequencies = get_actual_histogram_frequencies(scores_data)
    print(f"Found {len(actual_frequencies)} histograms in the actual data")
    
    # Calculate statistics for each histogram
    histogram_stats = []
    for hist in all_histograms:
        stats = calculate_histogram_stats(hist)
        stats['histogram'] = format_histogram(hist)
        stats['histogram_tuple'] = str(hist)  # For exact matching
        stats['frequency'] = actual_frequencies.get(hist, 0)
        stats['sorted_set'] = histogram_to_sorted_set(hist)
        stats['is_used'] = hist in actual_frequencies
        histogram_stats.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(histogram_stats)
    
    # Add some additional useful columns
    stats_df['log_frequency'] = np.log1p(stats_df['frequency'])  # log(1 + frequency)
    stats_df['sqrt_frequency'] = np.sqrt(stats_df['frequency'])   # For sizing points
    
    return stats_df

def save_processed_data(data_df):
    """Save processed data to CSV file."""
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    data_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved processed data to {OUTPUT_FILE}")
    
    # Print summary statistics
    used_patterns = data_df[data_df['is_used']].shape[0]
    total_patterns = data_df.shape[0]
    used_percentage = used_patterns / total_patterns * 100
    
    print(f"\nSummary:")
    print(f"Total possible patterns: {total_patterns}")
    print(f"Patterns used in show: {used_patterns} ({used_percentage:.1f}%)")
    print(f"Total task instances: {data_df['frequency'].sum()}")
    
    # Most frequent patterns
    print(f"\nTop 5 most frequent patterns:")
    top_patterns = data_df[data_df['frequency'] > 0].nlargest(5, 'frequency')
    for _, row in top_patterns.iterrows():
        print(f"  {row['sorted_set']}: {row['frequency']} times (mean={row['mean']:.2f})")

if __name__ == "__main__":
    # Process the data
    scoring_data = process_scoring_data()
    
    # Save processed data
    save_processed_data(scoring_data)
    
    print("Data processing complete!") 