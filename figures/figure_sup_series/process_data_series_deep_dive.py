#!/usr/bin/env python3
"""
Process data for Series Deep Dive Supplementary Figure

This script processes the scores.csv data to generate:
1. Ranking progression data for each task within a series
2. Cumulative score data for all contestants
3. Episode boundaries for proper visualization

Author: Taskmaster Analysis Team
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_scores_data():
    """Load the main scores CSV file"""
    print("Loading scores data...")
    df = pd.read_csv('../../data/raw/scores.csv')
    print(f"Loaded {len(df)} score records")
    return df

def get_series_info(df):
    """Get information about available series"""
    series_info = df.groupby('series').agg({
        'episode': ['min', 'max', 'nunique'],
        'task_id': 'nunique',
        'contestant_name': 'nunique'
    }).round(2)
    
    series_info.columns = ['min_episode', 'max_episode', 'num_episodes', 'num_tasks', 'num_contestants']
    print("\nSeries information:")
    print(series_info)
    return series_info

def process_series_data(df, series_num):
    """
    Process data for a specific series to get ranking progression and cumulative scores
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The scores dataframe
    series_num : int
        Series number to process
    
    Returns:
    --------
    dict
        Dictionary containing processed data for the series
    """
    print(f"\nProcessing Series {series_num}...")
    
    # Filter data for the specific series
    series_df = df[df['series'] == series_num].copy()
    
    if len(series_df) == 0:
        print(f"No data found for Series {series_num}")
        return None
    
    # Get unique contestants and tasks
    contestants = sorted(series_df['contestant_name'].unique())
    tasks = sorted(series_df['task_id'].unique())
    episodes = sorted(series_df['episode'].unique())
    
    print(f"  Contestants: {contestants}")
    print(f"  Episodes: {episodes}")
    print(f"  Tasks: {len(tasks)}")
    
    # Create task-episode mapping
    task_episode_map = series_df.groupby('task_id')['episode'].first().to_dict()
    
    # Initialize data structures
    cumulative_scores = {contestant: [] for contestant in contestants}
    rankings = {contestant: [] for contestant in contestants}
    task_positions = []
    episode_boundaries = []
    
    # Process each task in order
    current_totals = {contestant: 0 for contestant in contestants}
    
    for i, task_id in enumerate(tasks):
        task_data = series_df[series_df['task_id'] == task_id]
        
        # Get scores for this task
        task_scores = {}
        for _, row in task_data.iterrows():
            contestant = row['contestant_name']
            score = row['total_score']
            task_scores[contestant] = score
            current_totals[contestant] += score
        
        # Calculate cumulative scores after this task
        for contestant in contestants:
            cumulative_scores[contestant].append(current_totals[contestant])
        
        # Calculate rankings (1 = highest cumulative score)
        sorted_contestants = sorted(contestants, key=lambda x: current_totals[x], reverse=True)
        task_rankings = {}
        for rank, contestant in enumerate(sorted_contestants, 1):
            task_rankings[contestant] = rank
        
        for contestant in contestants:
            rankings[contestant].append(task_rankings[contestant])
        
        # Track task position and episode boundaries
        task_positions.append(i + 1)
        
        # Mark episode boundaries
        if i == 0 or task_episode_map[task_id] != task_episode_map[tasks[i-1]]:
            episode_boundaries.append(i + 1)
    
    # Add final episode boundary
    if len(tasks) + 1 not in episode_boundaries:
        episode_boundaries.append(len(tasks) + 1)
    
    return {
        'series': series_num,
        'contestants': contestants,
        'tasks': tasks,
        'episodes': episodes,
        'task_positions': task_positions,
        'episode_boundaries': episode_boundaries,
        'cumulative_scores': cumulative_scores,
        'rankings': rankings,
        'task_episode_map': task_episode_map,
        'final_scores': {contestant: current_totals[contestant] for contestant in contestants}
    }

def save_series_data(series_data, series_num):
    """Save processed series data to JSON file"""
    if series_data is None:
        return
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean data for JSON serialization
    clean_data = {}
    for key, value in series_data.items():
        if isinstance(value, dict):
            clean_data[key] = {k: convert_numpy(v) for k, v in value.items()}
        elif isinstance(value, list):
            clean_data[key] = [convert_numpy(item) for item in value]
        else:
            clean_data[key] = convert_numpy(value)
    
    # Save to file
    output_file = f'series_{series_num}_data.json'
    with open(output_file, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    print(f"  Saved data to {output_file}")

def create_summary_stats(all_series_data):
    """Create summary statistics across all series"""
    summary = {
        'total_series': len(all_series_data),
        'series_processed': [],
        'contestant_counts': {},
        'task_counts': {},
        'episode_counts': {}
    }
    
    for series_num, data in all_series_data.items():
        if data is not None:
            summary['series_processed'].append(int(series_num))
            summary['contestant_counts'][str(series_num)] = len(data['contestants'])
            summary['task_counts'][str(series_num)] = len(data['tasks'])
            summary['episode_counts'][str(series_num)] = len(data['episodes'])
    
    return summary

def main():
    """Main function to process series data"""
    print("=== Processing Series Deep Dive Data ===")
    
    # Load and process data
    df = load_scores_data()
    series_info = get_series_info(df)
    
    print(f"Available series: {sorted(series_info.index.tolist())}")
    
    # Process all available series
    all_series_data = {}
    for series_num in sorted(series_info.index.tolist()):
        print(f"\nProcessing Series {series_num}...")
        series_data = process_series_data(df, series_num)
        
        if series_data is not None:
            all_series_data[series_num] = series_data
            save_series_data(series_data, series_num)
            print(f"✓ Series {series_num} processed successfully")
        else:
            print(f"✗ Series {series_num} processing failed")
    
    # Create summary
    summary = create_summary_stats(all_series_data)
    
    # Save summary
    with open('series_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {len(all_series_data)} series")
    print(f"Series processed: {sorted(all_series_data.keys())}")

if __name__ == "__main__":
    main() 