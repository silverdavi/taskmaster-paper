#!/usr/bin/env python3
"""
Process scores.csv and break it into 18 CSVs (one per series) with the format:
ContestantID, ContestantName, Score_Task_1, Score_Task_2, ...

ContestantID is a running number from 1 to 90, where:
- 1-5 are Series 1 contestants
- 6-10 are Series 2 contestants
- etc.

Note: Alex Horne (the show's assistant) is excluded from the data processing.
"""

import pandas as pd
import os
import numpy as np
from pathlib import Path

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("data/processed/scores_by_series")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the scores data
    scores_df = pd.read_csv("data/raw/scores.csv")
    
    # Filter out Alex Horne (who is not a regular contestant but the show's assistant)
    scores_df = scores_df[scores_df['contestant_name'] != "Alex Horne"]
    
    # Map contestant names to IDs based on the specified rules
    # Get unique contestants by series
    contestants_by_series = {}
    for series in range(1, 19):
        series_scores = scores_df[scores_df['series'] == series]
        unique_contestants = sorted(series_scores['contestant_name'].unique())
        contestants_by_series[series] = unique_contestants
    
    # Create a mapping of contestant names to IDs
    contestant_id_map = {}
    contestant_name_map = {}  # Reverse mapping from ID to name
    current_id = 1
    
    for series in range(1, 19):
        for contestant in contestants_by_series[series]:
            contestant_id_map[contestant] = current_id
            contestant_name_map[current_id] = contestant
            current_id += 1
    
    # Add contestant_id column to the scores dataframe
    scores_df['contestant_id'] = scores_df['contestant_name'].map(contestant_id_map)
    
    # Process each series
    for series in range(1, 19):
        # Filter scores for this series
        series_scores = scores_df[scores_df['series'] == series]
        
        if len(series_scores) == 0:
            print(f"No data found for Series {series}")
            continue
            
        # Get unique tasks for this series
        tasks = series_scores['task_id'].unique()
        tasks = sorted(tasks)
        
        # Get unique contestants for this series
        contestants = series_scores['contestant_id'].unique()
        
        # Create a pivot table with contestants as rows and tasks as columns
        # First, create a DataFrame with the right structure
        result_df = pd.DataFrame(index=contestants)
        result_df.index.name = 'ContestantID'
        
        # Add contestant names
        result_df['ContestantName'] = result_df.index.map(contestant_name_map)
        
        # Add columns for each task
        for i, task_id in enumerate(tasks, 1):
            # Filter data for this task
            task_data = series_scores[series_scores['task_id'] == task_id]
            
            # Create a Series mapping contestant_id to score
            score_series = pd.Series(
                task_data['total_score'].values,
                index=task_data['contestant_id'].values
            )
            
            # Add to result DataFrame
            result_df[f'Score_Task_{i}'] = result_df.index.map(lambda x: score_series.get(x, np.nan))
        
        # Fill NaN values with 0 (for tasks a contestant didn't participate in)
        result_df = result_df.fillna(0)
        
        # Reset index to make ContestantID a column
        result_df = result_df.reset_index()
        
        # Sort by ContestantID
        result_df = result_df.sort_values('ContestantID')
        
        # Reorder columns to put ContestantID and ContestantName first
        cols = result_df.columns.tolist()
        cols = ['ContestantID', 'ContestantName'] + [col for col in cols if col not in ['ContestantID', 'ContestantName']]
        result_df = result_df[cols]
        
        # Save to CSV
        output_path = output_dir / f"series_{series}_scores.csv"
        result_df.to_csv(output_path, index=False)
        
        print(f"Processed Series {series}: {len(contestants)} contestants, {len(tasks)} tasks")
    
    # Create a readme file explaining the data format
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("# Scores by Series\n\n")
        f.write("This directory contains CSV files with contestant scores for each series of Taskmaster UK.\n\n")
        f.write("## File Format\n\n")
        f.write("Each file follows the format:\n")
        f.write("```\n")
        f.write("ContestantID, ContestantName, Score_Task_1, Score_Task_2, ...\n")
        f.write("```\n\n")
        f.write("Where:\n")
        f.write("- ContestantID is a running number from 1 to 90\n")
        f.write("- ContestantName is the contestant's name\n")
        f.write("- Score_Task_X is the score (0-5) for task X in that series\n\n")
        f.write("## Contestant ID Mapping\n\n")
        f.write("| Series | Contestant IDs | Contestants |\n")
        f.write("|--------|---------------|-------------|\n")
        
        start_id = 1
        for series in range(1, 19):
            if series in contestants_by_series:
                num_contestants = len(contestants_by_series[series])
                end_id = start_id + num_contestants - 1
                contestants_str = ", ".join(contestants_by_series[series])
                f.write(f"| {series} | {start_id}-{end_id} | {contestants_str} |\n")
                start_id = end_id + 1
    
    print(f"\nProcessed data saved to {output_dir}")
    print(f"Created README at {readme_path}")

if __name__ == "__main__":
    main() 