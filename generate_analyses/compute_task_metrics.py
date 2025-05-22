import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import entropy, wasserstein_distance, chisquare
import os
import warnings

# Ignore warnings for better output readability
warnings.filterwarnings('ignore')

def compute_task_metrics(scores):
    """
    Compute metrics for a single task given a list of scores.
    
    Args:
        scores (list): List of scores for a task (typically 5 values)
        
    Returns:
        dict: Dictionary of computed metrics
    """
    scores = np.array(scores)
    scores_sorted = np.sort(scores)
    
    # Perfect permutation for comparison
    perfect = np.arange(1, len(scores) + 1)
    
    # Basic descriptive stats
    metrics = {
        'score_min': np.min(scores),
        'score_max': np.max(scores),
        'score_mean': np.mean(scores),
        'score_median': np.median(scores),
        'score_std': np.std(scores, ddof=1),
        'score_range': np.max(scores) - np.min(scores)
    }
    
    # Distributional deviation
    metrics['l1_distance_from_perfect'] = np.sum(np.abs(scores_sorted - perfect))
    metrics['wasserstein_distance_from_perfect'] = wasserstein_distance(scores_sorted, perfect)
    metrics['is_permutation'] = np.array_equal(np.sort(scores), perfect)
    metrics['num_unique_scores'] = len(np.unique(scores))
    
    # Entropy and tie patterns
    # Create a histogram of scores
    bins = np.bincount(scores.astype(int))[1:] if scores.astype(int).min() >= 0 else None
    
    if bins is not None and len(bins) > 0:
        # Normalize the bins for entropy calculation (PMF)
        pmf = bins / len(scores)
        metrics['tie_entropy'] = entropy(pmf)
    else:
        metrics['tie_entropy'] = 0.0
    
    # Count ties
    unique, counts = np.unique(scores, return_counts=True)
    ties = np.sum(counts[counts > 1] - 1)  # Sum all instances beyond the first occurrence
    metrics['num_ties'] = ties
    
    # Create tie pattern string
    tie_pattern = []
    for val, count in zip(unique, counts):
        tie_pattern.extend([str(int(val))] * count)
    metrics['tie_pattern'] = "-".join(tie_pattern)
    
    # Chi-square test for uniformity
    try:
        # Use chi-square to test if the distribution differs from uniform
        # We need bins for each possible score value
        max_score = int(max(scores))
        hist = np.zeros(max_score)
        for s in scores:
            hist[int(s)-1] += 1
            
        # Expected frequency for uniform distribution
        expected = np.ones(max_score) * len(scores) / max_score
        
        # Calculate chi-square only if there are enough data points
        if len(scores) >= 5 and max_score > 1:
            chistat, p_value = chisquare(hist, expected)
            metrics['chisq_uniform_pval'] = p_value
        else:
            metrics['chisq_uniform_pval'] = np.nan
    except:
        metrics['chisq_uniform_pval'] = np.nan
        
    return metrics

def main():
    """
    Main function to process scores data and compute task-level metrics.
    """
    print("Starting task metric computation...")
    
    # Check if processed directory exists
    if not os.path.exists('data/processed/task'):
        os.makedirs('data/processed/task')
    
    # Load the raw scores data
    try:
        scores_df = pd.read_csv('data/scores.csv')
        print(f"Loaded raw scores data: {scores_df.shape[0]} rows, {scores_df.shape[1]} columns")
    except FileNotFoundError:
        print("Error: Raw scores data file 'data/scores.csv' not found!")
        return
    
    # Display the first few rows of the data to understand structure
    print("\nSample data (first 5 rows):")
    print(scores_df.head())
    
    # Map the actual column names to our expected names
    column_mapping = {
        'contestant_name': 'contestant',
        'total_score': 'score'
    }
    
    # Create new columns with the expected names
    for actual_col, expected_col in column_mapping.items():
        if actual_col in scores_df.columns:
            scores_df[expected_col] = scores_df[actual_col]
    
    # Check if expected columns exist after mapping
    required_columns = ['series', 'episode', 'task_id', 'contestant', 'score']
    missing_columns = [col for col in required_columns if col not in scores_df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {scores_df.columns.tolist()}")
        return
    
    # Create a DataFrame to store task-level metrics
    print("\nComputing task-level metrics...")
    
    # Group by series, episode, task_id to identify unique tasks
    task_groups = scores_df.groupby(['series', 'episode', 'task_id'])
    
    # Prepare to store task metrics
    all_task_metrics = []
    
    # Process each task
    for (series, episode, task_id), group in task_groups:
        # Extract scores for this task
        task_scores = group['score'].values
        
        # Skip if no scores available
        if len(task_scores) == 0:
            continue
            
        # Compute metrics
        metrics = compute_task_metrics(task_scores)
        
        # Add task identifiers
        metrics['series'] = series
        metrics['episode'] = episode
        metrics['task_id'] = task_id
        metrics['num_contestants'] = len(task_scores)
        
        # Add task title if available
        if 'task_title' in group.columns:
            metrics['task_title'] = group['task_title'].iloc[0]
        
        # Add to collection
        all_task_metrics.append(metrics)
    
    # Convert to DataFrame
    task_metrics_df = pd.DataFrame(all_task_metrics)
    
    # Reorder columns to put identifiers first
    id_cols = ['series', 'episode', 'task_id', 'num_contestants']
    if 'task_title' in task_metrics_df.columns:
        id_cols.append('task_title')
    reordered_cols = id_cols + [col for col in task_metrics_df.columns if col not in id_cols]
    task_metrics_df = task_metrics_df[reordered_cols]
    
    # Save to CSV
    output_path = 'data/processed/task/scores_metrics.csv'
    task_metrics_df.to_csv(output_path, index=False)
    
    print(f"\nProcessed {len(task_metrics_df)} tasks.")
    print(f"Task metrics saved to: {output_path}")
    
    # Display a summary of the computed metrics
    print("\nMetrics summary:")
    numeric_cols = task_metrics_df.select_dtypes(include=['number']).columns
    summary = task_metrics_df[numeric_cols].describe()
    print(summary)

if __name__ == "__main__":
    main() 