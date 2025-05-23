import os
import csv
import pandas as pd
import numpy as np

def load_series_data(filepath):
    """Load contestant data from a CSV file."""
    df = pd.read_csv(filepath)
    return df

def calculate_cumulative_scores_and_ranks(df):
    """Calculate cumulative scores and ranks for each contestant."""
    num_contestants = len(df)
    score_cols = [col for col in df.columns if col.startswith('Score_Task_')]
    
    # Initialize results
    all_cumulative_scores = []
    all_ranks = []
    max_total_scores = []
    
    # Process each contestant
    for i in range(num_contestants):
        # Get scores for this contestant
        scores = [df.iloc[i][col] for col in score_cols]
        # Remove NaN values
        scores = [s for s in scores if not pd.isna(s)]
        
        # Calculate cumulative scores
        cumulative_scores = [sum(scores[:j+1]) for j in range(len(scores))]
        all_cumulative_scores.append(cumulative_scores)
        
    # Calculate ranks for each task
    for t in range(max(len(cs) for cs in all_cumulative_scores)):
        # Get scores at this task for all contestants who have a score
        task_scores = [cs[t] if t < len(cs) else None for cs in all_cumulative_scores]
        
        # Rank contestants (higher score = better rank = lower number)
        valid_indices = [i for i, score in enumerate(task_scores) if score is not None]
        valid_scores = [task_scores[i] for i in valid_indices]
        
        # Convert to numpy array for easier ranking
        valid_scores_array = np.array(valid_scores)
        # argsort of argsort gives the rank (1-based)
        ranks = np.argsort(-valid_scores_array).argsort() + 1
        
        # Assign ranks to contestants
        for idx, rank_idx in enumerate(valid_indices):
            if len(all_ranks) <= rank_idx:
                all_ranks.append([])
            if len(all_ranks[rank_idx]) <= t:
                all_ranks[rank_idx].extend([None] * (t - len(all_ranks[rank_idx]) + 1))
            all_ranks[rank_idx][t] = ranks[idx]
    
    # Calculate max total score for each contestant
    for cs in all_cumulative_scores:
        max_total_scores.append(cs[-1] if cs else 0)
    
    return all_cumulative_scores, all_ranks, max_total_scores

def extract_features(CS, R, max_total_score):
    """Extract features from contestant performance data."""
    features = {}
    
    # Basic parameters
    T = len(CS)
    
    # Early and Late segments
    early_end = max(1, int(0.2 * T))
    late_start = max(early_end, int(0.8 * T))
    
    # Calculate first derivatives
    d_CS = [CS[i+1] - CS[i] for i in range(T - 1)]
    d_R = [R[i+1] - R[i] for i in range(T - 1)]
    
    # --- Feature 1: Early average score (first 20%) ---
    early_avg_score = sum(CS[:early_end]) / early_end if early_end > 0 else 0
    features['early_avg_score'] = early_avg_score
    
    # --- Feature 2: Late average score (last 20%) ---
    late_avg_score = sum(CS[late_start:]) / (T - late_start) if T > late_start else 0
    features['late_avg_score'] = late_avg_score
    
    # --- Feature 3: Score growth rate (linear regression slope) ---
    x_vals = list(range(T))
    x_mean = sum(x_vals) / T
    y_mean = sum(CS) / T
    numerator = sum((x_vals[i] - x_mean) * (CS[i] - y_mean) for i in range(T))
    denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(T))
    slope = numerator / denominator if denominator != 0 else 0
    features['score_growth_rate'] = slope
    
    # --- Feature 4: Score variance ---
    score_variance = sum((CS[i] - y_mean) ** 2 for i in range(T)) / T
    features['score_variance'] = score_variance
    
    # --- Feature 5: Average rank ---
    avg_rank = sum(R) / len(R)
    features['avg_rank'] = avg_rank
    
    # --- Feature 6: First rank ---
    first_rank = R[0]
    features['first_rank'] = first_rank
    
    # --- Feature 7: Last rank ---
    last_rank = R[-1]
    features['last_rank'] = last_rank
    
    # --- Feature 8: Rank variance ---
    rank_mean = sum(R) / len(R)
    rank_variance = sum((r - rank_mean) ** 2 for r in R) / len(R)
    features['rank_variance'] = rank_variance
    
    # --- Feature 9: Score acceleration (average of second derivatives) ---
    if len(d_CS) > 1:
        dd_CS = [d_CS[i+1] - d_CS[i] for i in range(len(d_CS) - 1)]
        score_accel = sum(dd_CS) / len(dd_CS) if dd_CS else 0
    else:
        score_accel = 0
    features['score_acceleration'] = score_accel
    
    # --- Feature 10: Rank changes (number of sign changes in rank derivative) ---
    sign_changes = 0
    for i in range(len(d_R) - 1):
        if d_R[i] * d_R[i+1] < 0:
            sign_changes += 1
    features['rank_changes'] = sign_changes
    
    # --- Feature 11: Early score ratio (early score / total score) ---
    early_score_ratio = early_avg_score * early_end / max_total_score if max_total_score > 0 else 0
    features['early_score_ratio'] = early_score_ratio
    
    # --- Feature 12: Late score ratio (late score / total score) ---
    late_score = sum(CS[late_start:])
    late_score_ratio = late_score / max_total_score if max_total_score > 0 else 0
    features['late_score_ratio'] = late_score_ratio
    
    # --- Feature 13: Average task score ---
    avg_task_score = CS[-1] / T if T > 0 else 0
    features['avg_task_score'] = avg_task_score
    
    # --- Feature 14: Score consistency (coefficient of variation of task scores) ---
    # Get individual task scores (not cumulative)
    if T > 1:
        task_scores = [CS[0]] + [CS[i] - CS[i-1] for i in range(1, T)]
        task_score_mean = sum(task_scores) / T
        task_score_std = (sum((s - task_score_mean) ** 2 for s in task_scores) / T) ** 0.5
        score_consistency = task_score_std / task_score_mean if task_score_mean > 0 else 0
    else:
        score_consistency = 0
    features['score_consistency'] = score_consistency
    
    # --- Feature 15: Comeback factor (difference between worst and final rank) ---
    if len(R) > 1:
        worst_rank = max(R)
        comeback_factor = worst_rank - last_rank
    else:
        comeback_factor = 0
    features['comeback_factor'] = comeback_factor

    return features

def extract_all_contestant_features():
    """Extract features for all contestants and save to CSV."""
    # Use absolute path from project root
    data_dir = os.path.join(os.getcwd(), '../../data/processed/scores_by_series')
    all_contestants = []
    
    # Process each series
    for i in range(1, 19):
        file_path = os.path.join(data_dir, f'series_{i}_scores.csv')
        print(f"Loading {file_path}")
        
        df = load_series_data(file_path)
        
        # Calculate cumulative scores and ranks
        cumulative_scores, ranks, max_total_scores = calculate_cumulative_scores_and_ranks(df)
        
        # Store contestant data
        for j in range(len(df)):
            contestant_id = df.iloc[j]['ContestantID']
            contestant_name = df.iloc[j]['ContestantName']
            
            # Extract features
            features = extract_features(
                cumulative_scores[j], 
                ranks[j], 
                max_total_scores[j]
            )
            
            # Create row with contestant info and features
            contestant_row = {
                'ContestantName': contestant_name,
                'ContestantID': contestant_id,
                'Series': i
            }
            contestant_row.update(features)
            
            # Special case for Series 1: Josh Widdicombe (ID 2) won, not Frank Skinner (ID 1)
            if i == 1 and contestant_id == 2:  # Josh Widdicombe
                contestant_row['last_rank'] = 1
            elif i == 1 and contestant_id == 1:  # Frank Skinner
                contestant_row['last_rank'] = 2
            
            all_contestants.append(contestant_row)
    
    # Save features to CSV in the figure5 directory
    output_file = 'contestant_features.csv'
    print(f"Saving features to {output_file}")
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        # Get all field names
        fieldnames = ['ContestantName', 'ContestantID', 'Series'] + list(all_contestants[0].keys() - {'ContestantName', 'ContestantID', 'Series'})
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_contestants)
    
    print(f"Saved features for {len(all_contestants)} contestants with {len(fieldnames) - 3} features each")
    return output_file

if __name__ == "__main__":
    extract_all_contestant_features() 