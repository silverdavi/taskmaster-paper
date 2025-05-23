import pandas as pd
import numpy as np
from collections import Counter

def perfect_archetype_assignment(input_csv='contestant_features.csv', output_csv='final_archetypes.csv'):
    """
    Assigns exactly one contestant per series to each of 5 archetypes,
    based on scoring functions that evaluate how well each contestant matches each archetype.
    """
    # Load contestant data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} contestants from {input_csv}")
    
    # Define our 5 archetypes
    archetypes = [
        'Steady Performer',  # Consistent high-rank with low variance
        'Late Bloomer',      # Improves throughout the series
        'Early Star',        # Starts strong but declines
        'Chaotic Wildcard',  # Unpredictable performance with high variance
        'Consistent Middle'  # Stays in the middle of the pack
    ]
    
    # Key features that define archetypes
    features = [
        'early_score_ratio',    # Early scores relative to total
        'late_score_ratio',     # Late scores relative to total
        'rank_variance',        # How much ranks vary over time
        'avg_rank',             # Average rank (1 is best)
        'score_variance',       # How much scores vary
        'score_growth_rate',    # Slope of scores over time
        'rank_changes',         # Number of times rank changes direction
        'score_consistency',    # Consistency of individual task scores
        'comeback_factor'       # Difference between worst and final rank
    ]
    
    # Normalize features to make them comparable
    df_normalized = df.copy()
    df_normalized[features] = (df[features] - df[features].mean()) / df[features].std()
    
    # Define scoring functions for each archetype
    def compute_archetype_scores(row):
        """Calculate how well a contestant matches each archetype"""
        return {
            'Steady Performer': -row['rank_variance'] - row['score_variance'] - row['rank_changes'] - row['avg_rank'],
            'Late Bloomer': row['late_score_ratio'] + row['score_growth_rate'] + row['comeback_factor'],
            'Early Star': row['early_score_ratio'] - row['score_growth_rate'] - row['late_score_ratio'],
            'Chaotic Wildcard': row['rank_variance'] + row['rank_changes'] + row['score_variance'],
            'Consistent Middle': -abs(row['avg_rank'] - 3) - row['rank_variance'] - abs(row['score_growth_rate'])
        }
    
    # Assign archetypes within each series
    assignments = []
    
    for series, group in df_normalized.groupby('Series'):
        # Calculate scores for each contestant in this series
        contestants = []
        for idx, row in group.iterrows():
            scores = compute_archetype_scores(row)
            contestants.append({
                'index': idx,
                'id': df.loc[idx, 'ContestantID'],
                'name': df.loc[idx, 'ContestantName'],
                'scores': scores
            })
        
        # Greedy assignment: for each archetype, pick the best remaining contestant
        used_indices = set()
        
        for archetype in archetypes:
            # Sort contestants by their score for this archetype (descending)
            sorted_contestants = sorted(
                [c for c in contestants if c['index'] not in used_indices],
                key=lambda c: c['scores'][archetype],
                reverse=True
            )
            
            # Select the best match
            best_match = sorted_contestants[0]
            used_indices.add(best_match['index'])
            
            # Record the assignment
            assignments.append({
                'Series': series,
                'ContestantID': best_match['id'],
                'ContestantName': best_match['name'],
                'Archetype': archetype,
                'Score': best_match['scores'][archetype]
            })
    
    # Convert to DataFrame and save
    result_df = pd.DataFrame(assignments)
    result_df.to_csv(output_csv, index=False)
    
    # Verify the assignments
    verify_assignments(result_df, archetypes)
    
    return result_df

def verify_assignments(df, archetypes):
    """Verify that each series has exactly one contestant per archetype"""
    # Count by series and archetype
    counts = df.groupby(['Series', 'Archetype']).size().reset_index(name='count')
    
    # Check if any count is not 1
    incorrect = counts[counts['count'] != 1]
    
    if len(incorrect) > 0:
        print("❌ Error: Some series have incorrect archetype distribution")
        print(incorrect)
    else:
        print("✅ Perfect 1:1 mapping: Each series has exactly one contestant per archetype")
    
    # Total counts per archetype
    archetype_counts = Counter(df['Archetype'])
    for archetype in archetypes:
        print(f"{archetype}: {archetype_counts[archetype]} contestants")

if __name__ == "__main__":
    perfect_archetypes = perfect_archetype_assignment()
    
    # Print sample of results
    print("\nSample assignments (first 10):")
    print(perfect_archetypes.head(10)) 