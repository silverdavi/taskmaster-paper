#!/usr/bin/env python3
"""
Stage 1: Episode-Level Data Preparation
Load sentiment and task features (inputs) and IMDB ratings (targets) for Figure 8a.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("../../data/raw")
OUTPUT_FILE = "episode_data.csv"

def load_target_data():
    """Load IMDB histogram data (our prediction targets)."""
    print("Loading IMDB histogram data (TARGETS)...")
    imdb_file = DATA_DIR / "taskmaster_histograms_corrected.csv"
    imdb_df = pd.read_csv(imdb_file)
    
    # Create episode identifier
    imdb_df['episode_id'] = imdb_df['season'].astype(str) + '_' + imdb_df['episode'].astype(str)
    
    # Keep only identifiers and targets
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    metadata_cols = ['season', 'episode', 'episode_id', 'title', 'imdb_id']
    
    targets = imdb_df[metadata_cols + target_cols].copy()
    
    print(f"  IMDB targets: {len(targets)} episodes")
    print(f"  Seasons: {targets['season'].min()}-{targets['season'].max()}")
    print(f"  Target variables: {target_cols}")
    
    return targets

def load_sentiment_features():
    """Load sentiment analysis data (input features)."""
    print("Loading sentiment data (INPUT FEATURES)...")
    sentiment_file = DATA_DIR / "sentiment.csv"
    
    if not sentiment_file.exists():
        print(f"  ❌ Sentiment file not found: {sentiment_file}")
        return pd.DataFrame()
    
    sentiment_df = pd.read_csv(sentiment_file)
    
    # Create episode identifier
    sentiment_df['episode_id'] = sentiment_df['series'].astype(str) + '_' + sentiment_df['episode'].astype(str)
    
    # Keep only sentiment features
    sentiment_features = [
        'avg_anger', 'avg_awkwardness', 'avg_frustration_or_despair',
        'avg_humor', 'avg_joy_or_excitement', 'avg_sarcasm', 'avg_self_deprecation'
    ]
    
    # Check which features exist
    available_features = [col for col in sentiment_features if col in sentiment_df.columns]
    missing_features = [col for col in sentiment_features if col not in sentiment_df.columns]
    
    if missing_features:
        print(f"  ⚠️  Missing sentiment features: {missing_features}")
    
    features = sentiment_df[['episode_id'] + available_features].copy()
    
    print(f"  Sentiment features: {len(features)} episodes")
    print(f"  Available features: {available_features}")
    
    return features

def load_task_features():
    """Load task-level data (input features)."""
    print("Loading task data (INPUT FEATURES)...")
    
    # Try different possible task files
    possible_files = [
        "taskmaster_UK_tasks.csv",
        "tasks.csv", 
        "long_task_scores.csv",
        "scores.csv"
    ]
    
    task_df = None
    for filename in possible_files:
        task_file = DATA_DIR / filename
        if task_file.exists():
            print(f"  Found: {filename}")
            try:
                task_df = pd.read_csv(task_file)
                print(f"  Task data: {len(task_df)} rows")
                print(f"  Columns: {list(task_df.columns)}")
                break
            except Exception as e:
                print(f"  Error reading {filename}: {e}")
                continue
    
    if task_df is None:
        print("  No task data files found - will skip task features")
        return pd.DataFrame()
    
    # Create episode identifier
    if 'series' in task_df.columns and 'episode' in task_df.columns:
        task_df['episode_id'] = task_df['series'].astype(str) + '_' + task_df['episode'].astype(str)
    elif 'season' in task_df.columns and 'episode' in task_df.columns:
        task_df['episode_id'] = task_df['season'].astype(str) + '_' + task_df['episode'].astype(str)
    else:
        print("  ❌ Cannot create episode_id from task data - missing series/episode columns")
        return pd.DataFrame()
    
    # Aggregate task features by episode
    print("  Aggregating task features by episode...")
    
    episode_features = []
    
    for episode_id in task_df['episode_id'].unique():
        episode_tasks = task_df[task_df['episode_id'] == episode_id]
        
        features = {'episode_id': episode_id}
        
        # Basic task counts
        features['num_tasks'] = len(episode_tasks)
        
        # Task type diversity (if available)
        if 'task_type' in task_df.columns:
            features['task_type_diversity'] = episode_tasks['task_type'].nunique()
            
        # Score-based features (look for score columns)
        score_cols = [col for col in task_df.columns if 'score' in col.lower() or 'point' in col.lower()]
        
        for score_col in score_cols:
            if episode_tasks[score_col].notna().sum() > 0:
                features[f'avg_{score_col}'] = episode_tasks[score_col].mean()
                features[f'std_{score_col}'] = episode_tasks[score_col].std()
                features[f'min_{score_col}'] = episode_tasks[score_col].min()
                features[f'max_{score_col}'] = episode_tasks[score_col].max()
                features[f'range_{score_col}'] = episode_tasks[score_col].max() - episode_tasks[score_col].min()
        
        episode_features.append(features)
    
    task_features_df = pd.DataFrame(episode_features)
    
    print(f"  Task features: {len(task_features_df)} episodes")
    print(f"  Feature columns: {[col for col in task_features_df.columns if col != 'episode_id']}")
    
    return task_features_df

def load_contestant_features():
    """Load contestant data and create episode-level features."""
    print("Loading contestant data (INPUT FEATURES)...")
    
    contestants_file = DATA_DIR / "contestants.csv"
    if not contestants_file.exists():
        print(f"  ❌ Contestants file not found: {contestants_file}")
        return pd.DataFrame()
    
    contestants_df = pd.read_csv(contestants_file)
    print(f"  Contestants: {len(contestants_df)} contestants across {contestants_df['series'].nunique()} series")
    
    # Create series-level contestant features that apply to all episodes in that series
    series_features = []
    
    for series_num in contestants_df['series'].unique():
        if pd.isna(series_num):
            continue
            
        series_contestants = contestants_df[contestants_df['series'] == series_num]
        
        # Demographics
        avg_age = series_contestants['age_during_taskmaster'].mean()
        prop_female = (series_contestants['gender'] == 'Female').mean()
        prop_comedians = series_contestants['occupation'].str.contains('Comedian', na=False).mean()
        nationality_diversity = series_contestants['nationality'].nunique()
        
        # Experience (simplified)
        years_active_values = []
        for years_str in series_contestants['years_active']:
            if pd.notna(years_str) and str(years_str) != 'Unknown':
                try:
                    if 'Since' in str(years_str):
                        start_year = int(str(years_str).split('Since ')[-1])
                        years = 2024 - start_year
                    elif '-' in str(years_str):
                        parts = str(years_str).split('-')
                        start_year = int(parts[0])
                        years = 2024 - start_year
                    else:
                        years = float(str(years_str))
                    years_active_values.append(years)
                except:
                    continue
        
        avg_experience = np.mean(years_active_values) if years_active_values else np.nan
        
        series_features.append({
            'series': int(series_num),
            'contestant_avg_age': avg_age,
            'contestant_prop_female': prop_female,
            'contestant_prop_comedians': prop_comedians,
            'contestant_nationality_diversity': nationality_diversity,
            'contestant_avg_experience': avg_experience
        })
    
    contestant_features_df = pd.DataFrame(series_features)
    
    print(f"  Contestant features: {len(contestant_features_df)} series")
    print(f"  Feature columns: {[col for col in contestant_features_df.columns if col != 'series']}")
    
    return contestant_features_df

def merge_episode_data():
    """Merge all episode-level data sources."""
    print("\n" + "="*50)
    print("EPISODE-LEVEL DATA PREPARATION")
    print("="*50)
    
    # Load all data sources
    targets = load_target_data()
    sentiment_features = load_sentiment_features()
    task_features = load_task_features()
    contestant_features = load_contestant_features()
    
    if len(targets) == 0:
        raise ValueError("No target data available!")
    
    print(f"\nMerging datasets...")
    print(f"  Starting with {len(targets)} episodes from IMDB data")
    
    # Start with targets
    merged = targets.copy()
    
    # Add sentiment features
    if len(sentiment_features) > 0:
        merged = merged.merge(sentiment_features, on='episode_id', how='left')
        print(f"  After sentiment merge: {len(merged)} episodes")
    else:
        print("  ⚠️  No sentiment features available")
    
    # Add task features
    if len(task_features) > 0:
        merged = merged.merge(task_features, on='episode_id', how='left')
        print(f"  After task merge: {len(merged)} episodes")
    else:
        print("  ⚠️  No task features available")
    
    # Add contestant features (by series)
    if len(contestant_features) > 0:
        merged = merged.merge(contestant_features, left_on='season', right_on='series', how='left')
        merged.drop('series', axis=1, inplace=True)  # Remove duplicate series column
        print(f"  After contestant merge: {len(merged)} episodes")
    else:
        print("  ⚠️  No contestant features available")
    
    # Handle missing values
    print(f"\nHandling missing values...")
    
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    feature_cols = [col for col in merged.columns 
                   if col not in target_cols + ['season', 'episode', 'episode_id', 'title', 'imdb_id']]
    
    missing_counts = merged[feature_cols].isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    
    if len(missing_features) > 0:
        print("  Missing values in features:")
        for feature, count in missing_features.items():
            print(f"    {feature}: {count} missing ({count/len(merged)*100:.1f}%)")
        
        # Fill missing values with median for numeric features
        for col in feature_cols:
            if merged[col].dtype in ['float64', 'int64'] and merged[col].isnull().sum() > 0:
                median_value = merged[col].median()
                merged[col].fillna(median_value, inplace=True)
                print(f"    Filled {col} with median: {median_value:.3f}")
    else:
        print("  No missing values in features!")
    
    # Final check - ensure we have some features
    available_features = [col for col in feature_cols if col in merged.columns]
    
    if len(available_features) == 0:
        raise ValueError("No input features available for modeling!")
    
    # Final data quality check
    print(f"\nFinal dataset:")
    print(f"  Episodes: {len(merged)}")
    print(f"  Input features: {len(available_features)}")
    print(f"  Target variables: {len(target_cols)}")
    print(f"  Missing values: {merged.isnull().sum().sum()}")
    
    print(f"\nInput features:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nTarget variables:")
    for i, target in enumerate(target_cols, 1):
        print(f"  {i:2d}. {target}")
    
    return merged

def save_episode_data():
    """Prepare and save episode-level dataset."""
    
    # Prepare data
    episode_data = merge_episode_data()
    
    # Save to CSV
    print(f"\nSaving to {OUTPUT_FILE}...")
    episode_data.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Episode-level data saved!")
    print(f"   File: {OUTPUT_FILE}")
    print(f"   Shape: {episode_data.shape}")
    
    # Data summary
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    feature_cols = [col for col in episode_data.columns 
                   if col not in target_cols + ['season', 'episode', 'episode_id', 'title', 'imdb_id']]
    
    print(f"\nDataset summary:")
    print(f"  Total columns: {len(episode_data.columns)}")
    print(f"  Input features: {len(feature_cols)}")
    print(f"  Target variables: {len(target_cols)}")
    print(f"  Metadata columns: {len(episode_data.columns) - len(feature_cols) - len(target_cols)}")
    
    return episode_data

if __name__ == "__main__":
    episode_data = save_episode_data() 