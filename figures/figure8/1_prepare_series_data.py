#!/usr/bin/env python3
"""
Stage 1: Series-Level Data Preparation
Load sentiment and contestant features (inputs) and IMDB ratings (targets) for Figure 8b.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
DATA_DIR = Path("../../data/raw")
OUTPUT_FILE = "series_data.csv"

def load_target_data():
    """Load IMDB histogram data and aggregate by series (our prediction targets)."""
    print("Loading IMDB histogram data (TARGETS)...")
    imdb_file = DATA_DIR / "taskmaster_histograms_corrected.csv"
    imdb_df = pd.read_csv(imdb_file)
    
    # Target columns
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    
    # Aggregate IMDB targets by series (mean across episodes in series) 
    series_targets = imdb_df.groupby('season')[target_cols].mean().reset_index()
    series_targets.rename(columns={'season': 'series'}, inplace=True)
    
    # Also include episode count per series for context
    episode_counts = imdb_df.groupby('season').size().reset_index(name='num_episodes')
    episode_counts.rename(columns={'season': 'series'}, inplace=True)
    
    targets = series_targets.merge(episode_counts, on='series')
    
    print(f"  IMDB targets: {len(targets)} series")
    print(f"  Series range: {targets['series'].min()}-{targets['series'].max()}")
    print(f"  Target variables: {target_cols}")
    
    return targets

def load_sentiment_features():
    """Load and aggregate sentiment features by series."""
    print("Loading sentiment data (INPUT FEATURES)...")
    
    sentiment_file = DATA_DIR / "sentiment.csv"
    if not sentiment_file.exists():
        print(f"  ❌ Sentiment file not found: {sentiment_file}")
        return pd.DataFrame()
    
    sentiment_df = pd.read_csv(sentiment_file)
    
    # Sentiment feature columns
    sentiment_cols = [
        'avg_anger', 'avg_awkwardness', 'avg_frustration_or_despair',
        'avg_humor', 'avg_joy_or_excitement', 'avg_sarcasm', 'avg_self_deprecation'
    ]
    
    # Check which features exist
    available_features = [col for col in sentiment_cols if col in sentiment_df.columns]
    
    if not available_features:
        print("  ❌ No sentiment features found")
        return pd.DataFrame()
    
    # Aggregate sentiment features by series (mean and std across episodes)
    series_sentiment = sentiment_df.groupby('series')[available_features].agg(['mean', 'std']).reset_index()
    
    # Flatten column names
    new_columns = ['series']
    for col in available_features:
        new_columns.extend([f'{col}_mean', f'{col}_std'])
    series_sentiment.columns = new_columns
    
    # Handle NaN standard deviations (single episode series)
    for col in series_sentiment.columns:
        if col.endswith('_std'):
            series_sentiment[col] = series_sentiment[col].fillna(0)
    
    print(f"  Sentiment features: {len(series_sentiment)} series")
    print(f"  Feature columns: {[col for col in series_sentiment.columns if col != 'series']}")
    
    return series_sentiment

def load_contestant_features():
    """Load contestant features by series."""
    print("Loading contestant data (INPUT FEATURES)...")
    
    contestants_file = DATA_DIR / "contestants.csv"
    if not contestants_file.exists():
        print(f"  ❌ Contestants file not found: {contestants_file}")
        return pd.DataFrame()
    
    contestants_df = pd.read_csv(contestants_file)
    print(f"  Contestants: {len(contestants_df)} contestants across {contestants_df['series'].nunique()} series")
    
    # Create series-level contestant features
    series_features = []
    
    for series_num in contestants_df['series'].unique():
        if pd.isna(series_num):
            continue
            
        series_contestants = contestants_df[contestants_df['series'] == series_num]
        
        # Demographics
        avg_age = series_contestants['age_during_taskmaster'].mean()
        std_age = series_contestants['age_during_taskmaster'].std()
        min_age = series_contestants['age_during_taskmaster'].min()
        max_age = series_contestants['age_during_taskmaster'].max()
        age_range = max_age - min_age if not pd.isna(max_age) and not pd.isna(min_age) else 0
        
        # Gender composition
        prop_female = (series_contestants['gender'] == 'Female').mean()
        prop_male = (series_contestants['gender'] == 'Male').mean()
        gender_diversity = series_contestants['gender'].nunique()
        
        # Nationality diversity
        nationality_diversity = series_contestants['nationality'].nunique()
        prop_british = series_contestants['nationality'].str.contains('British|English|Scottish|Welsh|Irish', na=False).mean()
        
        # Professional background
        prop_comedians = series_contestants['occupation'].str.contains('Comedian', na=False).mean()
        prop_actors = series_contestants['occupation'].str.contains('Actor|Actress', na=False).mean()
        prop_presenters = series_contestants['occupation'].str.contains('Presenter', na=False).mean()
        prop_writers = series_contestants['occupation'].str.contains('Writer', na=False).mean()
        occupation_diversity = series_contestants['occupation'].nunique()
        
        # Years of experience (handle missing values)
        years_active_values = []
        for years_str in series_contestants['years_active']:
            if pd.notna(years_str) and str(years_str) != 'Unknown':
                try:
                    # Extract numeric years from various formats
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
        std_experience = np.std(years_active_values) if len(years_active_values) > 1 else 0
        min_experience = min(years_active_values) if years_active_values else np.nan
        max_experience = max(years_active_values) if years_active_values else np.nan
        
        # Create series feature row
        series_features.append({
            'series': int(series_num),
            'num_contestants': len(series_contestants),
            
            # Age features
            'contestant_avg_age': avg_age,
            'contestant_std_age': std_age if not pd.isna(std_age) else 0,
            'contestant_age_range': age_range,
            
            # Gender features  
            'contestant_prop_female': prop_female,
            'contestant_prop_male': prop_male,
            'contestant_gender_diversity': gender_diversity,
            
            # Nationality features
            'contestant_nationality_diversity': nationality_diversity,
            'contestant_prop_british': prop_british,
            
            # Occupation features
            'contestant_prop_comedians': prop_comedians,
            'contestant_prop_actors': prop_actors,
            'contestant_prop_presenters': prop_presenters,
            'contestant_prop_writers': prop_writers,
            'contestant_occupation_diversity': occupation_diversity,
            
            # Experience features
            'contestant_avg_experience': avg_experience,
            'contestant_std_experience': std_experience,
            'contestant_experience_range': max_experience - min_experience if not pd.isna(max_experience) and not pd.isna(min_experience) else 0,
        })
    
    contestant_features_df = pd.DataFrame(series_features)
    
    print(f"  Contestant features: {len(contestant_features_df)} series")
    print(f"  Feature columns: {[col for col in contestant_features_df.columns if col not in ['series', 'num_contestants']]}")
    
    return contestant_features_df

def load_production_features():
    """Create basic production/temporal features."""
    print("Creating production features (INPUT FEATURES)...")
    
    # Create temporal and ordering features
    production_features = []
    
    for series_num in range(1, 19):  # Assuming 18 series
        production_features.append({
            'series': series_num,
            'series_order': series_num,  # Original air order
            'is_early_series': 1 if series_num <= 5 else 0,  # Early vs later series  
            'is_middle_series': 1 if 6 <= series_num <= 12 else 0,
            'is_recent_series': 1 if series_num >= 13 else 0,  # Recent series
            'series_squared': series_num ** 2,  # Non-linear time trend
        })
    
    production_df = pd.DataFrame(production_features)
    
    print(f"  Production features: {len(production_df)} series")
    print(f"  Feature columns: {[col for col in production_df.columns if col != 'series']}")
    
    return production_df

def merge_series_data():
    """Merge all series-level data sources."""
    print("\n" + "="*50)
    print("SERIES-LEVEL DATA PREPARATION")
    print("="*50)
    
    # Load all data sources
    targets = load_target_data()
    sentiment_features = load_sentiment_features()
    contestant_features = load_contestant_features()
    production_features = load_production_features()
    
    if len(targets) == 0:
        raise ValueError("No target data available!")
    
    print(f"\nMerging datasets...")
    print(f"  Starting with {len(targets)} series from IMDB data")
    
    # Start with targets
    merged = targets.copy()
    
    # Add sentiment features
    if len(sentiment_features) > 0:
        merged = merged.merge(sentiment_features, on='series', how='left')
        print(f"  After sentiment merge: {len(merged)} series")
    else:
        print("  ⚠️  No sentiment features available")
    
    # Add contestant features
    if len(contestant_features) > 0:
        merged = merged.merge(contestant_features, on='series', how='left')
        print(f"  After contestant merge: {len(merged)} series")
    else:
        print("  ⚠️  No contestant features available")
    
    # Add production features
    if len(production_features) > 0:
        merged = merged.merge(production_features, on='series', how='left')
        print(f"  After production merge: {len(merged)} series")
    else:
        print("  ⚠️  No production features available")
    
    # Handle missing values
    print(f"\nHandling missing values...")
    
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    feature_cols = [col for col in merged.columns 
                   if col not in target_cols + ['series', 'num_episodes', 'num_contestants']]
    
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
                merged.loc[merged[col].isnull(), col] = median_value
                print(f"    Filled {col} with median: {median_value:.3f}")
    else:
        print("  No missing values in features!")
    
    # Final check - ensure we have some features
    available_features = [col for col in feature_cols if col in merged.columns]
    
    if len(available_features) == 0:
        raise ValueError("No input features available for modeling!")
    
    # Final data quality check
    print(f"\nFinal dataset:")
    print(f"  Series: {len(merged)}")
    print(f"  Input features: {len(available_features)}")
    print(f"  Target variables: {len(target_cols)}")
    print(f"  Missing values: {merged.isnull().sum().sum()}")
    
    print(f"\nInput features:")
    for i, feat in enumerate(available_features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nTarget variables:")
    for i, target in enumerate(target_cols, 1):
        print(f"  {i:2d}. {target}")
    
    print(f"\n📊 Sample size warning:")
    print(f"  N = {len(merged)} is very small for machine learning")
    print(f"  Will require simple models and careful cross-validation")
    
    return merged

def save_series_data():
    """Prepare and save series-level dataset."""
    
    # Prepare data
    series_data = merge_series_data()
    
    # Save to CSV
    print(f"\nSaving to {OUTPUT_FILE}...")
    series_data.to_csv(OUTPUT_FILE, index=False)
    
    print(f"✅ Series-level data saved!")
    print(f"   File: {OUTPUT_FILE}")
    print(f"   Shape: {series_data.shape}")
    
    # Data summary
    target_cols = [f'hist{i}_pct' for i in range(1, 11)]
    feature_cols = [col for col in series_data.columns 
                   if col not in target_cols + ['series', 'num_episodes', 'num_contestants']]
    
    print(f"\nDataset summary:")
    print(f"  Total columns: {len(series_data.columns)}")
    print(f"  Input features: {len(feature_cols)}")
    print(f"  Target variables: {len(target_cols)}")
    print(f"  Metadata columns: 3 (series, num_episodes, num_contestants)")
    
    return series_data

if __name__ == "__main__":
    series_data = save_series_data() 