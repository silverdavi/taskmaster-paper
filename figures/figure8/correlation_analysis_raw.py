#!/usr/bin/env python3
"""
Raw Correlation Analysis: Input Features vs Mean IMDB Score
Calculate raw correlations between actual input data and mean IMDB ratings.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Configuration
DATA_DIR = Path("../../data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_FILE = "raw_correlations.json"
PLOT_FILE = "figure8b_raw_correlations.png"

def load_imdb_targets():
    """Load IMDB histogram data and compute mean scores."""
    print("Loading IMDB target data...")
    
    imdb_file = RAW_DIR / "taskmaster_histograms_corrected.csv"
    imdb_df = pd.read_csv(imdb_file)
    
    print(f"  IMDB data: {len(imdb_df)} episodes across {imdb_df['season'].nunique()} series")
    
    # Compute weighted mean IMDB score for each episode
    imdb_scores = []
    for _, row in imdb_df.iterrows():
        weighted_sum = 0
        total_votes = 0
        for rating in range(1, 11):
            votes = row[f'hist{rating}_votes']
            weighted_sum += rating * votes
            total_votes += votes
        
        mean_score = weighted_sum / total_votes if total_votes > 0 else np.nan
        imdb_scores.append({
            'season': int(row['season']),
            'episode': int(row['episode']), 
            'mean_imdb_score': float(mean_score),
            'total_votes': int(total_votes)
        })
    
    imdb_scores_df = pd.DataFrame(imdb_scores)
    print(f"  Mean IMDB scores: {imdb_scores_df['mean_imdb_score'].min():.2f} - {imdb_scores_df['mean_imdb_score'].max():.2f}")
    
    return imdb_scores_df

def load_task_data():
    """Load task data features."""
    print("\nLoading task data...")
    
    task_file = RAW_DIR / "taskmaster_UK_tasks.csv"
    if not task_file.exists():
        print(f"  ❌ Task file not found: {task_file}")
        return pd.DataFrame()
    
    task_df = pd.read_csv(task_file)
    print(f"  Task data: {len(task_df)} tasks")
    print(f"  Columns: {list(task_df.columns)}")
    
    # Check if we can aggregate by episode
    if 'series' in task_df.columns and 'episode' in task_df.columns:
        print("  ✅ Found series/episode columns for aggregation")
        # Group by episode and create features
        episode_features = []
        for (series, episode), group in task_df.groupby(['series', 'episode']):
            features = {
                'season': int(series),
                'episode': int(episode),
                'num_tasks': int(len(group))
            }
            
            # Add any numeric columns as features
            for col in task_df.columns:
                if task_df[col].dtype in ['int64', 'float64'] and col not in ['series', 'episode']:
                    features[f'task_{col}_mean'] = float(group[col].mean())
                    features[f'task_{col}_std'] = float(group[col].std()) if len(group) > 1 else 0.0
                    features[f'task_{col}_sum'] = float(group[col].sum())
            
            episode_features.append(features)
        
        task_features_df = pd.DataFrame(episode_features)
        print(f"  Task features: {len(task_features_df)} episodes, {len(task_features_df.columns)-2} features")
        return task_features_df
    else:
        print(f"  ⚠️  Task data cannot be aggregated by episode")
        # Try to use some aggregate features across all tasks
        print(f"  Creating series-level task aggregates...")
        
        # Look for boolean/binary task features
        binary_cols = [col for col in task_df.columns if task_df[col].dtype in ['bool', 'int64'] 
                      and task_df[col].nunique() <= 2]
        
        if len(binary_cols) > 0:
            # Create series-level proportions of task types
            series_task_features = []
            
            # Group by series if possible (extract from series_name or similar)
            if 'series_name' in task_df.columns:
                for series_name, group in task_df.groupby('series_name'):
                    try:
                        # Extract series number
                        series_num = int(''.join(filter(str.isdigit, str(series_name))))
                        
                        features = {'season': series_num}
                        
                        # Calculate proportions for binary features
                        for col in binary_cols[:10]:  # Limit to top 10
                            prop = group[col].mean()
                            features[f'task_prop_{col}'] = float(prop)
                        
                        features['total_series_tasks'] = int(len(group))
                        
                        # Expand to episode level (same for all episodes in series)
                        for episode in range(1, 11):
                            episode_features = features.copy()
                            episode_features['episode'] = episode
                            series_task_features.append(episode_features)
                            
                    except:
                        continue
            
            if series_task_features:
                task_features_df = pd.DataFrame(series_task_features)
                print(f"  Task features: {len(task_features_df)} episode-series combinations")
                return task_features_df
        
        print(f"  ❌ Cannot create meaningful task features")
        return pd.DataFrame()

def load_sentiment_data():
    """Load sentiment analysis features."""
    print("\nLoading sentiment data...")
    
    sentiment_file = RAW_DIR / "sentiment.csv"
    if not sentiment_file.exists():
        print(f"  ❌ Sentiment file not found: {sentiment_file}")
        return pd.DataFrame()
    
    sentiment_df = pd.read_csv(sentiment_file)
    print(f"  Sentiment data: {len(sentiment_df)} episodes")
    print(f"  Features: {[col for col in sentiment_df.columns if col.startswith('avg_')]}")
    
    # Rename series to season for consistency
    if 'series' in sentiment_df.columns:
        sentiment_df = sentiment_df.rename(columns={'series': 'season'})
    
    # Convert to standard types
    sentiment_df['season'] = sentiment_df['season'].astype(int)
    sentiment_df['episode'] = sentiment_df['episode'].astype(int)
    
    return sentiment_df

def load_scoring_data():
    """Load contestant scoring data."""
    print("\nLoading scoring data...")
    
    scores_dir = PROCESSED_DIR / "scores_by_series"
    if not scores_dir.exists():
        print(f"  ❌ Scores directory not found: {scores_dir}")
        return pd.DataFrame()
    
    all_scores = []
    score_files = list(scores_dir.glob("series_*_scores.csv"))
    print(f"  Found {len(score_files)} score files")
    
    for score_file in score_files:
        # Extract series number from filename
        series_num = int(score_file.stem.split('_')[1])
        
        try:
            scores_df = pd.read_csv(score_file)
            print(f"    Series {series_num}: {len(scores_df)} contestants, {len(scores_df.columns)-2} tasks")
            
            # Aggregate scores by episode
            score_cols = [col for col in scores_df.columns if col.startswith('Score_Task_')]
            
            # Calculate episode-level scoring features
            for episode in range(1, 11):  # Assume max 10 episodes per series
                episode_scores = []
                for task_col in score_cols:
                    task_num = int(task_col.split('_')[-1])
                    # Rough mapping: assume ~3-4 tasks per episode
                    if (episode-1)*3 <= task_num-1 < episode*3:
                        episode_scores.extend(scores_df[task_col].dropna().tolist())
                
                if episode_scores:
                    all_scores.append({
                        'season': int(series_num),
                        'episode': int(episode),
                        'score_mean': float(np.mean(episode_scores)),
                        'score_std': float(np.std(episode_scores)),
                        'score_range': float(max(episode_scores) - min(episode_scores)),
                        'score_max': float(max(episode_scores)),
                        'score_min': float(min(episode_scores)),
                        'num_scores': int(len(episode_scores))
                    })
        
        except Exception as e:
            print(f"    Error processing {score_file}: {e}")
    
    if all_scores:
        scoring_df = pd.DataFrame(all_scores)
        print(f"  Scoring features: {len(scoring_df)} episodes")
        return scoring_df
    
    return pd.DataFrame()

def load_contestant_data():
    """Load contestant demographic features."""
    print("\nLoading contestant data...")
    
    contestants_file = RAW_DIR / "contestants.csv"
    if not contestants_file.exists():
        print(f"  ❌ Contestants file not found: {contestants_file}")
        return pd.DataFrame()
    
    contestants_df = pd.read_csv(contestants_file)
    print(f"  Contestants: {len(contestants_df)} contestants across {contestants_df['series'].nunique()} series")
    
    # Aggregate by series (applies to all episodes in that series)
    series_features = []
    for series_num in contestants_df['series'].unique():
        if pd.isna(series_num):
            continue
            
        series_contestants = contestants_df[contestants_df['series'] == series_num]
        
        # Demographics
        avg_age = series_contestants['age_during_taskmaster'].mean()
        prop_female = (series_contestants['gender'] == 'Female').mean()
        prop_male = (series_contestants['gender'] == 'Male').mean()
        
        # Professional background
        prop_comedians = series_contestants['occupation'].str.contains('Comedian', na=False).mean()
        prop_actors = series_contestants['occupation'].str.contains('Actor|Actress', na=False).mean()
        prop_presenters = series_contestants['occupation'].str.contains('Presenter', na=False).mean()
        
        # Nationality diversity
        nationality_diversity = series_contestants['nationality'].nunique()
        prop_british = series_contestants['nationality'].str.contains('British|English|Scottish|Welsh|Irish', na=False).mean()
        
        series_features.append({
            'season': int(series_num),
            'contestant_avg_age': float(avg_age),
            'contestant_prop_female': float(prop_female),
            'contestant_prop_male': float(prop_male),
            'contestant_prop_comedians': float(prop_comedians),
            'contestant_prop_actors': float(prop_actors),
            'contestant_prop_presenters': float(prop_presenters),
            'contestant_nationality_diversity': int(nationality_diversity),
            'contestant_prop_british': float(prop_british),
            'num_contestants': int(len(series_contestants))
        })
    
    contestant_features_df = pd.DataFrame(series_features)
    print(f"  Contestant features: {len(contestant_features_df)} series")
    
    # Expand to episode level (same values for all episodes in a series)
    episode_features = []
    for _, series_row in contestant_features_df.iterrows():
        for episode in range(1, 11):  # Assume max 10 episodes per series
            episode_row = series_row.copy()
            episode_row['episode'] = int(episode)
            episode_features.append(episode_row)
    
    episode_contestant_df = pd.DataFrame(episode_features)
    return episode_contestant_df

def merge_all_data(imdb_scores, task_data, sentiment_data, scoring_data, contestant_data):
    """Merge all data sources."""
    print(f"\nMerging all data sources...")
    
    # Start with IMDB scores
    merged = imdb_scores.copy()
    print(f"  Starting with {len(merged)} episodes from IMDB data")
    
    # Merge each data source
    data_sources = [
        (task_data, "task"),
        (sentiment_data, "sentiment"), 
        (scoring_data, "scoring"),
        (contestant_data, "contestant")
    ]
    
    for data, name in data_sources:
        if len(data) > 0:
            before_len = len(merged)
            merged = merged.merge(data, on=['season', 'episode'], how='left')
            print(f"  After {name} merge: {len(merged)} episodes")
            if len(merged) != before_len:
                print(f"    ⚠️  Row count changed from {before_len} to {len(merged)}")
        else:
            print(f"  Skipping {name} - no data available")
    
    print(f"\nFinal merged dataset: {merged.shape}")
    return merged

def calculate_raw_correlations(merged_data):
    """Calculate raw correlation coefficients."""
    print(f"\nCalculating raw correlations...")
    
    # Identify feature columns (exclude metadata and target)
    exclude_cols = ['season', 'episode', 'mean_imdb_score', 'total_votes']
    feature_cols = [col for col in merged_data.columns 
                   if col not in exclude_cols and merged_data[col].dtype in ['int64', 'float64']]
    
    print(f"  Features to analyze: {len(feature_cols)}")
    
    correlations = {}
    valid_correlations = []
    
    target_values = merged_data['mean_imdb_score'].values
    
    for feature in feature_cols:
        feature_values = merged_data[feature].values
        
        # Remove rows where either feature or target is NaN
        valid_mask = ~(np.isnan(feature_values) | np.isnan(target_values))
        
        if valid_mask.sum() < 3:  # Need at least 3 valid points
            print(f"    Skipping {feature}: insufficient valid data ({valid_mask.sum()} points)")
            continue
        
        # Check for zero variance
        if np.std(feature_values[valid_mask]) < 1e-10:
            print(f"    Skipping {feature}: no variance")
            continue
            
        try:
            corr_coef = np.corrcoef(feature_values[valid_mask], target_values[valid_mask])[0, 1]
            
            if not np.isnan(corr_coef):
                correlations[feature] = {
                    'correlation': float(corr_coef),
                    'valid_points': int(valid_mask.sum()),
                    'total_points': int(len(feature_values))
                }
                valid_correlations.append(corr_coef)
            else:
                print(f"    Skipping {feature}: correlation is NaN")
                
        except Exception as e:
            print(f"    Error calculating correlation for {feature}: {e}")
    
    print(f"  Valid correlations computed: {len(valid_correlations)}")
    if valid_correlations:
        print(f"  Correlation range: {min(valid_correlations):.3f} to {max(valid_correlations):.3f}")
    
    return correlations, valid_correlations

def plot_correlation_histogram(correlations, valid_correlations):
    """Create histogram of raw correlation coefficients."""
    print(f"\nCreating correlation histogram...")
    
    # Single subplot for histogram only
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Main histogram
    n_bins = 25
    counts, bins, patches = ax.hist(valid_correlations, bins=n_bins, 
                                   alpha=0.7, color='steelblue', 
                                   edgecolor='black', linewidth=0.5,
                                   density=True, label='Observed correlations')
    
    # Add Gaussian fit
    mu, sigma = np.mean(valid_correlations), np.std(valid_correlations)
    x = np.linspace(min(valid_correlations), max(valid_correlations), 100)
    gaussian_fit = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, gaussian_fit, 'r-', linewidth=2, label=f'Gaussian fit (μ={mu:.3f}, σ={sigma:.3f})')
    
    # Add vertical line at zero
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No correlation (r=0)')
    
    # Add mean line
    ax.axvline(mu, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'Mean = {mu:.3f}')
    
    ax.set_xlabel('Raw Correlation Coefficient (r)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'N correlations = {len(valid_correlations)}\n'
    stats_text += f'Mean r = {np.mean(valid_correlations):.3f}\n'
    stats_text += f'Std r = {np.std(valid_correlations):.3f}\n'
    stats_text += f'Range = [{min(valid_correlations):.3f}, {max(valid_correlations):.3f}]'
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    return fig

def main():
    """Main analysis function."""
    print("="*60)
    print("RAW CORRELATION ANALYSIS: INPUT FEATURES vs MEAN IMDB SCORE")
    print("="*60)
    
    # Load all data sources
    imdb_scores = load_imdb_targets()
    task_data = load_task_data()
    sentiment_data = load_sentiment_data()
    scoring_data = load_scoring_data()
    contestant_data = load_contestant_data()
    
    # Merge all data
    merged_data = merge_all_data(imdb_scores, task_data, sentiment_data, scoring_data, contestant_data)
    
    # Calculate correlations
    correlations, valid_correlations = calculate_raw_correlations(merged_data)
    
    if len(valid_correlations) == 0:
        print("❌ No valid correlations computed!")
        return None, None
    
    # Create plots
    fig = plot_correlation_histogram(correlations, valid_correlations)
    
    # Save results
    results = {
        'correlations': correlations,
        'summary': {
            'total_features': len(correlations),
            'mean_correlation': float(np.mean(valid_correlations)),
            'std_correlation': float(np.std(valid_correlations)),
            'min_correlation': float(min(valid_correlations)),
            'max_correlation': float(max(valid_correlations)),
            'positive_correlations': int(sum(1 for c in valid_correlations if c > 0)),
            'negative_correlations': int(sum(1 for c in valid_correlations if c < 0)),
            'strong_positive': int(sum(1 for c in valid_correlations if c >= 0.5)),
            'strong_negative': int(sum(1 for c in valid_correlations if c <= -0.5))
        },
        'data_sources': {
            'imdb_episodes': len(imdb_scores),
            'task_episodes': len(task_data),
            'sentiment_episodes': len(sentiment_data),
            'scoring_episodes': len(scoring_data),
            'contestant_series': len(contestant_data) // 10,  # Rough estimate
            'final_merged': len(merged_data)
        }
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    fig.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Analysis completed!")
    print(f"   Correlations: {len(valid_correlations)} computed")
    print(f"   Mean correlation: {np.mean(valid_correlations):.3f}")
    print(f"   Results saved: {OUTPUT_FILE}")
    print(f"   Plot saved: {PLOT_FILE}")
    
    plt.show()
    
    return results, fig

if __name__ == "__main__":
    results, fig = main() 