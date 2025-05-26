#!/usr/bin/env python3
"""
Scoring Pattern Geometry vs IMDb Ratings Analysis
Tests whether the mathematical properties of task scoring patterns (μ, σ²) 
predict episode IMDb ratings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.plot_utils import load_config, apply_plot_style

def load_imdb_data():
    """Load IMDb ratings data."""
    print("Loading IMDb ratings data...")
    imdb_path = os.path.join('..', '..', 'data', 'raw', 'imdb_ratings.csv')
    imdb_data = pd.read_csv(imdb_path)
    print(f"  Loaded: {len(imdb_data)} episodes")
    return imdb_data

def load_scores_data():
    """Load task scores data."""
    print("Loading task scores data...")
    scores_path = os.path.join('..', '..', 'data', 'raw', 'long_task_scores.csv')
    scores_df = pd.read_csv(scores_path)
    
    # Rename columns to match expected format
    scores_df = scores_df.rename(columns={
        'SeriesID': 'series',
        'EpisodeID': 'episode', 
        'TaskID': 'task_id',
        'Score': 'score'
    })
    
    print(f"  Loaded: {len(scores_df)} task scores")
    print(f"  Series range: {scores_df['series'].min()}-{scores_df['series'].max()}")
    print(f"  Episodes: {scores_df['episode'].nunique()} unique episodes")
    print(f"  Tasks: {scores_df['task_id'].nunique()} unique tasks")
    return scores_df

def calculate_episode_scoring_geometry(scores_df):
    """Calculate mean and variance of scoring patterns for each episode."""
    print("Calculating episode-level scoring geometry...")
    
    episode_stats = []
    
    # Group by series and episode
    for (series, episode), group in scores_df.groupby(['series', 'episode']):
        # Get all task scores for this episode
        task_scores = []
        
        # Group by task to get score distributions
        for task_id, task_group in group.groupby('task_id'):
            scores = task_group['score'].values
            if len(scores) >= 3:  # Need at least 3 contestants for meaningful stats
                task_scores.append(scores)
        
        if task_scores:
            # Calculate statistics for each task
            task_means = [np.mean(scores) for scores in task_scores]
            task_vars = [np.var(scores, ddof=1) if len(scores) > 1 else 0 for scores in task_scores]
            
            # Episode-level aggregates
            episode_mean_mu = np.mean(task_means)
            episode_mean_var = np.mean(task_vars)
            episode_std_mu = np.std(task_means, ddof=1) if len(task_means) > 1 else 0
            episode_std_var = np.std(task_vars, ddof=1) if len(task_vars) > 1 else 0
            
            episode_stats.append({
                'series': series,
                'episode': episode,
                'episode_id': f"{series}_{episode}",
                'num_tasks': len(task_scores),
                'mean_mu': episode_mean_mu,
                'mean_var': episode_mean_var,
                'std_mu': episode_std_mu,
                'std_var': episode_std_var,
                'total_variance_explained': episode_mean_var / (episode_mean_var + episode_std_var) if (episode_mean_var + episode_std_var) > 0 else 0
            })
    
    episode_df = pd.DataFrame(episode_stats)
    print(f"  Calculated geometry for {len(episode_df)} episodes")
    return episode_df

def merge_data(imdb_data, episode_geometry):
    """Merge IMDb ratings with episode scoring geometry."""
    print("Merging IMDb and scoring geometry data...")
    
    # Create episode_id in IMDb data
    imdb_data['episode_id'] = imdb_data['series'].astype(str) + '_' + imdb_data['episode'].astype(str)
    
    # Merge datasets
    merged = pd.merge(imdb_data, episode_geometry, on='episode_id', how='inner', suffixes=('', '_geom'))
    
    print(f"  Merged dataset: {len(merged)} episodes")
    print(f"  IMDb rating range: {merged['imdb_rating'].min():.1f} - {merged['imdb_rating'].max():.1f}")
    print(f"  Mean μ range: {merged['mean_mu'].min():.2f} - {merged['mean_mu'].max():.2f}")
    print(f"  Mean variance range: {merged['mean_var'].min():.2f} - {merged['mean_var'].max():.2f}")
    
    return merged

def analyze_correlations(data):
    """Analyze correlations between scoring geometry and IMDb ratings."""
    print("\nAnalyzing correlations...")
    
    # Variables to analyze
    geometry_vars = ['mean_mu', 'mean_var', 'std_mu', 'std_var']
    rating_vars = ['imdb_rating', 'imdb_rating_relative']
    
    results = {}
    
    for rating_var in rating_vars:
        print(f"\n  {rating_var.upper()}:")
        results[rating_var] = {}
        
        for geom_var in geometry_vars:
            # Remove any NaN values
            clean_data = data[[rating_var, geom_var]].dropna()
            
            if len(clean_data) > 10:  # Need sufficient data
                # Pearson correlation
                pearson_r, pearson_p = pearsonr(clean_data[rating_var], clean_data[geom_var])
                
                # Spearman correlation
                spearman_r, spearman_p = spearmanr(clean_data[rating_var], clean_data[geom_var])
                
                results[rating_var][geom_var] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n': len(clean_data)
                }
                
                print(f"    {geom_var}:")
                print(f"      Pearson:  r={pearson_r:6.3f}, p={pearson_p:6.4f}")
                print(f"      Spearman: r={spearman_r:6.3f}, p={spearman_p:6.4f}")
                print(f"      n={len(clean_data)}")
    
    return results

def perform_regression_analysis(data):
    """Perform multiple regression analysis."""
    print("\nPerforming regression analysis...")
    
    # Prepare data
    geometry_vars = ['mean_mu', 'mean_var', 'std_mu', 'std_var']
    clean_data = data[['imdb_rating', 'imdb_rating_relative'] + geometry_vars].dropna()
    
    results = {}
    
    for target in ['imdb_rating', 'imdb_rating_relative']:
        print(f"\n  Predicting {target}:")
        
        X = clean_data[geometry_vars].values
        y = clean_data[target].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit regression
        reg = LinearRegression()
        reg.fit(X_scaled, y)
        
        # Predictions
        y_pred = reg.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        # F-test for overall significance
        n = len(y)
        p = len(geometry_vars)
        f_stat = (r2 / p) / ((1 - r2) / (n - p - 1))
        f_p = 1 - stats.f.cdf(f_stat, p, n - p - 1)
        
        results[target] = {
            'r2': r2,
            'f_stat': f_stat,
            'f_p': f_p,
            'coefficients': dict(zip(geometry_vars, reg.coef_)),
            'intercept': reg.intercept_,
            'n': n
        }
        
        print(f"    R² = {r2:.4f}")
        print(f"    F({p}, {n-p-1}) = {f_stat:.2f}, p = {f_p:.4f}")
        print(f"    Coefficients:")
        for var, coef in zip(geometry_vars, reg.coef_):
            print(f"      {var}: {coef:7.4f}")
    
    return results

def create_visualization(data, config):
    """Create visualization of scoring geometry vs IMDb ratings."""
    print("\nCreating visualization...")
    
    # Apply plot style
    apply_plot_style(config)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Scoring Pattern Geometry vs IMDb Ratings', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean μ vs IMDb rating
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(data['mean_mu'], data['imdb_rating'], 
                          alpha=0.6, s=50, c=data['series'], cmap='viridis')
    ax1.set_xlabel('Episode Mean Score (μ)')
    ax1.set_ylabel('IMDb Rating')
    ax1.set_title('Mean Score vs IMDb Rating')
    
    # Add trend line
    z1 = np.polyfit(data['mean_mu'], data['imdb_rating'], 1)
    p1 = np.poly1d(z1)
    ax1.plot(data['mean_mu'], p1(data['mean_mu']), "r--", alpha=0.8)
    
    # Plot 2: Mean variance vs IMDb rating
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(data['mean_var'], data['imdb_rating'], 
                          alpha=0.6, s=50, c=data['series'], cmap='viridis')
    ax2.set_xlabel('Episode Mean Variance (σ²)')
    ax2.set_ylabel('IMDb Rating')
    ax2.set_title('Mean Variance vs IMDb Rating')
    
    # Add trend line
    z2 = np.polyfit(data['mean_var'], data['imdb_rating'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(data['mean_var'], p2(data['mean_var']), "r--", alpha=0.8)
    
    # Plot 3: Mean μ vs relative IMDb rating
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(data['mean_mu'], data['imdb_rating_relative'], 
                          alpha=0.6, s=50, c=data['series'], cmap='viridis')
    ax3.set_xlabel('Episode Mean Score (μ)')
    ax3.set_ylabel('Relative IMDb Rating')
    ax3.set_title('Mean Score vs Relative Rating')
    
    # Add trend line
    z3 = np.polyfit(data['mean_mu'], data['imdb_rating_relative'], 1)
    p3 = np.poly1d(z3)
    ax3.plot(data['mean_mu'], p3(data['mean_mu']), "r--", alpha=0.8)
    
    # Plot 4: Mean variance vs relative IMDb rating
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(data['mean_var'], data['imdb_rating_relative'], 
                          alpha=0.6, s=50, c=data['series'], cmap='viridis')
    ax4.set_xlabel('Episode Mean Variance (σ²)')
    ax4.set_ylabel('Relative IMDb Rating')
    ax4.set_title('Mean Variance vs Relative Rating')
    
    # Add trend line
    z4 = np.polyfit(data['mean_var'], data['imdb_rating_relative'], 1)
    p4 = np.poly1d(z4)
    ax4.plot(data['mean_var'], p4(data['mean_var']), "r--", alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter1, ax=axes, shrink=0.8, aspect=20)
    cbar.set_label('Series Number')
    
    plt.tight_layout()
    return fig

def save_results(correlation_results, regression_results, data):
    """Save analysis results to files."""
    print("\nSaving results...")
    
    # Save correlation results
    corr_df = []
    for rating_var, geom_results in correlation_results.items():
        for geom_var, stats in geom_results.items():
            corr_df.append({
                'rating_variable': rating_var,
                'geometry_variable': geom_var,
                'pearson_r': stats['pearson_r'],
                'pearson_p': stats['pearson_p'],
                'spearman_r': stats['spearman_r'],
                'spearman_p': stats['spearman_p'],
                'n': stats['n']
            })
    
    corr_df = pd.DataFrame(corr_df)
    corr_df.to_csv('scoring_geometry_correlations.csv', index=False)
    
    # Save regression results
    reg_df = []
    for target, results in regression_results.items():
        reg_df.append({
            'target': target,
            'r2': results['r2'],
            'f_stat': results['f_stat'],
            'f_p': results['f_p'],
            'n': results['n'],
            'coef_mean_mu': results['coefficients']['mean_mu'],
            'coef_mean_var': results['coefficients']['mean_var'],
            'coef_std_mu': results['coefficients']['std_mu'],
            'coef_std_var': results['coefficients']['std_var'],
            'intercept': results['intercept']
        })
    
    reg_df = pd.DataFrame(reg_df)
    reg_df.to_csv('scoring_geometry_regression.csv', index=False)
    
    # Save merged dataset
    data.to_csv('episode_scoring_geometry_imdb.csv', index=False)
    
    print("  Results saved to:")
    print("    - scoring_geometry_correlations.csv")
    print("    - scoring_geometry_regression.csv") 
    print("    - episode_scoring_geometry_imdb.csv")

def main():
    """Main analysis function."""
    print("="*60)
    print("SCORING PATTERN GEOMETRY vs IMDb RATINGS ANALYSIS")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Load data
    imdb_data = load_imdb_data()
    scores_data = load_scores_data()
    
    # Calculate episode-level scoring geometry
    episode_geometry = calculate_episode_scoring_geometry(scores_data)
    
    # Merge datasets
    merged_data = merge_data(imdb_data, episode_geometry)
    
    # Analyze correlations
    correlation_results = analyze_correlations(merged_data)
    
    # Perform regression analysis
    regression_results = perform_regression_analysis(merged_data)
    
    # Create visualization
    fig = create_visualization(merged_data, config)
    
    # Save figure
    dpi = config['global'].get('dpi', 300)
    plt.savefig('figure6b_scoring_geometry_imdb.png', dpi=dpi, bbox_inches='tight')
    plt.savefig('figure6b_scoring_geometry_imdb.pdf', bbox_inches='tight')
    plt.close()
    
    # Save results
    save_results(correlation_results, regression_results, merged_data)
    
    print(f"\n✅ Analysis completed!")
    print(f"   Episodes analyzed: {len(merged_data)}")
    print(f"   Figure saved: figure6b_scoring_geometry_imdb.png/pdf")

if __name__ == "__main__":
    main() 