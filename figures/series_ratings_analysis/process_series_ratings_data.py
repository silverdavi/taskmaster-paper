#!/usr/bin/env python3
"""
Process data for Figure 1: Series-Level IMDb Ratings

This script:
1. Loads IMDb ratings data
2. Fits Gaussian distributions to ratings 2-9 for each series
3. Aggregates #1s and #10s counts
4. Performs PCA on the series-level metrics
5. Saves processed data directly to the figure1 folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats
from sklearn.decomposition import PCA
import json

# Add config to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "config"))
import plot_utils

# Define constants
FIGURE_NUM = 1
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
# Save processed data directly to figure1 folder
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def fit_gaussian_per_series(df_histograms):
    """
    Fit Gaussian distributions to ratings 2-9 for each series.
    
    Parameters:
    -----------
    df_histograms : pandas.DataFrame
        DataFrame with histogram data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with fitted parameters
    """
    # Create bins for ratings 2-9 (ignoring 1 and 10)
    rating_bins = np.arange(1.5, 10.5, 1)
    
    # Group by series
    results = []
    
    # Get unique series
    series_list = df_histograms['series'].unique()
    
    for series in series_list:
        # Filter for this series
        series_data = df_histograms[df_histograms['series'] == series]
        
        # Aggregate votes by rating (2-9)
        votes_by_rating = {}
        for rating in range(2, 10):
            col_name = f'hist{rating}_pct'
            votes_by_rating[rating] = series_data[col_name].sum() * series_data['total_votes'].sum() / 100
        
        # Convert to arrays for fitting
        x = np.array(list(votes_by_rating.keys()))
        y = np.array(list(votes_by_rating.values()))
        
        # Skip if no data
        if len(x) == 0 or y.sum() == 0:
            print(f"Warning: No data for series {series}, skipping")
            continue
        
        # Create weighted data for fitting
        # We need to repeat each rating by its frequency
        data_for_fit = np.repeat(x, y.astype(int))
        
        # Fit Gaussian using different approach
        if len(data_for_fit) > 0:
            mu = np.mean(data_for_fit)
            sigma = np.std(data_for_fit)
        else:
            # Fallback if there's not enough data
            mu = 0
            sigma = 0
            print(f"Warning: Not enough data for series {series}, using fallback values")
        
        # Calculate #1s and #10s (total percentage)
        pct_1s = (series_data['hist1_pct'] * series_data['total_votes']).sum() / series_data['total_votes'].sum()
        pct_10s = (series_data['hist10_pct'] * series_data['total_votes']).sum() / series_data['total_votes'].sum()
        
        # Calculate total votes and episode count
        total_votes = series_data['total_votes'].sum()
        episode_count = len(series_data)
        
        # Calculate overall mean rating (weighted by votes)
        overall_mean = 0
        for rating in range(1, 11):
            col_name = f'hist{rating}_pct'
            # Remember: hist1 is for rating 1, hist2 is for rating 2, etc.
            overall_mean += rating * (series_data[col_name] * series_data['total_votes']).sum() / total_votes
        
        # Store results
        results.append({
            'series': series,
            'mu': mu,
            'sigma': sigma,
            'pct_1s': pct_1s,
            'pct_10s': pct_10s,
            'total_votes': total_votes,
            'episode_count': episode_count,
            'overall_mean': overall_mean
        })
    
    return pd.DataFrame(results)

def perform_pca(df_series_metrics):
    """
    Perform PCA on series metrics.
    
    Parameters:
    -----------
    df_series_metrics : pandas.DataFrame
        DataFrame with series metrics
    
    Returns:
    --------
    tuple
        (DataFrame with PCA results, explained variance ratios, loadings DataFrame)
    """
    # Select features for PCA
    features = ['pct_1s', 'pct_10s', 'mu', 'sigma']
    X = df_series_metrics[features].values
    
    # Standardize the features
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    # Create result DataFrame
    df_pca = pd.DataFrame({
        'series': df_series_metrics['series'],
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1]
    })
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    # Calculate feature loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=features
    )
    
    return df_pca, explained_variance, loadings

def main():
    print("Processing data for Figure 1: Series-Level IMDb Ratings")
    
    # 1. Load raw data
    try:
        df_ratings = pd.read_csv(RAW_DIR / "imdb_ratings.csv")
        df_histograms = pd.read_csv(RAW_DIR / "taskmaster_histograms_corrected.csv")
        print(f"Loaded data: {len(df_histograms)} histogram records, {len(df_ratings)} ratings records")
        
        # Rename 'season' column to 'series' to match the expected column name
        if 'season' in df_histograms.columns and 'series' not in df_histograms.columns:
            df_histograms = df_histograms.rename(columns={'season': 'series'})
            print("Renamed 'season' column to 'series' in histograms data")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Make sure the required data files exist in data/raw/ directory")
        return
    
    # 2. Fit Gaussian distributions and aggregate #1s and #10s
    df_series_metrics = fit_gaussian_per_series(df_histograms)
    print(f"Processed metrics for {len(df_series_metrics)} series")
    
    # Calculate average IMDb rating per series from df_ratings
    imdb_avg_ratings = df_ratings.groupby('series')['imdb_rating'].mean().reset_index()
    print(f"Calculated average IMDb ratings for {len(imdb_avg_ratings)} series")
    
    # Merge with the series_metrics DataFrame
    df_series_metrics = pd.merge(df_series_metrics, imdb_avg_ratings, on='series', how='left')
    print(f"Added IMDb ratings to series metrics")
    
    # 3. Perform PCA
    df_pca, explained_variance, loadings = perform_pca(df_series_metrics)
    
    # 4. Save processed data directly to figure1 folder
    df_series_metrics.to_csv(OUTPUT_DIR / "series_metrics.csv", index=False)
    df_pca.to_csv(OUTPUT_DIR / "series_pca.csv", index=False)
    
    # Save loadings and explained variance
    loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")
    np.save(OUTPUT_DIR / "explained_variance.npy", explained_variance)
    
    # 5. Log metrics for caption
    metrics = {
        "best_series": int(df_series_metrics.loc[df_series_metrics['mu'].idxmax(), 'series']),
        "worst_series": int(df_series_metrics.loc[df_series_metrics['mu'].idxmin(), 'series']),
        "highest_mu": float(df_series_metrics['mu'].max()),
        "lowest_mu": float(df_series_metrics['mu'].min()),
        "max_pct_10s": float(df_series_metrics['pct_10s'].max()),
        "max_pct_10s_series": int(df_series_metrics.loc[df_series_metrics['pct_10s'].idxmax(), 'series']),
        "max_pct_1s": float(df_series_metrics['pct_1s'].max()),
        "max_pct_1s_series": int(df_series_metrics.loc[df_series_metrics['pct_1s'].idxmax(), 'series']),
        "pc1_explained_var": float(explained_variance[0]),
        "pc2_explained_var": float(explained_variance[1]),
        "total_explained_var": float(sum(explained_variance)),
        "total_series": len(df_series_metrics),
        "total_episodes": int(df_series_metrics['episode_count'].sum()),
        "total_votes": int(df_series_metrics['total_votes'].sum())
    }
    
    # Compute additional metrics
    metrics["avg_votes_per_series"] = metrics["total_votes"] / metrics["total_series"]
    metrics["avg_episodes_per_series"] = metrics["total_episodes"] / metrics["total_series"]
    
    # Compute correlations between mu and extreme ratings
    mu_vs_1s_corr = df_series_metrics[['mu', 'pct_1s']].corr().loc['mu', 'pct_1s']
    mu_vs_10s_corr = df_series_metrics[['mu', 'pct_10s']].corr().loc['mu', 'pct_10s']
    metrics["mu_vs_1s_corr"] = float(mu_vs_1s_corr)
    metrics["mu_vs_10s_corr"] = float(mu_vs_10s_corr)
    
    # Calculate PCA quadrants - how many series in each quadrant
    df_pca['quadrant'] = df_pca.apply(
        lambda row: (1 if row['PC1'] >= 0 else 2) if row['PC2'] >= 0 
                    else (3 if row['PC1'] < 0 else 4), 
        axis=1
    )
    quadrant_counts = df_pca['quadrant'].value_counts().to_dict()
    for q in range(1, 5):
        metrics[f"quadrant{q}_count"] = quadrant_counts.get(q, 0)
    
    # Save metrics directly to figure1 folder
    metrics_file = OUTPUT_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Data for Figure {FIGURE_NUM} processed successfully.")
    print(f"Processed data saved to {OUTPUT_DIR}")
    print(f"Best series: {metrics['best_series']} (μ={metrics['highest_mu']:.2f})")
    print(f"Worst series: {metrics['worst_series']} (μ={metrics['lowest_mu']:.2f})")
    print(f"PCA explained variance: {metrics['total_explained_var']:.1%}")

if __name__ == "__main__":
    main() 