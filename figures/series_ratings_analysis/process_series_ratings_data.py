#!/usr/bin/env python3
"""
Process data for Figure 1: Series-Level IMDb Ratings

This script:
1. Loads IMDb ratings data
2. Fits a mixture model: a_1*δ(1) + a_10*δ(10) + N(μ,σ) for ratings 2-9
3. Aggregates #1s and #10s counts
4. Performs PCA on the series-level metrics
5. Saves processed data directly to the figure1 folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats
from scipy.optimize import minimize
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

def fit_mixture_model_per_series(df_histograms):
    """
    Fit mixture model: a_1*δ(1) + a_10*δ(10) + N(μ,σ) to ratings for each series.
    
    Parameters:
    -----------
    df_histograms : pandas.DataFrame
        DataFrame with histogram data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with fitted parameters
    """
    # Group by series
    results = []
    
    # Get unique series
    series_list = df_histograms['series'].unique()
    
    for series in series_list:
        # Filter for this series
        series_data = df_histograms[df_histograms['series'] == series]
        
        # Get total votes for normalization
        total_votes = series_data['total_votes'].sum()
        
        # Calculate actual vote counts for each rating
        votes_by_rating = {}
        for rating in range(1, 11):
            col_name = f'hist{rating}_pct'
            votes_by_rating[rating] = (series_data[col_name] * series_data['total_votes']).sum() / 100
        
        # Extract votes for ratings 1 and 10 (these will be our delta functions)
        votes_1 = votes_by_rating[1]
        votes_10 = votes_by_rating[10]
        
        # Extract votes for ratings 2-9 (for Gaussian fitting)
        votes_2_to_9 = {k: v for k, v in votes_by_rating.items() if 2 <= k <= 9}
        
        # Create weighted data for fitting Gaussian to ratings 2-9
        x_gaussian = []
        for rating, count in votes_2_to_9.items():
            x_gaussian.extend([rating] * int(count))
        
        # Fit Gaussian to ratings 2-9
        if len(x_gaussian) > 0:
            mu_init = np.mean(x_gaussian)
            sigma_init = np.std(x_gaussian)
            
            # If we want to be more sophisticated, we could use MLE or optimization here
            # For now, using sample statistics as they work well
            mu = mu_init
            sigma = sigma_init if sigma_init > 0 else 0.1  # Avoid zero sigma
        else:
            # Fallback if there's not enough data
            mu = 5.5  # Middle of 2-9 range
            sigma = 1.0
            print(f"Warning: Not enough data for series {series}, using fallback values")
        
        # Calculate proportions for the mixture model
        total_2_to_9 = sum(votes_2_to_9.values())
        a_1 = votes_1 / total_votes  # Proportion of 1s
        a_10 = votes_10 / total_votes  # Proportion of 10s
        a_gaussian = total_2_to_9 / total_votes  # Proportion in Gaussian
        
        # Verify proportions sum to ~1 (allowing for rounding)
        prop_sum = a_1 + a_10 + a_gaussian
        if abs(prop_sum - 1.0) > 0.01:
            print(f"Warning: Proportions sum to {prop_sum} for series {series}")
        
        # Calculate percentages for backward compatibility
        pct_1s = a_1 * 100
        pct_10s = a_10 * 100
        
        # Calculate overall mean rating (weighted by votes)
        overall_mean = sum(rating * votes_by_rating[rating] for rating in range(1, 11)) / total_votes
        
        # Store results - now including mixture model parameters
        results.append({
            'series': series,
            'a_1': a_1,  # Proportion of delta at 1
            'a_10': a_10,  # Proportion of delta at 10
            'a_gaussian': a_gaussian,  # Proportion in Gaussian
            'mu': mu,  # Mean of Gaussian component
            'sigma': sigma,  # Std of Gaussian component
            'pct_1s': pct_1s,  # For backward compatibility
            'pct_10s': pct_10s,  # For backward compatibility
            'total_votes': total_votes,
            'episode_count': len(series_data),
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

def calculate_goodness_of_fit(df_histograms, df_series_metrics):
    """
    Calculate goodness of fit metrics comparing mixture model vs naive Gaussian fit.
    
    Parameters:
    -----------
    df_histograms : pandas.DataFrame
        DataFrame with histogram data
    df_series_metrics : pandas.DataFrame
        DataFrame with fitted mixture model parameters
    
    Returns:
    --------
    dict
        Dictionary with goodness of fit statistics
    """
    print("\nCalculating goodness of fit metrics...")
    
    mixture_residuals = []
    naive_residuals = []
    
    # Get unique series
    series_list = df_histograms['series'].unique()
    
    for series in series_list:
        # Get series data
        series_data = df_histograms[df_histograms['series'] == series]
        series_metrics = df_series_metrics[df_series_metrics['series'] == series].iloc[0]
        
        # Get total votes for normalization
        total_votes = series_data['total_votes'].sum()
        
        # Calculate observed proportions for each rating
        observed_props = []
        for rating in range(1, 11):
            col_name = f'hist{rating}_pct'
            prop = (series_data[col_name] * series_data['total_votes']).sum() / (100 * total_votes)
            observed_props.append(prop)
        
        # Calculate mixture model predictions
        mixture_props = []
        for rating in range(1, 11):
            if rating == 1:
                # Delta function at 1
                pred_prop = series_metrics['a_1']
            elif rating == 10:
                # Delta function at 10
                pred_prop = series_metrics['a_10']
            else:
                # Gaussian component for ratings 2-9
                mu = series_metrics['mu']
                sigma = series_metrics['sigma']
                # Calculate probability density for this rating (discrete approximation)
                # Use the area under the Gaussian curve from rating-0.5 to rating+0.5
                from scipy.stats import norm
                pred_prop = series_metrics['a_gaussian'] * (
                    norm.cdf(rating + 0.5, mu, sigma) - norm.cdf(rating - 0.5, mu, sigma)
                )
            mixture_props.append(pred_prop)
        
        # Calculate naive Gaussian fit to all data (ratings 1-10)
        # Create weighted data for all ratings
        all_ratings = []
        for rating in range(1, 11):
            col_name = f'hist{rating}_pct'
            count = (series_data[col_name] * series_data['total_votes']).sum() / 100
            all_ratings.extend([rating] * int(count))
        
        if len(all_ratings) > 0:
            naive_mu = np.mean(all_ratings)
            naive_sigma = np.std(all_ratings)
            if naive_sigma == 0:
                naive_sigma = 0.1  # Avoid division by zero
        else:
            naive_mu = 5.5
            naive_sigma = 1.0
        
        # Calculate naive Gaussian predictions
        naive_props = []
        for rating in range(1, 11):
            from scipy.stats import norm
            pred_prop = norm.cdf(rating + 0.5, naive_mu, naive_sigma) - norm.cdf(rating - 0.5, naive_mu, naive_sigma)
            naive_props.append(pred_prop)
        
        # Calculate residuals (observed - predicted)
        mixture_residual = np.array(observed_props) - np.array(mixture_props)
        naive_residual = np.array(observed_props) - np.array(naive_props)
        
        # Store residuals for this series
        mixture_residuals.extend(mixture_residual)
        naive_residuals.extend(naive_residual)
        
        # Print detailed comparison for first few series
        if series <= 3:
            print(f"\nSeries {series} detailed comparison:")
            print(f"  Observed:  {[f'{p:.3f}' for p in observed_props]}")
            print(f"  Mixture:   {[f'{p:.3f}' for p in mixture_props]}")
            print(f"  Naive:     {[f'{p:.3f}' for p in naive_props]}")
            print(f"  Mix Resid: {[f'{r:+.3f}' for r in mixture_residual]}")
            print(f"  Naive Res: {[f'{r:+.3f}' for r in naive_residual]}")
    
    # Convert to numpy arrays for analysis
    mixture_residuals = np.array(mixture_residuals)
    naive_residuals = np.array(naive_residuals)
    
    # Calculate goodness of fit statistics
    gof_stats = {
        # Mixture model residuals
        'mixture_residual_min': float(np.min(mixture_residuals)),
        'mixture_residual_max': float(np.max(mixture_residuals)),
        'mixture_residual_mean': float(np.mean(mixture_residuals)),
        'mixture_residual_median': float(np.median(mixture_residuals)),
        'mixture_residual_std': float(np.std(mixture_residuals)),
        'mixture_residual_mae': float(np.mean(np.abs(mixture_residuals))),  # Mean Absolute Error
        'mixture_residual_rmse': float(np.sqrt(np.mean(mixture_residuals**2))),  # Root Mean Square Error
        
        # Naive Gaussian residuals
        'naive_residual_min': float(np.min(naive_residuals)),
        'naive_residual_max': float(np.max(naive_residuals)),
        'naive_residual_mean': float(np.mean(naive_residuals)),
        'naive_residual_median': float(np.median(naive_residuals)),
        'naive_residual_std': float(np.std(naive_residuals)),
        'naive_residual_mae': float(np.mean(np.abs(naive_residuals))),  # Mean Absolute Error
        'naive_residual_rmse': float(np.sqrt(np.mean(naive_residuals**2))),  # Root Mean Square Error
        
        # Improvement metrics
        'mae_improvement': float(np.mean(np.abs(naive_residuals)) - np.mean(np.abs(mixture_residuals))),
        'rmse_improvement': float(np.sqrt(np.mean(naive_residuals**2)) - np.sqrt(np.mean(mixture_residuals**2))),
        'mae_improvement_pct': float((np.mean(np.abs(naive_residuals)) - np.mean(np.abs(mixture_residuals))) / np.mean(np.abs(naive_residuals)) * 100),
        'rmse_improvement_pct': float((np.sqrt(np.mean(naive_residuals**2)) - np.sqrt(np.mean(mixture_residuals**2))) / np.sqrt(np.mean(naive_residuals**2)) * 100),
    }
    
    return gof_stats

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
    
    # 2. Fit mixture model and aggregate #1s and #10s
    df_series_metrics = fit_mixture_model_per_series(df_histograms)
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
    
    # Calculate goodness of fit metrics
    gof_stats = calculate_goodness_of_fit(df_histograms, df_series_metrics)
    for k, v in gof_stats.items():
        metrics[k] = v
    
    # Save metrics directly to figure1 folder (including goodness of fit)
    metrics_file = OUTPUT_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Data for Figure {FIGURE_NUM} processed successfully.")
    print(f"Processed data saved to {OUTPUT_DIR}")
    print(f"Best series: {metrics['best_series']} (μ={metrics['highest_mu']:.2f})")
    print(f"Worst series: {metrics['worst_series']} (μ={metrics['lowest_mu']:.2f})")
    print(f"PCA explained variance: {metrics['total_explained_var']:.1%}")
    
    # Print goodness of fit summary
    print(f"\n=== GOODNESS OF FIT COMPARISON ===")
    print(f"Mixture Model Residuals:")
    print(f"  MAE: {gof_stats['mixture_residual_mae']:.4f}")
    print(f"  RMSE: {gof_stats['mixture_residual_rmse']:.4f}")
    print(f"  Range: [{gof_stats['mixture_residual_min']:.4f}, {gof_stats['mixture_residual_max']:.4f}]")
    print(f"  Mean: {gof_stats['mixture_residual_mean']:.4f}, Median: {gof_stats['mixture_residual_median']:.4f}")
    
    print(f"\nNaive Gaussian Residuals:")
    print(f"  MAE: {gof_stats['naive_residual_mae']:.4f}")
    print(f"  RMSE: {gof_stats['naive_residual_rmse']:.4f}")
    print(f"  Range: [{gof_stats['naive_residual_min']:.4f}, {gof_stats['naive_residual_max']:.4f}]")
    print(f"  Mean: {gof_stats['naive_residual_mean']:.4f}, Median: {gof_stats['naive_residual_median']:.4f}")
    
    print(f"\nImprovement (Mixture vs Naive):")
    print(f"  MAE improvement: {gof_stats['mae_improvement']:.4f} ({gof_stats['mae_improvement_pct']:.1f}%)")
    print(f"  RMSE improvement: {gof_stats['rmse_improvement']:.4f} ({gof_stats['rmse_improvement_pct']:.1f}%)")
    
    if gof_stats['mae_improvement'] > 0:
        print(f"  ✓ Mixture model fits better (lower error)")
    else:
        print(f"  ✗ Naive Gaussian fits better (lower error)")
    print(f"=====================================\n")

if __name__ == "__main__":
    main() 