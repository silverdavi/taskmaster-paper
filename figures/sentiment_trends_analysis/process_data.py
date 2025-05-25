#!/usr/bin/env python3
"""
Data processing for Figure 7: Sentiment Trends Analysis

This script:
1. Loads sentiment data and calculates series-level statistics
2. Performs trend analysis across series for each sentiment metric
3. Applies multiple testing correction for significance
4. Saves processed data for visualization

Output: figure7_sentiment_trends.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit
import warnings
from pathlib import Path

# Configuration
INPUT_FILE = Path("../../data/raw/sentiment.csv")
OUTPUT_DIR = Path(".")
OUTPUT_FILE = OUTPUT_DIR / "figure7_sentiment_trends.csv"

# Sentiment columns to analyze - ONLY SENTIMENT METRICS
SENTIMENT_COLS = [
    'avg_anger', 'avg_awkwardness', 'avg_frustration_or_despair', 
    'avg_humor', 'avg_joy_or_excitement', 'avg_sarcasm', 'avg_self_deprecation'
]

# Only analyze sentiment metrics
ALL_METRICS = SENTIMENT_COLS

def estimate_effective_tests(p_values, correlation_matrix=None):
    """
    Estimate effective number of independent tests using eigenvalue decomposition.
    
    Args:
        p_values: Array of p-values
        correlation_matrix: Correlation matrix of test statistics (optional)
        
    Returns:
        Effective number of independent tests
    """
    n_tests = len(p_values)
    
    if correlation_matrix is None:
        # If no correlation matrix provided, assume independence
        return n_tests
    
    # Use eigenvalue decomposition to estimate effective number of tests
    eigenvals = np.linalg.eigvals(correlation_matrix)
    eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
    
    # Effective number of tests (Li and Ji method)
    eff_tests = 1 + (n_tests - 1) * (1 - np.var(eigenvals) / n_tests)
    return max(1, min(n_tests, eff_tests))

def bonferroni_correction_corr(p_values, alpha=0.01, correlation_matrix=None):
    """Apply correlation-aware Bonferroni correction to p-values."""
    eff_tests = estimate_effective_tests(p_values, correlation_matrix)
    corrected_alpha = alpha / eff_tests
    corrected_p = np.array(p_values) * eff_tests
    corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0
    rejected = corrected_p < alpha
    return rejected, corrected_p

def fdr_correction(p_values, alpha=0.01):
    """Apply standard Benjamini-Hochberg FDR correction."""
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and get original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Apply BH procedure
    threshold = (np.arange(1, n+1) / n) * alpha
    rejected_sorted = sorted_p <= threshold
    
    # Find the largest k for which p_k <= (k/n) * alpha
    if np.any(rejected_sorted):
        max_k = np.max(np.where(rejected_sorted)[0])
        rejected_sorted[:max_k+1] = True
    
    # Map back to original order
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices] = rejected_sorted
    
    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_indices[i]] = min(1.0, sorted_p[i] * n / (i + 1))
    
    return rejected, adjusted_p

def fdr_correction_by(p_values, alpha=0.01):
    """
    Apply Benjamini-Yekutieli FDR correction for positive dependence.
    More conservative than BH but valid under positive dependence.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and get original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Harmonic number for BY correction
    c_n = np.sum(1.0 / np.arange(1, n + 1))
    
    # Apply BY procedure: more conservative threshold
    threshold = (np.arange(1, n+1) / n) * (alpha / c_n)
    rejected_sorted = sorted_p <= threshold
    
    # Find the largest k for which p_k <= (k/n) * (alpha/c_n)
    if np.any(rejected_sorted):
        max_k = np.max(np.where(rejected_sorted)[0])
        rejected_sorted[:max_k+1] = True
    
    # Map back to original order
    rejected = np.zeros(n, dtype=bool)
    rejected[sorted_indices] = rejected_sorted
    
    # Calculate adjusted p-values
    adjusted_p = np.zeros(n)
    for i in range(n):
        adjusted_p[sorted_indices[i]] = min(1.0, sorted_p[i] * n * c_n / (i + 1))
    
    return rejected, adjusted_p

def calculate_test_correlation_matrix(series_stats, metrics):
    """
    Calculate correlation matrix between test statistics.
    
    Args:
        series_stats: DataFrame with series-level statistics
        metrics: List of metrics to analyze
        
    Returns:
        Correlation matrix of the test statistics
    """
    # Extract mean values for each metric across series
    metric_values = []
    valid_metrics = []
    
    for metric in metrics:
        mean_col = f'{metric}_mean'
        if mean_col in series_stats.columns:
            values = series_stats[mean_col].values
            if not np.all(np.isnan(values)):
                metric_values.append(values)
                valid_metrics.append(metric)
    
    if len(metric_values) < 2:
        return None
    
    # Create correlation matrix
    data_matrix = np.array(metric_values).T  # Series x Metrics
    
    # Remove rows with any NaN values
    valid_rows = ~np.any(np.isnan(data_matrix), axis=1)
    if np.sum(valid_rows) < 3:
        return None
        
    clean_data = data_matrix[valid_rows, :]
    
    if clean_data.shape[0] < 3 or clean_data.shape[1] < 2:
        return None
    
    try:
        corr_matrix = np.corrcoef(clean_data.T)
        return corr_matrix
    except:
        return None

def load_sentiment_data():
    """Load the sentiment dataset."""
    try:
        sentiment_data = pd.read_csv(INPUT_FILE)
        print(f"Loaded sentiment data: {len(sentiment_data)} episodes")
        return sentiment_data
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        raise

def calculate_series_statistics(sentiment_data):
    """
    Calculate mean and standard deviation for each metric per series.
    
    Args:
        sentiment_data: DataFrame with sentiment data
        
    Returns:
        DataFrame with series-level statistics
    """
    # Group by series and calculate statistics
    series_stats = []
    
    for series in sorted(sentiment_data['series'].unique()):
        series_data = sentiment_data[sentiment_data['series'] == series]
        
        stats_row = {'series': series, 'n_episodes': len(series_data)}
        
        # Calculate mean and std for each metric
        for metric in ALL_METRICS:
            if metric in series_data.columns:
                stats_row[f'{metric}_mean'] = series_data[metric].mean()
                stats_row[f'{metric}_std'] = series_data[metric].std()
                stats_row[f'{metric}_sem'] = series_data[metric].sem()  # Standard error of mean
            else:
                stats_row[f'{metric}_mean'] = np.nan
                stats_row[f'{metric}_std'] = np.nan
                stats_row[f'{metric}_sem'] = np.nan
        
        series_stats.append(stats_row)
    
    return pd.DataFrame(series_stats)

def perform_trend_analysis(series_stats):
    """
    Perform weighted linear trend analysis for each sentiment metric across series.
    Uses standard errors as weights to account for within-series uncertainty.
    
    Args:
        series_stats: DataFrame with series-level statistics
        
    Returns:
        DataFrame with trend analysis results
    """
    trend_results = []
    
    # Get series numbers for regression
    series_numbers = series_stats['series'].values
    
    for metric in ALL_METRICS:
        mean_col = f'{metric}_mean'
        sem_col = f'{metric}_sem'  # Standard error of mean
        
        if mean_col not in series_stats.columns:
            continue
            
        # Get the mean values and standard errors across series
        metric_means = series_stats[mean_col].values
        metric_sems = series_stats[sem_col].values if sem_col in series_stats.columns else None
        
        # Skip if all NaN
        if np.all(np.isnan(metric_means)):
            continue
        
        # Remove NaN values for regression
        valid_mask = ~np.isnan(metric_means)
        if metric_sems is not None:
            valid_mask = valid_mask & ~np.isnan(metric_sems) & (metric_sems > 0)
        
        if np.sum(valid_mask) < 3:  # Need at least 3 points
            continue
            
        x_vals = series_numbers[valid_mask]
        y_vals = metric_means[valid_mask]
        
        # Prepare weights (inverse of standard errors)
        if metric_sems is not None and np.sum(valid_mask) > 0:
            sem_vals = metric_sems[valid_mask]
            # Use inverse of SEM as weights, add small constant to avoid division by zero
            weights = 1.0 / (sem_vals + np.mean(sem_vals) * 0.01)
        else:
            weights = None
        
        # Perform weighted linear regression using numpy.polyfit
        try:
            if weights is not None:
                # Weighted regression
                coeffs, cov_matrix = np.polyfit(x_vals, y_vals, 1, w=weights, cov=True)
                slope, intercept = coeffs
                slope_err = np.sqrt(cov_matrix[0, 0])
                
                # Calculate R-squared for weighted regression
                y_pred = slope * x_vals + intercept
                ss_res = np.sum(weights * (y_vals - y_pred) ** 2)
                ss_tot = np.sum(weights * (y_vals - np.average(y_vals, weights=weights)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate t-statistic and p-value
                t_stat = slope / slope_err if slope_err > 0 else 0
                df = len(x_vals) - 2
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df)) if df > 0 else 1.0
                
                # Correlation coefficient (approximation for weighted case)
                r_value = np.sqrt(r_squared) * np.sign(slope)
                
            else:
                # Unweighted regression (fallback)
                slope, intercept, r_value, p_value, slope_err = linregress(x_vals, y_vals)
                r_squared = r_value ** 2
                
        except Exception as e:
            print(f"Warning: Regression failed for {metric}: {e}")
            continue
        
        # Calculate additional statistics
        n_series = len(x_vals)
        
        # Classify trend direction
        if p_value < 0.05:
            if slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
        else:
            trend_direction = 'no_trend'
        
        # Calculate effect size (standardized slope)
        y_std = np.std(y_vals)
        x_range = np.max(x_vals) - np.min(x_vals)
        standardized_slope = (slope * x_range) / y_std if y_std > 0 else 0
        
        trend_results.append({
            'metric': metric,
            'metric_type': 'sentiment',  # All metrics are sentiment now
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'r_squared': r_squared,
            'p_value': p_value,
            'std_err': slope_err,
            'n_series': n_series,
            'trend_direction': trend_direction,
            'effect_size': abs(standardized_slope),
            'standardized_slope': standardized_slope,
            'mean_value': np.mean(y_vals),
            'std_value': np.std(y_vals),
            'min_value': np.min(y_vals),
            'max_value': np.max(y_vals),
            'weighted': weights is not None
        })
    
    return pd.DataFrame(trend_results)

def apply_multiple_testing_correction(trend_results, series_stats):
    """
    Apply multiple testing correction to p-values with standard FDR.
    
    Args:
        trend_results: DataFrame with trend analysis results
        series_stats: DataFrame with series-level statistics (for correlation estimation)
        
    Returns:
        DataFrame with corrected p-values
    """
    p_values = trend_results['p_value'].values
    
    # Standard FDR correction (Benjamini-Hochberg) - appropriate for correlated tests
    rejected_fdr, pvals_corrected_fdr = fdr_correction(p_values, alpha=0.01)
    
    # Simple Bonferroni for comparison
    rejected_bonf = p_values < (0.01 / len(p_values))
    pvals_corrected_bonf = p_values * len(p_values)
    pvals_corrected_bonf = np.minimum(pvals_corrected_bonf, 1.0)
    
    # Add corrected p-values to results
    trend_results = trend_results.copy()
    trend_results['p_value_fdr'] = pvals_corrected_fdr
    trend_results['p_value_bonferroni'] = pvals_corrected_bonf
    trend_results['significant_fdr'] = rejected_fdr
    trend_results['significant_bonferroni'] = rejected_bonf
    
    # Update trend direction based on corrected significance
    trend_results['trend_direction_fdr'] = trend_results.apply(
        lambda row: ('increasing' if row['slope'] > 0 else 'decreasing') 
        if row['significant_fdr'] else 'no_trend', axis=1
    )
    
    trend_results['trend_direction_bonferroni'] = trend_results.apply(
        lambda row: ('increasing' if row['slope'] > 0 else 'decreasing') 
        if row['significant_bonferroni'] else 'no_trend', axis=1
    )
    
    return trend_results

def format_metric_names(trend_results):
    """
    Add formatted metric names for better visualization.
    
    Args:
        trend_results: DataFrame with trend analysis results
        
    Returns:
        DataFrame with formatted names
    """
    # Mapping for better display names
    name_mapping = {
        'avg_anger': 'Anger',
        'avg_awkwardness': 'Awkwardness', 
        'avg_frustration_or_despair': 'Frustration/Despair',
        'avg_humor': 'Humor',
        'avg_joy_or_excitement': 'Joy/Excitement',
        'avg_sarcasm': 'Sarcasm',
        'avg_self_deprecation': 'Self-deprecation',
        'laughter_count': 'Laughter Count',
        'applause_count': 'Applause Count',
        'greg_mentions': 'Greg Mentions',
        'alex_mentions': 'Alex Mentions',
        'num_sentences': 'Number of Sentences',
        'num_words': 'Number of Words',
        'mean_sentence_length': 'Mean Sentence Length'
    }
    
    trend_results = trend_results.copy()
    trend_results['metric_display'] = trend_results['metric'].map(
        lambda x: name_mapping.get(x, x.replace('_', ' ').title())
    )
    
    return trend_results

def process_sentiment_trends():
    """
    Main processing function that analyzes sentiment trends across series.
    
    Returns:
        Tuple of (series_stats, trend_results) DataFrames
    """
    print("Processing sentiment trends for Figure 7...")
    
    # Load sentiment data
    sentiment_data = load_sentiment_data()
    
    # Calculate series-level statistics
    print("Calculating series-level statistics...")
    series_stats = calculate_series_statistics(sentiment_data)
    print(f"Analyzed {len(series_stats)} series")
    
    # Perform trend analysis
    print("Performing trend analysis...")
    trend_results = perform_trend_analysis(series_stats)
    print(f"Analyzed trends for {len(trend_results)} metrics")
    
    # Apply multiple testing correction
    print("Applying multiple testing correction...")
    trend_results = apply_multiple_testing_correction(trend_results, series_stats)
    
    # Format metric names
    trend_results = format_metric_names(trend_results)
    
    return series_stats, trend_results

def save_processed_data(series_stats, trend_results):
    """Save processed data to CSV files."""
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save trend results
    trend_results.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved trend analysis to {OUTPUT_FILE}")
    
    # Save series statistics
    series_stats_file = OUTPUT_DIR / "figure7_series_statistics.csv"
    series_stats.to_csv(series_stats_file, index=False)
    print(f"Saved series statistics to {series_stats_file}")
    
    # Print summary
    n_significant_fdr = trend_results['significant_fdr'].sum()
    n_significant_bonf = trend_results['significant_bonferroni'].sum()
    n_total = len(trend_results)
    
    print(f"\nSummary:")
    print(f"Total metrics analyzed: {n_total}")
    print(f"Significant trends (FDR corrected): {n_significant_fdr} ({n_significant_fdr/n_total*100:.1f}%)")
    print(f"Significant trends (Bonferroni corrected): {n_significant_bonf} ({n_significant_bonf/n_total*100:.1f}%)")
    
    # Show significant trends
    if n_significant_fdr > 0:
        print(f"\nSignificant trends (FDR corrected):")
        sig_trends = trend_results[trend_results['significant_fdr']].sort_values('p_value_fdr')
        for _, row in sig_trends.iterrows():
            direction = "↗" if row['slope'] > 0 else "↘"
            print(f"  {direction} {row['metric_display']}: slope={row['slope']:.4f}, p={row['p_value_fdr']:.4f}")

if __name__ == "__main__":
    # Process the data
    series_stats, trend_results = process_sentiment_trends()
    
    # Save processed data
    save_processed_data(series_stats, trend_results)
    
    print("Data processing complete!") 