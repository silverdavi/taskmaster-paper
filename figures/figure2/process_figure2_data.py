#!/usr/bin/env python3
"""
Process data for Figure 2: Episode Rating Trajectories by Position Patterns

This script:
1. Loads episode ratings data
2. Classifies series by episode rating patterns based on first (1), middle (2), and last (3) episodes
3. Groups episodes by position (First/Middle/Last)
4. Performs statistical analysis on pattern distributions
5. Saves processed data for plotting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import os
import sys
from scipy import stats as scipy_stats

# Set up paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = SCRIPT_DIR
CONFIG_FILE = ROOT_DIR / "config" / "plot_config.yaml"

# Input files
IMDB_RATINGS_FILE = RAW_DATA_DIR / "imdb_ratings.csv"

# Output files
PATTERN_EPISODES_FILE = PROCESSED_DATA_DIR / "episode_patterns.csv"
SERIES_PATTERNS_FILE = PROCESSED_DATA_DIR / "series_patterns.csv"
STATS_FILE = PROCESSED_DATA_DIR / "pattern_statistics.csv"


def load_data():
    """Load the raw data files needed for analysis."""
    try:
        # Load IMDb ratings data
        imdb_df = pd.read_csv(IMDB_RATINGS_FILE)
        print(f"Loaded {len(imdb_df)} episodes with IMDb ratings")
        
        return imdb_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def categorize_episodes(imdb_df):
    """
    Categorize episodes as First, Middle, or Last for each series.
    Returns a DataFrame with episode position information.
    """
    # Create a copy of the dataframe
    episodes = imdb_df.copy()
    
    # Get all series
    all_series = episodes['series'].unique()
    
    # Initialize position columns
    episodes['position'] = ''
    episodes['position_code'] = 0
    
    # For each series, identify first, middle, and last episodes
    for series in all_series:
        # Get episodes for this series
        series_episodes = episodes[episodes['series'] == series].sort_values('episode')
        
        # Get first and last episode numbers
        first_episode = series_episodes['episode'].min()
        last_episode = series_episodes['episode'].max()
        
        # Apply position labels
        episodes.loc[(episodes['series'] == series) & (episodes['episode'] == first_episode), 'position'] = 'First'
        episodes.loc[(episodes['series'] == series) & (episodes['episode'] == first_episode), 'position_code'] = 1
        
        episodes.loc[(episodes['series'] == series) & (episodes['episode'] == last_episode), 'position'] = 'Last'
        episodes.loc[(episodes['series'] == series) & (episodes['episode'] == last_episode), 'position_code'] = 3
        
        episodes.loc[(episodes['series'] == series) & 
                    (episodes['episode'] != first_episode) & 
                    (episodes['episode'] != last_episode), 'position'] = 'Middle'
        episodes.loc[(episodes['series'] == series) & 
                    (episodes['episode'] != first_episode) & 
                    (episodes['episode'] != last_episode), 'position_code'] = 2
    
    return episodes


def identify_rating_patterns(episodes_df):
    """
    Identify rating patterns for each series based on first, middle, and last episodes.
    Patterns are like "123" (rising), "213" (J-shaped), etc.
    """
    # Dictionary to store patterns for each series
    series_patterns = {}
    
    # Get all series
    all_series = episodes_df['series'].unique()
    
    for series in all_series:
        # Get episodes for this series
        series_episodes = episodes_df[episodes_df['series'] == series]
        
        # Get ratings for first, middle, and last positions
        first_rating = series_episodes[series_episodes['position'] == 'First']['imdb_rating'].values[0]
        
        # Calculate average rating for middle episodes
        middle_episodes = series_episodes[series_episodes['position'] == 'Middle']
        if len(middle_episodes) > 0:
            middle_rating = middle_episodes['imdb_rating'].mean()
        else:
            # If there are no middle episodes, use the average of first and last
            last_rating = series_episodes[series_episodes['position'] == 'Last']['imdb_rating'].values[0]
            middle_rating = (first_rating + last_rating) / 2
            
        last_rating = series_episodes[series_episodes['position'] == 'Last']['imdb_rating'].values[0]
        
        # Determine pattern based on ratings
        if first_rating <= middle_rating <= last_rating:
            pattern = "123"  # Rising
        elif first_rating <= last_rating <= middle_rating:
            pattern = "132"  # Rise then slight drop
        elif middle_rating <= first_rating <= last_rating:
            pattern = "213"  # J-shaped (dip then rise)
        elif middle_rating <= last_rating <= first_rating:
            pattern = "231"  # Middle dip then partial recovery
        elif last_rating <= first_rating <= middle_rating:
            pattern = "312"  # Decreasing then up
        elif last_rating <= middle_rating <= first_rating:
            pattern = "321"  # Consistently decreasing
        else:
            pattern = "Unknown"
        
        # Store pattern for this series
        series_patterns[series] = {
            'pattern': pattern,
            'first_rating': first_rating,
            'middle_rating': middle_rating,
            'last_rating': last_rating
        }
    
    return series_patterns


def map_episodes_to_patterns(episodes_df, series_patterns):
    """
    Map each episode to its series pattern and position.
    Returns a DataFrame with pattern and position information for each episode.
    """
    # Create a list to hold episode data
    episode_data = []
    
    for _, episode in episodes_df.iterrows():
        series = episode['series']
        pattern = series_patterns[series]['pattern']
        
        # Add to episode data
        episode_data.append({
            'series': series,
            'episode': episode['episode'],
            'episode_title': episode['episode_title'],
            'imdb_rating': episode['imdb_rating'],
            'pattern': pattern,
            'position': episode['position'],
            'position_code': episode['position_code']
        })
    
    # Create DataFrame
    episode_pattern_df = pd.DataFrame(episode_data)
    
    return episode_pattern_df


def create_series_pattern_summary(series_patterns):
    """
    Create a summary DataFrame of series patterns.
    """
    series_pattern_data = []
    
    for series, data in series_patterns.items():
        series_pattern_data.append({
            'series': series,
            'pattern': data['pattern'],
            'first_rating': data['first_rating'],
            'middle_rating': data['middle_rating'],
            'last_rating': data['last_rating'],
            'rating_diff_first_to_last': data['last_rating'] - data['first_rating'],
            'rating_diff_first_to_middle': data['middle_rating'] - data['first_rating'],
            'rating_diff_middle_to_last': data['last_rating'] - data['middle_rating']
        })
    
    series_pattern_df = pd.DataFrame(series_pattern_data)
    
    return series_pattern_df


def analyze_pattern_significance(series_patterns, series_pattern_df):
    """
    Analyze pattern distributions to determine statistical significance.
    
    This function:
    1. Calculates how many series follow each pattern
    2. Analyzes if certain patterns (e.g., 123, 213) are statistically more common
    3. Tests if rating patterns across series are statistically significant
    """
    # Initialize statistics dictionary
    pattern_stats = {
        'total_series': len(series_patterns),
        'patterns': {},
        'expected_random': 1/6,  # With 6 possible patterns, random would be 1/6 each
        'significance_tests': {}
    }
    
    # Count patterns across all series
    pattern_counts = series_pattern_df['pattern'].value_counts().to_dict()
    total_series = len(series_pattern_df)
    
    # Calculate the percentage for each pattern
    pattern_percentages = {pattern: count/total_series for pattern, count in pattern_counts.items()}
    
    # Store pattern counts and percentages
    pattern_stats['patterns'] = {
        pattern: {
            'count': count, 
            'percentage': pattern_percentages.get(pattern, 0)
        } for pattern, count in pattern_counts.items()
    }
    
    # Focus on specific patterns of interest
    rising_pattern_count = pattern_counts.get('123', 0)
    j_shaped_pattern_count = pattern_counts.get('213', 0)
    
    # Calculate how many series follow either 123 or 213 patterns
    key_patterns_count = rising_pattern_count + j_shaped_pattern_count
    key_patterns_percentage = key_patterns_count / total_series
    
    pattern_stats['key_patterns'] = {
        '123_and_213': {
            'count': key_patterns_count,
            'percentage': key_patterns_percentage,
            'out_of': total_series
        }
    }
    
    # Perform binomial test to see if the observed proportion is significantly different from random
    # Under random distribution, probability of getting either 123 or 213 would be 2/6 = 0.333
    random_probability = 2/6  # Two patterns out of six possible patterns
    
    # Use binomtest instead of binom_test in newer scipy versions
    try:
        # Newer scipy versions use binomtest
        binomtest_result = scipy_stats.binomtest(key_patterns_count, total_series, random_probability)
        p_value = binomtest_result.pvalue
    except AttributeError:
        # Fall back to older binom_test if available
        p_value = scipy_stats.binom_test(key_patterns_count, total_series, random_probability)
    
    pattern_stats['significance_tests']['binomial_test'] = {
        'test': 'Binomial Test for 123 and 213 patterns',
        'observed_count': key_patterns_count,
        'total_series': total_series,
        'observed_proportion': key_patterns_percentage,
        'expected_proportion': random_probability,
        'p_value': p_value,
        'significant_0.05': p_value < 0.05,
        'interpretation': "The proportion of series following '123' or '213' patterns is " +
                         ("significantly" if p_value < 0.05 else "not significantly") +
                         " different from what would be expected by random chance."
    }
    
    # Analyze rating differences between positions
    # Calculate mean differences and run t-tests
    
    # First to Last difference
    mean_first_to_last = series_pattern_df['rating_diff_first_to_last'].mean()
    t_stat, p_val = scipy_stats.ttest_1samp(series_pattern_df['rating_diff_first_to_last'], 0)
    
    pattern_stats['significance_tests']['first_to_last_ttest'] = {
        'test': 'One-sample t-test for First to Last rating difference',
        'mean_difference': mean_first_to_last,
        't_statistic': t_stat,
        'p_value': p_val,
        'significant_0.05': p_val < 0.05,
        'interpretation': f"The average rating difference from First to Last episode is {mean_first_to_last:.2f} and is " +
                         ("significantly" if p_val < 0.05 else "not significantly") +
                         " different from zero."
    }
    
    # First to Middle difference
    mean_first_to_middle = series_pattern_df['rating_diff_first_to_middle'].mean()
    t_stat, p_val = scipy_stats.ttest_1samp(series_pattern_df['rating_diff_first_to_middle'], 0)
    
    pattern_stats['significance_tests']['first_to_middle_ttest'] = {
        'test': 'One-sample t-test for First to Middle rating difference',
        'mean_difference': mean_first_to_middle,
        't_statistic': t_stat,
        'p_value': p_val,
        'significant_0.05': p_val < 0.05,
        'interpretation': f"The average rating difference from First to Middle episodes is {mean_first_to_middle:.2f} and is " +
                         ("significantly" if p_val < 0.05 else "not significantly") +
                         " different from zero."
    }
    
    # Middle to Last difference
    mean_middle_to_last = series_pattern_df['rating_diff_middle_to_last'].mean()
    t_stat, p_val = scipy_stats.ttest_1samp(series_pattern_df['rating_diff_middle_to_last'], 0)
    
    pattern_stats['significance_tests']['middle_to_last_ttest'] = {
        'test': 'One-sample t-test for Middle to Last rating difference',
        'mean_difference': mean_middle_to_last,
        't_statistic': t_stat,
        'p_value': p_val,
        'significant_0.05': p_val < 0.05,
        'interpretation': f"The average rating difference from Middle to Last episode is {mean_middle_to_last:.2f} and is " +
                         ("significantly" if p_val < 0.05 else "not significantly") +
                         " different from zero."
    }
    
    # Chi-square test for pattern distribution
    # For chi-square test, we need the same number of patterns in observed and expected
    # and the same total sum
    
    # Count the number of patterns we have
    num_patterns = len(pattern_counts)
    # Expected frequency for each pattern (assuming uniform distribution)
    expected_per_pattern = total_series / num_patterns
    
    # Prepare observed and expected arrays
    observed = np.array(list(pattern_counts.values()))
    expected = np.full(len(observed), expected_per_pattern)
    
    # Verify the sums match
    if abs(np.sum(observed) - np.sum(expected)) > 1e-10:
        print("Warning: Observed and expected sums don't match exactly")
        print(f"Observed sum: {np.sum(observed)}, Expected sum: {np.sum(expected)}")
    
    # Run chi-square test
    try:
        chi2, p_chi2 = scipy_stats.chisquare(observed, expected)
        
        pattern_stats['significance_tests']['chi_square_test'] = {
            'test': 'Chi-Square Test for Pattern Distribution',
            'chi2': chi2,
            'p_value': p_chi2,
            'significant_0.05': p_chi2 < 0.05,
            'interpretation': "The distribution of episode rating patterns is " +
                            ("significantly" if p_chi2 < 0.05 else "not significantly") +
                            " different from what would be expected by random chance."
        }
    except Exception as e:
        print(f"Warning: Chi-square test failed: {e}")
        pattern_stats['significance_tests']['chi_square_test'] = {
            'test': 'Chi-Square Test for Pattern Distribution',
            'error': str(e),
            'chi2': None,
            'p_value': None,
            'significant_0.05': None,
            'interpretation': "Could not perform chi-square test due to an error."
        }
    
    return pattern_stats


def main():
    """Main processing function."""
    print("Processing data for Figure 2: Episode Rating Trajectories by Position")
    
    # Load data
    imdb_df = load_data()
    
    # Categorize episodes by position
    print("Categorizing episodes by position...")
    episodes_df = categorize_episodes(imdb_df)
    
    # Identify rating patterns
    print("Identifying rating patterns...")
    series_patterns = identify_rating_patterns(episodes_df)
    
    # Map episodes to patterns
    print("Mapping episodes to patterns...")
    episode_pattern_df = map_episodes_to_patterns(episodes_df, series_patterns)
    
    # Create series pattern summary
    series_pattern_df = create_series_pattern_summary(series_patterns)
    
    # Analyze pattern significance
    print("Analyzing pattern significance...")
    pattern_stats = analyze_pattern_significance(series_patterns, series_pattern_df)
    
    # Create output directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save processed data
    episode_pattern_df.to_csv(PATTERN_EPISODES_FILE, index=False)
    series_pattern_df.to_csv(SERIES_PATTERNS_FILE, index=False)
    
    # Save statistics to a CSV file
    stats_df = pd.DataFrame({
        'statistic': [
            'total_series',
            'rising_pattern_count',
            'j_shaped_pattern_count',
            'key_patterns_count',
            'key_patterns_percentage',
            'binomial_test_p_value',
            'binomial_test_significant',
            'first_to_last_mean_diff',
            'first_to_last_p_value',
            'first_to_last_significant',
            'first_to_middle_mean_diff',
            'first_to_middle_p_value',
            'first_to_middle_significant',
            'middle_to_last_mean_diff',
            'middle_to_last_p_value',
            'middle_to_last_significant',
            'chi_square_statistic',
            'chi_square_p_value',
            'chi_square_significant'
        ],
        'value': [
            pattern_stats['total_series'],
            pattern_stats['patterns'].get('123', {}).get('count', 0),
            pattern_stats['patterns'].get('213', {}).get('count', 0),
            pattern_stats['key_patterns']['123_and_213']['count'],
            pattern_stats['key_patterns']['123_and_213']['percentage'],
            pattern_stats['significance_tests']['binomial_test']['p_value'],
            pattern_stats['significance_tests']['binomial_test']['significant_0.05'],
            pattern_stats['significance_tests']['first_to_last_ttest']['mean_difference'],
            pattern_stats['significance_tests']['first_to_last_ttest']['p_value'],
            pattern_stats['significance_tests']['first_to_last_ttest']['significant_0.05'],
            pattern_stats['significance_tests']['first_to_middle_ttest']['mean_difference'],
            pattern_stats['significance_tests']['first_to_middle_ttest']['p_value'],
            pattern_stats['significance_tests']['first_to_middle_ttest']['significant_0.05'],
            pattern_stats['significance_tests']['middle_to_last_ttest']['mean_difference'],
            pattern_stats['significance_tests']['middle_to_last_ttest']['p_value'],
            pattern_stats['significance_tests']['middle_to_last_ttest']['significant_0.05'],
            pattern_stats['significance_tests'].get('chi_square_test', {}).get('chi2', np.nan),
            pattern_stats['significance_tests'].get('chi_square_test', {}).get('p_value', np.nan),
            pattern_stats['significance_tests'].get('chi_square_test', {}).get('significant_0.05', False)
        ]
    })
    
    stats_df.to_csv(STATS_FILE, index=False)
    
    print(f"Saved processed data to {PATTERN_EPISODES_FILE}, {SERIES_PATTERNS_FILE}, and {STATS_FILE}")
    
    # Print summary
    pattern_counts = series_pattern_df['pattern'].value_counts()
    print("\nPattern distribution across series:")
    for pattern, count in pattern_counts.items():
        series_list = series_pattern_df[series_pattern_df['pattern'] == pattern]['series'].tolist()
        print(f"  {pattern} (n={count}): Series {', '.join(map(str, series_list))}")
    
    # Print statistical significance results
    print("\nStatistical Significance Analysis:")
    print(f"Total series analyzed: {pattern_stats['total_series']}")
    
    rising_count = pattern_stats['patterns'].get('123', {}).get('count', 0)
    j_shaped_count = pattern_stats['patterns'].get('213', {}).get('count', 0)
    
    print(f"Rising pattern (123): {rising_count} series ({rising_count/pattern_stats['total_series']*100:.1f}%)")
    print(f"J-shaped pattern (213): {j_shaped_count} series ({j_shaped_count/pattern_stats['total_series']*100:.1f}%)")
    
    key_patterns = pattern_stats['key_patterns']['123_and_213']
    print(f"Series with either pattern: {key_patterns['count']} out of {key_patterns['out_of']} ({key_patterns['percentage']*100:.1f}%)")
    
    binomial_test = pattern_stats['significance_tests']['binomial_test']
    print(f"Binomial test p-value: {binomial_test['p_value']:.4f}")
    print(f"Significance at α=0.05: {'Yes' if binomial_test['significant_0.05'] else 'No'}")
    print(f"Interpretation: {binomial_test['interpretation']}")
    
    # Print t-test results
    print("\nRating Differences:")
    first_to_last = pattern_stats['significance_tests']['first_to_last_ttest']
    print(f"First to Last: {first_to_last['mean_difference']:.2f} (p={first_to_last['p_value']:.4f}, "
          f"{'significant' if first_to_last['significant_0.05'] else 'not significant'})")
    
    first_to_middle = pattern_stats['significance_tests']['first_to_middle_ttest']
    print(f"First to Middle: {first_to_middle['mean_difference']:.2f} (p={first_to_middle['p_value']:.4f}, "
          f"{'significant' if first_to_middle['significant_0.05'] else 'not significant'})")
    
    middle_to_last = pattern_stats['significance_tests']['middle_to_last_ttest']
    print(f"Middle to Last: {middle_to_last['mean_difference']:.2f} (p={middle_to_last['p_value']:.4f}, "
          f"{'significant' if middle_to_last['significant_0.05'] else 'not significant'})")
    
    if 'chi_square_test' in pattern_stats['significance_tests']:
        chi_square = pattern_stats['significance_tests']['chi_square_test']
        if 'error' not in chi_square:
            print(f"\nChi-square test for rating patterns:")
            print(f"Chi-square statistic: {chi_square['chi2']:.4f}")
            print(f"p-value: {chi_square['p_value']:.4f}")
            print(f"Significance at α=0.05: {'Yes' if chi_square['significant_0.05'] else 'No'}")
            print(f"Interpretation: {chi_square['interpretation']}")
    
    print("\nData processing complete!")


if __name__ == "__main__":
    main() 