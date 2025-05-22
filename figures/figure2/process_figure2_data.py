#!/usr/bin/env python3
"""
Process data for Figure 2: Episode Rating Trajectories by Contestant Ranking Patterns

This script:
1. Loads episode ratings and contestant scores
2. Classifies series by contestant ranking patterns
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
SCORES_FILE = RAW_DATA_DIR / "scores.csv"

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
        
        # Load contestant scores data
        scores_df = pd.read_csv(SCORES_FILE)
        print(f"Loaded {len(scores_df)} task scores")
        
        return imdb_df, scores_df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def process_contestant_rankings(scores_df):
    """
    Calculate contestant rankings for each episode and series.
    Returns a DataFrame with rankings by series and episode.
    """
    # Group by task_episode_id and contestant to get total score per contestant per episode
    episode_scores = scores_df.groupby(['series', 'episode', 'contestant_name'])['total_score'].sum().reset_index()
    
    # Calculate rankings within each episode
    episode_rankings = episode_scores.copy()
    episode_rankings['rank'] = episode_rankings.groupby(['series', 'episode'])['total_score'].rank(method='min', ascending=False)
    
    # Calculate average rank for each contestant within their series
    contestant_avg_rank = episode_rankings.groupby(['series', 'contestant_name'])['rank'].mean().reset_index()
    
    # For each series, assign First (1), Middle (2), Last (3) status to contestants
    # based on their average rank across the series
    contestant_positions = contestant_avg_rank.copy()
    
    # Get the number of contestants in each series to determine position boundaries
    num_contestants = contestant_positions.groupby('series').size().reset_index(name='count')
    contestant_positions = contestant_positions.merge(num_contestants, on='series')
    
    # Assign position category (First, Middle, Last) based on average rank
    def assign_position(row):
        if row['count'] <= 3:
            # For series with 3 or fewer contestants, simple assignment
            if row['rank'] == 1:
                return 1  # First
            elif row['rank'] == row['count']:
                return 3  # Last
            else:
                return 2  # Middle
        else:
            # For series with more contestants, divide into three groups
            if row['rank'] <= row['count'] / 3:
                return 1  # First
            elif row['rank'] > 2 * row['count'] / 3:
                return 3  # Last
            else:
                return 2  # Middle
    
    contestant_positions['position'] = contestant_positions.apply(assign_position, axis=1)
    
    return episode_rankings, contestant_positions


def identify_ranking_patterns(episode_rankings, contestant_positions):
    """
    Identify ranking patterns for each series.
    Patterns like "123" (rising), "213" (J-shaped), etc.
    """
    # For each contestant, check their position pattern across episodes
    patterns_by_series = {}
    
    # Get all series
    series_list = episode_rankings['series'].unique()
    
    for series in series_list:
        # Get contestants in this series
        series_contestants = contestant_positions[contestant_positions['series'] == series]
        
        # Get episodes for this series
        series_episodes = episode_rankings[episode_rankings['series'] == series]
        
        # Sort episodes
        sorted_episodes = sorted(series_episodes['episode'].unique())
        
        # Divide episodes into three parts: First, Middle, Last
        first_third = sorted_episodes[:len(sorted_episodes)//3]
        last_third = sorted_episodes[-len(sorted_episodes)//3:]
        middle_third = sorted_episodes[len(sorted_episodes)//3:-len(sorted_episodes)//3]
        
        # Ensure middle third is not empty
        if not middle_third:
            # If we only have 2 parts, assign the middle episode to middle_third
            middle_index = len(sorted_episodes) // 2
            middle_third = [sorted_episodes[middle_index]]
            first_third = sorted_episodes[:middle_index]
            last_third = sorted_episodes[middle_index+1:]
        
        # For each contestant, get their average rank in each third
        contestant_patterns = {}
        
        for _, contestant_row in series_contestants.iterrows():
            contestant_name = contestant_row['contestant_name']
            
            # Get average rank in each third
            first_rank = series_episodes[
                (series_episodes['contestant_name'] == contestant_name) & 
                (series_episodes['episode'].isin(first_third))
            ]['rank'].mean()
            
            middle_rank = series_episodes[
                (series_episodes['contestant_name'] == contestant_name) & 
                (series_episodes['episode'].isin(middle_third))
            ]['rank'].mean()
            
            last_rank = series_episodes[
                (series_episodes['contestant_name'] == contestant_name) & 
                (series_episodes['episode'].isin(last_third))
            ]['rank'].mean()
            
            # Determine pattern based on ranks
            if first_rank <= middle_rank <= last_rank:
                pattern = "123"  # Rising (getting worse)
            elif first_rank <= last_rank <= middle_rank:
                pattern = "132"  # Rise then slight improvement
            elif middle_rank <= first_rank <= last_rank:
                pattern = "213"  # J-shaped (dip then continue declining)
            elif middle_rank <= last_rank <= first_rank:
                pattern = "231"  # Middle improvement then decline
            elif last_rank <= first_rank <= middle_rank:
                pattern = "312"  # Improving significantly at end
            elif last_rank <= middle_rank <= first_rank:
                pattern = "321"  # Consistently improving
            else:
                pattern = "Unknown"
            
            contestant_patterns[contestant_name] = {
                'pattern': pattern,
                'first_rank': first_rank,
                'middle_rank': middle_rank,
                'last_rank': last_rank
            }
        
        # Determine the dominant pattern for this series
        pattern_counts = {}
        for contestant, data in contestant_patterns.items():
            pattern = data['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Find the most common pattern
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        
        # Store series pattern
        patterns_by_series[series] = {
            'dominant_pattern': dominant_pattern,
            'contestant_patterns': contestant_patterns,
            'first_episodes': first_third,
            'middle_episodes': middle_third,
            'last_episodes': last_third,
            'pattern_distribution': pattern_counts
        }
    
    return patterns_by_series


def map_episodes_to_patterns(imdb_df, patterns_by_series):
    """
    Map each episode to its pattern and position.
    Returns a DataFrame with pattern and position information for each episode.
    """
    # Create a list to hold episode data
    episode_data = []
    
    for series in patterns_by_series:
        series_pattern = patterns_by_series[series]['dominant_pattern']
        
        # Get episodes for this series
        series_episodes = imdb_df[imdb_df['series'] == series]
        
        for _, episode_row in series_episodes.iterrows():
            episode_num = episode_row['episode']
            
            # Determine position (First, Middle, Last)
            if episode_num in patterns_by_series[series]['first_episodes']:
                position = "First"
                position_code = 1
            elif episode_num in patterns_by_series[series]['middle_episodes']:
                position = "Middle"
                position_code = 2
            elif episode_num in patterns_by_series[series]['last_episodes']:
                position = "Last"
                position_code = 3
            else:
                position = "Unknown"
                position_code = 0
            
            # Add to episode data
            episode_data.append({
                'series': series,
                'episode': episode_num,
                'episode_title': episode_row['episode_title'],
                'imdb_rating': episode_row['imdb_rating'],
                'pattern': series_pattern,
                'position': position,
                'position_code': position_code
            })
    
    # Create DataFrame
    episode_pattern_df = pd.DataFrame(episode_data)
    
    return episode_pattern_df


def create_series_pattern_summary(patterns_by_series):
    """
    Create a summary DataFrame of series patterns.
    """
    series_pattern_data = []
    
    for series, data in patterns_by_series.items():
        series_pattern_data.append({
            'series': series,
            'pattern': data['dominant_pattern'],
            'num_first_episodes': len(data['first_episodes']),
            'num_middle_episodes': len(data['middle_episodes']),
            'num_last_episodes': len(data['last_episodes'])
        })
    
    series_pattern_df = pd.DataFrame(series_pattern_data)
    
    return series_pattern_df


def analyze_pattern_significance(patterns_by_series, series_pattern_df):
    """
    Analyze pattern distributions to determine statistical significance.
    
    This function:
    1. Calculates how many series follow each pattern
    2. Analyzes if certain patterns (e.g., 123, 213) are statistically more common
    3. Tests if contestant patterns within series are statistically significant
    """
    # Initialize statistics dictionary
    pattern_stats = {
        'total_series': len(patterns_by_series),
        'total_contestants': 0,
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
    
    # Also analyze within-series contestant patterns
    contestant_pattern_counts = {}
    total_contestants = 0
    
    # Count pattern occurrences across all contestants
    for series, data in patterns_by_series.items():
        for contestant, contestant_data in data['contestant_patterns'].items():
            pattern = contestant_data['pattern']
            contestant_pattern_counts[pattern] = contestant_pattern_counts.get(pattern, 0) + 1
            total_contestants += 1
    
    pattern_stats['total_contestants'] = total_contestants
    
    # Calculate percentages for contestant patterns
    contestant_pattern_percentages = {pattern: count/total_contestants 
                                     for pattern, count in contestant_pattern_counts.items()}
    
    pattern_stats['contestant_patterns'] = {
        pattern: {
            'count': count, 
            'percentage': contestant_pattern_percentages.get(pattern, 0)
        } for pattern, count in contestant_pattern_counts.items()
    }
    
    # Chi-square test for contestant pattern distribution
    # For chi-square test, we need the same number of patterns in observed and expected
    # and the same total sum
    
    # Count the number of patterns we have
    num_patterns = len(contestant_pattern_counts)
    # Expected frequency for each pattern (assuming uniform distribution)
    expected_per_pattern = total_contestants / num_patterns
    
    # Prepare observed and expected arrays
    observed = np.array(list(contestant_pattern_counts.values()))
    expected = np.full(len(observed), expected_per_pattern)
    
    # Verify the sums match (should be total_contestants for both)
    if abs(np.sum(observed) - np.sum(expected)) > 1e-10:
        print("Warning: Observed and expected sums don't match exactly")
        print(f"Observed sum: {np.sum(observed)}, Expected sum: {np.sum(expected)}")
    
    # Run chi-square test
    try:
        chi2, p_chi2 = scipy_stats.chisquare(observed, expected)
        
        pattern_stats['significance_tests']['chi_square_test'] = {
            'test': 'Chi-Square Test for Contestant Pattern Distribution',
            'chi2': chi2,
            'p_value': p_chi2,
            'significant_0.05': p_chi2 < 0.05,
            'interpretation': "The distribution of contestant ranking patterns is " +
                            ("significantly" if p_chi2 < 0.05 else "not significantly") +
                            " different from what would be expected by random chance."
        }
    except Exception as e:
        print(f"Warning: Chi-square test failed: {e}")
        pattern_stats['significance_tests']['chi_square_test'] = {
            'test': 'Chi-Square Test for Contestant Pattern Distribution',
            'error': str(e),
            'chi2': None,
            'p_value': None,
            'significant_0.05': None,
            'interpretation': "Could not perform chi-square test due to an error."
        }
    
    return pattern_stats


def main():
    """Main processing function."""
    print("Processing data for Figure 2: Episode Rating Trajectories")
    
    # Load data
    imdb_df, scores_df = load_data()
    
    # Process contestant rankings
    print("Calculating contestant rankings...")
    episode_rankings, contestant_positions = process_contestant_rankings(scores_df)
    
    # Identify ranking patterns
    print("Identifying ranking patterns...")
    patterns_by_series = identify_ranking_patterns(episode_rankings, contestant_positions)
    
    # Map episodes to patterns
    print("Mapping episodes to patterns...")
    episode_pattern_df = map_episodes_to_patterns(imdb_df, patterns_by_series)
    
    # Create series pattern summary
    series_pattern_df = create_series_pattern_summary(patterns_by_series)
    
    # Analyze pattern significance
    print("Analyzing pattern significance...")
    pattern_stats = analyze_pattern_significance(patterns_by_series, series_pattern_df)
    
    # Create output directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Save processed data
    episode_pattern_df.to_csv(PATTERN_EPISODES_FILE, index=False)
    series_pattern_df.to_csv(SERIES_PATTERNS_FILE, index=False)
    
    # Save statistics to a JSON file
    stats_df = pd.DataFrame({
        'statistic': [
            'total_series',
            'total_contestants',
            'rising_pattern_count',
            'j_shaped_pattern_count',
            'key_patterns_count',
            'key_patterns_percentage',
            'binomial_test_p_value',
            'binomial_test_significant',
            'chi_square_statistic',
            'chi_square_p_value',
            'chi_square_significant'
        ],
        'value': [
            pattern_stats['total_series'],
            pattern_stats['total_contestants'],
            pattern_stats['patterns'].get('123', {}).get('count', 0),
            pattern_stats['patterns'].get('213', {}).get('count', 0),
            pattern_stats['key_patterns']['123_and_213']['count'],
            pattern_stats['key_patterns']['123_and_213']['percentage'],
            pattern_stats['significance_tests']['binomial_test']['p_value'],
            pattern_stats['significance_tests']['binomial_test']['significant_0.05'],
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
    
    if 'chi_square_test' in pattern_stats['significance_tests']:
        chi_square = pattern_stats['significance_tests']['chi_square_test']
        print(f"\nChi-square test for contestant patterns:")
        print(f"Chi-square statistic: {chi_square['chi2']:.4f}")
        print(f"p-value: {chi_square['p_value']:.4f}")
        print(f"Significance at α=0.05: {'Yes' if chi_square['significant_0.05'] else 'No'}")
        print(f"Interpretation: {chi_square['interpretation']}")
    
    print("\nData processing complete!")


if __name__ == "__main__":
    main() 