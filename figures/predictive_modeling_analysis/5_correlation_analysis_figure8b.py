#!/usr/bin/env python3
"""
Raw Correlation Analysis: Input Features vs IMDB Scores (Episode Level)
Direct correlation analysis with all available non-constant features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path
import sys

# Add config path
sys.path.append(str(Path(__file__).parent.parent.parent / 'config'))
from plot_utils import apply_plot_style, load_config

# Configuration
EPISODE_DATA_FILE = "episode_data.csv"
OUTPUT_CORRELATIONS_FILE = "raw_correlations.json"
OUTPUT_FIGURE_FILE = "s2_fig"

def load_data():
    """Load the episode-level data for correlation analysis."""
    print("Loading episode-level data for correlation analysis...")
    
    # Load episode data  
    episode_data = pd.read_csv(EPISODE_DATA_FILE)
    print(f"  Episode data loaded: {episode_data.shape}")
    
    return episode_data

def calculate_mean_imdb_scores(episode_data):
    """Calculate mean IMDB scores from histogram data."""
    print("Calculating mean IMDB scores from histograms...")
    
    # Histogram columns
    hist_cols = [f'hist{i}_pct' for i in range(1, 11)]
    rating_values = np.arange(1, 11)  # Ratings 1-10
    
    # Calculate weighted mean IMDB score for each episode
    mean_scores = []
    for idx, row in episode_data.iterrows():
        hist_percentages = row[hist_cols].values / 100.0  # Convert to fractions
        weighted_mean = np.sum(rating_values * hist_percentages)
        mean_scores.append(weighted_mean)
    
    episode_data['mean_imdb_score'] = mean_scores
    
    print(f"  Mean IMDB score range: {min(mean_scores):.2f} - {max(mean_scores):.2f}")
    return episode_data

def identify_input_features(episode_data):
    """Identify all available input features (exclude IMDB histogram data and identifiers)."""
    
    # Exclude histogram columns, identifiers, and metadata
    exclude_cols = (
        [f'hist{i}_pct' for i in range(1, 11)] +  # IMDB histograms (10 columns)
        ['season', 'episode', 'episode_id', 'title', 'imdb_id', 'mean_imdb_score']  # Identifiers and computed target
    )
    
    # Get all potential input features
    all_features = [col for col in episode_data.columns if col not in exclude_cols]
    
    # Filter to numeric features only and check variance
    input_features = []
    for feature in all_features:
        if episode_data[feature].dtype in ['int64', 'float64', 'bool']:
            # Check for sufficient variance (exclude constant features)
            if episode_data[feature].nunique() > 1 and episode_data[feature].std() > 1e-10:
                input_features.append(feature)
            else:
                print(f"    Excluding {feature}: constant or near-constant values")
    
    print(f"  Total columns: {len(episode_data.columns)}")
    print(f"  Excluded columns: {len(exclude_cols)} (histograms + identifiers)")
    print(f"  Potential input features: {len(all_features)}")
    print(f"  Valid input features: {len(input_features)}")
    
    print(f"\n📊 IDENTIFIED {len(input_features)} INPUT FEATURES:")
    for i, feat in enumerate(input_features, 1):
        print(f"    {i:2d}. {feat}")
    
    return input_features

def calculate_correlations(episode_data, input_features):
    """Calculate correlations between input features and mean IMDB scores."""
    print(f"\nCalculating correlations with mean IMDB scores...")
    
    correlations = {}
    valid_correlations = []
    
    target = episode_data['mean_imdb_score']
    
    for feature in input_features:
        try:
            # Calculate Pearson correlation
            corr, p_value = stats.pearsonr(episode_data[feature], target)
            
            # Store if valid (not NaN)
            if not np.isnan(corr):
                correlations[feature] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'abs_correlation': float(abs(corr))
                }
                valid_correlations.append(corr)
            
        except Exception as e:
            print(f"    Warning: Could not calculate correlation for {feature}: {e}")
    
    print(f"  Valid correlations calculated: {len(valid_correlations)}")
    print(f"  Correlation range: {min(valid_correlations):.3f} to {max(valid_correlations):.3f}")
    
    return correlations, valid_correlations

def plot_correlation_distribution(correlations, valid_correlations):
    """Create histogram of correlation distribution for Figure 8b."""
    print("\nCreating Figure 8b: Correlation distribution...")
    
    # Load configuration and apply styling
    config = load_config()
    
    # Create figure
    fig, ax = plt.subplots(figsize=tuple(config['global']['figure_size']))
    
    # Apply consistent styling
    apply_plot_style(fig, ax)
    
    # Create histogram
    n_bins = 15
    counts, bin_edges, patches = ax.hist(
        valid_correlations, 
        bins=n_bins, 
        alpha=0.7, 
        color=config['colors']['highlight']['neutral'],
        edgecolor='black',
        linewidth=0.5,
        density=True
    )
    
    # Fit and plot Gaussian
    mu, sigma = stats.norm.fit(valid_correlations)
    x_range = np.linspace(min(valid_correlations), max(valid_correlations), 100)
    gaussian_fit = stats.norm.pdf(x_range, mu, sigma)
    
    ax.plot(x_range, gaussian_fit, 
            color=config['colors']['highlight']['bad'], 
            linewidth=2, 
            label=f'Gaussian fit (μ={mu:.3f}, σ={sigma:.3f})')
    
    # Add reference lines
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Zero correlation')
    ax.axvline(x=mu, color=config['colors']['highlight']['bad'], linestyle=':', alpha=0.7, label='Mean')
    
    # Styling
    ax.set_xlabel('Correlation Coefficient', fontsize=config['fonts']['axis_label_size'])
    ax.set_ylabel('Density', fontsize=config['fonts']['axis_label_size'])
    ax.legend(fontsize=config['fonts']['legend_size'])
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'N = {len(valid_correlations)} features\nMean = {mu:.3f}\nStd = {sigma:.3f}'
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=config['fonts']['legend_size'])
    
    plt.tight_layout()
    
    return fig

def analyze_strongest_correlations(correlations, top_n=10):
    """Analyze the strongest correlations."""
    print(f"\nAnalyzing top {top_n} strongest correlations...")
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
    
    print(f"  🔝 TOP {top_n} STRONGEST CORRELATIONS:")
    print("     Feature                              | Correlation | P-value")
    print("     " + "-" * 70)
    
    for i, (feature, data) in enumerate(sorted_corr[:top_n], 1):
        corr = data['correlation']
        p_val = data['p_value']
        direction = "📈" if corr > 0 else "📉"
        
        print(f"  {i:2d}. {direction} {feature:35} | {corr:+8.3f} | {p_val:7.3f}")
    
    return sorted_corr[:top_n]

def save_results(correlations, valid_correlations):
    """Save correlation results to JSON file."""
    
    # Prepare summary
    summary = {
        'total_correlations': len(correlations),
        'mean_correlation': float(np.mean(valid_correlations)),
        'std_correlation': float(np.std(valid_correlations)),
        'min_correlation': float(min(valid_correlations)),
        'max_correlation': float(max(valid_correlations)),
        'correlations': correlations
    }
    
    # Save to JSON
    with open(OUTPUT_CORRELATIONS_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {OUTPUT_CORRELATIONS_FILE}")
    
    return summary

def main():
    """Main analysis function."""
    print("="*80)
    print("RAW CORRELATION ANALYSIS: ALL INPUT FEATURES vs IMDB SCORES (EPISODE LEVEL)")
    print("="*80)
    
    # Load data
    episode_data = load_data()
    
    # Calculate mean IMDB scores from histograms
    episode_data = calculate_mean_imdb_scores(episode_data)
    
    # Identify input features
    input_features = identify_input_features(episode_data)
    
    # Calculate correlations
    correlations, valid_correlations = calculate_correlations(episode_data, input_features)
    
    # Analyze strongest correlations
    top_correlations = analyze_strongest_correlations(correlations, top_n=10)
    
    # Create visualization
    fig = plot_correlation_distribution(correlations, valid_correlations)
    
    # Save results
    summary = save_results(correlations, valid_correlations)
    
    # Load config for saving
    config = load_config()
    
    # Save figure as both PNG and PDF with high DPI
    fig.savefig(f"{OUTPUT_FIGURE_FILE}.png", dpi=config['global']['dpi'], bbox_inches='tight')
    fig.savefig(f"{OUTPUT_FIGURE_FILE}.pdf", dpi=config['global']['dpi'], bbox_inches='tight')
    print(f"📊 Figure 8b saved: {OUTPUT_FIGURE_FILE}.png and {OUTPUT_FIGURE_FILE}.pdf")
    
    # Show plot
    plt.show()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Episodes analyzed: {len(episode_data)}")
    print(f"   Input features: {len(valid_correlations)}")
    print(f"   Mean correlation: {np.mean(valid_correlations):.3f}")
    print(f"   Standard deviation: {np.std(valid_correlations):.3f}")
    print(f"   Range: {min(valid_correlations):.3f} to {max(valid_correlations):.3f}")
    
    return correlations, valid_correlations, fig

if __name__ == "__main__":
    correlations, valid_correlations, fig = main() 