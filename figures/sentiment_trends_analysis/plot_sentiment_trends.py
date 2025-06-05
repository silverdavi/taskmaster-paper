#!/usr/bin/env python3
"""
Plotting script for Figure 7: Sentiment Trends Analysis

Creates two complementary visualizations:
1. Significant trends plot - shows individual significant trends with confidence intervals
2. All trends distribution plot - compact overview of all trend slopes and significance

Input: figure7_sentiment_trends.csv, figure7_series_statistics.csv
Output: figure7.png, figure7.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib.patches as mpatches

# Configuration
DATA_FILE = Path("figure7_sentiment_trends.csv")
SERIES_FILE = Path("figure7_series_statistics.csv")
OUTPUT_DIR = Path(".")
FIGURE_NAME = "figure7"

# Plot styling
plt.style.use('default')
sns.set_palette("Set2")

def load_processed_data():
    """Load the processed trend analysis data."""
    try:
        trend_data = pd.read_csv(DATA_FILE)
        series_data = pd.read_csv(SERIES_FILE)
        print(f"Loaded trend data: {len(trend_data)} metrics")
        print(f"Loaded series data: {len(series_data)} series")
        return trend_data, series_data
    except FileNotFoundError as e:
        print(f"Error: Could not find processed data files. Run process_data.py first.")
        raise

def load_raw_sentiment_data():
    """Load the raw sentiment dataset for violin plots."""
    try:
        sentiment_data = pd.read_csv("../../data/raw/sentiment.csv")
        print(f"Loaded raw sentiment data: {len(sentiment_data)} episodes")
        return sentiment_data
    except FileNotFoundError:
        print(f"Error: Could not find raw sentiment data file")
        raise

def create_significant_trends_plot(trend_data, series_data, ax):
    """
    Create plot showing significant trends with effect sizes and confidence intervals.
    
    Args:
        trend_data: DataFrame with trend analysis results
        series_data: DataFrame with series-level statistics
        ax: matplotlib axis to plot on
    """
    # Get significant trends (FDR corrected)
    sig_trends = trend_data[trend_data['significant_fdr']].copy()
    
    if len(sig_trends) == 0:
        ax.text(0.5, 0.5, 'No significant trends found\n(after FDR correction)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Significant Sentiment Trends Across Series')
        ax.set_xlabel('Trend Slope (change per series)')
        return
    
    # Sort by standardized effect size (absolute value)
    sig_trends = sig_trends.reindex(
        sig_trends['standardized_slope'].abs().sort_values(ascending=True).index
    )
    
    # Colors for different metric types
    colors = {'sentiment': '#2E86AB', 'other': '#A23B72'}
    
    # Create horizontal effect size plot with confidence intervals
    y_pos = np.arange(len(sig_trends))
    
    # Plot standardized slopes (effect sizes) with confidence intervals
    for i, (_, row) in enumerate(sig_trends.iterrows()):
        color = colors[row['metric_type']]
        
        # Use standardized slope for better interpretability
        effect_size = row['standardized_slope']
        
        # Calculate confidence interval for standardized effect
        # Approximate by scaling the slope CI
        y_std = row['std_value']
        x_range = 18 - 1  # Series range (approximate)
        scale_factor = x_range / y_std if y_std > 0 else 1
        
        ci_lower = effect_size - 1.96 * row['std_err'] * scale_factor
        ci_upper = effect_size + 1.96 * row['std_err'] * scale_factor
        
        # Plot the effect size as a horizontal point with CI
        ax.errorbar(effect_size, i, xerr=[[effect_size - ci_lower], [ci_upper - effect_size]], 
                   fmt='o', color=color, markersize=8, capsize=5, capthick=2, 
                   elinewidth=2, alpha=0.8)
        
        # Add significance stars based on p-value
        p_val = row['p_value_fdr']
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.005:
            stars = '**'
        elif p_val < 0.01:
            stars = '*'
        else:
            stars = ''
        
        # Position stars next to the error bar
        text_x = ci_upper + (np.max([ci_upper for _, r in sig_trends.iterrows()]) - 
                            np.min([ci_lower for _, r in sig_trends.iterrows()])) * 0.05
        
        if stars:
            ax.text(text_x, i, stars, va='center', ha='left', fontsize=12, 
                   fontweight='bold', color=color)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sig_trends['metric_display'], fontsize=10)
    ax.set_xlabel('Standardized Effect Size\n(Change over full series range, in standard deviations)', fontsize=11)
    ax.set_title('Significant Sentiment Trends\n(Effect Sizes with 95% Confidence Intervals)', 
                 fontsize=12, pad=15)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Create legend
    sentiment_patch = mpatches.Patch(color=colors['sentiment'], label='Sentiment Metrics')
    other_patch = mpatches.Patch(color=colors['other'], label='Other Metrics')
    
    # Add significance legend
    from matplotlib.lines import Line2D
    legend_elements = [
        sentiment_patch, other_patch,
        Line2D([0], [0], marker='o', color='black', linestyle='None', 
               markersize=6, label='95% Confidence Interval'),
        Line2D([0], [0], marker='', color='black', linestyle='None', 
               label='Significance: *** p<0.001, ** p<0.005, * p<0.01')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=9)
    
    # Adjust layout
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Add interpretation help
    ax.text(0.02, 0.98, 'Effect Size Interpretation:\n±0.2 = small, ±0.5 = medium, ±0.8 = large', 
            transform=ax.transAxes, va='top', ha='left', fontsize=8, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))

def create_all_trends_plot(trend_data, ax):
    """
    Create compact plot showing distribution of all trend slopes.
    
    Args:
        trend_data: DataFrame with trend analysis results
        ax: matplotlib axis to plot on
    """
    # Prepare data
    sentiment_data = trend_data[trend_data['metric_type'] == 'sentiment'].copy()
    other_data = trend_data[trend_data['metric_type'] == 'other'].copy()
    
    # Create scatter plot
    y_positions = []
    colors = []
    alphas = []
    sizes = []
    labels = []
    slopes = []
    
    # Process sentiment metrics
    for i, (_, row) in enumerate(sentiment_data.iterrows()):
        y_positions.append(i)
        slopes.append(row['slope'])
        
        # Color and alpha based on significance
        if row['significant_fdr']:
            colors.append('#D62728')  # Red for significant
            alphas.append(1.0)
            sizes.append(80)
        elif row['p_value'] < 0.01:  # Uncorrected significance
            colors.append('#FF7F0E')  # Orange for uncorrected
            alphas.append(0.8)
            sizes.append(60)
        else:
            colors.append('#1F77B4')  # Blue for non-significant
            alphas.append(0.5)
            sizes.append(40)
        
        labels.append(row['metric_display'])
    
    # Add separator
    separator_y = len(sentiment_data)
    
    # Process other metrics
    for i, (_, row) in enumerate(other_data.iterrows()):
        y_positions.append(separator_y + 1 + i)
        slopes.append(row['slope'])
        
        # Color and alpha based on significance
        if row['significant_fdr']:
            colors.append('#9467BD')  # Purple for significant
            alphas.append(1.0)
            sizes.append(80)
        elif row['p_value'] < 0.01:  # Uncorrected significance
            colors.append('#C5B0D5')  # Light purple for uncorrected
            alphas.append(0.8)
            sizes.append(60)
        else:
            colors.append('#8C564B')  # Brown for non-significant
            alphas.append(0.5)
            sizes.append(40)
        
        labels.append(row['metric_display'])
    
    # Create scatter plot
    scatter = ax.scatter(slopes, y_positions, c=colors, s=sizes, alpha=alphas, 
                        edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Trend Slope (change per series)', fontsize=11)
    ax.set_title('All Sentiment Trend Slopes\n(Significance and Effect Sizes)', 
                 fontsize=12, pad=15)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add separator line
    ax.axhline(y=separator_y + 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add section labels
    ax.text(-0.02, len(sentiment_data)/2, 'Sentiment\nMetrics', transform=ax.get_yaxis_transform(), 
            ha='right', va='center', fontweight='bold', fontsize=10)
    ax.text(-0.02, separator_y + 1 + len(other_data)/2, 'Other\nMetrics', 
            transform=ax.get_yaxis_transform(), ha='right', va='center', 
            fontweight='bold', fontsize=10)
    
    # Create custom legend
    legend_elements = [
        plt.scatter([], [], c='#D62728', s=80, label='Significant (FDR < 0.01)'),
        plt.scatter([], [], c='#FF7F0E', s=60, alpha=0.8, label='Uncorrected (p < 0.01)'),
        plt.scatter([], [], c='#1F77B4', s=40, alpha=0.5, label='Non-significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Adjust layout
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Invert y-axis to match hierarchical ordering
    ax.invert_yaxis()

def create_violin_plot(raw_sentiment_data, ax, remove_title=False):
    """
    Create violin plots showing distribution of all sentiment metrics across series.
    
    Args:
        raw_sentiment_data: DataFrame with raw episode-level sentiment data
        ax: matplotlib axis to plot on
        remove_title: Whether to remove the title from the plot
    """
    # Sentiment columns to plot
    sentiment_cols = [
        'avg_anger', 'avg_awkwardness', 'avg_frustration_or_despair', 
        'avg_humor', 'avg_joy_or_excitement', 'avg_sarcasm', 'avg_self_deprecation'
    ]
    
    # Display names
    display_names = [
        'Anger', 'Awkwardness', 'Frustration/\nDespair', 
        'Humor', 'Joy/\nExcitement', 'Sarcasm', 'Self-deprecation'
    ]
    
    # Prepare data for violin plots
    violin_data = []
    positions = []
    colors = []
    
    # Color palette - use different color for significant metric (awkwardness)
    base_color = '#7f7f7f'  # Gray for non-significant
    significant_color = '#D62728'  # Red for significant (awkwardness)
    
    for i, (col, name) in enumerate(zip(sentiment_cols, display_names)):
        if col in raw_sentiment_data.columns:
            data = raw_sentiment_data[col].dropna()
            violin_data.append(data)
            positions.append(i)
            
            # Highlight awkwardness (the significant metric)
            if 'awkwardness' in col:
                colors.append(significant_color)
            else:
                colors.append(base_color)
    
    # Create violin plots
    parts = ax.violinplot(violin_data, positions=positions, widths=0.7, 
                         showmeans=True, showmedians=True)
    
    # Color the violin plots
    for i, (pc, color) in enumerate(zip(parts['bodies'], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    # Style other violin plot elements
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1)
    
    # Customize the plot
    ax.set_xticks(positions)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Sentiment Level', fontsize=11)
    
    # Only add title if not removing it
    if not remove_title:
        ax.set_title('Distribution of All Sentiment Metrics\n(Across All Episodes)', 
                     fontsize=12, pad=15)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation for significant metric (smaller for standalone figure)
    awkwardness_idx = next(i for i, col in enumerate(sentiment_cols) if 'awkwardness' in col)
    if not remove_title:
        ax.annotate('Significant trend\n(increasing over time)', 
                    xy=(awkwardness_idx, ax.get_ylim()[1] * 0.9), 
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=significant_color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color=significant_color, lw=1.5))
    else:
        # Simpler annotation for standalone figure
        ax.annotate('Significant trend', 
                    xy=(awkwardness_idx, ax.get_ylim()[1] * 0.95), 
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=significant_color, alpha=0.3))

def create_figure7a():
    """Create Figure 7a: Significant sentiment trend visualization."""
    print("Creating Figure 7a: Significant Sentiment Trend...")
    
    # Load processed data
    trend_data, series_data = load_processed_data()
    
    # Get significant trends (FDR corrected)
    sig_trends = trend_data[trend_data['significant_fdr']].copy()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    if len(sig_trends) == 0:
        ax.text(0.5, 0.5, 'No significant sentiment trends found\n(FDR corrected p < 0.01)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    # Sort by trend direction and magnitude  
    sig_trends = sig_trends.sort_values(['slope'], ascending=False)
    
    # Plot the significant trend (awkwardness)
    trend_row = sig_trends.iloc[0]
    metric = trend_row['metric']
    metric_display = trend_row['metric_display']
    
    # Get series-level data for this metric
    series_means = []
    series_numbers = []
    
    for series_num in sorted(series_data['series'].unique()):
        mean_col = f'{metric}_mean'
        if mean_col in series_data.columns:
            mean_val = series_data[series_data['series'] == series_num][mean_col].iloc[0]
            if not np.isnan(mean_val):
                series_means.append(mean_val)
                series_numbers.append(series_num)
    
    # Create trend visualization
    slope = trend_row['slope']
    p_val = trend_row['p_value_fdr']
    r_squared = trend_row['r_squared']
    
    # Color for increasing sentiment
    color = '#D62728'  # Red
    
    # Plot the individual series points
    ax.plot(series_numbers, series_means, 'o', color=color, markersize=8, alpha=0.7, 
            markeredgecolor='black', markeredgewidth=0.5)
    
    # Add trend line
    x_trend = np.array([min(series_numbers), max(series_numbers)])
    y_trend = trend_row['intercept'] + slope * x_trend
    ax.plot(x_trend, y_trend, '-', color=color, linewidth=3, alpha=0.9)
    
    # Add confidence band
    std_err = trend_row['std_err']
    y_upper = (trend_row['intercept'] + 1.96 * std_err) + slope * x_trend
    y_lower = (trend_row['intercept'] - 1.96 * std_err) + slope * x_trend
    ax.fill_between(x_trend, y_lower, y_upper, color=color, alpha=0.15)
    
    # Add statistics text
    if p_val < 0.001:
        p_text = "p < 0.001"
    else:
        p_text = f"p = {p_val:.3f}"
    
    stats_text = f"{p_text}\nR² = {r_squared:.3f}\nSlope = {slope:.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, va='top', ha='left',
            fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    # Set labels and formatting (no title)
    ax.set_xlabel('Series Number', fontsize=12)
    ax.set_ylabel(f'{metric_display} Level', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Set x-axis to show all series
    ax.set_xlim(0.5, 18.5)
    ax.set_xticks([1, 5, 10, 15, 18])
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_figure7b():
    """Create Figure 7b: Violin plots of all sentiment metrics."""
    print("Creating Figure 7b: All Sentiment Distributions...")
    
    # Load raw sentiment data
    raw_sentiment_data = load_raw_sentiment_data()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create violin plot
    create_violin_plot(raw_sentiment_data, ax, remove_title=True)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def create_figure7():
    """Create both Figure 7a and 7b as separate figures."""
    # Create Figure 7a
    fig7a = create_figure7a()
    
    # Create Figure 7b  
    fig7b = create_figure7b()
    
    return fig7a, fig7b

def save_figures(fig7a, fig7b):
    """Save both figures in multiple formats."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Figure 7a
    png_path_7a = OUTPUT_DIR / "fig9.png"
    pdf_path_7a = OUTPUT_DIR / "fig9.pdf"
    
    fig7a.savefig(png_path_7a, dpi=300, bbox_inches='tight', 
                  facecolor='white', edgecolor='none')
    fig7a.savefig(pdf_path_7a, bbox_inches='tight', 
                  facecolor='white', edgecolor='none')
    
    # Save Figure 7b
    png_path_7b = OUTPUT_DIR / "s1_fig.png"
    pdf_path_7b = OUTPUT_DIR / "s1_fig.pdf"
    
    fig7b.savefig(png_path_7b, dpi=300, bbox_inches='tight', 
                  facecolor='white', edgecolor='none')
    fig7b.savefig(pdf_path_7b, bbox_inches='tight', 
                  facecolor='white', edgecolor='none')
    
    print(f"Saved Figure 7a as {png_path_7a} and {pdf_path_7a}")
    print(f"Saved Figure 7b as {png_path_7b} and {pdf_path_7b}")

def save_figure(fig):
    """Save the figure in multiple formats."""
    # This function is kept for backward compatibility but now unused
    pass

def print_summary(trend_data):
    """Print summary of the analysis."""
    print("\n" + "="*60)
    print("FIGURE 7 SUMMARY: SENTIMENT TRENDS ANALYSIS")
    print("="*60)
    
    n_total = len(trend_data)
    n_sig_fdr = trend_data['significant_fdr'].sum()
    n_sig_uncorr = (trend_data['p_value'] < 0.01).sum()
    
    print(f"Total metrics analyzed: {n_total}")
    print(f"Significant trends (uncorrected): {n_sig_uncorr} ({n_sig_uncorr/n_total*100:.1f}%)")
    print(f"Significant trends (FDR corrected): {n_sig_fdr} ({n_sig_fdr/n_total*100:.1f}%)")
    
    # Breakdown by metric type
    sentiment_trends = trend_data[trend_data['metric_type'] == 'sentiment']
    other_trends = trend_data[trend_data['metric_type'] == 'other']
    
    print(f"\nSentiment metrics: {len(sentiment_trends)} total, {sentiment_trends['significant_fdr'].sum()} significant")
    print(f"Other metrics: {len(other_trends)} total, {other_trends['significant_fdr'].sum()} significant")
    
    # Show significant trends
    if n_sig_fdr > 0:
        print(f"\nSignificant trends (FDR corrected):")
        sig_trends = trend_data[trend_data['significant_fdr']].sort_values('p_value_fdr')
        for _, row in sig_trends.iterrows():
            direction = "↗ Increasing" if row['slope'] > 0 else "↘ Decreasing"
            print(f"  {direction}: {row['metric_display']}")
            print(f"    Slope: {row['slope']:.4f}, R² = {row['r_squared']:.3f}, p = {row['p_value_fdr']:.4f}")
    else:
        print("\nNo significant trends found after FDR correction.")
    
    print("="*60)

if __name__ == "__main__":
    # Create and save the figures
    fig7a, fig7b = create_figure7()
    save_figures(fig7a, fig7b)
    
    # Load data for summary
    trend_data, _ = load_processed_data()
    print_summary(trend_data)
    
    # Show the plots
    plt.show()
    
    print("Figures 7a and 7b complete!") 