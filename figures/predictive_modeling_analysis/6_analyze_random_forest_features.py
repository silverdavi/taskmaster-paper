#!/usr/bin/env python3
"""
Analyze Random Forest Feature Importance and Correlations
Determine which features to maximize/minimize for high IMDB scores.
"""

import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add config path
sys.path.append(str(Path(__file__).parent.parent.parent / 'config'))
from plot_utils import apply_plot_style, get_palette, load_config

def load_data():
    """Load the episode data and model results."""
    print("Loading episode data and Random Forest results...")
    
    # Load model results
    with open('episode_model_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    # Load episode data
    episode_data = pd.read_csv('episode_data.csv')
    
    # Calculate mean IMDB score from histogram data
    hist_cols = ['hist1_pct', 'hist2_pct', 'hist3_pct', 'hist4_pct', 'hist5_pct', 
                 'hist6_pct', 'hist7_pct', 'hist8_pct', 'hist9_pct', 'hist10_pct']
    
    rating_values = np.arange(1, 11)  # Ratings 1-10
    
    # Calculate weighted mean IMDB score for each episode
    mean_scores = []
    for idx, row in episode_data.iterrows():
        hist_percentages = row[hist_cols].values / 100.0  # Convert percentages to fractions
        weighted_mean = np.sum(rating_values * hist_percentages)
        mean_scores.append(weighted_mean)
    
    episode_data['mean_imdb_score'] = mean_scores
    
    # Get Random Forest results for top_5 features
    rf_results = results['all_results']['top_5']['results']['Random Forest']
    features_used = results['all_results']['top_5']['features_used']
    
    print(f"Features used: {features_used}")
    print(f"Feature importance keys: {list(rf_results['feature_importance'].keys())}")
    print(f"Mean IMDB score range: {episode_data['mean_imdb_score'].min():.2f} - {episode_data['mean_imdb_score'].max():.2f}")
    
    return episode_data, rf_results, features_used

def analyze_feature_importance_and_correlations(episode_data, rf_results, features_used):
    """Analyze feature importance and direct correlations with IMDB scores."""
    
    # Get feature importance (it's a dictionary)
    feature_importance_dict = rf_results['feature_importance']
    
    # Calculate direct correlations between features and mean IMDB score
    correlations = {}
    importance_values = []
    
    for feature in features_used:
        if feature in episode_data.columns:
            correlation = episode_data[feature].corr(episode_data['mean_imdb_score'])
            correlations[feature] = correlation
            importance_values.append(float(feature_importance_dict[feature]))
        else:
            correlations[feature] = np.nan
            importance_values.append(float(feature_importance_dict.get(feature, 0)))
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'feature': features_used,
        'importance': importance_values,
        'correlation': [correlations.get(feat, np.nan) for feat in features_used]
    })
    
    # Sort by importance
    analysis_df = analysis_df.sort_values('importance', ascending=False)
    
    return analysis_df

def create_recommendation_summary(analysis_df):
    """Create actionable recommendations for maximizing IMDB scores."""
    
    print("\n" + "="*80)
    print("🎯 ACTIONABLE INSIGHTS: HOW TO MAXIMIZE EPISODE IMDB SCORES")
    print("="*80)
    
    # Features to maximize (positive correlation)
    positive_features = analysis_df[analysis_df['correlation'] > 0].sort_values('importance', ascending=False)
    
    # Features to minimize (negative correlation)  
    negative_features = analysis_df[analysis_df['correlation'] < 0].sort_values('importance', ascending=False)
    
    print("\n📈 FEATURES TO MAXIMIZE (positive correlation with IMDB):")
    print("-" * 60)
    for _, row in positive_features.iterrows():
        feature = row['feature']
        importance = row['importance']
        correlation = row['correlation']
        
        # Clean feature name for readability
        clean_name = feature.replace('_', ' ').title()
        
        print(f"  • {clean_name}")
        print(f"    - Importance: {importance:.3f} (rank {analysis_df[analysis_df['feature'] == feature].index[0] + 1})")
        print(f"    - Correlation: +{correlation:.3f}")
        print(f"    - Strategy: INCREASE this factor")
        print()
    
    print("\n📉 FEATURES TO MINIMIZE (negative correlation with IMDB):")
    print("-" * 60)
    for _, row in negative_features.iterrows():
        feature = row['feature']
        importance = row['importance']
        correlation = row['correlation']
        
        # Clean feature name for readability
        clean_name = feature.replace('_', ' ').title()
        
        print(f"  • {clean_name}")
        print(f"    - Importance: {importance:.3f} (rank {analysis_df[analysis_df['feature'] == feature].index[0] + 1})")
        print(f"    - Correlation: {correlation:.3f}")
        print(f"    - Strategy: DECREASE this factor")
        print()
    
    return positive_features, negative_features

def create_feature_analysis_plot(analysis_df):
    """Create a visualization showing feature importance vs correlation."""
    
    # Load configuration and apply styling
    config = load_config()
    
    # Increase figure width to give more space for labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Apply consistent styling to each axis separately
    apply_plot_style(fig, ax1)
    apply_plot_style(fig, ax2)
    
    # Get colors from config
    colors = config['colors']['highlight']
    
    # Left plot: Feature Importance
    importance_colors = [colors['good'] if corr > 0 else colors['bad'] for corr in analysis_df['correlation']]
    bars1 = ax1.barh(range(len(analysis_df)), analysis_df['importance'], 
                     color=importance_colors, alpha=0.7)
    
    ax1.set_yticks(range(len(analysis_df)))
    ax1.set_yticklabels([feat.replace('_', ' ').title() for feat in analysis_df['feature']])
    ax1.set_xlabel('Feature Importance', fontsize=config['fonts']['axis_label_size'])
    ax1.set_title('Random Forest Feature Importance\n(Green=Positive Correlation, Red=Negative)', 
                  fontsize=config['fonts']['title_size'])
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add importance values
    for i, (bar, imp) in enumerate(zip(bars1, analysis_df['importance'])):
        # Place text on bars with white background for consistency
        x_pos = imp * 0.8  # 80% along the bar
        y_pos = bar.get_y() + bar.get_height()/2
        
        ax1.text(x_pos, y_pos, f'{imp:.3f}', 
                va='center', ha='center', 
                fontsize=config['fonts']['tick_label_size'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Right plot: Correlation with IMDB scores
    correlation_colors = [colors['good'] if corr > 0 else colors['bad'] for corr in analysis_df['correlation']]
    bars2 = ax2.barh(range(len(analysis_df)), analysis_df['correlation'], color=correlation_colors, alpha=0.7)
    
    ax2.set_yticks(range(len(analysis_df)))
    ax2.set_yticklabels([feat.replace('_', ' ').title() for feat in analysis_df['feature']])
    ax2.set_xlabel('Correlation with IMDB Score', fontsize=config['fonts']['axis_label_size'])
    ax2.set_title('Direct Correlation with IMDB Scores\n(Green=Maximize, Red=Minimize)', 
                  fontsize=config['fonts']['title_size'])
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add correlation values with better spacing
    for i, (bar, corr) in enumerate(zip(bars2, analysis_df['correlation'])):
        # Place text on top of bars with white background
        y_pos = bar.get_y() + bar.get_height()/2
        
        # Position text at the end of the bar (inside the bar)
        if corr > 0:
            x_pos = corr * 0.8  # 80% along the bar
        else:
            x_pos = corr * 0.8  # 80% along the bar (negative direction)
        
        ax2.text(x_pos, y_pos, f'{corr:+.3f}', 
                va='center', ha='center', 
                fontsize=config['fonts']['tick_label_size'],
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    
    # Adjust layout to prevent overlapping
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.4)
    
    return fig

def provide_strategic_recommendations(analysis_df, episode_data, rf_results):
    """Provide specific strategic recommendations based on the analysis."""
    
    print("\n" + "="*80)
    print("🏆 STRATEGIC RECOMMENDATIONS FOR HIGH-SCORING EPISODES")
    print("="*80)
    
    # Get the most important features
    top_features = analysis_df.head(3)
    
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        feature = row['feature']
        correlation = row['correlation']
        importance = row['importance']
        
        print(f"\n{i}. {feature.replace('_', ' ').title()} (Importance: {importance:.3f})")
        
        if feature in episode_data.columns:
            # Get statistical summary
            feature_stats = episode_data[feature].describe()
            
            if correlation > 0:
                print(f"   ✅ MAXIMIZE this factor (correlation: +{correlation:.3f})")
                print(f"   📊 Current range: {feature_stats['min']:.2f} to {feature_stats['max']:.2f}")
                print(f"   🎯 Target: Aim for values above {feature_stats['75%']:.2f} (75th percentile)")
            else:
                print(f"   ❌ MINIMIZE this factor (correlation: {correlation:.3f})")
                print(f"   📊 Current range: {feature_stats['min']:.2f} to {feature_stats['max']:.2f}")
                print(f"   🎯 Target: Aim for values below {feature_stats['25%']:.2f} (25th percentile)")
    
    # Overall strategy
    print(f"\n🎯 OVERALL STRATEGY:")
    print(f"   • Focus on the top 3 features (explain {top_features['importance'].sum():.1%} of model decisions)")
    print(f"   • These features alone could potentially improve IMDB scores significantly")
    print(f"   • Random Forest model explains {rf_results['test_r2']:.1%} of IMDB score variance")

def main():
    """Main analysis function."""
    print("="*80)
    print("🤖 RANDOM FOREST FEATURE ANALYSIS FOR IMDB OPTIMIZATION")
    print("="*80)
    
    # Load data
    episode_data, rf_results, features_used = load_data()
    
    # Analyze features
    analysis_df = analyze_feature_importance_and_correlations(episode_data, rf_results, features_used)
    
    print("\n📋 FEATURE ANALYSIS SUMMARY:")
    print(analysis_df.round(3))
    
    # Create recommendations
    positive_features, negative_features = create_recommendation_summary(analysis_df)
    
    # Create visualization
    fig = create_feature_analysis_plot(analysis_df)
    
    # Load config for saving
    config = load_config()
    
    # Save plot as both PNG and PDF with high DPI
    fig.savefig('fig11.png', dpi=config['global']['dpi'], bbox_inches='tight')
    fig.savefig('fig11.pdf', dpi=config['global']['dpi'], bbox_inches='tight')
    print(f"\n✅ Feature analysis plot saved: fig11.png and fig11.pdf")
    
    # Strategic recommendations
    provide_strategic_recommendations(analysis_df, episode_data, rf_results)
    
    plt.show()
    
    return analysis_df, fig

if __name__ == "__main__":
    analysis_df, fig = main() 