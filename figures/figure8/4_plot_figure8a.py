#!/usr/bin/env python3
"""
Stage 4: Plot Figure 8a - Episode-Level ML Results
Create visualization showing model performance comparison.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add config path
sys.path.append(str(Path(__file__).parent.parent.parent / 'config'))
from plot_utils import apply_plot_style, get_palette, load_config

# Configuration
RESULTS_FILE = "episode_model_results.pkl"
FEATURES_FILE = "episode_selected_features.json"
OUTPUT_FILE = "figure8a_episode_ml"

def load_results():
    """Load ML results and feature selection."""
    print("Loading episode-level ML results...")
    
    # Load model results
    with open(RESULTS_FILE, 'rb') as f:
        results = pickle.load(f)
    
    # Load feature selection
    with open(FEATURES_FILE, 'r') as f:
        features = json.load(f)
    
    print(f"  Feature sets evaluated: {len(results['all_results'])}")
    print(f"  Models per set: {len(results['summary_table']) // len(results['all_results'])}")
    print(f"  Total features available: {features['summary']['total_features']}")
    
    return results, features

def create_figure8a(results, features):
    """Create Figure 8a with model performance comparison only."""
    print("\nCreating Figure 8a...")
    
    # Load configuration and apply styling
    config = load_config()
    
    # Create figure with single subplot
    fig, ax = plt.subplots(1, 1, figsize=tuple(config['global']['figure_size']))
    
    # Apply consistent styling
    apply_plot_style(fig, ax)
    
    # Model Performance Comparison
    # Extract data from summary table
    summary_data = results['summary_table']
    
    # Group by model type
    models = {}
    for entry in summary_data:
        model_name = entry['model']
        if model_name not in models:
            models[model_name] = {'test_r2': [], 'cv_r2': [], 'feature_sets': []}
        models[model_name]['test_r2'].append(entry['test_r2'])
        models[model_name]['cv_r2'].append(float(entry['cv_r2']))  # Convert numpy float
        models[model_name]['feature_sets'].append(entry['feature_set'])
    
    # Calculate average performance per model
    model_names = []
    avg_test_r2 = []
    avg_cv_r2 = []
    best_test_r2 = []
    
    for model_name, data in models.items():
        model_names.append(model_name)
        avg_test_r2.append(np.mean(data['test_r2']))
        avg_cv_r2.append(np.mean(data['cv_r2']))
        best_test_r2.append(np.max(data['test_r2']))
    
    x_pos = np.arange(len(model_names))
    width = 0.25
    
    # Use colors from config
    colors = config['colors']['highlight']
    
    bars1 = ax.bar(x_pos - width, avg_cv_r2, width, 
                   label='Avg CV R²', alpha=0.7, color=colors['bad'])  # Red for CV (typically lower)
    bars2 = ax.bar(x_pos, avg_test_r2, width,
                   label='Avg Test R²', alpha=0.7, color=colors['neutral'])  # Blue for average test
    bars3 = ax.bar(x_pos + width, best_test_r2, width,
                   label='Best Test R²', alpha=0.7, color=colors['good'])  # Green for best
    
    ax.set_xlabel('Model Type', fontsize=config['fonts']['axis_label_size'])
    ax.set_ylabel('R² Score', fontsize=config['fonts']['axis_label_size'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=config['fonts']['legend_size'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits to show negative values if any
    all_scores = avg_cv_r2 + avg_test_r2 + best_test_r2
    y_min = min(all_scores) - 0.05
    y_max = max(all_scores) + 0.05
    ax.set_ylim(y_min, y_max)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', 
                    fontsize=config['fonts']['tick_label_size'])
    
    # Highlight best overall performance
    best_overall_idx = np.argmax(best_test_r2)
    best_model = model_names[best_overall_idx]
    best_score = best_test_r2[best_overall_idx]
    
    ax.text(0.02, 0.98, f'Best Model: {best_model}\nBest R² = {best_score:.3f}',
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='gold', alpha=0.7),
             fontsize=config['fonts']['legend_size'], fontweight='bold')
    
    plt.tight_layout()
    
    return fig

def add_summary_stats(results, features):
    """Add summary statistics to the plot."""
    
    # Calculate summary statistics
    n_episodes = results.get('data_info', {}).get('n_episodes', features['summary']['episodes'])
    n_features_total = features['summary']['total_features']
    
    # Find best performance
    best_test_r2 = max([entry['test_r2'] for entry in results['summary_table']])
    best_entry = [e for e in results['summary_table'] if e['test_r2'] == best_test_r2][0]
    
    print(f"\n📊 FIGURE 8A SUMMARY:")
    print(f"   Episodes analyzed: {n_episodes}")
    print(f"   Total features available: {n_features_total}")
    print(f"   Best model: {best_entry['model']} with {best_entry['feature_set']}")
    print(f"   Best test R²: {best_test_r2:.3f}")
    print(f"   Explained variance: {best_test_r2*100:.1f}%")
    
    return {
        'n_episodes': n_episodes,
        'n_features': n_features_total,
        'best_model': best_entry['model'],
        'best_feature_set': best_entry['feature_set'],
        'best_r2': best_test_r2,
        'explained_variance': best_test_r2 * 100
    }

def main():
    """Main plotting function."""
    print("="*60)
    print("PLOTTING FIGURE 8A: EPISODE-LEVEL ML RESULTS")
    print("="*60)
    
    # Load results
    results, features = load_results()
    
    # Create plot
    fig = create_figure8a(results, features)
    
    # Add summary statistics
    summary = add_summary_stats(results, features)
    
    # Load config for saving
    config = load_config()
    
    # Save plot as both PNG and PDF with high DPI
    fig.savefig(OUTPUT_FILE + '.png', dpi=config['global']['dpi'], bbox_inches='tight')
    fig.savefig(OUTPUT_FILE + '.pdf', dpi=config['global']['dpi'], bbox_inches='tight')
    print(f"\n✅ Figure 8a saved: {OUTPUT_FILE}.png and {OUTPUT_FILE}.pdf")
    
    # Show plot
    plt.show()
    
    return fig, summary

if __name__ == "__main__":
    fig, summary = main() 