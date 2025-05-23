#!/usr/bin/env python3
"""
Plot Results for Figure 8: Episode-Level Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load analysis results."""
    with open('analysis_results.pkl', 'rb') as f:
        return pickle.load(f)

def plot_model_comparison(results):
    """Plot model performance comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ['Keras NN', 'Random Forest']
    sets = ['train', 'validation', 'test']
    
    # R² comparison
    for i, model_name in enumerate(['keras', 'random_forest']):
        r2_scores = [results['results'][model_name][s]['r2'] for s in sets]
        axes[0].plot(sets, r2_scores, 'o-', label=models[i], linewidth=2, markersize=8)
    
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Model Performance: R² Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    
    # MAE comparison
    for i, model_name in enumerate(['keras', 'random_forest']):
        mae_scores = [results['results'][model_name][s]['mae'] for s in sets]
        axes[1].plot(sets, mae_scores, 'o-', label=models[i], linewidth=2, markersize=8)
    
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Model Performance: MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Feature Importance
    feature_names = list(results['feature_importance'].keys())
    importance_values = list(results['feature_importance'].values())
    
    y_pos = np.arange(len(feature_names))
    axes[2].barh(y_pos, importance_values, color='skyblue', alpha=0.8)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([name.replace('avg_', '') for name in feature_names])
    axes[2].set_xlabel('Feature Importance')
    axes[2].set_title('Random Forest Feature Importance')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('episode_level_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary_figure(results):
    """Create a summary figure showing key findings."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model comparison bar chart
    models = ['Keras NN', 'Random Forest']
    test_r2 = [results['results']['keras']['test']['r2'], 
               results['results']['random_forest']['test']['r2']]
    test_mae = [results['results']['keras']['test']['mae'], 
                results['results']['random_forest']['test']['mae']]
    
    x = np.arange(len(models))
    axes[0, 0].bar(x, test_r2, color=['orange', 'green'], alpha=0.8)
    axes[0, 0].set_ylabel('Test R² Score')
    axes[0, 0].set_title('Model Performance on Test Set')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # MAE comparison
    axes[0, 1].bar(x, test_mae, color=['orange', 'green'], alpha=0.8)
    axes[0, 1].set_ylabel('Test MAE')
    axes[0, 1].set_title('Mean Absolute Error on Test Set')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Feature importance
    feature_names = list(results['feature_importance'].keys())
    importance_values = list(results['feature_importance'].values())
    
    axes[1, 0].barh(range(len(feature_names)), importance_values, 
                    color='skyblue', alpha=0.8)
    axes[1, 0].set_yticks(range(len(feature_names)))
    axes[1, 0].set_yticklabels([name.replace('avg_', '').title() for name in feature_names])
    axes[1, 0].set_xlabel('Importance Score')
    axes[1, 0].set_title('Feature Importance (Random Forest)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Data summary
    data_info = [
        f"Episodes: {results['data']['X_train'].shape[0] + results['data']['X_val'].shape[0] + results['data']['X_test'].shape[0]}",
        f"Features: {len(results['feature_importance'])}",
        f"Series: 1-18",
        f"Train/Val/Test: {results['data']['X_train'].shape[0]}/{results['data']['X_val'].shape[0]}/{results['data']['X_test'].shape[0]}",
        "",
        "Key Finding:",
        "Episode sentiment cannot",
        "predict IMDB ratings",
        "",
        f"Best R² = {max(test_r2):.3f}",
        "(Worse than baseline)"
    ]
    
    axes[1, 1].text(0.1, 0.9, '\n'.join(data_info), transform=axes[1, 1].transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Analysis Summary')
    
    plt.suptitle('Figure 8a: Episode-Level IMDB Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure8a_episode_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main plotting function."""
    print("Loading results and creating plots...")
    
    results = load_results()
    
    print("Creating model comparison plot...")
    plot_model_comparison(results)
    
    print("Creating summary figure...")
    plot_summary_figure(results)
    
    print("✅ All plots saved successfully!")
    print("Generated files:")
    print("  - episode_level_performance.png")
    print("  - figure8a_episode_summary.png")

if __name__ == "__main__":
    main() 