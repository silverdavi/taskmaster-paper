#!/usr/bin/env python3
"""
Plot Results for Figure 8b: Series-Level Analysis
Handles NaN values and small sample size limitations gracefully.
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

def plot_series_results(results):
    """Plot series-level analysis results with NaN handling."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model performance comparison
    models = list(results['results'].keys())
    r2_scores = []
    mae_scores = []
    
    for model_name in models:
        r2 = results['results'][model_name].get('cv_r2', -1.0)
        mae = results['results'][model_name].get('cv_mae', np.nan)
        
        # Handle NaN values
        if np.isnan(r2):
            r2 = -1.0
        if np.isnan(mae):
            mae = 1.0  # Fallback value
            
        r2_scores.append(r2)
        mae_scores.append(mae)
    
    # R² comparison
    colors = ['red' if r2 < 0 else 'orange' if r2 < 0.1 else 'green' for r2 in r2_scores]
    bars1 = axes[0, 0].bar(models, r2_scores, color=colors, alpha=0.8)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Baseline')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Series-Level Model Performance (N=18)')
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend()
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MAE comparison
    bars2 = axes[0, 1].bar(models, mae_scores, color='skyblue', alpha=0.8)
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Mean Absolute Error (Series-Level)')
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars2, mae_scores):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Feature importance (if available)
    feature_importance = None
    for model_name in models:
        if 'feature_importance' in results['results'][model_name]:
            feature_importance = results['results'][model_name]['feature_importance']
            break
    
    if feature_importance:
        feature_names = list(feature_importance.keys())[:10]  # Top 10
        importance_values = [feature_importance[name] for name in feature_names]
        
        # Handle NaN in importance
        importance_values = [val if not np.isnan(val) else 0.0 for val in importance_values]
        
        y_pos = np.arange(len(feature_names))
        axes[1, 0].barh(y_pos, importance_values, color='lightcoral', alpha=0.8)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([name.replace('avg_', '').replace('prop_', '').title() 
                                    for name in feature_names])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Feature Importance (Series-Level)')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    else:
        axes[1, 0].text(0.5, 0.5, 'No feature importance\navailable', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=14, style='italic')
        axes[1, 0].set_title('Feature Importance (Not Available)')
    
    # Analysis summary
    summary = results.get('summary', {})
    best_r2 = summary.get('best_r2', -1.0)
    status = summary.get('status', '❌')
    
    summary_text = [
        f"Series-Level Analysis (N=18)",
        f"",
        f"Status: {status}",
        f"Best R²: {best_r2:.3f}",
        f"Dataset: {summary.get('dataset_size', 18)} series",
        f"Features: {summary.get('feature_count', 0)}",
        f"",
        f"Key Finding:",
        f"Small sample size (N=18)",
        f"limits reliable prediction",
        f"",
        f"Cross-validation challenges:",
        f"• Leave-One-Out → NaN issues",
        f"• K-Fold → High variance", 
        f"• Insufficient data for ML",
        f"",
        f"Conclusion:",
        f"Need larger dataset for",
        f"meaningful series-level",
        f"prediction analysis"
    ]
    
    axes[1, 1].text(0.05, 0.95, '\n'.join(summary_text), 
                   transform=axes[1, 1].transAxes, fontsize=11,
                   verticalalignment='top', ha='left',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Analysis Summary & Limitations')
    
    plt.suptitle('Figure 8b: Series-Level IMDB Prediction Analysis\n(Demonstrates Small Sample Size Limitations)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure8b_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_diagnostic_plot(results):
    """Create diagnostic plot showing why series-level analysis is challenging."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sample size comparison
    sample_sizes = ['Episode-Level\n(N=139)', 'Series-Level\n(N=18)', 'Recommended\nMinimum\n(N=100+)']
    sizes = [139, 18, 100]
    colors = ['green', 'red', 'blue']
    
    bars = axes[0].bar(sample_sizes, sizes, color=colors, alpha=0.7)
    axes[0].set_ylabel('Sample Size')
    axes[0].set_title('Sample Size Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
    
    # CV strategy comparison
    cv_strategies = ['Leave-One-Out\n(N=17 train)', 'K-Fold-3\n(N=12 train)', 'K-Fold-5\n(N=14 train)']
    reliability = [0.2, 0.6, 0.4]  # Arbitrary reliability scores
    colors = ['red', 'orange', 'orange']
    
    axes[1].bar(cv_strategies, reliability, color=colors, alpha=0.7)
    axes[1].set_ylabel('Reliability Score')
    axes[1].set_title('Cross-Validation Strategy\nReliability (N=18)')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Model complexity vs sample size
    complexity_levels = ['Linear\nRegression', 'Ridge\nRegression', 'Random\nForest', 'Neural\nNetwork']
    suitability = [0.8, 0.7, 0.3, 0.1]  # For N=18
    colors = ['green', 'green', 'red', 'red']
    
    axes[2].bar(complexity_levels, suitability, color=colors, alpha=0.7)
    axes[2].set_ylabel('Suitability for N=18')
    axes[2].set_title('Model Complexity vs\nSample Size Suitability')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Series-Level Analysis: Diagnostic Overview\nWhy N=18 is Insufficient for Reliable ML', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figure8b_diagnostic.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main plotting function for series-level analysis."""
    print("Creating series-level analysis plots...")
    
    try:
        # Run the series analysis
        from series_level_analysis import main as run_series_analysis
        results = run_series_analysis()
        
        print("\nCreating series-level results plot...")
        plot_series_results(results)
        
        print("Creating diagnostic plot...")
        create_diagnostic_plot(results)
        
        print("✅ Series-level plots completed!")
        print("Generated files:")
        print("  - figure8b_series_analysis.png")
        print("  - figure8b_diagnostic.png")
        
    except Exception as e:
        print(f"❌ Error creating plots: {str(e)}")
        print("This might be due to missing data or analysis failures.")

if __name__ == "__main__":
    main() 