#!/usr/bin/env python3
"""
Copy all generated figures to the figures/all_figures directory with publication-ready names.

This script copies figures from their module subdirectories to the figures/all_figures/
directory with the standardized naming convention for publication.
"""

import shutil
from pathlib import Path

def copy_figures_for_publication():
    """Copy all figures to figures/all_figures directory with publication names."""
    
    # Define the mapping from source to destination (now to all_figures subfolder)
    figure_mapping = {
        # Fig 1A - Ridge plot
        'figures/series_ratings_analysis/fig1a.png': 'figures/all_figures/fig1a.png',
        'figures/series_ratings_analysis/fig1a.pdf': 'figures/all_figures/fig1a.pdf',
        
        # Fig 1B - PCA plot  
        'figures/series_ratings_analysis/fig1b.png': 'figures/all_figures/fig1b.png',
        'figures/series_ratings_analysis/fig1b.pdf': 'figures/all_figures/fig1b.pdf',
        
        # Fig 2 - Task type distribution
        'figures/task_characteristics_analysis/fig2.png': 'figures/all_figures/fig2.png',
        'figures/task_characteristics_analysis/fig2.pdf': 'figures/all_figures/fig2.pdf',
        
        # Fig 3 - Series distribution
        'figures/task_characteristics_analysis/fig3.png': 'figures/all_figures/fig3.png',
        'figures/task_characteristics_analysis/fig3.pdf': 'figures/all_figures/fig3.pdf',
        
        # Fig 4 - Spider plot
        'figures/task_skill_profiles/fig4.png': 'figures/all_figures/fig4.png',
        'figures/task_skill_profiles/fig4.pdf': 'figures/all_figures/fig4.pdf',
        
        # Fig 5 - Geographic origins
        'figures/contestant_geographic_origins/fig5.png': 'figures/all_figures/fig5.png',
        'figures/contestant_geographic_origins/fig5.pdf': 'figures/all_figures/fig5.pdf',
        
        # Fig 6 - Episode trajectories
        'figures/episode_rating_trajectories/fig6.png': 'figures/all_figures/fig6.png',
        'figures/episode_rating_trajectories/fig6.pdf': 'figures/all_figures/fig6.pdf',
        
        # Fig 7 - Performance archetypes
        'figures/contestant_performance_archetypes/fig7.png': 'figures/all_figures/fig7.png',
        'figures/contestant_performance_archetypes/fig7.pdf': 'figures/all_figures/fig7.pdf',
        
        # Fig 8 - Scoring patterns
        'figures/scoring_pattern_geometry/fig8.png': 'figures/all_figures/fig8.png',
        'figures/scoring_pattern_geometry/fig8.pdf': 'figures/all_figures/fig8.pdf',
        
        # Fig 9 - Sentiment trends
        'figures/sentiment_trends_analysis/fig9.png': 'figures/all_figures/fig9.png',
        'figures/sentiment_trends_analysis/fig9.pdf': 'figures/all_figures/fig9.pdf',
        
        # Fig 10 - Episode ML prediction
        'figures/predictive_modeling_analysis/fig10.png': 'figures/all_figures/fig10.png',
        'figures/predictive_modeling_analysis/fig10.pdf': 'figures/all_figures/fig10.pdf',
        
        # Fig 11 - Random forest features
        'figures/predictive_modeling_analysis/fig11.png': 'figures/all_figures/fig11.png',
        'figures/predictive_modeling_analysis/fig11.pdf': 'figures/all_figures/fig11.pdf',
        
        # S1 Fig - Sentiment trends supplementary
        'figures/sentiment_trends_analysis/s1_fig.png': 'figures/all_figures/s1_fig.png',
        'figures/sentiment_trends_analysis/s1_fig.pdf': 'figures/all_figures/s1_fig.pdf',
        
        # S2 Fig - Correlation analysis
        'figures/predictive_modeling_analysis/s2_fig.png': 'figures/all_figures/s2_fig.png',
        'figures/predictive_modeling_analysis/s2_fig.pdf': 'figures/all_figures/s2_fig.pdf',
    }
    
    # Add S3 Fig series (1-18)
    for i in range(1, 19):
        figure_mapping[f'figures/individual_series_analysis/s3_fig_{i}.png'] = f'figures/all_figures/s3_fig_{i}.png'
        figure_mapping[f'figures/individual_series_analysis/s3_fig_{i}.pdf'] = f'figures/all_figures/s3_fig_{i}.pdf'
    
    print("Copying figures to figures/all_figures directory for publication...")
    print("=" * 60)
    
    copied_count = 0
    missing_count = 0
    
    for source_path, dest_path in figure_mapping.items():
        source = Path(source_path)
        dest = Path(dest_path)
        
        if source.exists():
            # Create destination directory if it doesn't exist
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source, dest)
            print(f"✅ Copied: {source} → {dest}")
            copied_count += 1
        else:
            print(f"❌ Missing: {source}")
            missing_count += 1
    
    print("=" * 60)
    print(f"Summary: {copied_count} files copied, {missing_count} files missing")
    
    if missing_count > 0:
        print("\nTo generate missing figures, run:")
        print("python generate_all_figures.py")
    
    print("\nPublication-ready figures are now available in the figures/all_figures/ directory!")
    print("\nFigure naming convention:")
    print("• Main figures: fig1a.png, fig1b.png, fig2.png, ..., fig11.png")
    print("• Supplementary figures: s1_fig.png, s2_fig.png")
    print("• Series figures: s3_fig_1.png, s3_fig_2.png, ..., s3_fig_18.png")
    
    # Show directory structure
    all_figures_dir = Path("figures/all_figures")
    if all_figures_dir.exists():
        print(f"\n📁 Directory structure:")
        print(f"figures/all_figures/")
        
        # Count files by type
        main_figs = len(list(all_figures_dir.glob("fig*.png")))
        supp_figs = len(list(all_figures_dir.glob("s1_fig.png"))) + len(list(all_figures_dir.glob("s2_fig.png")))
        series_figs = len(list(all_figures_dir.glob("s3_fig_*.png")))
        
        print(f"├── Main figures: {main_figs} PNG files")
        print(f"├── Supplementary figures: {supp_figs} PNG files")
        print(f"└── Series figures: {series_figs} PNG files")
        print(f"Total: {main_figs + supp_figs + series_figs} PNG files (+ PDF versions)")

if __name__ == "__main__":
    copy_figures_for_publication() 