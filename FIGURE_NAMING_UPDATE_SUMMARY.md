# Figure Naming Convention Update Summary

## Overview
All figure generation scripts have been updated to use the new simplified naming convention for publication. The changes ensure consistent, clean figure names that are easy to reference in LaTeX documents.

## Updated Naming Convention

### Main Figures
| LaTeX/PDF Figure | Old Filename | New Filename |
|------------------|--------------|--------------|
| Fig 1A | figure1_ridge_output.png | fig1a.png |
| Fig 1B | figure1_pca_output.png | fig1b.png |
| Fig 2 | figure3.png | fig2.png |
| Fig 3 | figure3_series_distribution.png | fig3.png |
| Fig 4 | figure_sup9_spider_plot.png | fig4.png |
| Fig 5 | figure4.png | fig5.png |
| Fig 6 | figure2_output.png | fig6.png |
| Fig 7 | figure5_output.png | fig7.png |
| Fig 8 | figure6.png | fig8.png |
| Fig 9 | figure7a.png | fig9.png |
| Fig 10 | figure8a_episode_ml.png | fig10.png |
| Fig 11 | random_forest_feature_analysis.png | fig11.png |

### Supplementary Figures
| LaTeX/PDF Figure | Old Filename | New Filename |
|------------------|--------------|--------------|
| S1 Fig | figure7b.png | s1_fig.png |
| S2 Fig | figure8b_raw_correlations.png | s2_fig.png |

### Series Figures (S3 Fig)
| LaTeX/PDF Figure | Old Filename | New Filename |
|------------------|--------------|--------------|
| S3 Fig (Series 1) | series_1_deep_dive.png | s3_fig_1.png |
| S3 Fig (Series 2) | series_2_deep_dive.png | s3_fig_2.png |
| ... | ... | ... |
| S3 Fig (Series 18) | series_18_deep_dive.png | s3_fig_18.png |

## Scripts Updated

### 1. Series Ratings Analysis
- **File**: `figures/series_ratings_analysis/plot_seaborn_ridgeline_decomposed.py`
- **Change**: `figure1_ridge_output` → `fig1a`

- **File**: `figures/series_ratings_analysis/plot_series_ratings_analysis.py`
- **Change**: `figure1_pca_output` → `fig1b`

### 2. Task Characteristics Analysis
- **File**: `figures/task_characteristics_analysis/plot_task_characteristics.py`
- **Changes**: 
  - `figure3` → `fig2`
  - `figure3_series_distribution` → `fig3`

### 3. Task Skill Profiles
- **File**: `figures/task_skill_profiles/create_skill_spider_plot.py`
- **Change**: `figure_sup9_spider_plot` → `fig4`

### 4. Geographic Origins
- **File**: `figures/contestant_geographic_origins/plot_geographic_origins.py`
- **Change**: `figure4` → `fig5`

### 5. Episode Rating Trajectories
- **File**: `figures/episode_rating_trajectories/plot_episode_trajectories.py`
- **Change**: `figure2_output` → `fig6`

### 6. Performance Archetypes
- **File**: `figures/contestant_performance_archetypes/plot_performance_archetypes.py`
- **Change**: `figure5_output` → `fig7`

### 7. Scoring Pattern Geometry
- **File**: `figures/scoring_pattern_geometry/plot_scoring_patterns.py`
- **Change**: `figure6` → `fig8`

### 8. Sentiment Trends Analysis
- **File**: `figures/sentiment_trends_analysis/plot_sentiment_trends.py`
- **Changes**:
  - `figure7a` → `fig9`
  - `figure7b` → `s1_fig`

### 9. Predictive Modeling Analysis
- **File**: `figures/predictive_modeling_analysis/4_plot_figure8a.py`
- **Change**: `figure8a_episode_ml` → `fig10`

- **File**: `figures/predictive_modeling_analysis/5_correlation_analysis_figure8b.py`
- **Change**: `figure8b_raw_correlations` → `s2_fig`

- **File**: `figures/predictive_modeling_analysis/6_analyze_random_forest_features.py`
- **Change**: `random_forest_feature_analysis` → `fig11`

- **File**: `figures/predictive_modeling_analysis/run_all.py`
- **Change**: Updated output file references

### 10. Individual Series Analysis
- **File**: `figures/individual_series_analysis/create_series_progression_plots.py`
- **Change**: `series_{n}_deep_dive` → `s3_fig_{n}`

## Copy Script Updated
- **File**: `copy_figures_for_publication.py`
- **Change**: Updated all source and destination paths to match new naming convention
- **Added**: Support for S3 Fig series (1-18)

## Benefits of New Naming Convention

1. **Consistency**: All figures follow a simple `fig{n}` pattern
2. **Clarity**: Figure numbers directly correspond to LaTeX figure numbers
3. **Simplicity**: No complex descriptive names that can become outdated
4. **Organization**: Clear separation between main figures, supplementary figures, and series figures
5. **LaTeX-friendly**: Easy to reference in LaTeX documents with `\ref{fig:1a}`, `\ref{fig:2}`, etc.

## Usage

To generate all figures with the new naming convention:

```bash
# Generate all figures
python generate_all_figures.py

# Copy to root figures directory (optional)
python copy_figures_for_publication.py
```

All figures will be generated in their respective module directories with the new names and can optionally be copied to the root `figures/` directory for easy access. 