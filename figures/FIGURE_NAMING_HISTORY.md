# Figure Naming History and Reference

This document preserves the original figure numbering system and maps it to the new descriptive folder names. This maintains the logical order and progression of the analysis while using more intuitive naming.

## Original → Descriptive Mapping

| Original Name | New Descriptive Name | Description |
|---------------|---------------------|-------------|
| `figure1` | `series_ratings_analysis` | IMDb ratings analysis with ridge plots and PCA |
| `figure2` | `episode_rating_trajectories` | Rating patterns across episode positions |
| `figure3` | `task_characteristics_analysis` | Task activity types vs judgment patterns |
| `figure4` | `contestant_geographic_origins` | Geographic distribution of contestant birthplaces |
| `figure5` | `contestant_performance_archetypes` | Performance clustering and dynamics |
| `figure6` | `scoring_pattern_geometry` | Geometric analysis of scoring distributions |
| `figure7` | `sentiment_trends_analysis` | Sentiment evolution over time |
| `figure8` | `predictive_modeling_analysis` | ML and correlation analysis |
| `figure_sup9` | `task_skill_profiles` | Spider plots of task skill requirements |
| `figure_sup_series` | `individual_series_analysis` | Detailed series progression analysis |

## Detailed Plot/Subplot Inventory

### Figure 1: `series_ratings_analysis`
- **Panel A**: Ridge plot showing rating distributions per series
  - Gaussian fits for ratings 2-9
  - Red spikes for 1-star ratings
  - Green spikes for 10-star ratings
  - IMDb official ratings with yellow styling
- **Panel B**: PCA plot of series relationships
  - PC1 vs PC2 scatter plot
  - Loading vectors for 4 metrics (% 1s, % 10s, mean 2-9, std 2-9)
  - Color-coding by mean rating quality
  - Series labels and annotations

### Figure 2: `episode_rating_trajectories`
- **Single Panel**: Violin plots by rating pattern
  - X-axis: Rating patterns (123, 213, 231, 312)
  - Y-axis: Episode ratings
  - Three violins per pattern (First, Middle, Last episodes)
  - Color scheme: mustard yellow, orange, deep red
  - Statistical annotation box with key findings

### Figure 3: `task_characteristics_analysis`
- **Main Panel**: Grouped bar chart
  - X-axis: Activity types (Creative, Physical, Mental, Social)
  - Y-axis: Number of tasks
  - Grouped bars: Objective vs Subjective judgment
  - Colors: Steel blue (Objective), Golden rod (Subjective)
- **Supplementary Panel**: Stacked bar chart
  - Series distribution of activity types across series 1-18
  - Stacked proportions showing task type evolution

### Figure 4: `contestant_geographic_origins`
- **Single Panel**: Geographic visualization
  - British Isles map background
  - Heatmap overlay showing contestant density
  - Contour lines highlighting concentration patterns
  - Legend showing country counts (UK/Ireland vs international)
  - Coordinate transformation from lat/lon to pixel positions

### Figure 5: `contestant_performance_archetypes`
- **Grid Layout**: 3×6 grid (18 series)
  - Each cell: One series with 5 contestants positioned by archetype
  - Spatial positions: 
    - Steady Performer (top-left)
    - Late Bloomer (top-right)
    - Early Star (bottom-left)
    - Chaotic Wildcard (bottom-right)
    - Consistent Middle (center)
  - Gold circles highlighting series winners
  - Directional labels with arrows

### Figure 6: `scoring_pattern_geometry`
- **Single Panel**: Scatter plot
  - X-axis: Mean score (scoring generosity)
  - Y-axis: Variance (score spread)
  - Color: Skew (red=positive, blue=negative, white=symmetric)
  - Background points: All possible scoring patterns
  - Black circles: Actually used patterns (size ∝ frequency)

### Figure 7: `sentiment_trends_analysis`
- **Panel A**: Significant trend visualization
  - Linear trend plot for awkwardness over series
  - Individual series points with error bars
  - Trend line with 95% confidence interval
  - Statistical annotations (slope, p-value, R²)
- **Panel B**: Sentiment distributions
  - Violin plots for all 7 sentiment metrics
  - Awkwardness highlighted in red, others in gray
  - Median, mean, and probability density shown

### Figure 8: `predictive_modeling_analysis`
- **Panel A**: Episode-level ML analysis
  - Model performance comparison (Linear, Ridge, Random Forest)
  - R² scores and feature importance
  - Cross-validation results
- **Panel B**: Series-level correlation analysis
  - Histogram of correlation coefficients
  - Gaussian fit overlay
  - Distribution statistics (μ=-0.025, σ=0.199)

### Figure Sup 9: `task_skill_profiles`
- **Single Panel**: Spider/radar plot
  - 8 skill dimensions as axes (0.0-1.0 scale)
  - 4 polarized tasks as different colored lines
  - Skills: Creativity, Physical Coordination, Problem Solving, Time Pressure, Originality, Entertainment, Strategic Planning, Adaptability
  - Multiline legend with full task titles

### Figure Sup Series: `individual_series_analysis`
- **Per Series (18 total)**: Two-panel layout
  - **Top Panel**: Ranking progression
    - X-axis: Task number
    - Y-axis: Ranking position (1-5, inverted)
    - Episode boundaries with gray shading
    - Episode labels ("Ep 1", "Ep 2", etc.)
  - **Bottom Panel**: Cumulative scores
    - Line plot showing score accumulation
    - Same color scheme as ranking plot
    - Full width traditional line plot

## Analysis Flow and Logic

The original numbering followed a logical progression through different aspects of Taskmaster analysis:

### Core Figures (1-8)
1. **Series Overview** (`series_ratings_analysis`) - Start with high-level series ratings
2. **Episode Patterns** (`episode_rating_trajectories`) - Drill down to episode-level patterns
3. **Task Structure** (`task_characteristics_analysis`) - Analyze the building blocks (tasks)
4. **Contestant Demographics** (`contestant_geographic_origins`) - Who are the contestants?
5. **Performance Dynamics** (`contestant_performance_archetypes`) - How do contestants perform?
6. **Scoring Mechanics** (`scoring_pattern_geometry`) - How does the scoring system work?
7. **Content Evolution** (`sentiment_trends_analysis`) - How has the show evolved emotionally?
8. **Predictive Analysis** (`predictive_modeling_analysis`) - What drives success?

### Supplementary Figures
- **Task Details** (`task_skill_profiles`) - Deep dive into task skill requirements
- **Series Details** (`individual_series_analysis`) - Deep dive into individual series

## Folder Structure

```
figures/
├── FIGURE_NAMING_HISTORY.md                    # This file
├── series_ratings_analysis/                    # Figure 1
├── episode_rating_trajectories/                # Figure 2
├── task_characteristics_analysis/              # Figure 3
├── contestant_geographic_origins/              # Figure 4
├── contestant_performance_archetypes/          # Figure 5
├── scoring_pattern_geometry/                   # Figure 6
├── sentiment_trends_analysis/                  # Figure 7
├── predictive_modeling_analysis/               # Figure 8
├── task_skill_profiles/                        # Figure Sup 9
└── individual_series_analysis/                 # Figure Sup Series
```

## Benefits of Descriptive Naming

1. **Self-documenting** - Folder names immediately convey content
2. **Easier navigation** - No need to remember what "figure3" contains
3. **Better organization** - Related analyses are clearly grouped
4. **Future-proof** - Can add new analyses without renumbering
5. **Professional appearance** - More suitable for academic work

## Usage Notes

- When referencing figures in papers, use the descriptive names
- The original logical order is preserved in this document
- Each folder contains its own README.md with detailed documentation
- Script names within folders also use descriptive naming

## Date of Renaming

This renaming was completed to move away from numbered figures before the paper was written, allowing for more flexible and intuitive organization of the analysis. 