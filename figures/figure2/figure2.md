# Figure 2: Episode Rating Trajectories by Position Patterns

## Overview

Figure 2 visualizes how IMDb ratings change across the span of Taskmaster series, tracking the rating patterns from first episodes through middle episodes to series finales. This figure uses violin plots to show rating distributions for episodes categorized by different series rating trajectories.

## Key Components

1. **Violin Plot Visualization**:
   - X-axis: Different rating patterns (123, 213, 231, 312)
   - Y-axis: Episode ratings (ranging ~6.5 to 9.2)
   - For each pattern, three violins showing rating distributions by position:
     - First episode (mustard yellow)
     - Middle episodes (orange)
     - Last episode (deep red)
   - Within each violin:
     - White dot: median rating
     - Thick bar: interquartile range
     - Thin line: full range excluding outliers

2. **Series Grouping**:
   - Under each pattern group, list of the series numbers that follow that pattern
   - Pattern descriptions:
     - 123: Rising pattern (ratings improve throughout the series)
     - 213: J-shaped pattern (middle episodes rated lower than beginning and end)
     - 231, 312: Other unique patterns

3. **Statistical Analysis**:
   - Centrally placed annotation box showing key statistical findings
   - Prevalence of Rising (123) and J-shaped (213) patterns: 16/18 series (88.9%)
   - Mean rating increase from first to last episode: 0.283 (p=0.0003)
   - Mean rating increase from middle to last episode: 0.276 (p=0.0003)

## Implementation Details

### Data Processing (`process_figure2_data.py`)
- Processes IMDb episode ratings
- Categorizes episodes as First, Middle, or Last
- Classifies series by rating pattern based on comparing first (1), middle (2), and last (3) episode ratings
- Performs statistical analysis:
  - Binomial test to determine if certain patterns (123, 213) are significantly more common
  - T-tests to analyze rating differences between first, middle, and last episodes
  - Chi-square test to assess overall pattern distribution

### Visualization (`plot_figure2.py`)
- Creates violin plots with position-based coloring (yellow, orange, red)
- Uses clean design with no gridlines and neutral-colored annotations
- Displays statistical significance results within the plot area
- Formats axes and labels using configuration from `plot_config.yaml`

## Interpretation

This figure illustrates how episode ratings typically change across the course of a series. Key insights include:

- Most series (16 out of 18, 88.9%) follow either a rising (123) or J-shaped (213) pattern
- There is a statistically significant increase in ratings from first to last episodes (mean increase of 0.283, p=0.0003)
- Middle-to-last episode transitions show a similar significant improvement (mean increase of 0.276, p=0.0003)
- First-to-middle transitions show minimal change, suggesting that significant improvement typically happens in the final episode

The prevalence of these patterns suggests that Taskmaster series tend to either improve consistently throughout or dip in the middle before finishing with a strong finale. The statistical significance of these patterns provides evidence that this is not merely due to random variation in ratings. 