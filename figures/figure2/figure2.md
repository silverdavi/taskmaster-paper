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

## Implementation Details

### Data Processing (`process_figure2_data.py`)
- Processes IMDb episode ratings
- Categorizes episodes as First, Middle, or Last
- Classifies series by rating pattern based on comparing first (1), middle (2), and last (3) episode ratings
- Performs statistical analysis:
  - Tests if certain patterns (123, 213) are significantly more common
  - Analyzes rating differences between first, middle, and last episodes

### Visualization (`plot_figure2.py`)
- Creates violin plots with appropriate coloring
- Adds annotations for series under each pattern
- Displays statistical significance results
- Formats axes, labels, and styling using configuration from `plot_config.yaml`

## Interpretation

This figure illustrates how episode ratings typically change across the course of a series. Key insights include:
- Whether ratings tend to improve over the course of a series
- Whether middle episodes are typically rated differently from series premieres and finales
- The prevalence of different rating trajectory patterns across all Taskmaster series

The analysis reveals that most series (16 out of 18) follow either a rising (123) or J-shaped (213) pattern, with a statistically significant increase in ratings from first to last episodes, suggesting that series tend to either improve consistently or dip in the middle before a strong finale. 