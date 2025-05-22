# Figure 2: Episode Rating Trajectories by Contestant Ranking Patterns

## Overview

Figure 2 visualizes how episode IMDb ratings correlate with patterns of contestant rankings across different series of Taskmaster. This figure uses violin plots to show rating distributions for episodes categorized by different contestant ranking trajectories.

## Key Components

1. **Violin Plot Visualization**:
   - X-axis: Different ranking patterns (123, 213, 231, 312)
   - Y-axis: Episode ratings (ranging ~6.5 to 9.2)
   - For each pattern, three violins showing rating distributions by position:
     - First position (mustard yellow)
     - Middle position (orange)
     - Last position (deep red)
   - Within each violin:
     - White dot: median rating
     - Thick bar: interquartile range
     - Thin line: full range excluding outliers

2. **Series Grouping**:
   - Under each pattern group, list of the series numbers that follow that pattern
   - Pattern descriptions:
     - 123: Rising pattern (contestants start low, improve steadily)
     - 213: J-shaped pattern (drop then rebound)
     - 231, 312: Other unique patterns

## Implementation Details

### Data Processing (`process_figure2_data.py`)
- Processes raw episode ratings and contestant rankings
- Classifies series by ranking pattern
- Groups episodes by position (First/Middle/Last)
- Calculates statistics for each combination

### Visualization (`plot_figure2.py`)
- Creates violin plots with appropriate coloring
- Adds annotations for series under each pattern
- Formats axes, labels, and styling using configuration from `plot_config.yaml`

## Interpretation

This figure illustrates how contestant ranking patterns across episodes may correlate with viewer reception (as measured by IMDb ratings). Key insights include:
- Which ranking patterns appear in which series
- How episode positions (first, middle, last) correlate with ratings across different patterns
- Whether certain trajectory patterns tend to yield higher or lower ratings overall

The visualization helps identify whether certain narrative structures in contestant performance correlate with audience enjoyment. 