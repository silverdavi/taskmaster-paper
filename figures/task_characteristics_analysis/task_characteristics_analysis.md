# Task Characteristics Analysis

## Overview

This figure analyzes the nature of 917 tasks across all 18 series of Taskmaster UK, examining:

1. **Panel A: 2×2 Grid Analysis** - Shows the distribution of tasks across two key dimensions:
   - Activity Type (Creative vs Physical)
   - Judgment Type (Objective vs Subjective)

2. **Panel B: Series-Level Trends** - Tracks how task characteristics have evolved over time

## Key Results

### Overall Task Statistics
- **Total tasks analyzed**: 917
- **Creative tasks**: 43.5% (399 tasks)
- **Physical tasks**: 48.1% (441 tasks)
- **Objective judgment**: 55.9% (513 tasks)
- **Subjective judgment**: 41.5% (381 tasks)

### Task Type Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| **Assignment Types** |
| Solo | 806 | 87.9% |
| Team | 111 | 12.1% |
| Special | 12 | 1.3% |
| Split | 13 | 1.4% |
| Tiebreaker | 27 | 2.9% |
| **Format Types** |
| Prize | 167 | 18.2% |
| Filmed | 569 | 62.1% |
| Homework | 10 | 1.1% |
| Live | 171 | 18.7% |
| **Activity Types** |
| Creative | 399 | 43.5% |
| Mental | 380 | 41.4% |
| Physical | 441 | 48.1% |
| Social | 155 | 16.9% |

### 2×2 Grid Quadrant Analysis

The four quadrants represent different task combinations:

1. **Creative-Objective (Top Left)**: 182 tasks (19.9%)
   - Example: "Make the best thing out of ice"
   - Clear criteria but creative execution

2. **Physical-Objective (Top Right)**: 258 tasks (28.1%)
   - Example: "Get the egg in the egg cup from the furthest distance"
   - Most common type - clear physical challenges with measurable outcomes

3. **Creative-Subjective (Bottom Left)**: 196 tasks (21.4%)
   - Example: "Impress the mayor"
   - Artistic tasks judged on quality/humor

4. **Physical-Subjective (Bottom Right)**: 150 tasks (16.4%)
   - Example: "Do the most spectacular thing with this pommel horse"
   - Physical performance judged on entertainment value

### Trend Analysis

No significant trends were found over the 18 series:

| Task Type | Kendall's τ | p-value | Trend | % Change |
|-----------|------------|---------|-------|----------|
| Creative | 0.262 | 0.129 | No trend | +42.3% |
| Mental | -0.170 | 0.324 | No trend | +1.1% |
| Physical | -0.296 | 0.088 | No trend | -46.6% |
| Social | -0.157 | 0.363 | No trend | +85.0% |

Despite some apparent percentage changes, none reach statistical significance, suggesting the show maintains a consistent balance of task types.

## Implementation Details

### Data Processing (`process_task_characteristics_data.py`)

The script:
1. Loads task data from multiple sources:
   - `taskmaster_UK_tasks.csv`: Task categorizations
   - `_OL_tasks.csv`: Additional task attributes
   - `long_task_scores.csv`: Task-series mappings
2. Merges datasets to create comprehensive task database
3. Categorizes tasks along multiple dimensions:
   - Assignment type (solo, team, etc.)
   - Format (prize, filmed, live, homework)
   - Activity type (creative, mental, physical, social)
   - Judgment type (objective, subjective)
4. Performs trend analysis using Kendall's tau
5. Generates quadrant data for 2×2 analysis

### Plotting (`plot_task_characteristics.py`)

Creates two panels:

**Panel A: 2×2 Grid**
- Four quadrants with task counts and percentages
- Bubble visualization showing relative proportions
- Clear axis labels explaining dimensions
- Color-coded quadrants for visual distinction

**Panel B: Trend Lines**
- Series-by-series proportions for each task type
- Smooth trend lines with confidence intervals
- Color-coded by task characteristic
- Grid lines for readability

## Output Files

- `data/summary_stats.json`: Overall task statistics
- `data/bubble_data.json`: Quadrant analysis data
- `data/series_data.json`: Series-level breakdowns
- `data/trend_analysis.json`: Statistical trend results
- `figure3.pdf/png`: Main 2×2 grid figure
- `figure3_series_distribution.pdf/png`: Trend analysis figure
- `metrics.json`: Key metrics for caption
- `caption.txt`: Auto-generated figure caption

## Insights for Paper

1. **Balanced task design**: The show maintains roughly equal proportions of creative (43.5%) and physical (48.1%) tasks, providing variety.

2. **Objective bias**: More tasks have objective (55.9%) rather than subjective (41.5%) judging criteria, possibly to maintain fairness.

3. **Physical-Objective dominance**: The most common task type (28.1%) combines physical challenges with clear success criteria.

4. **Consistent format**: Despite 18 series, task characteristics show no significant trends, indicating a successful formula that doesn't need major adjustments.

5. **Format distribution**: Most tasks are pre-filmed (62.1%), with prize tasks (18.2%) and live tasks (18.7%) providing variety in each episode.

6. **Solo focus**: The vast majority of tasks (87.9%) are individual challenges, maintaining the competitive element while occasional team tasks (12.1%) provide collaborative moments. 