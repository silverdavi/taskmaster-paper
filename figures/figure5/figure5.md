# Figure 5: Contestant Dynamics and Clustering

## Overview

Figure 5 visualizes contestant performance dynamics across all 18 series of Taskmaster UK, categorizing contestants into five distinct archetypes based on their performance patterns. The visualization presents a 3×6 grid (18 series), where each cell represents one series with the five contestants positioned according to their performance archetype.

## Data Processing Pipeline

The data processing and visualization pipeline consists of three main steps:

1. **Feature Extraction** (`extract_features.py`)
2. **Archetype Assignment** (`perfect_archetypes.py`) 
3. **Visualization** (`plot_figure5.py`)

### Step 1: Feature Extraction

The `extract_features.py` script processes the series score data to extract 15 performance features for each contestant:

```
ContestantID → [Performance Features] → contestant_features.csv
```

**Key Features Extracted:**

| Feature | Description |
|---------|-------------|
| `early_avg_score` | Average score in the first 20% of tasks |
| `late_avg_score` | Average score in the last 20% of tasks |
| `score_growth_rate` | Linear regression slope of scores over time |
| `score_variance` | Variance in cumulative scores |
| `avg_rank` | Average ranking throughout the series |
| `first_rank` | Contestant's rank after the first task |
| `last_rank` | Contestant's final rank |
| `rank_variance` | Variance in contestant's rank over time |
| `score_acceleration` | Second derivative of scores (acceleration) |
| `rank_changes` | Number of direction changes in rank trajectory |
| `early_score_ratio` | Ratio of early scores to total score |
| `late_score_ratio` | Ratio of late scores to total score |
| `avg_task_score` | Average score per task |
| `score_consistency` | Coefficient of variation of task scores |
| `comeback_factor` | Difference between worst and final rank |

### Step 2: Archetype Assignment

The `perfect_archetypes.py` script assigns exactly one contestant per series to each of the five archetypes:

```
contestant_features.csv → [Archetype Assignment] → final_archetypes.csv
```

**Archetype Decision Rules:**

Each contestant is scored on how well they match each archetype using a weighted combination of features. The scoring functions are:

1. **Steady Performer**: 
   ```
   Score = -rank_variance - score_variance - rank_changes - avg_rank
   ```
   *Contestants who maintain consistently high performance with minimal variance*

2. **Late Bloomer**:
   ```
   Score = late_score_ratio + score_growth_rate + comeback_factor
   ```
   *Contestants who start slow but improve significantly over time*

3. **Early Star**:
   ```
   Score = early_score_ratio - score_growth_rate - late_score_ratio
   ```
   *Contestants who start strong but lose momentum*

4. **Chaotic Wildcard**:
   ```
   Score = rank_variance + rank_changes + score_variance
   ```
   *Contestants with unpredictable performance and high variance*

5. **Consistent Middle**:
   ```
   Score = -abs(avg_rank - 3) - rank_variance - abs(score_growth_rate)
   ```
   *Contestants who maintain a middle-of-the-pack position with little variation*

### Step 3: Visualization

The `plot_figure5.py` script creates a 3×6 grid visualization positioning contestants according to their archetypes:

```
final_archetypes.csv → [Visualization] → figure5_output.pdf/png
```

**Spatial Positioning:**
- **Steady Performer**: Top-left (0.25, 0.75)
- **Late Bloomer**: Top-right (0.75, 0.75)
- **Early Star**: Bottom-left (0.25, 0.25)
- **Chaotic Wildcard**: Bottom-right (0.75, 0.25)
- **Consistent Middle**: Center (0.5, 0.5)

**Visualization Features:**
- Directional labels with arrows showing "→ Higher Scores" (vertical axis) and "→ Chaotic Performance" (horizontal axis)
- Gold circles highlighting the series winners (contestants with `last_rank=1`)
- Contestant names with white background for better readability
- Light gray title bars connecting subplot titles to their respective series
- Sufficient spacing between rows for better visual separation

## Archetype Distribution Analysis

The archetype assignments reveal interesting patterns across all 18 series:

- Each series contains exactly one contestant of each archetype
- Some contestants strongly exemplify their archetype (high scores)
- Others are more borderline cases (lower scores)

### Notable Examples:

1. **Exemplary Steady Performers**:
   - Romesh Ranganathan (Series 1)
   - Doc Brown (Series 2)
   - Kerry Godliman (Series 7)

2. **Dramatic Late Bloomers**:
   - Josh Widdicombe (Series 1, winner)
   - Richard Herring (Series 10, winner)
   - Mike Wozniak (Series 11)

3. **Quintessential Early Stars**:
   - Tim Key (Series 1)
   - Joe Wilkinson (Series 2)
   - Sara Pascoe (Series 3)

4. **Memorable Chaotic Wildcards**:
   - Frank Skinner (Series 1)
   - Jon Richardson (Series 2)
   - Rhod Gilbert (Series 7)

5. **Definitive Consistent Middles**:
   - Roisin Conaty (Series 1)
   - Richard Osman (Series 2)
   - Hugh Dennis (Series 4)

## Interpretation and Insights

The archetype classification reveals that:

1. Each series has a balance of different contestant types, potentially by design
2. Performance trajectories vary significantly between contestants
3. The five archetypes represent fundamentally different approaches to task completion
4. The spatial visualization makes it easy to identify patterns across series

This classification system provides a framework for understanding contestant dynamics and may explain why certain contestants are remembered differently by fans, despite achieving similar final rankings.

## Technical Notes

- The feature extraction process analyzes both absolute scores and relative rankings
- The archetype assignment algorithm optimizes for having exactly one contestant of each type per series
- Small random jitter is added to positions in the visualization to prevent overlapping
- The 2D coordinate system maps to performance consistency (x-axis) and improvement trajectory (y-axis) 