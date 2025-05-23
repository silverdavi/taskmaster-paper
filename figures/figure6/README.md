# Figure 6: Scoring Pattern Analysis

## Overview

Figure 6 visualizes the **geometry of Taskmaster scoring patterns**, showing how the show's 5-contestant scoring system creates a rich space of possible distributions and revealing which patterns are actually used in practice.

## Description

This figure presents a scatter plot where each point represents a possible way to distribute scores among 5 contestants (with scores ranging from 0-5). The visualization reveals:

- **X-axis**: Mean score (how generous the scoring was overall)
- **Y-axis**: Variance (how much the scores varied between contestants)
- **Color**: Skew (whether scores were skewed toward high or low values)
- **Black circles**: Patterns that actually occurred in the show (size ∝ frequency)
- **Background points**: All theoretically possible scoring patterns

## Key Insights

1. **Limited Usage**: Despite hundreds of possible scoring patterns, the show uses only a subset
2. **Clustering**: Certain scoring patterns are much more common than others
3. **Geometric Structure**: The space of scoring patterns has clear geometric constraints
4. **Scoring Philosophy**: The distribution reveals implicit rules about how tasks are scored

## Files

### `process_data.py`
**Purpose**: Processes raw scoring data to generate all possible scoring patterns and their usage statistics.

**What it does**:
- Generates all possible histograms for 5 contestants with scores 0-5
- Extracts actual scoring patterns from the dataset
- Calculates statistical properties (mean, variance, skew) for each pattern
- Counts frequency of use for each pattern in the actual data

**Input**: `../../data/raw/scores.csv`
**Output**: `figure6_scoring_patterns.csv`

**Usage**:
```bash
cd figures/figure6
python process_data.py
```

### `plot_figure6.py`
**Purpose**: Creates the scoring pattern visualization.

**What it does**:
- Loads processed scoring pattern data
- Creates scatter plot showing all possible vs. actual patterns
- Adds appropriate legends, labels, and colorbar
- Saves both PNG and PDF versions of the figure

**Input**: `figure6_scoring_patterns.csv`
**Output**: 
- `figure6.png`
- `figure6.pdf`

**Usage**:
```bash
cd figures/figure6
python plot_figure6.py
```

### `caption_figure6.txt`
**Purpose**: Contains the detailed caption and interpretation for Figure 6.

**What it includes**:
- Explanation of the visualization components
- Key statistical findings
- Examples of important scoring patterns
- Interpretation of the geometric structure

## Running the Analysis

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Step-by-step
1. **Process the data**:
   ```bash
   python process_data.py
   ```
   
2. **Generate the figure**:
   ```bash
   python plot_figure6.py
   ```

### Quick run (both steps):
```bash
python process_data.py && python plot_figure6.py
```

## Output Files

- **`figure6.png`**: Main figure in PNG format
- **`figure6.pdf`**: Main figure in PDF format  
- **`figure6_scoring_patterns.csv`**: Processed data with all pattern statistics
- **`caption_figure6.txt`**: Detailed caption and interpretation

## Interpretation Guide

### Reading the Visualization

1. **Background cloud**: Shows all mathematically possible scoring patterns
2. **Black circles**: Patterns actually used (larger = more frequent)
3. **Color intensity**: 
   - Red = positive skew (more low scores)
   - Blue = negative skew (more high scores)
   - White = symmetric distribution

### Common Patterns

- **High frequency, low variance**: "Safe" scoring with most contestants getting similar scores
- **High variance**: "Dramatic" tasks with wide score spreads
- **Extreme skew**: Tasks where most contestants failed or most succeeded

### Analysis Questions

1. **Coverage**: What percentage of possible patterns does the show actually use?
2. **Preferences**: Are there systematic biases in scoring patterns?
3. **Outliers**: Which rare patterns represent particularly unusual tasks?
4. **Evolution**: Do scoring patterns change over series/seasons?

## Technical Notes

### Scoring System
- Taskmaster uses 0-5 point scoring system
- Most tasks have exactly 5 contestants
- Ties are possible (multiple contestants can get the same score)

### Statistical Measures
- **Mean**: Average score (0-5)
- **Variance**: Spread of scores (0 = all same, higher = more varied)
- **Skew**: Asymmetry (-∞ to +∞, 0 = symmetric)

### Data Quality
- Filters out tasks with non-standard scoring
- Excludes tasks with ≠5 contestants
- Handles missing or invalid score data

## Customization

### Modifying Score Range
To analyze different scoring systems, change `MAX_SCORE` in `process_data.py`:
```python
MAX_SCORE = 10  # For 0-10 scoring system
```

### Visual Styling
Modify the configuration in `plot_figure6.py`:
```python
config = {
    'colormap': 'viridis',    # Different color scheme
    'figure_size': (16, 10),  # Different size
    'dpi': 600               # Higher resolution
}
```

## Related Analysis

This figure pairs well with:
- **Figure 2**: Episode difficulty trajectories
- **Figure 3**: Task category analysis  
- **Figure 7**: Statistical correlations

## Contact

For questions about this analysis or suggestions for improvements, please refer to the main project documentation. 