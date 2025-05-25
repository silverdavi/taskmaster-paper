# Scoring Pattern Geometry Analysis

## Overview

This figure visualizes the mathematical landscape of all possible scoring patterns in Taskmaster by analyzing 252 unique score distributions that can occur when five contestants compete in a task. The analysis reveals which patterns are actually used in the show versus which are theoretically possible but never occur.

## Key Results

### Pattern Usage Statistics
- **Total possible patterns**: 252 unique distributions
- **Patterns actually used**: 98 (38.9%)
- **Patterns never used**: 154 (61.1%)
- **Most common pattern**: [0,1,1,1,1,1] - All contestants score differently (353 occurrences)

### Pattern Categories

1. **High-Usage Patterns** (>20 occurrences):
   - All different scores: 353 tasks
   - Mixed 2-3-4-5 scores: 35 tasks
   - Consensus patterns: 26-27 tasks

2. **Moderate-Usage Patterns** (5-20 occurrences):
   - Various competitive distributions
   - Often involve 2-3 contestants with similar scores

3. **Rare Patterns** (1-4 occurrences):
   - Extreme distributions (all 0s, all 5s)
   - Highly skewed patterns

4. **Never-Used Patterns** (0 occurrences):
   - Patterns that create unfair or uninteresting dynamics
   - Overly homogeneous distributions

### Mathematical Properties

The patterns are characterized by three key metrics:

1. **Mean Score**: Average points awarded (0-5 scale)
   - Used patterns range: 0.0 to 5.0
   - Most common mean: 2.5-3.5

2. **Variance**: Spread of scores
   - Low variance: Similar performance (boring)
   - High variance: Clear winners/losers (exciting)
   - Sweet spot: 1.5-2.5 variance

3. **Skewness**: Asymmetry of distribution
   - Negative skew: Few low scores
   - Positive skew: Few high scores
   - Most used patterns: Slightly negative skew

### Top 10 Most Frequent Patterns

| Pattern | Frequency | Mean | Variance | Description |
|---------|-----------|------|----------|-------------|
| [0,1,1,1,1,1] | 353 | 3.0 | 2.0 | All different scores |
| [1,0,1,1,1,1] | 35 | 2.8 | 2.96 | One pair, others different |
| [0,0,2,1,1,1] | 27 | 3.2 | 1.36 | One pair at 3, others spread |
| [0,0,2,0,1,1] | 26 | 2.4 | 4.24 | Two at 3, two at 4, one at 5 |
| [1,0,0,2,1,1] | 20 | 3.2 | 1.76 | Spread distribution |
| [0,1,1,0,1,2] | 15 | 3.2 | 2.56 | Multiple pairs |
| [0,1,0,1,2,1] | 14 | 3.4 | 1.04 | Tight clustering |
| [1,1,0,1,2,1] | 12 | 3.6 | 1.04 | Upper-middle clustering |
| [0,0,3,0,0,2] | 11 | 3.8 | 0.96 | Binary outcome |
| [0,0,2,0,0,3] | 10 | 3.0 | 6.0 | Extreme spread |

### Geometric Visualization Insights

The 3D scatter plot reveals:

1. **Used patterns form a connected structure**: Not randomly distributed but follow logical progressions

2. **Unused patterns create "voids"**: Clear gaps where certain combinations don't work for game dynamics

3. **High-frequency patterns cluster**: The most-used patterns are near the center of the feasible region

4. **Edge patterns are rare**: Extreme distributions (all same score, maximum spread) are seldom used

## Implementation Details

### Data Processing (`process_data.py`)

The script:
1. Generates all 252 possible score distributions for 5 contestants
2. Loads actual task scores from the dataset
3. Counts frequency of each pattern in real data
4. Calculates statistical properties (mean, variance, skewness)
5. Identifies used vs. unused patterns

### Visualization (`plot_scoring_patterns.py`)

Creates a 3D scatter plot showing:
- X-axis: Mean score
- Y-axis: Variance
- Z-axis: Skewness
- Point size: Log-scaled frequency (larger = more common)
- Color: Frequency gradient (yellow to red)
- Gray points: Theoretically possible but never used

## Output Files

- `scoring_patterns_data.csv`: All 252 patterns with statistics
- `figure6.pdf/png`: 3D visualization
- `scoring_patterns_caption.txt`: Auto-generated caption

## Insights for Paper

1. **Strategic Pattern Selection**: The show uses only 39% of possible patterns, suggesting careful task design to create engaging competitive dynamics.

2. **Avoidance of Extremes**: Patterns where everyone scores the same (boring) or where scores are maximally spread (unfair) are rarely used.

3. **Preference for Differentiation**: The most common pattern has all different scores, maximizing competitive tension and clear rankings.

4. **Mathematical Constraints Shape Entertainment**: The geometry of used patterns shows how mathematical properties translate to entertainment value.

5. **Emergent Structure**: The connected nature of used patterns suggests an implicit "grammar" of fair and engaging score distributions.

6. **Design Principles**: Successful patterns balance:
   - Sufficient variance to create winners/losers
   - Not so much variance that outcomes seem predetermined
   - Slight negative skew (more high scores than low)
   - Mean scores in the middle range (2.5-3.5) 