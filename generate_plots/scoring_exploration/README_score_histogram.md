# Taskmaster Scoring Pattern Analysis

This analysis explores the geometric space of possible scoring patterns in the British comedy panel show Taskmaster, visualizing all possible ways 5 contestants can receive scores from 0-6.

## Visualization Approach

The visualization represents each possible scoring histogram as a point in a 2D space:

- **X-axis**: Mean score 
- **Y-axis**: Variance (higher values = more varied scores)
- **Color**: Skew (using a coolwarm colormap, where red = positive skew/low scores dominant, blue = negative skew/high scores dominant)
- **Size of black circles**: Frequency in the actual show data (larger = more occurrences)

Points without black circles represent valid scoring patterns that never occurred in the show.

## Data Coverage Statistics

### Score Range 0-5
- Total possible scoring patterns: 252
- Patterns used in the show: 103 (40.9%)
- Patterns used at least twice: 58 (23.0%)

### Score Range 0-6
- Total possible scoring patterns: 462
- Patterns used in the show: 111 (24.0%)
- Patterns used at least twice: 58 (12.6%)

This shows that while the show used a good variety of scoring patterns, there are still many mathematically possible patterns that never appeared.

## Key Observations

1. **Most Common Pattern**: The perfectly balanced pattern {1,2,3,4,5} with histogram [0,1,1,1,1,1,0] occurred 353 times, far more than any other pattern. This represents the ideal discriminatory scoring where each contestant receives a different score from 1-5.

2. **Extreme Patterns**: Some notable extreme patterns that did appear:
   - All-or-nothing: {0,0,0,0,5} with histogram [4,0,0,0,0,1,0] occurred 22 times
   - High variance: {0,0,0,4,5} with histogram [3,0,0,0,1,1,0] occurred 21 times
   - Low variance: {2,2,3,4,5} with histogram [0,0,2,1,1,1,0] occurred 27 times

3. **Distribution of Points**: The visualization reveals clusters of commonly used patterns, suggesting that some regions of the scoring space were preferred by the hosts.

4. **Score 6**: While the show allowed for scores of 6, they were rarely used. Most patterns with non-zero frequencies end with 0 in the 6-score position.

## Methodology

The analysis was conducted using the following steps:

1. Generated all mathematically possible histograms for 5 contestants receiving scores from 0-6
2. Calculated statistics (mean, variance, skew) for each histogram
3. Loaded the actual Taskmaster scoring data and computed frequencies of each scoring pattern
4. Visualized the data using matplotlib, highlighting patterns that occurred in the actual show

## Files

- `taskmaster_scoring_geometry_0-6.png/pdf`: Basic visualization of all possible patterns
- `taskmaster_scoring_geometry_0-6_labeled.png/pdf`: Enhanced visualization with labels for notable patterns
- `visualize_histograms.py`: Python script to generate the visualizations
- `data/scores.csv`: Raw data of all contestant scores

## X-Y Plot Information Capture

The mean-variance plot effectively captures the key aspects of the scoring distributions:

- **Completeness**: The 2D projection represents 100% of the possible scoring patterns (462 for scores 0-6)
- **Discrimination**: 
  - For patterns with at least 1 occurrence: The plot visually distinguishes 111/111 (100%) of patterns
  - For patterns with at least 2 occurrences: The plot visually distinguishes 58/58 (100%) of patterns

Adding the color dimension (skew) further enhances the information content by showing the asymmetry of the distributions, making it possible to distinguish patterns with similar mean and variance but different shapes. 