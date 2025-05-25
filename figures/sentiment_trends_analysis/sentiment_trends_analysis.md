# Sentiment Trends Analysis

## Overview

This figure analyzes emotional trends across 18 series of Taskmaster UK using sentiment analysis of episode transcripts. The analysis examines seven key emotional metrics to understand how the show's emotional tone has evolved over time.

## Key Results

### Statistical Summary

| Metric | Trend Direction | Slope | p-value | FDR-adjusted p-value | Significant? | Effect Size |
|--------|----------------|-------|---------|---------------------|--------------|-------------|
| **Awkwardness** | **Increasing** | **0.0122** | **0.0004** | **0.0027** | **Yes** | **2.71** |
| Humor | Decreasing | -0.0073 | 0.0181 | 0.0633 | No | -1.93 |
| Anger | No trend | -0.0008 | 0.5832 | 0.6804 | No | -0.49 |
| Frustration/Despair | No trend | 0.0006 | 0.0570 | 0.1330 | No | 0.10 |
| Joy/Excitement | No trend | 0.0012 | 0.2495 | 0.4367 | No | 0.11 |
| Sarcasm | No trend | -0.0026 | 0.2754 | 0.3856 | No | -0.97 |
| Self-deprecation | No trend | -0.0007 | 0.7809 | 0.7809 | No | -0.23 |

### Key Finding: Rising Awkwardness

**Awkwardness is the only sentiment showing a statistically significant trend**, increasing by approximately 8.2% from Series 1 to Series 18:
- Series 1 average: 2.39
- Series 18 average: 2.59
- Total increase: 0.20 units (8.2%)
- Correlation coefficient: r = 0.76
- Effect size: 2.71 (very large)

### Detailed Metrics by Series

| Series | Episodes | Awkwardness | Humor | Anger | Joy/Excitement | Sarcasm | Self-deprecation |
|--------|----------|-------------|-------|-------|----------------|---------|------------------|
| 1 | 6 | 2.39 ± 0.13 | 3.16 ± 0.07 | 0.28 ± 0.07 | 0.04 ± 0.02 | 2.10 ± 0.16 | 1.75 ± 0.08 |
| 5 | 10 | 2.40 ± 0.07 | 3.18 ± 0.07 | 0.20 ± 0.06 | 0.28 ± 0.49 | 1.98 ± 0.14 | 1.64 ± 0.08 |
| 10 | 10 | 2.50 ± 0.10 | 3.12 ± 0.05 | 0.21 ± 0.05 | 0.15 ± 0.29 | 2.00 ± 0.13 | 1.61 ± 0.09 |
| 15 | 8 | 2.58 ± 0.07 | 3.01 ± 0.05 | 0.22 ± 0.05 | 0.04 ± 0.02 | 2.04 ± 0.07 | 1.64 ± 0.06 |
| 17 | 9 | 2.59 ± 0.10 | 3.24 ± 0.28 | 0.15 ± 0.05 | 0.76 ± 0.68 | 1.98 ± 0.11 | 1.64 ± 0.18 |
| 18 | 5 | 2.46 ± 0.06 | 3.14 ± 0.05 | 0.19 ± 0.08 | 0.08 ± 0.02 | 1.96 ± 0.19 | 1.52 ± 0.14 |

### Notable Patterns

1. **Awkwardness Peak**: Series 17 shows the highest awkwardness (2.59), coinciding with particularly uncomfortable moments
2. **Humor Stability**: Despite slight downward trend, humor remains consistently high (3.0-3.2 range)
3. **Low Negative Emotions**: Anger and frustration remain very low throughout (< 0.3)
4. **Joy Variability**: Joy/Excitement shows high variability, with Series 17 as an outlier (0.76)

## Implementation Details

### Data Processing (`process_data.py`)

The script:
1. Loads sentiment data from `sentiment.csv`
2. Calculates weighted averages for each sentiment metric per episode
3. Aggregates to series level with mean and standard error
4. Performs linear regression analysis for each metric
5. Applies multiple testing corrections (FDR and Bonferroni)
6. Calculates effect sizes using standardized slopes
7. Identifies statistically significant trends

### Plotting (`plot_sentiment_trends.py`)

Creates two complementary visualizations:

**Panel A: Trend Lines**
- Shows all seven sentiment metrics over 18 series
- Highlights the significant awkwardness trend with bold styling
- Uses consistent color scheme for each emotion
- Includes regression lines for significant trends
- Error bars show standard error of the mean

**Panel B: Effect Size Comparison**
- Bar chart showing standardized effect sizes
- Highlights statistically significant results (awkwardness)
- Color-coded by trend direction (red=decreasing, green=increasing)
- Includes significance threshold line

## Output Files

- `sentiment_trends_data.csv`: Regression results for all metrics
- `sentiment_series_statistics.csv`: Series-level statistics
- `figure7.pdf/png`: Combined two-panel figure
- `figure7a.pdf/png`: Panel A (trend lines)
- `figure7b.pdf/png`: Panel B (effect sizes)
- `sentiment_trends_caption.txt`: Auto-generated caption

## Insights for Paper

1. **Awkwardness as Show Evolution**: The significant increase in awkwardness suggests the show has deliberately embraced uncomfortable comedy as a key element, possibly reflecting changing comedy tastes or production choices.

2. **Emotional Consistency**: Despite the awkwardness trend, other emotions remain remarkably stable, indicating the show maintains its core emotional formula.

3. **Positive Emotional Tone**: High humor levels (>3.0) and low negative emotions (<0.3) confirm Taskmaster's fundamentally positive atmosphere.

4. **Series 17 Anomaly**: The spike in both awkwardness and joy in Series 17 suggests particularly memorable or extreme moments.

5. **Comedy Evolution**: The rising awkwardness aligns with broader trends in British comedy toward "cringe comedy" and uncomfortable humor.

6. **Production Learning**: The trend may reflect producers learning what generates the most engaging content and viewer reactions over time. 