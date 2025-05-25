# Series-Level IMDb Ratings Analysis

## Overview

This figure consists of two panels that visualize the IMDb ratings across all Taskmaster series:

1. **Panel A: Ridge plot** showing the distribution of ratings for each series, decomposed using a mixture model:
   - Delta functions (spikes) at ratings 1 and 10: a₁·δ(1) + a₁₀·δ(10)
   - A Gaussian distribution for ratings 2-9: N(μ, σ)
   - Official IMDb rating labels with IMDb-style yellow formatting

2. **Panel B: PCA plot** showing how series relate to each other based on their rating profiles:
   - PC1 and PC2 derived from four key metrics: percentage of 1s, percentage of 10s, mean of ratings 2-9, standard deviation of ratings 2-9
   - Loading vectors showing the influence of each feature
   - Color-coding based on mean rating quality

## Key Results

### Overall Statistics
- **Total series analyzed**: 18
- **Total episodes**: 154
- **Total votes**: 32,607
- **Average votes per series**: 1,811.5
- **Average episodes per series**: 8.6

### Best and Worst Performing Series
- **Highest-rated series**: Series 7 (μ = 7.88, IMDb = 8.3)
- **Lowest-rated series**: Series 10 (μ = 7.27, IMDb = 7.52)

### Extreme Ratings Analysis
- **Most 10-star ratings**: Series 7 (29.9%)
- **Most 1-star ratings**: Series 18 (10.8%)
- **Correlation between mean rating and 1-star percentage**: -0.53
- **Correlation between mean rating and 10-star percentage**: 0.71

### PCA Results
- **PC1 explains**: 66.7% of variance
- **PC2 explains**: 22.1% of variance
- **Total variance explained**: 88.8%

### Series Distribution by Quadrant
- Quadrant 1 (high quality, high engagement): 7 series
- Quadrant 2 (high quality, low controversy): 2 series
- Quadrant 3 (lower quality, polarizing): 5 series
- Quadrant 4 (lower quality, consensus): 4 series

### Detailed Series Metrics

| Series | Gaussian Mean (μ) | Std Dev (σ) | % 1-stars | % 10-stars | IMDb Rating | Episodes | Total Votes | a₁ | a₁₀ | a_gaussian |
|--------|------------------|-------------|-----------|------------|-------------|----------|-------------|-----|-----|------------|
| 1      | 7.88            | 1.09        | 0.76%     | 21.42%     | 8.12        | 6        | 2,063       | 0.008 | 0.214 | 0.778 |
| 2      | 7.82            | 1.21        | 1.42%     | 22.91%     | 8.10        | 5        | 1,514       | 0.014 | 0.229 | 0.757 |
| 3      | 7.86            | 1.02        | 1.07%     | 22.67%     | 8.10        | 5        | 1,401       | 0.011 | 0.227 | 0.763 |
| 4      | 7.86            | 1.07        | 1.04%     | 26.16%     | 8.16        | 8        | 2,229       | 0.010 | 0.262 | 0.728 |
| 5      | 7.81            | 1.10        | 1.12%     | 28.17%     | 8.11        | 8        | 2,140       | 0.011 | 0.282 | 0.708 |
| 6      | 7.42            | 1.24        | 2.07%     | 19.12%     | 7.64        | 10       | 2,275       | 0.021 | 0.191 | 0.788 |
| 7      | 7.88            | 1.10        | 1.39%     | 29.90%     | 8.30        | 8        | 2,154       | 0.014 | 0.299 | 0.687 |
| 8      | 7.55            | 1.11        | 1.09%     | 21.12%     | 7.72        | 9        | 1,927       | 0.011 | 0.211 | 0.778 |
| 9      | 7.69            | 1.06        | 1.52%     | 25.42%     | 7.93        | 10       | 2,093       | 0.015 | 0.254 | 0.731 |
| 10     | 7.27            | 1.37        | 2.70%     | 21.54%     | 7.52        | 8        | 1,709       | 0.027 | 0.215 | 0.757 |
| 11     | 7.72            | 1.12        | 0.27%     | 27.05%     | 7.95        | 10       | 2,181       | 0.003 | 0.271 | 0.726 |
| 12     | 7.73            | 1.04        | 0.15%     | 25.38%     | 7.93        | 10       | 1,864       | 0.002 | 0.254 | 0.745 |
| 13     | 7.74            | 1.12        | 1.05%     | 24.91%     | 7.97        | 10       | 1,811       | 0.011 | 0.249 | 0.740 |
| 14     | 7.75            | 0.99        | 0.08%     | 24.32%     | 7.94        | 7        | 1,196       | 0.001 | 0.243 | 0.756 |
| 15     | 7.43            | 1.32        | 2.60%     | 19.22%     | 7.74        | 10       | 1,645       | 0.026 | 0.192 | 0.782 |
| 16     | 7.52            | 1.17        | 0.31%     | 18.37%     | 7.68        | 10       | 1,595       | 0.003 | 0.184 | 0.813 |
| 17     | 7.31            | 1.15        | 1.48%     | 14.15%     | 7.51        | 10       | 1,286       | 0.015 | 0.141 | 0.844 |
| 18     | 7.25            | 1.42        | 10.77%    | 21.66%     | 7.54        | 10       | 1,524       | 0.108 | 0.217 | 0.676 |

## Implementation Details

### Data Processing (`process_series_ratings_data.py`)

The data processing script:

1. Loads IMDb rating data from `taskmaster_histograms_corrected.csv` with fixes for column naming (histogram columns were reversed)
2. For each series, fits a mixture model:
   - Extracts proportions for ratings 1 and 10 as delta functions (a₁ and a₁₀)
   - Fits a Gaussian distribution N(μ, σ) to ratings 2-9
   - The complete model is: a₁·δ(1) + a₁₀·δ(10) + a_gaussian·N(μ, σ)
   - Verifies that a₁ + a₁₀ + a_gaussian ≈ 1.0
3. Performs PCA on these four metrics:
   - Percentage of 1-star ratings (`pct_1s`)
   - Percentage of 10-star ratings (`pct_10s`)
   - Mean of ratings 2-9 (`mu`)
   - Standard deviation of ratings 2-9 (`sigma`)
4. Calculates additional statistics for the figure caption
5. Saves processed data directly to the figure folder

### Plotting (`plot_series_ratings_analysis.py`)

The plotting script creates:

#### Panel A: Ridge Plot
- One ridge per series, ordered by series number
- Gaussian curves fitted to ratings 2-9
- Red spikes at rating 1 (size proportional to percentage)
- Green spikes at rating 10 (size proportional to percentage)
- Series labels with mean rating (μ) annotated
- Official IMDb ratings displayed with IMDb yellow styling

#### Panel B: PCA Plot
- Each series as a point in 2D space determined by PCA
- Series numbers as labels with white outlines
- Color gradient from red (low mean) to green (high mean)
- Loading vectors (blue arrows) showing feature influence
- Focused axis limits for clarity

## Output Files

- **Processed Data**:
  - `series_metrics.csv` - Series-level metrics
  - `series_pca.csv` - PCA coordinates
  - `pca_loadings.csv` - Feature loadings
  - `explained_variance.npy` - Explained variance ratios
  - `metrics.json` - Key metrics used in the caption

- **Figure Output**:
  - `figure1_ridge_output.pdf/png` - Ridge plot panel
  - `figure1_pca_output.pdf/png` - PCA plot panel
  - `series_ratings_caption.txt` - Figure caption

## Expected Insights

This figure reveals:
- Series 7 stands out as the highest-rated series with the most 10-star ratings
- Series 18 is notable for having an extremely high percentage of 1-star ratings (10.8%)
- There's a strong positive correlation (0.71) between mean rating and percentage of 10-star ratings
- Later series (15-18) show more variable ratings and lower overall scores
- The PCA analysis separates series primarily by their mean rating (PC1) and rating polarization (PC2)
- High-quality series tend to have more enthusiastic fans (more 10s) rather than fewer detractors (fewer 1s) 