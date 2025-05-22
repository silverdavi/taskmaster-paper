# Figure 1: Series-Level IMDb Ratings

## Overview

Figure 1 consists of two panels that visualize the IMDb ratings across all Taskmaster series:

1. **Panel A: Ridge plot** showing the distribution of ratings for each series, decomposed into:
   - A Gaussian fit for ratings 2-9
   - Red spikes for 1-star ratings and green spikes for 10-star ratings
   - Official IMDb rating labels with IMDb-style yellow formatting

2. **Panel B: PCA plot** showing how series relate to each other based on their rating profiles:
   - PC1 and PC2 derived from four key metrics: percentage of 1s, percentage of 10s, mean of ratings 2-9, standard deviation of ratings 2-9
   - Loading vectors showing the influence of each feature
   - Color-coding based on mean rating quality

## Implementation Details

### Data Processing (`process_data_figure1.py`)

The data processing script:

1. Loads IMDb rating data from `taskmaster_histograms_corrected.csv` with fixes for column naming (histogram columns were reversed)
2. For each series:
   - Fits a Gaussian distribution to ratings 2-9
   - Calculates percentage of 1-star and 10-star ratings
   - Computes mean and standard deviation for the Gaussian
   - Records episode count and total votes
   - Calculates overall mean rating
   - Adds official IMDb ratings
3. Performs PCA on these four metrics:
   - Percentage of 1-star ratings (`pct_1s`)
   - Percentage of 10-star ratings (`pct_10s`)
   - Mean of ratings 2-9 (`mu`)
   - Standard deviation of ratings 2-9 (`sigma`)
4. Calculates additional statistics for the figure caption:
   - Series with highest/lowest mean ratings
   - Series with most extreme ratings (1s and 10s)
   - Correlations between mean rating and extreme ratings
   - Distribution of series across PCA quadrants
5. Saves processed data directly to the figure1 folder:
   - `series_metrics.csv` - Series-level metrics
   - `series_pca.csv` - PCA coordinates
   - `pca_loadings.csv` - Feature loadings
   - `explained_variance.npy` - Explained variance ratios
   - `metrics.json` - Metrics for caption

### Plotting (`plot_figure1.py`)

The plotting script creates:

#### Panel A: Ridge Plot

This panel shows:

- One ridge per series, ordered by series number
- Gaussian curves fitted to ratings 2-9, showing the "central tendency" of ratings
- Red spikes at rating 1 showing percentage of 1-star ratings (size proportional to percentage)
- Green spikes at rating 10 showing percentage of 10-star ratings (size proportional to percentage)
- Series labels with mean rating (μ) annotated
- Official IMDb ratings displayed with IMDb yellow styling
- Simplified labels that only show prefixes (μ=, IMDb=) on the first series
- No axis borders or explanatory text boxes for cleaner presentation

Key features:

- Marker size scales with percentage of extreme ratings
- IMDb labels positioned based on their rating values
- White outlines around text for better readability
- Integration with central styling configuration
- Clear visual distinction between fitted Gaussian means and official IMDb ratings

#### Panel B: PCA Plot

This panel shows:

- Each series as a point in 2D space determined by PCA
- Series numbers as labels with white outlines for visibility
- Color gradient from red (low mean) to green (high mean)
- Loading vectors (blue arrows) showing the influence of each feature, elongated by 20% for better visibility
- Feature labels with intelligent positioning
- Focused axis limits (-4 to 3 on x-axis, -2.2 to 3 on y-axis)
- Annotation labels for extreme series

Key improvements:

- More focused visualization with clearer feature vectors
- Optimized positioning of series and feature labels
- Better use of space with adjusted axis limits
- Proper PDF output with correct shading
- Maintained detailed statistics for extreme examples

## Code Structure

The implementation follows a strict separation of concerns:

1. `process_data_figure1.py`:
   - Handles ALL data processing and transformation
   - Performs statistical analysis and dimension reduction
   - Saves processed data and metrics directly to the figure1 folder
   - Contains the "business logic" of the figure

2. `plot_figure1.py`:
   - Handles ONLY visualization
   - Takes processed data as input (assumes processing is complete)
   - Loads data directly from the figure1 folder
   - Creates visualizations with consistent styling from `plot_config.yaml`
   - Saves outputs in both PDF and PNG formats
   - Contains NO data processing logic

## Output Files

The implementation produces the following files in the figure1 directory:

- **Processed Data**:
  - `series_metrics.csv` - Series-level metrics
  - `series_pca.csv` - PCA coordinates
  - `pca_loadings.csv` - Feature loadings
  - `explained_variance.npy` - Explained variance ratios

- **Figure Output**:
  - `figure1.pdf` - The final figure in PDF format
  - `figure1.png` - The final figure in PNG format
  - `metrics.json` - Key metrics used in the caption

## Expected Insights

This figure reveals:

- Which series have the highest/lowest mean ratings
- How the distribution of ratings varies across series
- Which series have the most polarizing ratings (high percentages of 1s and 10s)
- How series cluster based on their rating profiles
- Whether later series show different rating patterns than earlier ones
- The relationship between mean ratings and extreme ratings
- Discrepancies between fitted Gaussian means and official IMDb ratings

The PCA plot in particular helps identify which aspects of the rating distribution (mean, variance, extremes) contribute most to differences between series. 