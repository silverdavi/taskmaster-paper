# Figure 7: Sentiment Trends Analysis

This folder contains the analysis for Figure 7, which examines how sentiment characteristics have evolved across the 18 series of Taskmaster.

## Overview

The analysis performs linear trend analysis on sentiment metrics across series while controlling for multiple testing. This represents a focused investigation into the temporal evolution of emotional content in Taskmaster episodes.

## Current Analysis Structure

### Figure 7a: Significant Sentiment Trend
- **Single significant trend**: Awkwardness increasing over time
- **Statistical rigor**: FDR correction at 1% level
- **Key finding**: Only 1 of 7 sentiment metrics shows temporal change

### Figure 7b: Sentiment Distributions  
- **All sentiment metrics**: Complete emotional landscape visualization
- **Context**: Shows distributions of all 7 sentiment dimensions
- **Highlighting**: Emphasizes awkwardness as the unique temporal trend

## Files

### Core Analysis
- `process_data.py` - Data processing script that analyzes trends
- `plot_figure7.py` - Plotting script that creates Figure 7a and 7b
- `README.md` - This documentation file

### Outputs
- `figure7a.png/pdf` - Significant trend visualization (awkwardness)
- `figure7b.png/pdf` - Violin plots of all sentiment distributions
- `figure7.png/pdf` - Combined figure with both panels
- `caption_figure7a.txt` - Figure 7a caption and explanation
- `caption_figure7b.txt` - Figure 7b caption and explanation
- `caption_figure7.txt` - Combined figure caption

### Data
- `figure7_sentiment_trends.csv` - Trend analysis results
- `figure7_series_statistics.csv` - Series-level statistics

## Analysis Pipeline

### 1. Data Processing (`process_data.py`)

**Input**: `../../data/raw/sentiment.csv`

**Process**:
1. **Series Statistics**: Calculate mean and standard error for each sentiment metric per series
2. **Weighted Trend Analysis**: Perform linear regression of series means vs. series number using inverse standard errors as weights
3. **Multiple Testing Correction**: Apply Benjamini-Hochberg FDR correction at 1% level across 7 sentiment metrics
4. **Effect Size Calculation**: Compute standardized effect sizes for trends

**Sentiment Metrics Analyzed**:
- anger, awkwardness, frustration/despair, humor, joy/excitement, sarcasm, self-deprecation

**Output**: Two CSV files with trend results and series statistics

### 2. Visualization (`plot_figure7.py`)

**Creates separate visualizations**:

**Figure 7a - Significant Sentiment Trend**:
- Linear trend plot with individual series points
- Trend line with 95% confidence interval  
- Statistical annotations (slope, p-value, R²)
- Focus on awkwardness as the single significant trend

**Figure 7b - All Sentiment Distributions**:
- Violin plots showing full distributions across all episodes
- Highlights awkwardness (red) vs. other sentiments (gray)
- Shows median, mean, and probability density for each metric

## Key Findings

### Statistical Results
- **1 significant trend** out of 7 sentiment metrics (FDR < 0.01)
- **Awkwardness increasing**: slope = 0.012 units/series, p = 0.0027, R² = 0.576
- **Strong effect**: 57.6% of variance in awkwardness explained by series number
- **Robust methodology**: Weighted regression with FDR correction

### Substantive Interpretation
- **Temporal evolution**: Episodes have become systematically more cringe-inducing over time
- **Unique pattern**: All other emotional dimensions remain stable across 18 series
- **Production implications**: Suggests evolution in contestant selection, task design, or comedic emphasis

## Usage

```bash
# Run the complete analysis
python process_data.py
python plot_figure7.py

# Individual figure creation
python plot_figure7.py  # Creates both 7a and 7b separately
```

## Dependencies

- pandas, numpy, scipy, matplotlib, seaborn
- Custom plot utilities from `../../config/plot_utils.py`
- Configuration from `../../config/plot_config.yaml`

## Notes

- Analysis focuses exclusively on sentiment metrics (emotional content)
- Uses weighted regression to account for within-series variability
- FDR correction controls for multiple comparisons across sentiment dimensions
- Represents the most rigorous temporal analysis of Taskmaster's emotional evolution 