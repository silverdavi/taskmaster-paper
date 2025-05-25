# Taskmaster Paper Figure Generation

This repository contains the code for generating figures for the Taskmaster data analysis paper.

## Directory Structure

```
taskmaster_paper/
├── config/                          # Configuration files
│   ├── plot_config.yaml             # Visual settings for plots
│   └── plot_utils.py                # Shared plotting utilities
├── data/
│   ├── raw/                         # Original CSV files
│   └── processed/                   # Processed data organized by figure
├── figures/                         # One directory per figure
│   ├── figure1/                     # Series-Level IMDb Ratings
│   ├── figure2/                     # Episode Rating Trajectories
│   └── ...
├── generate_all_figures.py          # Master script for generating all figures
└── requirements.txt                 # Python dependencies
```

## Setup

1. Create a Python environment (Python 3.8+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the raw data files in the `data/raw/` directory:
   - contestants.csv
   - scores.csv
   - _OL_tasks._csv
   - sentiment.csv
   - imdb_ratings.csv
   - taskmaster_histograms_corrected.csv
   - taskmaster_UK_tasks.csv
   - taskmaster_uk_episodes.csv
   - Cont_lon_lat.tsv
   - long_task_scores.csv

## Usage

### Generate All Figures

To generate all figures:

```bash
python generate_all_figures.py
```

### Generate a Specific Figure

To generate a specific figure (e.g., Figure 1):

```bash
python generate_all_figures.py --figure 1
```

### Process Data Only

To only process data without generating plots:

```bash
python generate_all_figures.py --process-only
```

### Plot Only

To only generate plots without reprocessing data:

```bash
python generate_all_figures.py --plot-only
```

### Collect Captions

To collect all figure captions into a single document:

```bash
python generate_all_figures.py --captions-only
```

## Figure Descriptions

1. **Figure 1: Series-Level IMDb Ratings**
   - Ridge plot of rating distributions
   - PCA plot of series reception quality profiles

2. **Figure 2: Episode Rating Trajectories**
   - Bar summary of trajectory types
   - Violin plots of episode thirds by type

3. **Figure 3: Task Type Landscape**
   - t-SNE of task locations
   - Task type distribution
   - Radar plots of selected tasks
   - Heatmap of location vs. task type

4. **Figure 4: Task Scoring Patterns**
   - Histogram of individual scores
   - Mean vs. variance by series

5. **Figure 5: Contestant Demographics and Dynamics**
   - Demographic summary plots
   - Sorted average scores
   - Score progression line plots

6. **Figure 6: Sentiment Profiles**
   - Trends of key sentiments
   - Correlation matrix or scatter plots

7. **Figure 7: Synthesis of All Correlations**
   - Boxplots of scores by task type/location
   - Sentiment vs. metrics scatter plots
   - MDS plot and correlation table

## Implementation Guide

To implement a new figure:

1. Create process_data.py and plot_figure.py scripts in the appropriate figure directory
2. Follow the established patterns for data processing and plotting
3. Use the utilities in plot_utils.py for consistent styling

## Contributing

When contributing new figures or modifying existing ones:

1. Follow the established code style
2. Ensure all plots use the consistent styling from plot_config.yaml
3. Document all metrics needed for figure captions
4. Update README.md with any new figure descriptions