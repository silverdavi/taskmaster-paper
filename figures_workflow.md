# Taskmaster Paper: Figures Implementation Plan

## 1. Directory Structure

```
taskmaster_paper/
├── config/
│   ├── plot_config.yaml                # Main configuration file for all plots
│   └── plot_utils.py                   # Shared utility functions for plotting
├── data/
│   └── raw/                            # Original CSV files
│       ├── contestants.csv
│       ├── scores.csv
│       ├── tasks.csv
│       ├── sentiment.csv
│       ├── imdb_ratings.csv
│       ├── taskmaster_histograms_corrected.csv
│       ├── taskmaster_UK_tasks.csv
│       └── taskmaster_uk_episodes.csv
├── figures/
│   ├── figure1/                        # Series-Level IMDb Ratings
│   │   ├── process_data_figure1.py     # Data processing script
│   │   ├── plot_figure1.py             # Plotting script
│   │   ├── figure1.md                  # Documentation
│   │   ├── series_metrics.csv          # Processed data
│   │   ├── series_pca.csv              # Processed data
│   │   ├── pca_loadings.csv            # Processed data
│   │   ├── explained_variance.npy      # Processed data
│   │   ├── metrics.json                # Metrics for caption
│   │   ├── figure1_output.pdf          # Generated figure
│   │   └── caption.txt                 # Generated caption
│   ├── figure2/                        # Episode Rating Trajectories
│   │   ├── process_data_figure2.py
│   │   ├── plot_figure2.py
│   │   └── ...
│   └── ...
└── notebooks/                          # Optional exploratory notebooks
    ├── explore_figure1.ipynb
    └── ...
```

## 2. Configuration Setup

### `config/plot_config.yaml`

```yaml
# Global plot settings
global:
  dpi: 300
  figure_size: [10, 6]
  font_family: "Arial"
  output_format: "pdf"
  grid: True
  version: "1.0"

# Color schemes
colors:
  series_colormap: "viridis"  # For 18 series
  archetype_palette: "Set2"
  task_type_palette: "Paired"
  highlight:
    good: "#2ecc71"  # Green
    neutral: "#3498db"  # Blue
    bad: "#e74c3c"  # Red
  
  sentiment:
    humor: "#ff7f0e"
    sarcasm: "#d62728"
    awkwardness: "#9467bd"
    joy: "#2ca02c"

# Text settings
fonts:
  title_size: 16
  axis_label_size: 12
  tick_label_size: 10
  legend_size: 10

# Line and marker styles
styles:
  line_width: 1.5
  marker_size: 6
  markers: ["o", "s", "^", "d", "P", "*"]
```

## 3. Processing and Plotting Workflow

Each figure follows a two-step process with a strict separation of concerns:

### Step 1: Data Processing (`process_data_figureX.py`)

This script:
1. Loads raw data from `data/raw/`
2. Performs necessary transformations and analyses
3. Saves processed data files directly to the figure directory
4. Generates metrics for the figure caption and saves them to `metrics.json`

### Step 2: Plotting (`plot_figureX.py`)

This script:
1. Loads processed data from the figure directory
2. Creates the figure with all subplots
3. Applies consistent styling
4. Saves the output image as `figureX_output.{format}`
5. Generates and saves the caption

## 4. Naming Conventions

- **Processing Script**: `process_data_figure{X}.py`
- **Plotting Script**: `plot_figure{X}.py`
- **Output Figure**: `figure{X}_output.pdf`
- **Processed Data**: Descriptive names (e.g., `series_metrics.csv`, `pca_results.csv`)
- **Metrics**: `metrics.json`
- **Caption**: `caption.txt`
- **Documentation**: `figure{X}.md`

## 5. Implementation Plan for Each Figure

### Figure 1: Series-Level IMDb Ratings

- **Data Processing**:
  - Load `imdb_ratings.csv` and `taskmaster_histograms_corrected.csv`
  - Fit Gaussian distributions to ratings 2-9 for each series
  - Aggregate #1s and #10s counts
  - Perform PCA on the series-level metrics
  - Save: `series_metrics.csv`, `series_pca.csv`, `pca_loadings.csv`, `explained_variance.npy`
  
- **Plotting**:
  - Create ridge plot of rating distributions (subplot A)
  - Plot PCA results with series colored by reception quality (subplot B)
  - Generate output: `figure1_output.pdf`, `caption.txt`

### Figure 2: Episode Rating Trajectories

- **Data Processing**:
  - Analyze episode ratings within each series
  - Categorize series by trajectory patterns (Rising, J-shape, etc.)
  - Calculate statistics for each trajectory type
  - Save processed data directly to figure2 folder
  
- **Plotting**:
  - Bar chart of trajectory types (subplot A)
  - Violin plots of episode thirds by type (subplot B)
  - Generate output: `figure2_output.pdf`, `caption.txt`

### Figure 3: Task Type Landscape

- **Data Processing**:
  - Load `tasks.csv` and `taskmaster_UK_tasks.csv`
  - Perform t-SNE on task locations
  - Analyze task type distributions
  - Select extreme/unique tasks for radar plots
  - Save processed data directly to figure3 folder
  
- **Plotting**:
  - t-SNE plot of locations (subplot A)
  - Distribution of task types (subplot B)
  - Radar plots of selected tasks (subplot C)
  - Heatmap of location vs. task type (subplot D)
  - Generate output: `figure3_output.pdf`, `caption.txt`

### Figure 4: Task Scoring Patterns

- **Data Processing**:
  - Load `scores.csv`
  - Calculate score distributions
  - Compute mean and variance of scores per series
  - Save processed data directly to figure4 folder
  
- **Plotting**:
  - Histogram of individual scores (subplot A)
  - Scatter plot of mean vs. variance by series (subplot B)
  - Generate output: `figure4_output.pdf`, `caption.txt`

### Figure 5: Contestant Demographics and Dynamics

- **Data Processing**:
  - Load `contestants.csv` and `scores.csv`
  - Analyze demographic distributions
  - Calculate average scores per contestant
  - Extract score progression for Series 7
  - Save processed data directly to figure5 folder
  
- **Plotting**:
  - Demographic summary plots (subplot A)
  - Sorted average scores (subplot B)
  - Line plots of score progression (subplot C)
  - Generate output: `figure5_output.pdf`, `caption.txt`

### Figure 6: Sentiment Profiles

- **Data Processing**:
  - Load `sentiment.csv`
  - Analyze sentiment trends over series
  - Calculate correlation matrix of sentiments
  - Save processed data directly to figure6 folder
  
- **Plotting**:
  - Line plots of key sentiments (subplot A)
  - Correlation matrix or scatter plots (subplot B)
  - Generate output: `figure6_output.pdf`, `caption.txt`

### Figure 7: Synthesis of All Correlations

- **Data Processing**:
  - Merge data from all sources
  - Calculate correlations between key variables
  - Perform MDS on task composition
  - Save processed data directly to figure7 folder
  
- **Plotting**:
  - Boxplots of scores by task type/location (subplot A)
  - Scatter plots of sentiment vs. metrics (subplot B)
  - MDS plot and correlation table (subplot C)
  - Generate output: `figure7_output.pdf`, `caption.txt`

## 6. Execution Strategy

1. **First Implementation Phase**:
   - Create the `config` directory with `plot_config.yaml` and `plot_utils.py`
   - Set up the directory structure
   - Implement Figure 1 as a proof of concept

2. **Parallel Development**:
   - Assign figures to team members for concurrent development
   - Use the templates to ensure consistency

3. **Quality Control**:
   - Review each figure for adherence to style guidelines
   - Verify that all metrics are properly logged for captions
   - Check for consistent colormaps and styling

4. **Integration**:
   - Create a master script to generate all figures
   - Compile all captions into a single document for the paper

5. **Version Control**:
   - Tag versions of the figures as they are finalized
   - Document any significant changes or improvements 