# Taskmaster Paper: Figures Overview and Generation Framework

## I. Overview and Workflow Guidelines

### A. Configuration File (`plot_config.py` or `config.yaml`)

A central configuration file will control visual consistency across all figures:

* **Fonts**: Title, axis, legend fonts (e.g., Arial or Times New Roman)
* **Colormaps**:

  * **Series Colormap (18 series)**: Discrete, perceptually ordered (e.g., `viridis`, custom)
  * **Categorical Palettes**: For archetypes, task types, sentiments (e.g., `Set2`, `Paired`)
  * **Highlight Colors**: Green (good), red (bad)
  * **Sentiment Colors**: Assigned uniquely
* **Figure Aesthetics**:

  * Default size (e.g., 6x4 in), DPI 300
  * Line/marker styles
  * Grid visibility

### B. Directory Structure

```
taskmaster_paper/
├── config/
│   └── plot_config.py or plot_config.yaml
├── data/
│   ├── raw/                # Original CSVs
│   └── processed/          # By figure
│       ├── figure1/
│       └── figure2/
├── figures/
│   ├── figure1/
│   │   ├── process_figure1_data.py
│   │   ├── plot_figure1.py
│   │   └── figure1_output.png/pdf
│   └── ...
```

### C. Per-Figure Script Workflow

#### 1. Data Processing (`process_figureX_data.py`):

* Load raw CSVs from `data/raw/`
* Clean, transform, analyze (PCA, clustering, regression, etc.)
* Output to `data/processed/figureX/`
* Log key stats (e.g., p-values, R^2)

#### 2. Plot Generation (`plot_figureX.py`):

* Load processed data + global config
* Plot with `matplotlib`, `seaborn`
* Save high-res image to `figures/figureX/`

---

## II. Figures Summary with Descriptions and Captions

Each figure includes:

* Folder location
* Required processed data files
* Subplot descriptions
* Caption placeholder with variables to be filled in from script output

### Figure 1: Series-Level IMDb Ratings

* **Subplots**:

  * (A) Ridge plot: Distributions (Gaussian fit of 2–9) + #1s/#10s markers
  * (B) PCA: Series colored by reception quality profile

### Figure 2: Episode Rating Trajectories

* (A) Bar summary of trajectory types (Rising, J-shape, etc.)
* (B) Violin plots of episode thirds by type

### Figure 3: Task Type Landscape

* (A) t-SNE of locations
* (B) Task type and solo/group distribution
* (C) Radar plots of selected extreme/unique tasks
* (D) Heatmap of location vs. task type

### Figure 4: Task Scoring Patterns

* (A) Histogram of individual task scores (0–5)
* (B) Mean vs. variance of scores per series, colored by IMDb

### Figure 5: Contestant Demographics and Dynamics

* (A) Summary plots: gender, age, occupation
* (B) Sorted average scores per contestant (color = series)
* (C) Line plots of score progression in Series 7 (appendix for all)

### Figure 6: Sentiment Profiles

* (A) Trends of key sentiments over series
* (B) Correlation matrix or 2–4 key sentiment scatter plots

### Figure 7: Synthesis of All Correlations

* (A) Boxplots of scores by task type/location
* (B) Scatter plots: sentiment vs. score/IMDb/variance
* (C) MDS of task composition + correlation table

---

## III. Final Notes

* All processing scripts must log metrics needed for figure captions.
* Use consistent colormap and font styles across all plots.
* Track and label outlier series/contestants explicitly.
* Include version tags in each plot file (e.g., v1.0, date-stamped).
* Appendices may include additional time-series, full correlation matrices, etc.

This document should serve as the centralized reference for all figure design, script coordination, and consistency enforcement for the Taskmaster data paper.
