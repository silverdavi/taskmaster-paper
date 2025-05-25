# A Quantitative Exploration of Taskmaster: Data Analysis and Visualization

This repository contains the complete codebase for a comprehensive quantitative analysis of the British comedy panel show Taskmaster. The project explores contestant archetypes, task dynamics, audience reception patterns, and comedic sentiment through rigorous statistical analysis and data visualization.

>  **For complete research methodology and findings, see [PaperGuide.MD](PaperGuide.MD)**

## Project Overview

This analysis leverages multiple datasets to provide empirical insights into Taskmaster's enduring appeal and operational mechanics. The study employs advanced statistical techniques including mixture model fitting, principal component analysis, clustering algorithms, and sentiment analysis to understand the show's unique blend of structure and comedic unpredictability.

> ANALYSIS: **For detailed analysis framework and module documentation, see [FiguresGuide.MD](FiguresGuide.MD)**

## Directory Structure

```
taskmaster-paper/
├── config/                                    # Configuration and utilities
│   ├── plot_config.yaml                      # Visual settings for all plots
│   └── plot_utils.py                         # Shared plotting utilities
├── data/
│   ├── raw/                                  # Original datasets with documentation
│   │   ├── DATA_SOURCES_AND_METHODOLOGY.md  # Comprehensive data documentation
│   │   ├── sentiment_analysis.py            # Reference script for sentiment extraction
│   │   ├── contestants.csv                  # Contestant demographics and performance
│   │   ├── imdb_ratings.csv                 # Official IMDb episode ratings
│   │   ├── taskmaster_histograms_corrected.csv # IMDb vote distributions (1-10)
│   │   ├── sentiment.csv                    # Episode-level sentiment analysis
│   │   ├── _OL_tasks.csv                    # GPT-4o classified task types
│   │   ├── taskmaster_uk_episodes.csv       # Episode metadata
│   │   ├── Cont_lon_lat.tsv                 # Contestant geographic coordinates
│   │   └── ...
│   ├── processed/                           # Processed data organized by analysis
│   └── taskmaster_data_documentation.md     # Data processing overview
├── figures/                                 # Analysis modules (one per research question)
│   ├── series_ratings_analysis/             # IMDb rating distributions & mixture models
│   ├── episode_rating_trajectories/         # Within-series rating patterns
│   ├── task_characteristics_analysis/       # Task typology and demand analysis
│   ├── contestant_geographic_origins/       # Geographic distribution analysis
│   ├── contestant_performance_archetypes/   # Performance-based clustering
│   ├── sentiment_trends_analysis/           # Comedic sentiment patterns
│   ├── predictive_modeling_analysis/        # Episode success prediction models
│   ├── scoring_pattern_geometry/            # Task scoring system analysis
│   ├── task_skill_profiles/                 # Task-skill requirement mapping
│   └── individual_series_analysis/          # Series-specific deep dives
├── PaperGuide.MD                           # Complete research methodology and findings
├── FiguresGuide.MD                         # Figure generation and interpretation guide
├── generate_all_figures.py                # Master script for all analyses
└── requirements.txt                        # Python dependencies
```

> DATA: **For comprehensive data statistics and insights, see [data/taskmaster_data_documentation.md](data/taskmaster_data_documentation.md)**

## Key Features

### Advanced Statistical Methods
- **Mixture Model Fitting**: IMDb rating distributions modeled as delta functions (ratings 1 & 10) + Gaussian (ratings 2-9)
- **Goodness of Fit Analysis**: Quantitative comparison of mixture vs. naive Gaussian models
- **Principal Component Analysis**: Series-level reception quality profiling
- **Clustering Analysis**: Contestant archetype identification using demographic and performance data
- **Sentiment Analysis**: GPT-4o powered extraction of comedic sentiment patterns

### Comprehensive Data Documentation
- Detailed methodology for each dataset with reliability assessments
- Transparent discussion of AI-assisted data collection limitations
- Correlation validation between raw vote histograms and official IMDb scores (>99% correlation)

>  **For detailed data sources and collection methodology, see [data/raw/DATA_SOURCES_AND_METHODOLOGY.md](data/raw/DATA_SOURCES_AND_METHODOLOGY.md)**

### Reproducible Research Framework
- Modular analysis structure with clear separation of data processing and visualization
- Consistent styling and configuration across all figures
- Comprehensive goodness-of-fit metrics and statistical validation

>  **For implementation workflow and quality standards, see [figures_workflow.md](figures_workflow.md)**

## Setup and Installation

1. **Create Python Environment** (Python 3.8+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation**: Raw data files are included in `data/raw/` with comprehensive documentation

>  **For current project status and development roadmap, see [ISSUES_AND_ROADMAP.md](ISSUES_AND_ROADMAP.md)**

## Usage

### Generate All Analyses
```bash
python generate_all_figures.py
```

### Generate Specific Analysis Module
```bash
python generate_all_figures.py --module series_ratings_analysis
```

### Process Data Only (No Visualization)
```bash
python generate_all_figures.py --process-only
```

### Generate Plots Only (Skip Data Processing)
```bash
python generate_all_figures.py --plot-only
```

### List All Available Modules
```bash
python generate_all_figures.py --list
```

> FEATURE: **For complete usage examples and module descriptions, run the above command or see [FiguresGuide.MD](FiguresGuide.MD)**

## Key Research Findings

### Series-Level Reception Analysis
- **Mixture Model Performance**: Mean Absolute Error of 1.8% vs. 4.1% for naive Gaussian
- **Rating Distribution Patterns**: Clear identification of polarizing vs. consensus episodes
- **Series Quality Metrics**: Quantitative profiling using #1s, #10s, μ, and σ parameters

### Task Characteristics and Performance
- **Task Typology**: Data-driven classification based on creativity, physicality, and technical demands
- **Performance Correlations**: Systematic analysis of contestant archetype success across task types
- **Scoring System Analysis**: Comprehensive evaluation of the 0-5 point scoring distribution

### Audience Reception Dynamics
- **Episode Trajectory Patterns**: Statistical identification of "Rising," "J-shape," and other rating patterns
- **Sentiment-Reception Correlations**: Quantitative links between comedic sentiment and audience appreciation
- **Geographic and Demographic Influences**: Analysis of contestant diversity impact on show reception

> TREND: **For detailed findings and statistical analysis, see [PaperGuide.MD](PaperGuide.MD)**

## Data Sources and Reliability

### High Reliability (Statistical Analysis)
- **IMDb Data**: Official ratings and vote distributions (>99% correlation validation)
- **Episode Metadata**: Verified information from taskmaster.info

### Medium Reliability (Exploratory Analysis)
- **Contestant Demographics**: Web-sourced with manual verification
- **Geographic Data**: Official birthplace coordinates

### Exploratory Only (No Statistical Conclusions)
- **Task Classifications**: GPT-4o assisted categorization for pattern identification
- **Sentiment Analysis**: GPT-4o extraction with >80% accuracy vs. <60% for lexicon methods

> ANALYSIS: **For complete data reliability assessment and collection methodology, see [data/raw/DATA_SOURCES_AND_METHODOLOGY.md](data/raw/DATA_SOURCES_AND_METHODOLOGY.md)**

> DATA: **For comprehensive data statistics (90 contestants, 917 tasks, 154 episodes), see [data/taskmaster_data_documentation.md](data/taskmaster_data_documentation.md)**

## Analysis Modules

The project includes 10 comprehensive analysis modules:

1. **Series Ratings Analysis**: Mixture model fitting and goodness of fit analysis
2. **Episode Rating Trajectories**: Within-series rating pattern analysis  
3. **Task Characteristics Analysis**: Task typology and demand analysis
4. **Geographic Origins**: Contestant birthplace and cultural analysis
5. **Performance Archetypes**: Clustering analysis of contestant types
6. **Sentiment Trends**: Comedic sentiment pattern analysis
7. **Predictive Modeling**: Episode success prediction models
8. **Scoring Pattern Geometry**: Task scoring system analysis
9. **Task Skill Profiles**: Task-skill requirement mapping
10. **Individual Series Analysis**: Series-specific deep dive analysis

> ANALYSIS: **For detailed module documentation and implementation status, see [FiguresGuide.MD](FiguresGuide.MD)**

>  **For module naming history and evolution, see [figures/FIGURE_NAMING_HISTORY.md](figures/FIGURE_NAMING_HISTORY.md)**

## Contributing

When contributing to this analysis:

1. Follow the established modular structure in `figures/`
2. Use consistent styling from `config/plot_config.yaml`
3. Document all statistical methods and assumptions
4. Include goodness-of-fit metrics for model validation
5. Update relevant documentation files

>  **For detailed workflow standards and quality assurance, see [figures_workflow.md](figures_workflow.md)**

## Academic Context

This work contributes to the quantitative analysis of comedy television, providing empirical support for theories of humor (Incongruity, Superiority, Relief) and demonstrating methodologies for analyzing unscripted entertainment formats. The framework established here can serve as a model for studying other panel game shows or comedic formats.

>  **For complete academic context and research contributions, see [PaperGuide.MD](PaperGuide.MD)**

## Citation

If you use this work in academic research, please cite the associated paper and acknowledge the comprehensive dataset compilation and analysis methodology developed in this repository.

## License

This project is for academic research purposes. Raw data sources are acknowledged in `data/raw/DATA_SOURCES_AND_METHODOLOGY.md`.

---

## Documentation Index

- **[PaperGuide.MD](PaperGuide.MD)**: Complete research methodology, findings, and academic framework
- **[FiguresGuide.MD](FiguresGuide.MD)**: Analysis module documentation and implementation guide
- **[figures_workflow.md](figures_workflow.md)**: Implementation workflow and quality standards
- **[data/taskmaster_data_documentation.md](data/taskmaster_data_documentation.md)**: Comprehensive data statistics and insights
- **[data/raw/DATA_SOURCES_AND_METHODOLOGY.md](data/raw/DATA_SOURCES_AND_METHODOLOGY.md)**: Data collection methodology and reliability assessment
- **[ISSUES_AND_ROADMAP.md](ISSUES_AND_ROADMAP.md)**: Issues tracker and development roadmap
- **[figures/FIGURE_NAMING_HISTORY.md](figures/FIGURE_NAMING_HISTORY.md)**: Evolution of analysis module naming and structure