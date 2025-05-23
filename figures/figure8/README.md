# Figure 8: Episode-Level ML Analysis & Series-Level Correlation Analysis

This directory contains the complete Figure 8 analysis with two complementary approaches:

## 🤖 Episode-Level ML Analysis (Figure 8a)

**Objective**: Predict IMDB score distributions using machine learning models

### Pipeline:
1. **`1_prepare_episode_data.py`** - Data preparation and feature engineering
2. **`2_feature_selection_episode.py`** - Feature selection using mutual information
3. **`3_model_episode.py`** - Train and evaluate ML models
4. **`4_plot_figure8a.py`** - Generate Figure 8a visualization

### Models Used:
- **Linear Regression** - Baseline linear model
- **Ridge Regression** - Regularized linear model  
- **Random Forest** - Ensemble tree-based model

### Best Results:
- **Random Forest with top 5 features**: R² = 0.385
- **Key features**: contestant_avg_age, avg_awkwardness, contestant_avg_experience
- **Episodes analyzed**: 154

## 📊 Series-Level Correlation Analysis (Figure 8b)

**Objective**: Analyze raw correlations between input features and IMDB scores

### Pipeline:
1. **`1_prepare_series_data.py`** - Aggregate data at series level
2. **`correlation_analysis_raw.py`** - Calculate correlations with IMDB histograms as target

### Key Findings:
- **45 valid correlations** between input features and mean IMDB scores
- **Strongest negative**: contestant_prop_actors (r = -0.547)
- **Strongest positive**: contestant_avg_age (r = +0.397)
- **Distribution**: Nearly normal around zero (μ=-0.025, σ=0.199)

## 🎯 Random Forest Feature Analysis

**Additional Analysis**: `analyze_rf_features.py` - Deep dive into Random Forest insights

### Key Insights for Maximizing IMDB Scores:
1. **👴 Maximize**: Contestant average age (39.5% importance, +0.396 correlation)
2. **😬 Minimize**: Average awkwardness (32.6% importance, -0.151 correlation)  
3. **📺 Maximize**: Contestant experience (16.2% importance, +0.019 correlation)

## 📁 Data Files

### Input Data:
- **`episode_data.csv`** - Prepared episode-level data (N=154)
- **`series_data.csv`** - Prepared series-level data
- **`episode_selected_features.json`** - Feature selection results

### Model Results:
- **`episode_model_results.pkl`** - Complete ML model results
- **`raw_correlations.json`** - Correlation analysis results

## 📈 Output Figures

- **`figure8a_episode_ml.png`** - Model performance comparison (3 models)
- **`figure8b_raw_correlations.png`** - Correlation distribution histogram
- **`random_forest_feature_analysis.png`** - Detailed feature importance analysis

## 🔬 Methodology

### Episode-Level (ML Approach):
- **Target**: IMDB histogram percentages (10-dimensional)
- **Features**: Demographics, sentiment, experience metrics
- **Validation**: 5-fold cross-validation + holdout test set
- **Metric**: R² score for explained variance

### Series-Level (Correlation Approach):  
- **Target**: Mean IMDB score computed from vote histograms
- **Features**: Raw input data from multiple sources
- **Analysis**: Direct Pearson correlations
- **Visualization**: Distribution of correlation coefficients

## 🏆 Key Scientific Findings

1. **Professional Polish Matters**: Older, experienced comedians score higher
2. **Awkwardness Hurts**: Smooth performances outperform awkward ones
3. **Demographics Drive Ratings**: Age and profession are strongest predictors
4. **ML vs Correlation**: Both approaches reveal consistent patterns

---

**Figure 8 demonstrates that Taskmaster episode success is largely predictable from contestant characteristics, with Random Forest achieving 38.5% explained variance.**

## Key Insight: Appropriate Methods for Data Scale

This analysis demonstrates choosing the right statistical approach:
- **Episode-level**: Sufficient data for machine learning
- **Series-level**: Simple correlation analysis provides interpretable insights

## File Structure

### Episode-Level Pipeline (Figure 8a)
1. **`1_prepare_episode_data.py`** - Data preparation (154 episodes, 12 features)
2. **`2_feature_selection_episode.py`** - Information entropy feature selection  
3. **`3_model_episode.py`** - ML modeling with train/test split

### Series-Level Pipeline (Figure 8b)
1. **`1_prepare_series_data.py`** - Data preparation (18 series, aggregated features)
2. **`correlation_analysis_raw.py`** - **Raw correlation analysis** (final approach)

### Generated Data Files
- **`episode_data.csv`** - Episode-level features and targets
- **`series_data.csv`** - Series-level aggregated data
- **`episode_selected_features.json`** - Episode feature selection results
- **`episode_model_results.pkl`** - Episode ML model results
- **`raw_correlations.json`** - Series correlation analysis results

### Output Figures
- **`figure8b_raw_correlations.png`** - **Final Figure 8b** (correlation histogram with Gaussian fit)

## Key Results

### Episode-Level (Successful ML)
- **Best Model**: Random Forest (R² = 0.385)
- **Top Predictors**: contestant_avg_age (39.5%), avg_awkwardness (32.6%), contestant_avg_experience (16.2%)
- **Sample Size**: Adequate for reliable ML (N=154)

### Series-Level (Correlation Analysis)
- **Strongest Correlation**: contestant_prop_actors (r = -0.547) - more actors → lower IMDB ratings
- **Key Findings**: 
  - Older contestants → higher ratings (r = +0.397)
  - Solo tasks preferred over team tasks
  - Special/unique tasks boost ratings
- **Distribution**: Normal around zero (μ=-0.025, σ=0.199)

## Data Sources (INPUT Features)

All analyses use the same input data sources:
- **`data/raw/taskmaster_UK_tasks.csv`** - Task characteristics
- **`data/raw/sentiment.csv`** - Sentiment analysis  
- **`data/processed/scores_by_series/series_*_scores.csv`** - Contestant scoring
- **`data/raw/contestants.csv`** - Contestant demographics

**TARGET (OUTPUT)**: Mean IMDB scores computed from `data/raw/taskmaster_histograms_corrected.csv`

## Usage

### Run Episode-Level Analysis:
```bash
python 1_prepare_episode_data.py
python 2_feature_selection_episode.py  
python 3_model_episode.py
```

### Run Series-Level Analysis:
```bash
python 1_prepare_series_data.py
python correlation_analysis_raw.py
```

## Scientific Value

This analysis provides methodological education:
1. **Sample size considerations**: When to use ML vs simple statistics
2. **Feature consistency**: Same input features analyzed at different scales  
3. **Interpretable results**: Both approaches yield meaningful insights
4. **Appropriate methodology**: Matching analysis complexity to data constraints

## Key Finding

**Contestant demographics and task structure consistently predict IMDB ratings** across both episode and series levels, with older contestants and comedians correlating with higher ratings, while actors correlate with lower ratings. 