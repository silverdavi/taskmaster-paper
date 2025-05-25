# Figure 8: Predictive Modeling Analysis

This directory contains the complete analysis pipeline for Figure 8 of the Taskmaster paper, which demonstrates machine learning approaches to predicting IMDB episode ratings.

## 📁 Directory Structure

### Core Pipeline Scripts (Run in Order)
1. **`1_prepare_episode_data.py`** - Data preparation and feature engineering
2. **`2_feature_selection_episode.py`** - Feature selection using mutual information
3. **`3_model_episode_analysis.py`** - Train and evaluate ML models
4. **`4_plot_figure8a.py`** - Generate Figure 8a visualization
5. **`5_correlation_analysis_figure8b.py`** - Generate Figure 8b correlation analysis
6. **`6_analyze_random_forest_features.py`** - Deep dive into Random Forest insights

### Generated Data Files
- **`episode_data.csv`** - Prepared episode-level dataset (154 episodes, 12 features)
- **`episode_selected_features.json`** - Feature selection results and rankings
- **`episode_model_results.pkl`** - Complete ML model results and predictions
- **`raw_correlations.json`** - Correlation analysis results for Figure 8b

### Output Figures
- **`figure8a_episode_ml.png/pdf`** - Episode-level ML performance comparison
- **`figure8b_raw_correlations.png/pdf`** - Correlation distribution analysis
- **`random_forest_feature_analysis.png/pdf`** - Feature importance insights

### Documentation
- **`FIGURE8_OUTPUT_SUMMARY.md`** - Complete analysis summary
- **`FIGURE8B_FINAL_SUMMARY.md`** - Figure 8b specific results
- **`RANDOM_FOREST_INSIGHTS.md`** - Strategic insights from Random Forest analysis

## 🚀 Quick Start

### Run Complete Analysis
```bash
# Step 1: Prepare data
python 1_prepare_episode_data.py

# Step 2: Select features  
python 2_feature_selection_episode.py

# Step 3: Train models
python 3_model_episode_analysis.py

# Step 4: Create Figure 8a
python 4_plot_figure8a.py

# Step 5: Create Figure 8b
python 5_correlation_analysis_figure8b.py

# Step 6: Analyze Random Forest insights
python 6_analyze_random_forest_features.py
```

### Or Run All at Once
```bash
# Run the complete pipeline
for script in 1_prepare_episode_data.py 2_feature_selection_episode.py 3_model_episode_analysis.py 4_plot_figure8a.py 5_correlation_analysis_figure8b.py 6_analyze_random_forest_features.py; do
    echo "Running $script..."
    python $script
done
```

## 📊 Analysis Overview

### Figure 8a: Episode-Level ML Analysis
**Objective**: Predict IMDB score distributions using machine learning

**Approach**:
- **Dataset**: 154 episodes with 12 features
- **Models**: Linear Regression, Ridge Regression, Random Forest
- **Target**: IMDB histogram percentages (10-dimensional)
- **Validation**: 5-fold cross-validation + holdout test

**Key Results**:
- **Best Model**: Random Forest (R² = 0.385)
- **Top Predictors**: contestant_avg_age (39.5%), avg_awkwardness (32.6%), contestant_avg_experience (16.2%)
- **Insight**: Professional polish matters more than amateur charm

### Figure 8b: Correlation Analysis
**Objective**: Analyze raw correlations between input features and IMDB scores

**Approach**:
- **Dataset**: 154 episodes with 45 input features
- **Target**: Mean IMDB score computed from vote histograms
- **Analysis**: Direct Pearson correlations
- **Visualization**: Distribution of correlation coefficients

**Key Results**:
- **Strongest Negative**: contestant_prop_actors (r = -0.547)
- **Strongest Positive**: contestant_avg_age (r = +0.397)
- **Distribution**: Nearly normal around zero (μ=-0.025, σ=0.199)

## 🎯 Key Scientific Findings

### 1. Professional Polish Beats Amateur Charm
- **Older contestants** (41+ years) correlate with higher ratings
- **Professional comedians** enhance entertainment value
- **TV experience** matters for audience satisfaction

### 2. Awkwardness Hurts Ratings
- **Smooth, confident performances** score better than awkward ones
- **Professional backgrounds** (comedians) outperform actors
- **Polished entertainment** preferred over raw authenticity

### 3. Predictable Patterns
- **38.5% of IMDB variance** explained by contestant characteristics
- **Age and experience** are strongest predictors
- **Demographics drive ratings** more than task content

## 🔬 Methodology

### Data Sources
- **`data/raw/taskmaster_UK_tasks.csv`** - Task characteristics
- **`data/raw/sentiment.csv`** - Sentiment analysis
- **`data/processed/scores_by_series/series_*_scores.csv`** - Contestant scoring
- **`data/raw/contestants.csv`** - Contestant demographics
- **`data/raw/taskmaster_histograms_corrected.csv`** - IMDB ratings (target)

### Feature Engineering
- **Sentiment metrics**: Anger, awkwardness, humor, sarcasm, etc.
- **Contestant demographics**: Age, experience, profession, gender
- **Task characteristics**: Activity types, judgment patterns
- **Aggregation**: Episode-level means and proportions

### Model Selection
- **Feature Selection**: Mutual information + entropy analysis
- **Cross-Validation**: 5-fold CV with stratification
- **Model Comparison**: Linear, Ridge, Random Forest
- **Evaluation**: R², MAE, feature importance

## 📈 Usage Examples

### Load Results for Further Analysis
```python
import pickle
import pandas as pd

# Load model results
with open('episode_model_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Load episode data
episode_data = pd.read_csv('episode_data.csv')

# Get Random Forest performance
rf_results = results['all_results']['top_5']['results']['Random Forest']
print(f"Random Forest R²: {rf_results['test_r2']:.3f}")
```

### Access Feature Importance
```python
# Get feature importance from Random Forest
feature_importance = rf_results['feature_importance']
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("Top 5 Features:")
for feature, importance in top_features[:5]:
    print(f"  {feature}: {importance:.3f}")
```

## 🏆 Strategic Insights

### For Maximizing IMDB Scores:
1. **Cast older, experienced performers** (40+ years)
2. **Include professional comedians** in each series
3. **Minimize awkward moments** during filming/editing
4. **Prioritize TV veterans** over newcomers
5. **Balance or favor female representation**

### Scientific Value:
- **Demonstrates predictable patterns** in entertainment ratings
- **Shows demographic effects** on audience satisfaction
- **Provides actionable insights** for casting decisions
- **Validates professional polish hypothesis**

---

**This analysis demonstrates that Taskmaster episode success is largely predictable from contestant characteristics, with Random Forest achieving 38.5% explained variance in IMDB ratings.** 