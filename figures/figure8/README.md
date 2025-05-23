# Figure 8: IMDB Rating Prediction Analysis

This directory contains the complete pipeline for Figure 8, demonstrating two different analytical approaches for IMDB rating prediction based on sample size considerations.

## Analysis Overview

### Episode-Level Analysis (N=154) - Machine Learning
- **Approach**: Successful ML prediction with proper cross-validation
- **Result**: Random Forest achieving R² = 0.385 (38% explained variance)
- **Conclusion**: Demonstrates meaningful predictive relationships

### Series-Level Analysis (N=154) - Raw Correlation  
- **Approach**: Raw correlation analysis between input features and mean IMDB scores
- **Result**: 45 correlations ranging from -0.547 to +0.397
- **Conclusion**: Shows relationship patterns without overfitting risks

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