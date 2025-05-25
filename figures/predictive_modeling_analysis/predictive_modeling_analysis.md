# Predictive Modeling Analysis

## Overview

This figure uses machine learning to predict IMDB ratings and identify the most important factors for Taskmaster's success. The analysis operates at two levels:

1. **Panel A: Episode-Level Machine Learning** - Predicts individual episode ratings using contestant and task features
2. **Panel B: Series-Level Correlation Analysis** - Examines relationships between series characteristics and average ratings

## Key Results

### Episode-Level ML Performance (154 episodes)

| Model | Cross-Val R² | Test R² | Interpretation |
|-------|--------------|---------|----------------|
| Linear Regression | 0.120 | 0.134 | Weak predictive power |
| Ridge Regression | 0.120 | 0.134 | No improvement over linear |
| **Random Forest** | **0.368** | **0.385** | **Moderate predictive power** |

The Random Forest model explains 38.5% of rating variance, suggesting that while ratings are somewhat predictable, significant randomness remains.

### Top Predictive Features

Based on Random Forest feature importance analysis:

| Feature | Importance | Impact | Recommendation |
|---------|------------|---------|----------------|
| **Contestant Age** | 39.5% | +0.396 corr | Cast older contestants (40+) |
| **Awkwardness** | 32.6% | -0.151 corr | Minimize awkward moments |
| **Experience** | 16.2% | +0.019 corr | Prioritize TV veterans (25+ years) |
| **% Comedians** | 6.5% | +0.331 corr | Include professional comedians |
| **% Female** | 5.2% | +0.108 corr | Maintain gender balance |

### Series-Level Correlations (18 series)

Analyzed 35 features against mean IMDB scores:

**Strongest Negative Correlations:**
1. **Series squared** (-0.660, p=0.003): Later series perform worse
2. **Gender diversity** (-0.629, p=0.005): Homogeneous casts rate higher
3. **% Actors** (-0.621, p=0.006): Non-comedians reduce ratings
4. **Series order** (-0.597, p=0.009): Chronological decline confirmed
5. **Self-deprecation variability** (-0.541, p=0.020): Consistency preferred

**Strongest Positive Correlations:**
1. **Early series indicator** (+0.443, p=0.066): First 6 series rated higher
2. **Average age** (+0.435, p=0.071): Older contestants preferred
3. **% Comedians** (+0.331, p=0.179): Comedy professionals boost ratings
4. **Episode count** (-0.431, p=0.074): Fewer episodes = higher quality

### The "Golden Formula" for High Ratings

Based on the combined analysis:

**Cast Profile:**
- Average age: 41+ years
- Experience: 25+ years in entertainment
- Profession: Majority professional comedians
- Gender: Balanced or slight female majority
- Personality: Polished, minimal awkwardness

**Production Strategy:**
- Keep series shorter (6-8 episodes)
- Edit to minimize awkward moments
- Focus on professional execution
- Maintain consistent tone

### Insights by Feature Category

**Demographic Features:**
- Age is the single strongest predictor (39.5% importance)
- Professional comedians significantly boost ratings
- Gender balance has minor positive effect
- Actor-heavy casts underperform

**Emotional/Sentiment Features:**
- Awkwardness is the second-strongest predictor (32.6%)
- High variability in emotions reduces ratings
- Consistent tone preferred over wild swings

**Format Features:**
- Series position shows strong negative trend
- Shorter series (fewer episodes) rate higher
- Early series (1-6) significantly outperform later ones

**Task Features:**
- Individual task characteristics have minimal impact
- Overall contestant quality matters more than task design

## Implementation Details

### Data Preparation (`1_prepare_episode_data.py`)
- Loads and merges contestant, task, and rating data
- Creates derived features (means, proportions, indicators)
- Handles missing data and outliers
- Outputs clean dataset with 154 episodes

### Feature Selection (`2_feature_selection_episode.py`)
- Tests correlation-based and mutual information methods
- Identifies top predictive features
- Removes redundant/constant features
- Selects optimal feature set for modeling

### Machine Learning (`3_model_episode_analysis.py`)
- Implements three ML algorithms
- Uses 5-fold cross-validation
- Tests on held-out data (20%)
- Saves model results and feature importance

### Visualization (`4_plot_figure8a.py`, `5_correlation_analysis_figure8b.py`)
- Creates publication-ready figures
- Panel A: Model comparison bar chart
- Panel B: Correlation distribution histogram
- Additional: Random Forest feature analysis

## Output Files

- `episode_data.csv`: Cleaned episode-level dataset
- `episode_model_results.pkl`: ML model performance metrics
- `raw_correlations.json`: Series-level correlation results
- `figure8a_episode_ml.pdf/png`: ML performance comparison
- `figure8b_raw_correlations.pdf/png`: Correlation analysis
- `random_forest_feature_analysis.pdf/png`: Feature importance breakdown

## Insights for Paper

1. **Professionalism Wins**: The strongest predictors (age, experience, comedian status) all point to professional polish outperforming amateur charm.

2. **Awkwardness Hurts**: Despite comedy trends toward "cringe," too much awkwardness significantly reduces ratings.

3. **Diminishing Returns**: The strong negative correlation with series number suggests either viewer fatigue or difficulty maintaining quality over 18 series.

4. **Demographics Matter**: Contestant characteristics (39.5% + 16.2% = 55.7% of model importance) far outweigh task design in determining success.

5. **Consistency Preferred**: Low variability in emotional tone correlates with higher ratings, suggesting viewers prefer predictable comfort over wild swings.

6. **The 40+ Advantage**: Older contestants bring experience, confidence, and established fan bases that translate to higher ratings. 