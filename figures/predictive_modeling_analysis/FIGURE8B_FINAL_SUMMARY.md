# Figure 8b Final Summary: Raw Correlation Analysis

## Corrected Analysis Overview

Successfully completed **Figure 8b** using proper raw correlation analysis between actual input features and mean IMDB scores, as requested.

## Key Corrections Made

1. **Fixed Target Variable**: Used `taskmaster_histograms_corrected.csv` as OUTPUT (target) not input
2. **Used Raw Input Data**: Loaded actual data from:
   - `data/raw/taskmaster_UK_tasks.csv` - Task characteristics 
   - `data/raw/sentiment.csv` - Sentiment analysis
   - `data/processed/scores_by_series/series_*_scores.csv` - Contestant scoring
   - `data/raw/contestants.csv` - Contestant demographics
3. **Raw Correlations**: Computed actual correlation coefficients (positive and negative)
4. **Proper Target**: Mean IMDB score computed from histogram vote distributions

## Final Results

### Dataset Summary
- **Episodes analyzed**: 154 episodes
- **Features analyzed**: 45 features from all input data sources
- **Target variable**: Mean IMDB score (6.68 - 8.86 range)
- **Correlation range**: -0.547 to +0.397

### Correlation Distribution 
- **Total correlations**: 45
- **Mean correlation**: -0.025 (near zero, as expected)
- **Standard deviation**: 0.199
- **Positive correlations**: 19 (42%)
- **Negative correlations**: 26 (58%)
- **Strong correlations (|r| ≥ 0.5)**: 1 (2%)

## Top Correlations Found

### Strongest Positive Correlations
1. **contestant_avg_age**: r = +0.397 (Older contestants → higher IMDB)
2. **task_prop_is_special**: r = +0.371 (More special tasks → higher IMDB)
3. **task_prop_is_solo**: r = +0.358 (More solo tasks → higher IMDB)
4. **contestant_prop_comedians**: r = +0.331 (More comedians → higher IMDB)

### Strongest Negative Correlations  
1. **contestant_prop_actors**: r = -0.547 (More actors → lower IMDB)  **STRONGEST**
2. **task_prop_is_team**: r = -0.358 (More team tasks → lower IMDB)
3. **task_prop_is_prize**: r = -0.331 (More prize tasks → lower IMDB)
4. **task_prop_is_live**: r = -0.277 (More live tasks → lower IMDB)

### Notable Sentiment Correlations
- **mean_sentence_length**: r = +0.242 (Longer sentences → higher IMDB)
- **num_sentences**: r = -0.221 (More sentences → lower IMDB)
- **greg_mentions**: r = -0.208 (More Greg mentions → lower IMDB)
- **total_humor**: r = -0.164 (More humor → lower IMDB, surprisingly)

## Key Insights

### 1. Contestant Demographics Matter Most
- **Age effect**: Older contestants correlate with higher ratings
- **Professional bias**: Actors negatively impact ratings, comedians positively
- **Gender effects**: Males show slight positive correlation

### 2. Task Structure Effects
- **Solo vs Team**: Solo tasks preferred over team tasks  
- **Special tasks**: Unique/special tasks boost ratings
- **Live tasks**: Live studio tasks may reduce ratings

### 3. Surprising Sentiment Findings
- More humor content correlates with *lower* IMDB scores
- Longer, fewer sentences preferred over many short ones
- Greg mentions correlate negatively (fatigue effect?)

### 4. Scoring Shows Weak Correlations
- Task scoring metrics show minimal correlation with IMDB ratings
- Suggests scoring and entertainment value are somewhat independent

## Statistical Interpretation

### Distribution Shape
The correlation histogram shows:
- **Roughly normal distribution** centered near zero
- **Moderate spread** (std = 0.199) indicating meaningful relationships exist
- **Few strong correlations** suggesting complex, multifactorial ratings
- **Slight negative skew** (more negative than positive correlations)

### Scientific Value
This analysis demonstrates:
1. **Realistic correlation magnitudes** for entertainment data
2. **Multiple weak-to-moderate effects** rather than single strong predictors  
3. **Meaningful patterns** across different data types (demographics, tasks, sentiment)
4. **Appropriate methodology** for exploratory correlation analysis

## Comparison with Previous Analysis

| Aspect | Previous (Wrong) | Corrected (Right) |
|--------|-----------------|-------------------|
| **Input data** | Used IMDB histograms as features | Used actual raw input data |
| **Correlations** | Absolute values only | Raw positive/negative values |
| **Target** | Multiple IMDB histogram components | Single mean IMDB score |
| **Sample size** | N=18 (series-level) | N=154 (episode-level) |
| **Results** | 385 artificial correlations | 45 meaningful correlations |

## Files Generated

### Primary Output
- **`figure8b_raw_correlations.png`** - Main Figure 8b (correlation histogram + top features)

### Analysis Pipeline  
- **`correlation_analysis_raw.py`** - Complete raw correlation analysis
- **`raw_correlations.json`** - Detailed correlation results

### Key Finding
The strongest relationship is that **episodes with more actors as contestants tend to have lower IMDB ratings** (r = -0.547), while **episodes with older contestants tend to have higher ratings** (r = +0.397).

## Conclusion

Figure 8b now properly demonstrates raw correlation analysis between actual Taskmaster input features and IMDB ratings. The results reveal meaningful but moderate relationships, with contestant demographics and task structure being most predictive of viewer ratings. The analysis provides dozens of correlations as requested, showing the complex multifactorial nature of entertainment ratings. 