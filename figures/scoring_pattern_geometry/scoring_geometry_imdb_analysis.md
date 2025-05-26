# Scoring Pattern Geometry vs IMDb Ratings Analysis

## Overview

This analysis tests whether the mathematical properties of task scoring patterns within episodes predict audience reception as measured by IMDb ratings. The hypothesis is that scoring dynamics (captured by mean μ and variance σ² of task score distributions) might affect how entertaining or engaging episodes are perceived to be.

## Key Findings

### **No Significant Correlations Found**

The analysis of 132 episodes reveals **no statistically significant relationships** between scoring pattern geometry and IMDb ratings:

#### Individual Correlations (all p > 0.05):
- **Mean Score (μ)** vs IMDb Rating: r = -0.012, p = 0.890
- **Mean Variance (σ²)** vs IMDb Rating: r = -0.118, p = 0.179  
- **Score Variability (std μ)** vs IMDb Rating: r = 0.078, p = 0.372
- **Variance Variability (std σ²)** vs IMDb Rating: r = -0.056, p = 0.524

#### Multiple Regression Results:
- **Absolute IMDb Rating**: R² = 0.017, F(4,127) = 0.55, **p = 0.698**
- **Relative IMDb Rating**: R² = 0.011, F(4,127) = 0.36, **p = 0.835**

## Interpretation

### What This Means

1. **Scoring Dynamics Don't Predict Ratings**: The mathematical structure of how contestants score on tasks within an episode does not significantly influence audience perception of episode quality.

2. **Entertainment Value is Independent**: Episode entertainment appears to be driven by factors other than the competitive dynamics captured by scoring patterns.

3. **Content Over Competition**: Audience appreciation likely depends more on:
   - Task creativity and humor
   - Contestant personalities and interactions  
   - Production quality and editing
   - Memorable moments and comedy

### Statistical Power

With 132 episodes analyzed, this study has sufficient power to detect medium-to-large effect sizes. The consistently small correlations (|r| < 0.12) and high p-values suggest genuine absence of relationship rather than insufficient sample size.

## Data Summary

- **Episodes Analyzed**: 132 (from Series 1-15)
- **Task Scores**: 3,750 individual task scores
- **IMDb Rating Range**: 7.3 - 8.8
- **Mean Score Range**: 2.28 - 3.80 (out of 5)
- **Variance Range**: 1.10 - 3.02
- **Histogram Variance Explained by (μ, σ²)**: 68.4% (range: 45.5% - 89.5%)

## Methodological Notes

### Scoring Geometry Calculation
For each episode, we calculated:
- **Mean μ**: Average of task mean scores
- **Mean σ²**: Average of task score variances  
- **Std μ**: Variability in task means across episode
- **Std σ²**: Variability in task variances across episode

### Statistical Approach
- Pearson and Spearman correlations for individual relationships
- Multiple linear regression for combined predictive power
- Standardized coefficients for interpretability

## Implications for Research

### Negative Results Are Valuable

This null finding is scientifically important because it:

1. **Refutes Intuitive Hypothesis**: While one might expect competitive dynamics to affect entertainment value, the data shows otherwise.

2. **Guides Future Research**: Suggests focusing on qualitative factors (humor, creativity, personality) rather than quantitative scoring patterns.

3. **Supports Show Design**: Indicates that producers can focus on task creativity without worrying about scoring distributions affecting ratings.

### Broader Context

This finding aligns with entertainment research showing that audience engagement often depends more on:
- Narrative elements and character development
- Emotional resonance and humor
- Production values and pacing
- Rather than competitive fairness or mathematical properties

## Files Generated

- `figure6b_scoring_geometry_imdb.png/pdf`: Visualization of relationships
- `scoring_geometry_correlations.csv`: Detailed correlation statistics
- `scoring_geometry_regression.csv`: Multiple regression results
- `episode_scoring_geometry_imdb.csv`: Complete merged dataset

## Conclusion

**The mathematical geometry of scoring patterns does not predict episode IMDb ratings.** This negative result provides valuable insight that entertainment value in Taskmaster is independent of the competitive dynamics captured by scoring statistics, suggesting that content quality and production factors are the primary drivers of audience appreciation. 