# Data Sources and Methodology

This document provides detailed information about the data sources, collection methods, and reliability assessments for all datasets used in the Taskmaster analysis.

## Overview

The Taskmaster dataset comprises multiple data sources collected through various methodologies, ranging from official databases to AI-assisted content analysis. Each dataset serves specific analytical purposes and has different reliability characteristics that inform how we use them in our statistical analyses.

## Dataset Descriptions

### 1. IMDb Ratings Data

#### `imdb_ratings.csv`
**Source**: Official IMDb database  
**Description**: Official IMDb ratings for each of the 154 Taskmaster UK episodes  
**Methodology**: Direct extraction from IMDb's public API/website  
**Reliability**:  (Official, authoritative source)

This dataset contains the official IMDb rating for each episode, which represents IMDb's proprietary weighted average of user votes. These ratings serve as our primary dependent variable for audience reception analysis.

**Key Features**:
- Episode-level ratings (1-10 scale)
- Official IMDb methodology (proprietary weighting)
- Complete coverage of all 154 episodes
- High reliability and external validity

---

### 2. Detailed Rating Distributions

#### `taskmaster_histograms_corrected.csv`
**Source**: IMDb website vote histograms  
**Description**: Raw vote distributions (1-10 stars) for each of the 154 episodes  
**Methodology**: Web scraping of IMDb's detailed vote breakdowns  
**Reliability**:  (Official IMDb data, validated against official ratings)

This dataset provides the granular vote distributions that underlie IMDb's official ratings, enabling more sophisticated distributional analysis.

**Validation**: 
- Correlation with official IMDb scores: >99%
- This near-perfect correlation validates the accuracy of our histogram data extraction

**Advanced Analysis**:
We fitted a **tri-peak mixture model** to these distributions:
- **Model**: a₁·δ(1) + a₁₀·δ(10) + a_gaussian·N(μ,σ)
- **Rationale**: Visual inspection reveals consistent patterns of extreme ratings (1s and 10s) plus a bell curve for middle ratings (2-9)
- **Performance**: 
  - Mixture model MAE: 1.94%
  - Naive single Gaussian MAE: 3.73%
  - **Improvement**: 48.1% reduction in prediction error

This tri-peak model captures the bimodal nature of online rating distributions more accurately than traditional approaches.

---

### 3. Episode and Task Metadata

#### `taskmaster_uk_episodes.csv`
**Source**: taskmaster.info website  
**Description**: Comprehensive episode metadata and task classifications  
**Methodology**: Systematic extraction from taskmaster.info database  
**Reliability**:  (Fan-maintained but comprehensive and cross-referenced)

Contains hard classifications and objective data about episodes and tasks, including:
- Task types (objective vs. subjective scoring)
- Episode air dates and descriptions
- Task categories and formats
- Links to original taskmaster.info sources for verification

**Data Quality**: 
- Objective classifications (less prone to interpretation bias)
- Cross-referenced with multiple sources
- Community-maintained with high accuracy standards

---

### 4. Contestant Demographics

#### `contestants.csv`
**Source**: Web search and public records  
**Description**: Demographic and professional information for all 90 contestants  
**Methodology**: Manual research using web searches, Wikipedia, and public databases  
**Reliability**:  (Best effort with acknowledged limitations)

**Important Limitations**:
- Not verifiably 100% accurate due to contestant privacy
- Some demographic information may be incomplete or outdated
- Occupation classifications involve subjective judgment

**Usage Guidelines**:
- Used for basic demographic analysis and occupation classification
- Trends and patterns are more reliable than individual data points
- Results interpreted with appropriate caveats about data completeness

#### `Cont_lon_lat.tsv`
**Source**: Official city records and geographic databases  
**Description**: Longitude and latitude coordinates for contestants' official birth cities  
**Methodology**: Geocoding of verified birth locations  
**Reliability**:  (Geographic data is objective and verifiable)

---

### 5. Task Classification (Exploratory Only)

#### `OL_tasks.csv`
**Source**: GPT-4o analysis of episode scripts  
**Description**: AI-generated task classifications and characteristics  
**Methodology**: Large language model analysis of episode transcripts  
**Reliability**:  (Exploratory only - no accuracy metrics available)

**Critical Limitations**:
- No established metrics to assess classification accuracy
- AI interpretation may introduce systematic biases
- Results not independently validated

**Usage Restrictions**:
- **Used ONLY for exploration and hypothesis generation**
- **NOT used for statistical conclusions or distribution analysis**
- **NOT used as basis for quantitative findings**
- Used to identify interesting examples for manual verification

**Methodology Note**: While GPT-4o shows promise for content analysis, the lack of ground truth validation data means these classifications cannot support statistical inference.

---

### 6. Sentiment Analysis

#### `sentiment.csv`
**Source**: GPT-4o analysis of episode transcripts  
**Description**: Episode-level sentiment metrics extracted from dialogue  
**Methodology**: AI-powered sentiment analysis using GPT-4o  
**Reliability**:  (Limited scope due to methodological constraints)

**Methodology**:
- GPT-4o sentiment extraction from episode transcripts
- Episode-by-episode analysis of dialogue content
- Multiple sentiment dimensions (humor, awkwardness, joy, etc.)

**Accuracy Assessment**:
Based on published research¹, GPT-4 achieves:
- **95.3% accuracy** in sentiment classification
- Significantly outperforms traditional lexicon-based methods (37.2% accuracy)
- Superior to classical machine learning approaches (66-71% accuracy)

**Implementation Limitations**:
- Requires access to episode transcripts (not included in repository)
- Requires OpenAI API access
- `sentiment_analysis.py` provided for reference but non-functional without transcripts

**Usage Guidelines**:
- Sentiment analysis kept as **minor component** of overall analysis
- **Not the cornerstone** of this work due to data access limitations
- Results interpreted with appropriate methodological caveats

---

## Data Quality Summary

| Dataset | Reliability | Primary Use | Limitations |
|---------|-------------|-------------|-------------|
| `imdb_ratings.csv` |  | Primary dependent variable | None significant |
| `taskmaster_histograms_corrected.csv` |  | Advanced distributional analysis | None significant |
| `taskmaster_uk_episodes.csv` |  | Task/episode metadata | Fan-maintained source |
| `contestants.csv` |  | Demographic analysis | Privacy limitations, incomplete data |
| `Cont_lon_lat.tsv` |  | Geographic analysis | None significant |
| `OL_tasks.csv` |  | Exploration only | No validation metrics |
| `sentiment.csv` |  | Supplementary analysis | Access limitations, minor role |

## Statistical Methodology Notes

### Mixture Model Validation
Our tri-peak mixture model for rating distributions represents a significant methodological contribution:
- **Theoretical justification**: Online rating distributions consistently show extreme rating spikes
- **Empirical validation**: 48.1% improvement over naive Gaussian fitting
- **Visual confirmation**: Clear tri-peak patterns visible across all series

### Correlation Validation
The >99% correlation between our extracted histograms and official IMDb ratings provides strong validation of our data extraction methodology and supports the reliability of our distributional analyses.

## References

¹ Zhang, Y., et al. (2024). "Accuracy of ChatGPT in sentiment analysis: A comparative study." *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10865189/

---

## Usage Guidelines for Researchers

1. **High-confidence analyses**: Use IMDb ratings and histogram data for primary statistical conclusions
2. **Demographic trends**: Use contestant data for broad patterns, not individual-level claims
3. **Exploratory insights**: Use AI-generated classifications for hypothesis generation only
4. **Methodological transparency**: Always report data source limitations in results
5. **Replication**: All extraction methodologies documented for reproducibility

This multi-source approach provides robust foundations for statistical analysis while maintaining appropriate skepticism about data quality limitations. 