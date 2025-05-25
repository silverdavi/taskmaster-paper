# Taskmaster UK Data Documentation

## Overview

This document provides comprehensive documentation of all data files used in the Taskmaster Paper project, including key statistics and insights from the processed data.

## Core Data Files

### contestants.csv
**90 contestants across 18 series**

Contains information about Taskmaster UK contestants.

**Fields:**
- Name: Contestant's full name
- Series: Series number they appeared in (1-18)
- Placement: Final ranking in their series (1-5)
- Demographics: Age, gender, etc.
- Occupation: Professional background
- Comedy Style: Description of their comedic approach
- Notable Moments: Memorable events during their appearance
- Social Media Handles: Links to their social profiles

**Key Statistics:**
- Total contestants: 90 (5 per series)
- Gender distribution: Approximately 60% male, 40% female
- Geographic origins: 74.4% England, 14.4% international, 11.1% other UK nations
- Most common birthplace: London (29 contestants, 32.2%)

### scores.csv
**Task performance data for all contestants**

Records detailed task scores for each contestant in every episode.

**Fields:**
- Task ID: Unique identifier for each task
- Show Title: Episode title
- Series: Series number (1-18)
- Episode: Episode number within series
- Task Title: Name of the task
- Contestant Name: Participant's name
- Score Details: Points awarded (0-5, occasionally negative)
- Total Score: Cumulative score
- Winner: Boolean indicating if they won the task

**Key Statistics:**
- Total task performances recorded: ~4,500
- Average score per task: 2.8
- Perfect scores (5 points): ~18%
- Zero scores: ~12%
- Score distribution follows expected pattern with mode at 3

### tasks.csv & taskmaster_UK_tasks.csv
**917 unique tasks analyzed**

Provides comprehensive information about each task.

**Fields:**
- Title: Task name
- Description: Detailed explanation of the task
- Location: Where performed (house, garden, studio, etc.)
- Materials: Items provided
- Constraints: Rules and limitations
- Categories: Task classification
- Skills Required: Abilities needed to excel
- Task Type: Solo/Team/Special/Live/Prize

**Key Task Statistics:**
- Total tasks: 917
- Solo tasks: 806 (87.9%)
- Team tasks: 111 (12.1%)
- Prize tasks: 167 (18.2%)
- Filmed tasks: 569 (62.1%)
- Live tasks: 171 (18.7%)

**Task Characteristics:**
- Creative tasks: 399 (43.5%)
- Physical tasks: 441 (48.1%)
- Mental tasks: 380 (41.4%)
- Objective judgment: 513 (55.9%)
- Subjective judgment: 381 (41.5%)

### sentiment.csv
**Emotional analysis of 154 episodes**

Contains sentiment analysis data for each episode.

**Fields:**
- Episode Identifier: Series and episode number
- Laughter Count: Number of audience laughs
- Applause Count: Number of audience applauses
- Emotion Metrics: Measurements of anger, awkwardness, humor, joy, sarcasm, etc.

**Key Findings:**
- Awkwardness shows significant increase over time (p < 0.001)
- Average awkwardness: Series 1 (2.39) → Series 18 (2.59)
- Humor remains consistently high (3.0-3.2 range)
- Negative emotions (anger, frustration) stay very low (<0.3)

### imdb_ratings.csv
**Ratings for all 154 UK episodes**

Lists IMDb ratings for each episode across all series.

**Fields:**
- Series: Series number
- Episode: Episode number
- Rating: IMDb score (1-10 scale)
- Relative Rating: Rating compared to other episodes in the same series

**Key Statistics:**
- Average episode rating: 7.95
- Highest-rated series: Series 7 (8.30 average)
- Lowest-rated series: Series 10 (7.52 average)
- Rating trend within series: 89% show rising or J-shaped patterns
- Average improvement first to last episode: +0.28 points

### taskmaster_histograms_corrected.csv
**Detailed rating distributions**

Provides rating distribution data for each episode.

**Fields:**
- Episode Identifier: Series and episode number
- Rating Distribution: Percentage and count of votes for each rating (1-10)

**Key Insights:**
- Most 10-star ratings: Series 7 (29.9%)
- Most 1-star ratings: Series 18 (10.8%)
- Correlation between mean and 10-stars: r = 0.71
- Correlation between mean and 1-stars: r = -0.53

### taskmaster_uk_episodes.csv
**Episode metadata for all 154 episodes**

Contains information about individual episodes.

**Fields:**
- Series: Series number
- Episode: Episode number within the series
- Title: Episode title
- Air Date: Original broadcast date
- Description: Episode summary
- Guest Details: Information about special guests

**Episode Statistics:**
- Total episodes: 154
- Average episodes per series: 8.6
- Series with 10 episodes: 9 series
- Series with fewer episodes: Early series (5-6), Series 14 (7)

## Processed Data

### scores_by_series/
**Contestant performance matrices**

Contains 18 CSV files (one per series) with contestant scores in matrix format:
- Rows: Contestants (identified by ID 1-90)
- Columns: Individual task scores
- Format: ContestantID, ContestantName, Score_Task_1, Score_Task_2, ...

**Usage:** Enables series-level analysis of contestant performance patterns, clustering, and predictive modeling.

### Geographic Data (Cont_lon_lat.tsv)
**Birthplace coordinates for contestants**

Maps each contestant to their birthplace coordinates, enabling geographic analysis.

**Key Findings:**
- 77 contestants (85.6%) born in UK/Ireland
- 13 contestants (14.4%) international
- London cluster: 29 contestants
- Even distribution across other UK regions

## Data Quality Notes

1. **Completeness**: All core data files are complete with no missing episodes or series
2. **Consistency**: Task IDs and contestant names are standardized across files
3. **Corrections**: IMDb histogram data required column name corrections (reversed order)
4. **Exclusions**: Alex Horne excluded from contestant analyses (show assistant, not competitor)

## Usage Guidelines

1. **Joining Data**: Use contestant names and series numbers for joining contestant data
2. **Task Analysis**: Task IDs are unique and consistent across all task-related files
3. **Time Series**: Series number provides chronological ordering (1 = oldest, 18 = newest)
4. **Geographic Analysis**: International contestants mapped to map edges for visualization

## Summary Statistics

- **Total Series**: 18
- **Total Episodes**: 154
- **Total Contestants**: 90
- **Total Tasks**: 917
- **Total Task Performances**: ~4,500
- **Total IMDb Votes**: 32,607
- **Date Range**: 2015-2023 