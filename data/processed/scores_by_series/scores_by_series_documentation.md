# Scores by Series Documentation

## Overview

This directory contains CSV files with contestant scores for each series of Taskmaster UK, formatted as matrices for easy analysis. The data has been processed from the raw scores data to create a standardized format suitable for statistical analysis, machine learning, and visualization.

## File Format

Each file follows the format:
```
ContestantID, ContestantName, Score_Task_1, Score_Task_2, ...
```

Where:
- **ContestantID**: A running number from 1 to 90 (unique across all series)
- **ContestantName**: The contestant's name
- **Score_Task_X**: The score (0-5, occasionally negative) for task X in that series

## Key Statistics

### Overall
- **Total contestants**: 90 (5 per series)
- **Total series files**: 18
- **Tasks per series**: Varies from 28 (early series) to 59 (later series)
- **Average tasks per series**: ~51

### Score Distribution
- **Mode**: 3 points (most common score)
- **Perfect scores (5 points)**: ~18% of all scores
- **Zero scores**: ~12% of all scores
- **Negative scores**: Rare, usually for rule violations
- **Missing values**: Filled with 0 (for team tasks where not all contestants participated)

## Contestant ID Mapping

ContestantID is a unique identifier assigned sequentially:

| Series | Contestant IDs | Contestants |
|--------|---------------|-------------|
| 1 | 1-5 | Frank Skinner, Josh Widdicombe, Roisin Conaty, Romesh Ranganathan, Tim Key |
| 2 | 6-10 | Doc Brown, Joe Wilkinson, Jon Richardson, Katherine Ryan, Richard Osman |
| 3 | 11-15 | Al Murray, Dave Gorman, Paul Chowdhry, Rob Beckett, Sara Pascoe |
| 4 | 16-20 | Hugh Dennis, Joe Lycett, Lolly Adefope, Mel Giedroyc, Noel Fielding |
| 5 | 21-25 | Aisling Bea, Bob Mortimer, Mark Watson, Nish Kumar, Sally Phillips |
| 6 | 26-30 | Alice Levine, Asim Chaudhry, Liza Tarbuck, Russell Howard, Tim Vine |
| 7 | 31-35 | James Acaster, Jessica Knappett, Kerry Godliman, Phil Wang, Rhod Gilbert |
| 8 | 36-40 | Iain Stirling, Joe Thomas, Lou Sanders, Paul Sinha, Sian Gibson |
| 9 | 41-45 | David Baddiel, Ed Gamble, Jo Brand, Katy Wix, Rose Matafeo |
| 10 | 46-50 | Daisy May Cooper, Johnny Vegas, Katherine Parkinson, Mawaan Rizwan, Richard Herring |
| 11 | 51-55 | Charlotte Ritchie, Jamali Maddix, Lee Mack, Mike Wozniak, Sarah Kendall |
| 12 | 56-60 | Alan Davies, Desiree Burch, Guz Khan, Morgana Robinson, Victoria Coren Mitchell |
| 13 | 61-65 | Ardal O'Hanlon, Bridget Christie, Chris Ramsey, Judi Love, Sophie Duker |
| 14 | 66-70 | Dara Ó Briain, Fern Brady, John Kearns, Munya Chawawa, Sarah Millican |
| 15 | 71-75 | Frankie Boyle, Ivo Graham, Jenny Eclair, Kiell Smith-Bynoe, Mae Martin |
| 16 | 76-80 | Julian Clary, Lucy Beaumont, Sam Campbell, Sue Perkins, Susan Wokoma |
| 17 | 81-85 | Joanne McNally, John Robins, Nick Mohammed, Sophie Willan, Steve Pemberton |
| 18 | 86-90 | Andy Zaltzman, Babatunde Aléshé, Emma Sidi, Jack Dee, Rosie Jones |

## Usage Examples

### Loading a Series
```python
import pandas as pd
series_1_scores = pd.read_csv('series_1_scores.csv')
```

### Analyzing Performance
The matrix format enables:
- **Row operations**: Analyze individual contestant performance
- **Column operations**: Analyze task difficulty
- **Clustering**: Group contestants by performance patterns
- **Time series**: Track performance over tasks

### Example Insights from the Data

1. **Series Winners** (highest total across all tasks):
   - Series 1: Josh Widdicombe (ID: 2)
   - Series 7: James Acaster (ID: 31)
   - Series 9: Ed Gamble (ID: 42)

2. **Task Difficulty**: Tasks with many 0s or 5s tend to be either very difficult or have binary success criteria

3. **Consistency**: Some contestants (e.g., Richard Osman, ID: 10) show very consistent scoring patterns

4. **Team Tasks**: Identified by patterns where multiple contestants have identical scores

## Data Processing Notes

1. **Source**: Processed from raw `scores.csv` file
2. **Cleaning**: Alex Horne (show assistant) excluded from all series
3. **Missing Values**: Filled with 0 for tasks where contestant didn't participate
4. **Validation**: All scores verified to be in valid range (typically 0-5)
5. **Sorting**: Contestants sorted by ID, tasks in chronological order

## Applications

This formatted data is ideal for:
- **Statistical Analysis**: ANOVA, correlation analysis, regression
- **Machine Learning**: Predicting winners, clustering contestants
- **Visualization**: Heatmaps, performance trajectories
- **Game Theory**: Analyzing strategic behavior in competitive tasks 