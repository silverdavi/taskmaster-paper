# Data Files Documentation

This document provides an overview of all data files used in the Taskmaster Paper project.

## contestants.csv

Contains information about Taskmaster UK contestants across 18 series.

**Fields:**
- Name: Contestant's full name
- Series: Series number they appeared in
- Placement: Final ranking in their series
- Demographics: Age, gender, etc.
- Occupation: Professional background
- Comedy Style: Description of their comedic approach
- Notable Moments: Memorable events during their appearance
- Social Media Handles: Links to their social profiles

## scores.csv

Records detailed task scores for each contestant in every episode.

**Fields:**
- Task ID: Unique identifier for each task
- Show Title: Episode title
- Series: Series number
- Episode: Episode number
- Task Title: Name of the task
- Contestant Name: Participant's name
- Score Details: Points awarded
- Total Score: Cumulative score
- Winner: Boolean indicating if they won the task

## tasks.csv

Provides comprehensive information about each task.

**Fields:**
- Title: Task name
- Description: Detailed explanation of the task
- Location: Where the task was performed
- Materials: Items provided for the task
- Constraints: Rules and limitations
- Categories: Task classification
- Skills Required: Abilities needed to excel
- Task Type: Classification of task format

## sentiment.csv

Contains sentiment analysis data for each episode.

**Fields:**
- Episode Identifier: Series and episode number
- Laughter Count: Number of audience laughs
- Applause Count: Number of audience applauses
- Emotion Metrics: Measurements of anger, awkwardness, humor, joy, sarcasm, etc.

## imdb_ratings.csv

Lists IMDb ratings for each episode across all series.

**Fields:**
- Series: Series number
- Episode: Episode number
- Rating: IMDb score
- Relative Rating: Rating compared to other episodes in the same series

## taskmaster_histograms_corrected.csv

Provides detailed rating distribution data for each episode.

**Fields:**
- Episode Identifier: Series and episode number
- Rating Distribution: Percentage and count of votes for each rating (1-10)

## taskmaster_UK_tasks.csv

Contains categorized information about each task.

**Fields:**
- Task Identifier: Unique ID for the task
- Task Type: Solo/Team classification
- Special: Whether it's a special task
- Prize Task: Whether it's a prize task
- Filmed: Whether it was pre-recorded
- Homework: Whether it was completed at home
- Live: Whether it was performed live
- Creative: Whether it required creative skills
- Mental: Whether it required mental skills
- Physical: Whether it required physical skills

## taskmaster_uk_episodes.csv

Contains information about individual episodes.

**Fields:**
- Series: Series number
- Episode: Episode number within the series
- Title: Episode title
- Air Date: Original broadcast date
- Description: Episode summary
- Guest Details: Information about special guests 