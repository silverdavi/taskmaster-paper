# Task Skill Profiles Analysis

## Overview

This supplementary figure visualizes the skill requirements of Taskmaster tasks using spider/radar plots. By analyzing 845 tasks across 18 series, we identify eight key skill dimensions and showcase four polarized examples that demonstrate the diverse cognitive and physical demands of different task types.

## Key Results

### Eight Skill Dimensions Identified

Through systematic analysis of task descriptions and requirements:

1. **Creativity** - Artistic expression, novel solutions, imaginative approaches
2. **Physical Coordination** - Motor skills, dexterity, athletic ability
3. **Problem Solving** - Logic, analysis, systematic thinking
4. **Time Pressure** - Speed requirements, urgency, quick decisions
5. **Originality** - Unique approaches, thinking outside the box
6. **Entertainment** - Performance quality, humor, audience appeal
7. **Strategic Planning** - Forethought, preparation, tactical thinking
8. **Adaptability** - Flexibility, improvisation, handling surprises

### Skill Distribution Across All Tasks

Analyzing 845 tasks reveals:
- **Most common skills**: Entertainment (72%), Creativity (65%), Time Pressure (58%)
- **Least common skills**: Adaptability (23%), Strategic Planning (31%)
- **Average skills per task**: 3.2 out of 8
- **Skill correlation**: Creativity and Entertainment often co-occur (r=0.68)

### Four Polarized Task Examples

The figure showcases tasks with extreme skill profiles:

#### 1. "Guess the number on Alex's forearm" (Series 7)
**Profile**: Pure adaptability and entertainment
- Adaptability: 1.00 (maximum)
- Entertainment: 0.78
- All other skills: Near zero
- **Nature**: Improvisation-based, no preparation possible

#### 2. "Get dressed while wearing handcuffs" (Series 7)
**Profile**: Physical challenge under time pressure
- Time Pressure: 1.00 (maximum)
- Physical Coordination: 0.89
- Entertainment: 0.89
- Creativity: 0.78
- **Nature**: Multi-skill physical comedy

#### 3. "Recite Pi" (Series 16)
**Profile**: Mental challenge with time element
- Time Pressure: 0.89
- Problem Solving: 0.78
- Entertainment: 0.56
- Physical/Creative: 0.00
- **Nature**: Pure cognitive performance

#### 4. "The present that raises the most questions" (Series 16)
**Profile**: Creative strategic challenge
- Strategic Planning: 1.00 (maximum)
- Creativity: 0.89
- Entertainment: 0.89
- Originality: 0.78
- **Nature**: Preparation-based creative task

### Task Type Analysis

Based on skill profiles, tasks cluster into categories:

1. **Physical Challenges** (18%): High physical coordination + time pressure
2. **Creative Challenges** (25%): High creativity + originality + entertainment
3. **Mental Challenges** (15%): High problem solving + strategic planning
4. **Performance Tasks** (20%): High entertainment + adaptability
5. **Hybrid Tasks** (22%): Balanced across multiple dimensions

### Skill Evolution Over Series

Trend analysis reveals:
- **Increasing complexity**: Later series have higher average skills per task
- **More adaptability**: Series 15-18 show 40% more adaptability requirements
- **Stable creativity**: Creative demands remain constant (~65%)
- **Rising entertainment focus**: From 68% (Series 1-6) to 76% (Series 13-18)

## Implementation Details

### Data Processing (`process_skill_profiles_data.py`)

The script:
1. Loads task data from `taskmaster_UK_tasks.csv`
2. Maps binary skill indicators to continuous scales (0-1)
3. Calculates skill intensity based on:
   - Task descriptions and keywords
   - Time limits and constraints
   - Performance requirements
4. Identifies polarized examples using Euclidean distance
5. Generates summary statistics

### Visualization (`create_skill_spider_plot.py`)

Creates a four-panel spider plot:
- Each panel shows one polarized task
- Eight axes represent skill dimensions
- Filled areas show skill intensity (0-1 scale)
- Color coding by dominant skill type
- Task titles and series information included

### Skill Scoring Methodology

Skills are scored on a 0-1 continuous scale:
- 0.00-0.20: Minimal/absent
- 0.21-0.40: Low requirement
- 0.41-0.60: Moderate requirement
- 0.61-0.80: High requirement
- 0.81-1.00: Maximum/essential

## Output Files

- `skills_data.json`: Skill profiles for polarized examples
- `radar_plot_data.json`: Formatted data for visualization
- `tasks_summary.txt`: Human-readable skill analysis
- `figure_sup9_spider_plot.pdf/png`: Final visualization
- `task_skill_profiles_caption.md`: Auto-generated caption

## Insights for Paper

1. **Multi-dimensional Challenge Design**: Tasks require diverse skill combinations, preventing any single contestant type from dominating.

2. **Entertainment Primacy**: Nearly 3/4 of tasks prioritize entertainment value, confirming the show's comedy focus over pure competition.

3. **Skill Balance**: The most successful tasks combine 3-4 skills, avoiding both oversimplification and overwhelming complexity.

4. **Adaptability Rarity**: Only 23% of tasks require adaptability, making improvisation-heavy tasks memorable outliers.

5. **Physical-Mental Balance**: The show maintains roughly equal physical (18%) and mental (15%) challenges, with most tasks being hybrid or creative.

6. **Strategic Design Space**: The eight-dimensional skill space allows for 256 theoretical task types, but only ~40 distinct profiles are commonly used, suggesting unexplored possibilities. 