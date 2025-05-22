# Figure 3: Taskmaster Task Characteristics Analysis

## Overview

Figure 3 provides a visual analysis of Taskmaster tasks, exploring the relationships between different task characteristics. The visualization consists of two components:

1. A grouped bar chart showing the distribution of Objective and Subjective judgments across four activity types (Creative, Physical, Mental, Social)
2. A supplementary stacked bar chart showing how task activity types are distributed across different series

## Key Components

### Main Grouped Bar Chart Visualization

The grouped bar chart presents the distribution of judgment types across activity categories:

**X-axis: Activity Type**
This axis represents the type of effort or skill a task requires, presented in the order: Creative, Physical, Mental, Social. Each task can belong to one or more of these categories:

- **Creative**: Tasks requiring imagination, artistic expression, or invention
- **Physical**: Tasks requiring bodily movement, coordination, or action
- **Mental**: Tasks requiring logic, memory, knowledge, or deduction
- **Social**: Tasks requiring interaction, persuasion, or cooperation with others

**Y-axis: Number of Tasks**
This shows the count of tasks in each category, separated by judgment type:

- **Objective**: Tasks scored by clear, measurable criteria (shown in steel blue)
- **Subjective**: Tasks judged based on opinions, humor, creativity, etc. (shown in warm gold)

Each activity type has two bars side by side, allowing direct comparison of how tasks are judged within each activity category.

### Supplementary Series Distribution Chart

The stacked bar chart shows how task activity types are distributed across different Taskmaster series:

- Each bar represents a series (1-18)
- Different colors represent different activity types (Creative, Mental, Physical, Social)
- Height of each segment represents the proportion of tasks in that series with that activity type
- Since tasks can have multiple activity types, the total proportions typically sum to >1.0

## Implementation Details

### Data Processing (`process_figure3_data.py`)
- Processes the Taskmaster UK tasks dataset
- Extracts counts for all Activity Type and Judgment Type combinations
- Calculates summary statistics for all task characteristics
- Computes proportions of task types across different series

### Visualization (`plot_figure3.py`)
- Creates a clean, modern grouped bar chart showing judgment types across activity categories
- Uses neutral colors (steel blue for Objective, golden rod for Subjective) to avoid political or evaluative connotations
- Creates a supplementary stacked bar chart showing task type distribution by series
- Uses consistent color scheme and modern styling for both visualizations

## Interpretation

This figure helps to understand the design space of Taskmaster tasks by revealing:

1. **Judgment Distribution**: How different activity types tend to be evaluated
   - Are physical tasks mostly objectively judged?
   - Are creative tasks typically subjectively evaluated?

2. **Activity Prevalence**: The relative frequency of different task types
   - Which activity types are most common in the show?
   - How do objective and subjective judgments vary across activities?

3. **Series Evolution**: How the distribution of task types has changed across different series
   - Have certain activity types become more or less prevalent over time?
   - Are there noticeable shifts in the types of tasks used in different series?

The analysis reveals that Physical tasks and Objective evaluation are most common across all tasks, while Creative tasks are often evaluated Subjectively. Tasks can be (and often are) evaluated using both objective and subjective criteria simultaneously.

This visualization provides insights into the task design patterns that define the Taskmaster format and how these patterns have evolved over time. 