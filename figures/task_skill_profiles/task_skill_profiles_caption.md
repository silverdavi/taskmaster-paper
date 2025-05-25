# Figure Supplementary 9: Task Skill Intensity Profiles

## Title
Task Skill Intensity Profiles: Polarized Examples from Taskmaster

## Caption
Spider plot showing continuous skill intensity profiles (0.0-1.0 scale) for four polarized Taskmaster tasks selected based on Euclidean distance in 8-dimensional skill space. Each axis represents a different skill dimension mapped from existing task scoring data: Creativity (creativity_required_score), Physical Coordination (physical_demand_score), Problem Solving (technical_difficulty_score), Time Pressure (time_pressure_score), Originality (weirdness_score), Entertainment (entertainment_value_score), Strategic Planning (preparation_possible_score), and Adaptability (luck_factor_score). The four tasks demonstrate distinct skill profiles: (1) "Guess the number on Alex's forearm" - pure adaptability/entertainment task with high Adaptability (1.00) and Entertainment (0.78); (2) "Get dressed while wearing handcuffs" - intense physical-temporal challenge with high Time Pressure (1.00) and Physical Coordination (0.89); (3) "Recite Pi" - mental/memory challenge with high Problem Solving (0.78) and Time Pressure (0.89); (4) "The present that raises the most questions" - creative strategic task with high Strategic Planning (1.00) and Creativity (0.89). The continuous approach creates smooth, interpretable curves that reveal the multidimensional nature of task requirements, contrasting specialist tasks (high in 1-2 skills) with more generalist tasks requiring balanced capabilities across multiple dimensions.

## Methods Note
Task selection used Euclidean distance calculations in 8-dimensional skill space to identify genuinely different skill profiles. Original scores (1-10 scale) were normalized to 0.0-1.0 range using the formula: (raw_score - 1) / 9. Four examples were chosen for optimal visual clarity while avoiding overcrowded spider plots.

## Data Source
Analysis based on 845 Taskmaster tasks with complete scoring data from the _OL_tasks.csv dataset. 