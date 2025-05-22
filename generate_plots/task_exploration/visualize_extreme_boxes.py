"""
Taskmaster Extreme Skill Examples Visualization Script

This script identifies and visualizes tasks from the Taskmaster UK dataset that represent
extreme examples for each skill dimension - tasks that score high in one skill but low in
another, and vice versa. It generates box plots and radar charts to visually represent
these extreme task examples.

Required Input Files:
- data/processed/task/tasks_standardized_revised.csv: Contains the standardized task data with
  skill scores, task types, and location categories.

Output:
- PNG, SVG, and PDF visualizations saved to the visualizations/skill_extremes directory.
- A CSV file with details of the extreme task examples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from math import pi

# Create output directory for visualizations
output_dir = "visualizations/skill_extremes"
os.makedirs(output_dir, exist_ok=True)

# Set figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
sns.set_style("whitegrid")

# Define input file path
input_file = 'data/processed/task/tasks_standardized_final'

# Function to save figures in multiple formats
def save_figure(fig, base_filename, dpi=300):
    """Save the figure in multiple formats (PNG, SVG, PDF)"""
    # Save as PNG (raster)
    fig.savefig(f"{base_filename}.png", dpi=dpi, bbox_inches='tight')
    # Save as SVG (vector)
    fig.savefig(f"{base_filename}.svg", bbox_inches='tight')
    # Save as PDF (vector)
    fig.savefig(f"{base_filename}.pdf", bbox_inches='tight')
    print(f"Saved visualization to {base_filename}.png, {base_filename}.svg, and {base_filename}.pdf")

# Load the standardized data
print(f"Loading standardized data from {input_file}...")
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} tasks.")

# Select ALL feature columns for analysis
skill_columns = [
    'std_weirdness_score', 
    'std_creativity_required_score', 
    'std_physical_demand_score', 
    'std_technical_difficulty_score', 
    'std_entertainment_value_score', 
    'std_time_pressure_score', 
    'std_preparation_possible_score', 
    'std_luck_factor_score'
]

# Check for missing values in skill columns
missing_values = df[skill_columns].isnull().sum().sum()
if missing_values > 0:
    print(f"Warning: Found {missing_values} missing values. Filling with column means.")
    df[skill_columns] = df[skill_columns].fillna(df[skill_columns].mean())

# Clean skill names for display
clean_skill_names = [col.replace('std_', '').replace('_score', '').replace('_', ' ').title() for col in skill_columns]
skill_name_map = dict(zip(skill_columns, clean_skill_names))

print("\nFinding extreme skill score examples for each skill dimension...")

# Create a dictionary to hold our extreme task examples
extreme_examples = {
    'high_in': {},  # High in primary skill, low in some other skill
    'low_in': {}    # Low in primary skill, high in some other skill
}

# For each skill, find examples that are high in this skill but low in another
for primary_skill in skill_columns:
    primary_name = skill_name_map[primary_skill]
    print(f"\nFinding extremes for {primary_name}:")
    
    # High in this skill (top 20%)
    high_threshold = np.percentile(df[primary_skill], 80)
    high_tasks = df[df[primary_skill] >= high_threshold].copy()
    
    # Find a task that's high in this skill but low in some other skill
    for secondary_skill in skill_columns:
        if secondary_skill != primary_skill:
            secondary_name = skill_name_map[secondary_skill]
            
            # Find tasks that are low in the secondary skill (bottom 20%)
            low_threshold = np.percentile(df[secondary_skill], 20)
            extreme_tasks = high_tasks[high_tasks[secondary_skill] <= low_threshold]
            
            if len(extreme_tasks) > 0:
                # Sort by the difference between primary and secondary scores to get the most extreme
                extreme_tasks.loc[:, 'score_diff'] = extreme_tasks[primary_skill] - extreme_tasks[secondary_skill]
                extreme_tasks = extreme_tasks.sort_values('score_diff', ascending=False)
                
                key = f"High in {primary_name}"
                if key not in extreme_examples['high_in']:
                    best_task = extreme_tasks.iloc[0]
                    extreme_examples['high_in'][key] = (best_task, f"Low in {secondary_name}")
                    print(f"  Found task high in {primary_name}, low in {secondary_name}: {best_task['task_title']}")
                    break  # Found one example for this primary skill

    # Low in this skill (bottom 20%)
    low_threshold = np.percentile(df[primary_skill], 20)
    low_tasks = df[df[primary_skill] <= low_threshold].copy()
    
    # Find a task that's low in this skill but high in some other skill
    for secondary_skill in skill_columns:
        if secondary_skill != primary_skill:
            secondary_name = skill_name_map[secondary_skill]
            
            # Find tasks that are high in the secondary skill (top 20%)
            high_threshold = np.percentile(df[secondary_skill], 80)
            extreme_tasks = low_tasks[low_tasks[secondary_skill] >= high_threshold]
            
            if len(extreme_tasks) > 0:
                # Sort by the difference between secondary and primary scores to get the most extreme
                extreme_tasks.loc[:, 'score_diff'] = extreme_tasks[secondary_skill] - extreme_tasks[primary_skill]
                extreme_tasks = extreme_tasks.sort_values('score_diff', ascending=False)
                
                key = f"Low in {primary_name}"
                if key not in extreme_examples['low_in']:
                    best_task = extreme_tasks.iloc[0]
                    extreme_examples['low_in'][key] = (best_task, f"High in {secondary_name}")
                    print(f"  Found task low in {primary_name}, high in {secondary_name}: {best_task['task_title']}")
                    break  # Found one example for this primary skill

# Print summary of what we found
high_count = len(extreme_examples['high_in'])
low_count = len(extreme_examples['low_in'])
print(f"\nFound {high_count} 'high in X' examples and {low_count} 'low in X' examples")

# Create a more visual representation using a grid of boxes with bar charts for skill scores
# Using gridspec for better layout control
fig = plt.figure(figsize=(22, 28))
gs = GridSpec(4, 4, figure=fig)
plt.suptitle('Extreme Task Examples for Each Skill Dimension', fontsize=24, y=0.98)

# Colors for high and low skills
high_color = '#2ca02c'  # Green
low_color = '#d62728'   # Red
neutral_color = '#7f7f7f'  # Gray

# Define a function to create a task box with barplot
def create_task_box(ax, task, primary_skill, secondary_skill, contrast_type):
    # Set the title
    primary_name = skill_name_map[primary_skill]
    secondary_name = skill_name_map[secondary_skill]
    
    if contrast_type == 'high_low':
        title = f"HIGH in {primary_name}, LOW in {secondary_name}"
        title_color = high_color
    else:  # low_high
        title = f"LOW in {primary_name}, HIGH in {secondary_name}"
        title_color = low_color
    
    ax.set_title(title, fontsize=14, color=title_color, fontweight='bold')
    
    # Add task info at the top
    info_text = f"Task: {task['task_title']}\nType: {task['standardized_task_type']} | Location: {task['location_category']}"
    ax.text(0.5, 0.95, info_text, ha='center', va='top', fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Create horizontal bar chart for skills
    y_pos = np.arange(len(skill_columns))
    skill_scores = [task[col] for col in skill_columns]
    
    # Create colors list - highlight primary and secondary skills
    bar_colors = []
    for i, col in enumerate(skill_columns):
        if col == primary_skill:
            if contrast_type == 'high_low':
                bar_colors.append(high_color)
            else:
                bar_colors.append(low_color)
        elif col == secondary_skill:
            if contrast_type == 'high_low':
                bar_colors.append(low_color)
            else:
                bar_colors.append(high_color)
        else:
            bar_colors.append(neutral_color)
    
    # Plot horizontal bars
    bars = ax.barh(y_pos, skill_scores, color=bar_colors, alpha=0.7)
    
    # Add skill values as text
    for i, v in enumerate(skill_scores):
        if v > 5:  # For high values, position text inside the bar
            ax.text(v - 0.5, i, str(v), ha='right', va='center', color='white', fontweight='bold')
        else:  # For low values, position text outside the bar
            ax.text(v + 0.2, i, str(v), ha='left', va='center', color='black')
    
    # Set labels and limits
    ax.set_yticks(y_pos)
    ax.set_yticklabels(clean_skill_names)
    ax.set_xlim(0, 10.5)
    ax.set_xlabel('Skill Score')
    
    # Add brief task description at the bottom
    desc = task['task_description']
    if len(desc) > 200:
        desc = desc[:200] + "..."
    ax.text(0.5, 0.05, desc, ha='center', va='bottom', fontsize=10, 
            bbox=dict(facecolor='#f0f0f0', alpha=0.9, boxstyle='round,pad=0.5'),
            wrap=True)
    
    # Add a border around the plot
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth(1)

# Populate 16 boxes - 8 "high in X" examples on top, 8 "low in X" examples on bottom
# First, process the "high in X" tasks - top half
for i, (key, (task, contrast)) in enumerate(extreme_examples['high_in'].items()):
    # Calculate position (top half of the grid)
    row = i // 4
    col = i % 4
    
    # Create the box subplot
    ax = fig.add_subplot(gs[row, col])
    
    # Determine the skills to highlight
    primary_skill = [s for s in skill_columns if skill_name_map[s] in key.replace("High in ", "")][0]
    secondary_skill = [s for s in skill_columns if skill_name_map[s] in contrast.replace("Low in ", "")][0]
    
    # Create the task box
    create_task_box(ax, task, primary_skill, secondary_skill, 'high_low')

# Then, process the "low in X" tasks - bottom half
for i, (key, (task, contrast)) in enumerate(extreme_examples['low_in'].items()):
    # Calculate position (bottom half of the grid)
    row = 2 + (i // 4)
    col = i % 4
    
    # Create the box subplot
    ax = fig.add_subplot(gs[row, col])
    
    # Determine the skills to highlight
    primary_skill = [s for s in skill_columns if skill_name_map[s] in key.replace("Low in ", "")][0]
    secondary_skill = [s for s in skill_columns if skill_name_map[s] in contrast.replace("High in ", "")][0]
    
    # Create the task box
    create_task_box(ax, task, primary_skill, secondary_skill, 'low_high')

# Add a legend at the top
handles = [
    Rectangle((0, 0), 1, 1, color=high_color, alpha=0.7),
    Rectangle((0, 0), 1, 1, color=low_color, alpha=0.7),
    Rectangle((0, 0), 1, 1, color=neutral_color, alpha=0.7)
]
labels = ['High Skill Score', 'Low Skill Score', 'Other Skills']

fig.legend(handles=handles, labels=labels, loc='upper center', 
           bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(top=0.92)  # Adjust to make room for the title

# Save the box visualization in multiple formats
base_filename = f'{output_dir}/extreme_tasks_boxes'
save_figure(fig, base_filename)

# Also create a detailed visualization with radar charts for selected examples
fig2 = plt.figure(figsize=(20, 16))
gs2 = GridSpec(2, 4, figure=fig2)
plt.suptitle('Radar Chart Visualization of Extreme Tasks', fontsize=24, y=0.98)

# Prepare angles for radar chart
categories = clean_skill_names
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create radar chart for selected examples (most extreme for each type)
def create_radar(ax, task, primary_skill, secondary_skill, contrast_type):
    # Set axes limits
    ax.set_ylim(0, 10)
    
    # Draw category labels at the right position
    plt.xticks(angles[:-1], categories, size=11)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2, 4, 6, 8], ["2", "4", "6", "8"], color="grey", size=8)
    
    # Get skill values and repeat first value to close the loop
    values = [task[col] for col in skill_columns]
    values += values[:1]
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    
    # Fill area
    if contrast_type == 'high_low':
        fill_color = high_color
    else:
        fill_color = low_color
    
    ax.fill(angles, values, fill_color, alpha=0.1)
    
    # Add task title
    ax.set_title(f"{task['task_title']}", fontsize=13, wrap=True)
    
    # Add skill annotations on the radar
    for i, angle in enumerate(angles[:-1]):
        skill_name = categories[i]
        skill_val = values[i]
        
        # Highlight primary and secondary skills
        if skill_columns[i] == primary_skill:
            marker_color = high_color if contrast_type == 'high_low' else low_color
            marker_size = 180
            text_weight = 'bold'
            offset = 0.7 if skill_val > 5 else -0.7
            text_color = high_color if contrast_type == 'high_low' else low_color
        elif skill_columns[i] == secondary_skill:
            marker_color = low_color if contrast_type == 'high_low' else high_color
            marker_size = 180
            text_weight = 'bold'
            offset = 0.7 if skill_val > 5 else -0.7
            text_color = low_color if contrast_type == 'high_low' else high_color
        else:
            marker_color = 'gray'
            marker_size = 80
            text_weight = 'normal'
            offset = 0  # No offset for regular skills
            text_color = 'gray'
        
        # Add marker at skill point
        ax.scatter(angle, skill_val, s=marker_size, c=marker_color, alpha=0.7, zorder=10)
        
        # Add skill value text
        if skill_columns[i] == primary_skill or skill_columns[i] == secondary_skill:
            text_x = (skill_val + offset) * np.cos(angle)
            text_y = (skill_val + offset) * np.sin(angle)
            ax.text(text_x, text_y, f"{skill_val}", fontsize=11, 
                    ha='center', va='center', fontweight=text_weight, color=text_color)

# Function to find the most extreme task examples
def get_most_extreme_examples():
    high_examples = []
    low_examples = []
    
    # Process high in X tasks
    for key, (task, contrast) in extreme_examples['high_in'].items():
        primary_skill = [s for s in skill_columns if skill_name_map[s] in key.replace("High in ", "")][0]
        secondary_skill = [s for s in skill_columns if skill_name_map[s] in contrast.replace("Low in ", "")][0]
        score_diff = task[primary_skill] - task[secondary_skill]
        high_examples.append((task, primary_skill, secondary_skill, score_diff))
    
    # Process low in X tasks
    for key, (task, contrast) in extreme_examples['low_in'].items():
        primary_skill = [s for s in skill_columns if skill_name_map[s] in key.replace("Low in ", "")][0]
        secondary_skill = [s for s in skill_columns if skill_name_map[s] in contrast.replace("High in ", "")][0]
        score_diff = task[secondary_skill] - task[primary_skill]
        low_examples.append((task, primary_skill, secondary_skill, score_diff))
    
    # Sort by score difference and get top examples
    high_examples.sort(key=lambda x: x[3], reverse=True)
    low_examples.sort(key=lambda x: x[3], reverse=True)
    
    return high_examples[:4], low_examples[:4]

# Get the most extreme examples
high_extremes, low_extremes = get_most_extreme_examples()

# Create radar plots for top high examples
for i, (task, primary_skill, secondary_skill, _) in enumerate(high_extremes):
    ax = fig2.add_subplot(gs2[0, i], polar=True)
    create_radar(ax, task, primary_skill, secondary_skill, 'high_low')

# Create radar plots for top low examples
for i, (task, primary_skill, secondary_skill, _) in enumerate(low_extremes):
    ax = fig2.add_subplot(gs2[1, i], polar=True)
    create_radar(ax, task, primary_skill, secondary_skill, 'low_high')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the radar chart visualization in multiple formats
base_filename = f'{output_dir}/extreme_tasks_radar'
save_figure(fig2, base_filename)

# Save the tasks to CSV as before
print("Creating CSV with extreme task examples...")
all_extreme_tasks = []

for contrast_type, contrasts in extreme_examples.items():
    for primary, (task, secondary) in contrasts.items():
        task_data = task.copy()
        task_data['primary_contrast'] = primary
        task_data['secondary_contrast'] = secondary
        task_data['contrast_type'] = 'High-Low' if contrast_type == 'high_in' else 'Low-High'
        all_extreme_tasks.append(task_data)

extreme_df = pd.DataFrame(all_extreme_tasks)
csv_columns = [
    'task_unique_id', 'task_title', 'task_description', 'standardized_task_type', 
    'location_category', 'primary_contrast', 'secondary_contrast', 'contrast_type'
] + skill_columns

csv_path = f'{output_dir}/extreme_task_examples.csv'
extreme_df[csv_columns].to_csv(csv_path, index=False)
print(f"CSV file saved with extreme task examples to {csv_path}")

print("All visualizations completed!") 