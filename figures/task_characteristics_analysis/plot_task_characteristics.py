import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = SCRIPT_DIR / "data"

# Add the project root to the Python path
sys.path.append(str(PROJECT_ROOT))
from config.plot_utils import load_config, apply_plot_style, get_palette

def plot_activity_judgment_bars(bubble_data, summary_stats, output_path_base=None):
    """
    Create a clean, modern grouped bar chart showing judgment types across all activity types.
    """
    if output_path_base is None:
        output_path_base = SCRIPT_DIR / 'fig2'
    
    # Set up the figure using the project configuration
    config = load_config()
    fig_size = (7, 4)  # More compact size for the grouped bar chart
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Apply consistent styling from the project config
    apply_plot_style(fig, ax)
    
    # Use specified order: Creative, Physical, Mental, Social
    activities = ['creative', 'physical', 'mental', 'social']
    judgments = ['objective', 'subjective']
    
    # Prepare data for plotting
    data = {}
    for activity in activities:
        data[activity] = {
            'objective': 0,
            'subjective': 0
        }
    
    # Count tasks for each activity-judgment combination
    for item in bubble_data:
        activity = item['activity']
        judgment = item['judgment']
        count = item['count']
        if activity in data and judgment in data[activity]:
            data[activity][judgment] = count
    
    # Use more neutral, semantically aligned colors
    objective_color = '#4682B4'  # Steel blue for objective
    subjective_color = '#DAA520'  # Golden rod for subjective
    
    # Set up data for grouped bar chart
    x = np.arange(len(activities))
    width = 0.4  # Slightly wider bars for denser appearance
    
    # Get counts for the bars in sorted order
    objective_counts = [data[activity]['objective'] for activity in activities]
    subjective_counts = [data[activity]['subjective'] for activity in activities]
    
    # Create the grouped bars
    objective_bars = ax.bar(x - width/2, objective_counts, width, label='Objective', 
                           color=objective_color, alpha=0.85, edgecolor='white', linewidth=0.5)
    subjective_bars = ax.bar(x + width/2, subjective_counts, width, label='Subjective', 
                            color=subjective_color, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Add count labels to the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(objective_bars)
    add_labels(subjective_bars)
    
    # Customize the chart appearance
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in activities], fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tasks', fontsize=11)
    ax.legend(fontsize=10, frameon=False, loc='upper right')
    
    # Remove the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid lines only on the y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout - use tighter margins
    plt.tight_layout(pad=1.0)
    
    # Save both PDF and PNG versions of the figure
    plt.savefig(f"{output_path_base}.pdf", bbox_inches='tight', dpi=config['global']['dpi'])
    plt.savefig(f"{output_path_base}.png", bbox_inches='tight', dpi=config['global']['dpi'])
    plt.close()
    
    print(f"Figures saved to {output_path_base}.pdf and {output_path_base}.png")
    
    # Create an additional plot showing the distribution of task types by series
    plot_series_distribution(summary_stats, config)

def plot_series_distribution(summary_stats, config):
    """
    Create a supplementary visualization showing how task types are distributed
    across different series.
    """
    # Load series data
    try:
        with open(DATA_DIR / 'series_data.json', 'r') as f:
            series_data = json.load(f)
    except FileNotFoundError:
        print("Series data not found. Skipping series distribution plot.")
        return
    
    # Filter out Champion of Champions and only keep Series 1-18
    filtered_series_data = []
    for item in series_data:
        if "Champion" not in item['series'] and "Series" in item['series']:
            try:
                series_num = int(item['series'].split()[1])
                if 1 <= series_num <= 18:
                    filtered_series_data.append(item)
            except (ValueError, IndexError):
                continue
    
    # Sort the series by numeric value (1-18)
    filtered_series_data.sort(key=lambda x: int(x['series'].split()[1]))
    
    # Extract series numbers for x-axis
    series_names = [item['series'] for item in filtered_series_data]
    series_nums = []
    for name in series_names:
        try:
            # Extract just the number
            num = name.split(" ")[1]
            series_nums.append(num)
        except:
            series_nums.append(name)
    
    # Extract activity proportions
    creative_props = [item['activity_proportions']['creative'] for item in filtered_series_data]
    mental_props = [item['activity_proportions']['mental'] for item in filtered_series_data]
    physical_props = [item['activity_proportions']['physical'] for item in filtered_series_data]
    social_props = [item['activity_proportions']['social'] for item in filtered_series_data]
    
    # Set up the figure with a modern style - denser
    fig_size = (9, 4)  # Denser proportion
    fig, ax = plt.subplots(figsize=fig_size)
    apply_plot_style(fig, ax)
    
    # Create the stacked bar chart
    width = 0.9  # Wider bars for denser look
    x = np.arange(len(series_nums))
    
    # Use colors from the task_type_palette in the config but with better alpha for modern look
    palette = get_palette('task_type_palette', 4)
    
    # Create bars with cleaner edges
    ax.bar(x, creative_props, width, label='Creative', color=palette[0], alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.bar(x, mental_props, width, bottom=creative_props, label='Mental', color=palette[1], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    bottom = np.array(creative_props) + np.array(mental_props)
    ax.bar(x, physical_props, width, bottom=bottom, label='Physical', color=palette[2], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    bottom = bottom + np.array(physical_props)
    ax.bar(x, social_props, width, bottom=bottom, label='Social', color=palette[3], alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # Set labels with modern fonts
    ax.set_xlabel('Series', fontsize=11)
    ax.set_ylabel('Proportion of Tasks', fontsize=11)
    
    # Set x-ticks
    ax.set_xticks(x)
    ax.set_xticklabels(series_nums, rotation=45, fontsize=9)
    
    # Add a legend with a cleaner style
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=10)
    
    # Remove the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add subtle grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Annotate with note about task categories overlapping - cleaner style
    ax.annotate('Note: Tasks can belong to multiple categories, so proportions sum to >1.0', 
               xy=(0.98, 0.02), xycoords='axes fraction',
               ha='right', va='bottom', fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8, ec='none'))
    
    # Adjust layout - use tighter margins
    plt.tight_layout(pad=1.0)
    
    # Save both PDF and PNG versions of the figure
    output_base = SCRIPT_DIR / 'fig3'
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight', dpi=config['global']['dpi'])
    plt.savefig(f"{output_base}.png", bbox_inches='tight', dpi=config['global']['dpi'])
    plt.close()
    
    print(f"Series distribution figures saved to {output_base}.pdf and {output_base}.png")

    # Save metrics for caption
    metrics = {
        'total_tasks': summary_stats['total_tasks'],
        'pct_creative': round(summary_stats['activity_counts']['creative'] / summary_stats['total_tasks'] * 100, 1),
        'pct_physical': round(summary_stats['activity_counts']['physical'] / summary_stats['total_tasks'] * 100, 1),
        'pct_objective': round(summary_stats['judgment_counts']['objective'] / summary_stats['total_tasks'] * 100, 1),
        'pct_subjective': round(summary_stats['judgment_counts']['subjective'] / summary_stats['total_tasks'] * 100, 1)
    }
    
    with open(SCRIPT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    # Load the data
    try:
        with open(DATA_DIR / 'bubble_data.json', 'r') as f:
            bubble_data = json.load(f)
        
        with open(DATA_DIR / 'summary_stats.json', 'r') as f:
            summary_stats = json.load(f)
            
        plot_activity_judgment_bars(bubble_data, summary_stats)
        
    except FileNotFoundError:
        print("Data files not found. Run process_task_characteristics_data.py first.")
        # Import the renamed module
        import process_task_characteristics_data
        bubble_data, summary_stats, _ = process_task_characteristics_data.process_data()
        plot_activity_judgment_bars(bubble_data, summary_stats) 