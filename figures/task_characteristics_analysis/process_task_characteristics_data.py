import pandas as pd
import numpy as np
import json
import os
from scipy import stats
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = SCRIPT_DIR / "data"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def process_data():
    """Process Taskmaster UK tasks data for Figure 3 visualization."""
    # Load the dataset
    tasks_df = pd.read_csv(DATA_DIR / 'taskmaster_UK_tasks.csv')
    
    # Define our categories
    activity_types = ['is_creative', 'is_mental', 'is_physical', 'is_social']
    
    # Focusing only on objective and subjective judgment types
    judgment_types = ['is_objective', 'is_subjective']
    
    # Create a dictionary to store our results
    bubble_data = []
    
    # For each activity type and judgment type combination
    for activity in activity_types:
        activity_name = activity.replace('is_', '')
        for judgment in judgment_types:
            judgment_name = judgment.replace('is_', '')
            
            # Count tasks that have both this activity type AND this judgment type
            count = sum((tasks_df[activity] == True) & (tasks_df[judgment] == True))
            
            # Only add if there are any tasks with this combination
            if count > 0:
                bubble_data.append({
                    'activity': activity_name,
                    'judgment': judgment_name,
                    'count': int(count)
                })
    
    # Calculate additional statistics for the entire dataset
    total_tasks = len(tasks_df)
    
    # Count by assignment type
    assignment_counts = {
        'solo': sum(tasks_df['is_solo']),
        'team': sum(tasks_df['is_team']),
        'special': sum(tasks_df['is_special']),
        'split': sum(tasks_df['is_split']),
        'tiebreaker': sum(tasks_df['is_tiebreaker'])
    }
    
    # Count by task format
    format_counts = {
        'prize': sum(tasks_df['is_prize']),
        'filmed': sum(tasks_df['is_filmed']),
        'homework': sum(tasks_df['is_homework']),
        'live': sum(tasks_df['is_live'])
    }
    
    # Count by activity type
    activity_counts = {
        'creative': sum(tasks_df['is_creative']),
        'mental': sum(tasks_df['is_mental']),
        'physical': sum(tasks_df['is_physical']),
        'social': sum(tasks_df['is_social'])
    }
    
    # Count by judgment type (only subjective and objective)
    judgment_counts = {
        'objective': sum(tasks_df['is_objective']),
        'subjective': sum(tasks_df['is_subjective'])
    }
    
    # Count by number of task briefs
    brief_counts = {
        'single': sum(tasks_df['is_single']),
        'multiple': sum(tasks_df['is_multiple'])
    }
    
    # Count by task originality
    originality_counts = {
        'original': sum(tasks_df['is_original']),
        'adapted': sum(tasks_df['is_adapted'])
    }
    
    # Calculate proportions of different task types per series
    series_data = []
    series_list = tasks_df['series_name'].unique()
    
    # Filter out Champion of Champions and other special series
    regular_series = []
    for series in series_list:
        if "Champion" not in series and "Series" in series:
            try:
                series_num = int(series.split()[1])
                if 1 <= series_num <= 18:
                    regular_series.append(series)
            except (ValueError, IndexError):
                continue
    
    # Sort by series number
    regular_series.sort(key=lambda x: int(x.split()[1]))
    
    # Calculate proportions for each series
    for series in regular_series:
        series_tasks = tasks_df[tasks_df['series_name'] == series]
        series_count = len(series_tasks)
        
        series_info = {
            'series': series,
            'task_count': series_count,
            'activity_proportions': {
                'creative': sum(series_tasks['is_creative']) / series_count,
                'mental': sum(series_tasks['is_mental']) / series_count,
                'physical': sum(series_tasks['is_physical']) / series_count,
                'social': sum(series_tasks['is_social']) / series_count
            },
            'judgment_proportions': {
                'objective': sum(series_tasks['is_objective']) / series_count,
                'subjective': sum(series_tasks['is_subjective']) / series_count
            }
        }
        series_data.append(series_info)
    
    # Analyze trends in task type proportions across series
    trend_analysis = analyze_trends(series_data)
    
    # Save all data
    with open(OUTPUT_DIR / 'bubble_data.json', 'w') as f:
        json.dump(bubble_data, f, indent=2)
    
    # Save summary statistics
    summary_stats = {
        'total_tasks': total_tasks,
        'assignment_counts': assignment_counts,
        'format_counts': format_counts,
        'activity_counts': activity_counts,
        'judgment_counts': judgment_counts,
        'brief_counts': brief_counts,
        'originality_counts': originality_counts
    }
    
    with open(OUTPUT_DIR / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save series data
    with open(OUTPUT_DIR / 'series_data.json', 'w') as f:
        json.dump(series_data, f, indent=2)
    
    # Save trend analysis
    with open(OUTPUT_DIR / 'trend_analysis.json', 'w') as f:
        json.dump(trend_analysis, f, indent=2)
    
    print(f"Processed {total_tasks} tasks.")
    print(f"Bubble data saved to {OUTPUT_DIR / 'bubble_data.json'}")
    print(f"Summary statistics saved to {OUTPUT_DIR / 'summary_stats.json'}")
    print(f"Series data saved to {OUTPUT_DIR / 'series_data.json'}")
    print(f"Trend analysis saved to {OUTPUT_DIR / 'trend_analysis.json'}")
    
    return bubble_data, summary_stats, series_data

def analyze_trends(series_data):
    """
    Analyze if there are significant trends in task type proportions across series.
    Uses Mann-Kendall trend test to detect monotonic trends.
    
    Args:
        series_data: List of dictionaries containing task type proportions by series
        
    Returns:
        Dictionary with trend analysis results
    """
    # Extract series numbers and activity proportions
    series_nums = [int(item['series'].split()[1]) for item in series_data]
    
    # Get the activity proportions for each series
    activity_data = {
        'creative': [item['activity_proportions']['creative'] for item in series_data],
        'mental': [item['activity_proportions']['mental'] for item in series_data],
        'physical': [item['activity_proportions']['physical'] for item in series_data],
        'social': [item['activity_proportions']['social'] for item in series_data]
    }
    
    # Dictionary to store results
    trend_results = {}
    
    # Significance level
    alpha = 0.05
    
    # Perform Mann-Kendall trend test for each activity type
    for activity, proportions in activity_data.items():
        # Perform the trend test
        result = stats.kendalltau(series_nums, proportions)
        tau, p_value = result
        
        # Convert numpy types to Python native types for JSON serialization
        tau = float(tau)
        p_value = float(p_value)
        
        # Determine if there's a significant trend
        significant = bool(p_value < alpha)
        
        # Determine the direction of the trend
        if significant:
            if tau > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
        else:
            trend = "no trend"
        
        # Also perform linear regression to quantify the change
        slope, intercept, r_value, p_value_reg, std_err = stats.linregress(series_nums, proportions)
        
        # Convert numpy types to Python native types for JSON serialization
        slope = float(slope)
        intercept = float(intercept)
        r_value = float(r_value)
        p_value_reg = float(p_value_reg)
        std_err = float(std_err)
        
        # Calculate percent change from first to last series
        first_value = float(proportions[0])
        last_value = float(proportions[-1])
        if first_value > 0:
            percent_change = ((last_value - first_value) / first_value) * 100
        else:
            percent_change = float('inf') if last_value > 0 else 0
        
        # Store results
        trend_results[activity] = {
            'tau': tau,
            'p_value': p_value,
            'significant': significant,
            'trend': trend,
            'slope': slope,
            'percent_change': percent_change
        }
    
    return trend_results

if __name__ == "__main__":
    process_data() 