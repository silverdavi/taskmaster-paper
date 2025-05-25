#!/usr/bin/env python3
"""
Process data for Figure Supplementary 9: Spider Plot of Task Skills (VERSION 2)

This script uses the existing continuous scoring columns to create meaningful
fractional skill profiles rather than binary ones.

Author: Taskmaster Analysis Team
"""

import pandas as pd
import numpy as np
import json
import warnings
from sklearn.metrics.pairwise import euclidean_distances
warnings.filterwarnings('ignore')

def load_and_process_data():
    """Load tasks data and extract continuous skill scores"""
    print("Loading tasks data...")
    
    # Load the tasks CSV
    df = pd.read_csv('../../data/raw/_OL_tasks.csv')
    print(f"Loaded {len(df)} tasks")
    
    # Check for missing values in score columns
    score_columns = [
        'creativity_required_score', 'physical_demand_score', 'technical_difficulty_score',
        'time_pressure_score', 'weirdness_score', 'entertainment_value_score',
        'preparation_possible_score', 'luck_factor_score'
    ]
    
    # Filter out tasks with missing scores
    df_clean = df.dropna(subset=score_columns).copy()
    print(f"Retained {len(df_clean)} tasks with complete scoring data")
    
    return df_clean

def create_skill_profiles(df):
    """Create meaningful skill profiles from existing scores"""
    
    # Map scoring columns to interpretable skill dimensions
    skill_mappings = {
        'Creativity': 'creativity_required_score',
        'Physical Coordination': 'physical_demand_score', 
        'Problem Solving': 'technical_difficulty_score',
        'Time Pressure': 'time_pressure_score',
        'Originality': 'weirdness_score',
        'Entertainment': 'entertainment_value_score',
        'Strategic Planning': 'preparation_possible_score',  # Higher prep possible = more strategic
        'Adaptability': 'luck_factor_score'  # Higher luck factor = need more adaptability
    }
    
    print(f"Creating skill profiles using {len(skill_mappings)} dimensions...")
    
    # Extract skill scores (normalize to 0-1 scale)
    skill_data = []
    for idx, row in df.iterrows():
        skills = {}
        for skill_name, score_column in skill_mappings.items():
            # Normalize scores to 0-1 range (assuming original scale is 1-10)
            raw_score = row[score_column]
            normalized_score = (raw_score - 1) / 9  # Convert 1-10 to 0-1
            skills[skill_name] = max(0, min(1, normalized_score))  # Ensure 0-1 bounds
        
        task_info = {
            'task_title': row['task_title'],
            'task_id': row.get('task_unique_id', idx),
            'series': row['series'],
            'episode': row['episode'],
            'skills': skills,
            'skill_vector': list(skills.values())
        }
        skill_data.append(task_info)
    
    skill_names = list(skill_mappings.keys())
    return skill_data, skill_names

def find_polarized_tasks_continuous(skill_data, skill_names, n_examples=4):
    """Find tasks with very different continuous skill profiles"""
    print("Finding polarized tasks using continuous skill profiles...")
    
    # Create skill matrix
    skill_matrix = np.array([task['skill_vector'] for task in skill_data])
    
    # Calculate pairwise Euclidean distances
    distances = euclidean_distances(skill_matrix)
    
    # Find tasks that are most different from each other
    max_distance_pairs = []
    n_tasks = len(skill_data)
    
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            distance = distances[i, j]
            max_distance_pairs.append((i, j, distance))
    
    # Sort by distance (most different first)
    max_distance_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Select diverse examples
    selected_indices = set()
    polarized_examples = []
    
    # Add the most different pair
    for i, j, dist in max_distance_pairs:
        if len(selected_indices) < n_examples:
            if i not in selected_indices:
                selected_indices.add(i)
                task = skill_data[i]
                task['distance_score'] = dist
                polarized_examples.append(task)
            if j not in selected_indices and len(selected_indices) < n_examples:
                selected_indices.add(j)
                task = skill_data[j]
                task['distance_score'] = dist
                polarized_examples.append(task)
        else:
            break
    
    # Add some extreme cases (highest and lowest in different dimensions)
    if len(polarized_examples) < n_examples:
        # Find tasks with extreme profiles
        skill_matrix = np.array([task['skill_vector'] for task in skill_data])
        
        for skill_idx, skill_name in enumerate(skill_names):
            if len(polarized_examples) >= n_examples:
                break
                
            # Highest in this skill
            max_idx = np.argmax(skill_matrix[:, skill_idx])
            if max_idx not in selected_indices:
                task = skill_data[max_idx].copy()
                task['selection_reason'] = f"Highest {skill_name}"
                polarized_examples.append(task)
                selected_indices.add(max_idx)
    
    # Limit to desired number
    polarized_examples = polarized_examples[:n_examples]
    
    print(f"Selected {len(polarized_examples)} polarized task examples:")
    for i, task in enumerate(polarized_examples):
        print(f"  {i+1}. {task['task_title']} (Series {task['series']})")
        # Show top skills
        skills_sorted = sorted(task['skills'].items(), key=lambda x: x[1], reverse=True)
        top_skills = [f"{name}: {score:.2f}" for name, score in skills_sorted[:3]]
        print(f"      Top skills: {', '.join(top_skills)}")
    
    return polarized_examples

def save_continuous_data(skill_names, polarized_examples):
    """Save processed continuous data for plotting"""
    print("Saving continuous skill data...")
    
    # Save comprehensive data
    output_data = {
        'skill_names': skill_names,
        'polarized_examples': polarized_examples,
        'analysis_metadata': {
            'skill_dimensions': len(skill_names),
            'polarized_examples_count': len(polarized_examples),
            'score_range': '0.0 to 1.0 (normalized)',
            'method': 'continuous_euclidean_distance'
        }
    }
    
    # Convert numpy types for JSON serialization
    for task in output_data['polarized_examples']:
        # Convert skill values
        task['skills'] = {k: float(v) for k, v in task['skills'].items()}
        task['skill_vector'] = [float(x) for x in task['skill_vector']]
        if 'distance_score' in task:
            task['distance_score'] = float(task['distance_score'])
    
    with open('skills_data.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save data formatted for spider plot
    radar_data = {
        'skills': skill_names,
        'tasks': []
    }
    
    for task in polarized_examples:
        task_data = {
            'title': task['task_title'],
            'series': int(task['series']),
            'episode': int(task['episode']),
            'values': [float(task['skills'][skill]) for skill in skill_names]
        }
        radar_data['tasks'].append(task_data)
    
    with open('radar_plot_data.json', 'w') as f:
        json.dump(radar_data, f, indent=2)
    
    print("Data saved:")
    print("  - skills_data.json")
    print("  - radar_plot_data.json")

def main():
    """Main processing function"""
    print("=== Processing Figure Supplementary 9 Data (Continuous Version) ===")
    print("Using existing continuous scoring data for meaningful skill profiles...")
    
    # Load and process data
    df = load_and_process_data()
    
    # Create continuous skill profiles
    skill_data, skill_names = create_skill_profiles(df)
    
    # Find polarized examples
    polarized_examples = find_polarized_tasks_continuous(skill_data, skill_names)
    
    # Save processed data
    save_continuous_data(skill_names, polarized_examples)
    
    print("\n=== Processing Complete ===")
    print(f"Processed {len(df)} tasks with continuous skill profiles")
    print(f"Identified {len(skill_names)} skill dimensions")
    print(f"Selected {len(polarized_examples)} polarized task examples")
    print("\nThis will create much more meaningful spider plots with smooth, interpretable shapes!")
    
    return True

if __name__ == "__main__":
    main() 