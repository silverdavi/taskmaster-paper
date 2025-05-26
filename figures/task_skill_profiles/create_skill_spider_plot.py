#!/usr/bin/env python3
"""
Create Spider Plot for Figure Supplementary 9: Continuous Task Skill Profiles

This script creates a radar/spider plot showing the CONTINUOUS skill intensity
profiles for polarized task examples from the Taskmaster dataset.

Author: Taskmaster Analysis Team
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import seaborn as sns

def load_data():
    """Load the processed continuous radar plot data"""
    with open('radar_plot_data.json', 'r') as f:
        data = json.load(f)
    return data

def create_continuous_spider_plot(data, save_path=None):
    """Create a professional spider/radar plot with continuous values"""
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use more distinct colors for better visibility
    colors = ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A']  # Red, Blue, Green, Orange, Purple
    
    # Extract data
    skills = data['skills']
    tasks = data['tasks']
    
    # Number of variables
    N = len(skills)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(16, 12), subplot_kw=dict(projection='polar'))
    
    def wrap_text(text, max_width=25):
        """Wrap text to multiple lines for legend"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    # Plot each task
    for i, task in enumerate(tasks):
        values = task['values'][:]  # Copy the values
        values += values[:1]  # Complete the circle
        
        # Create multiline label for legend
        full_title = task['title']
        multiline_title = wrap_text(full_title, max_width=30)
        
        # Plot the polygon with continuous values
        ax.plot(angles, values, 'o-', linewidth=3, label=multiline_title, 
                color=colors[i % len(colors)], markersize=5, alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    # Add skill labels with better formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(skills, fontsize=22, fontweight='bold')
    
    # Set the range for the radial axis (0 to 1 for normalized scores)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18)
    ax.grid(True, alpha=0.4)
    
    # Position legend outside the plot with multiline support
    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=22, 
               frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to accommodate legend
    plt.tight_layout()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Continuous spider plot saved to: {save_path}")
    
    return fig, ax

def create_detailed_summary(data):
    """Create a detailed text summary of the continuous skill profiles"""
    
    text_summary = []
    text_summary.append("=== POLARIZED TASK EXAMPLES (CONTINUOUS SKILL PROFILES) ===\n")
    
    skills = data['skills']
    
    for i, task in enumerate(data['tasks']):
        text_summary.append(f"{i+1}. {task['title']}")
        text_summary.append(f"   Series {task['series']}, Episode {task['episode']}")
        text_summary.append(f"   Skill Profile (0.0-1.0 scale):")
        
        # Show all skills with their scores
        for j, skill in enumerate(skills):
            score = task['values'][j]
            # Create a visual bar representation
            bar_length = int(score * 20)  # Scale to 20 characters
            bar = "█" * bar_length + "░" * (20 - bar_length)
            text_summary.append(f"     {skill:18s}: {score:.2f} |{bar}|")
        
        # Identify dominant skills (> 0.7)
        dominant_skills = []
        for j, score in enumerate(task['values']):
            if score > 0.7:
                dominant_skills.append(f"{skills[j]} ({score:.2f})")
        
        if dominant_skills:
            text_summary.append(f"   Dominant Skills: {', '.join(dominant_skills)}")
        
        text_summary.append("")  # Empty line
    
    # Add analysis summary
    text_summary.append("=== ANALYSIS ===")
    text_summary.append("The continuous skill profiles show clear differentiation between tasks:")
    text_summary.append("- Tasks have smooth, interpretable intensity profiles")
    text_summary.append("- Different tasks emphasize different skill combinations")
    text_summary.append("- Polarized tasks show contrasting skill requirements")
    text_summary.append("")
    
    return "\n".join(text_summary)

def main():
    """Main function to create the continuous spider plot"""
    print("=== Creating Continuous Spider Plot for Figure Supplementary 9 ===")
    
    # Load data
    data = load_data()
    print(f"Loaded data for {len(data['skills'])} skills and {len(data['tasks'])} tasks")
    
    # Show value ranges
    all_values = []
    for task in data['tasks']:
        all_values.extend(task['values'])
    
    print(f"Skill intensity range: {min(all_values):.3f} to {max(all_values):.3f}")
    print("Using continuous fractional values for meaningful spider plot shapes!")
    
    # Create spider plot
    fig, ax = create_continuous_spider_plot(data, 'figure_sup9_spider_plot.png')
    
    # Also save as PDF
    fig.savefig('figure_sup9_spider_plot.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Create detailed summary
    summary = create_detailed_summary(data)
    
    # Save summary
    with open('tasks_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\nDetailed task profiles:")
    print(summary[:1000] + "...\n[Full summary saved to file]")
    
    # Close the plot
    plt.close(fig)
    
    print("\n=== Visualization Complete ===")
    print("Files created:")
    print("  - figure_sup9_spider_plot.png")
    print("  - figure_sup9_spider_plot.pdf") 
    print("  - tasks_summary.txt")
    print("\nNow THIS is a proper spider plot with meaningful continuous values!")

if __name__ == "__main__":
    main() 