"""
Taskmaster Location Visualization Script

This script generates visualizations of task distributions by location categories from the
Taskmaster UK dataset. It creates t-SNE plots showing the relationships between tasks, as well
as additional visualizations for skill distributions and task type distributions.

Required Input Files:
- data/processed/task/tasks_standardized_final.csv: Contains the standardized task data with
  location categories, skill scores, and task types.

Output:
- PNG, SVG, and PDF visualizations saved to the visualizations/location_improved directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.colors as mcolors

# Create output directory for visualizations
output_dir = "visualizations/location_improved"
os.makedirs(output_dir, exist_ok=True)

# Set figure aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Define input file path
input_file = 'data/processed/task/tasks_standardized_final.csv'

# Load the standardized data
print(f"Loading standardized data with fixed locations from {input_file}...")
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} tasks.")

# Select skill columns for analysis
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

# Define the preferred order for location categories
location_order = ["Inside House", "Outside House", "Studio/Stage", "Other Locations"]

# Count location categories
location_counts = df['location_category'].value_counts()
print("\nLocation category counts:")
print(location_counts)

# Count detailed locations
detailed_counts = df['standardized_location'].value_counts().head(10)
print("\nTop standardized locations:")
print(detailed_counts)

# Set up colormaps for each location category
category_base_colors = {
    "Inside House": "Blues",
    "Outside House": "Greens", 
    "Studio/Stage": "Oranges",
    "Other Locations": "Purples"
}

print("Preparing for t-SNE dimensionality reduction...")

# We'll use both skill scores and location features for t-SNE, with higher weight for categories
# First, create one-hot encodings for locations with higher weight for categories
location_cat_features = pd.get_dummies(df['location_category'], prefix='loc_cat')
location_specific_features = pd.get_dummies(df['standardized_location'], prefix='loc_spec')

# Apply weights: even lower weights to make clusters less tight
weight_category = 1.5  # Lower weight for less clustering
weight_specific = 0.3  # Even lower weight for specific

# Scale the features by weight
location_cat_weighted = location_cat_features * weight_category
location_spec_weighted = location_specific_features * weight_specific

# Combine features: skill scores + weighted location features 
X_combined = pd.concat([
    df[skill_columns].reset_index(drop=True),
    location_cat_weighted.reset_index(drop=True),
    location_spec_weighted.reset_index(drop=True)
], axis=1)

X_features = X_combined.values
print(f"Using {X_features.shape[0]} data points with {X_features.shape[1]} features for t-SNE.")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Add random noise to increase jitter
np.random.seed(42)  # For reproducibility
jitter_scale = 0.1  # Scale of the noise
X_with_jitter = X_scaled + np.random.normal(0, jitter_scale, X_scaled.shape)

# Run t-SNE with adjusted parameters for less clustering
print("Running t-SNE with jitter...")
tsne = TSNE(
    n_components=2, 
    random_state=42, 
    perplexity=70,  # Higher perplexity for even less tight clustering
    learning_rate='auto',  # Let TSNE choose the optimal learning rate
    n_iter=2000,  # Slightly fewer iterations
    metric='euclidean',  # Standard distance metric
    init='pca'  # Use PCA initialization for better global structure
)
tsne_results = tsne.fit_transform(X_with_jitter)
print(f"t-SNE output shape: {tsne_results.shape}")

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

# Create a comprehensive figure with multiple visualizations
fig = plt.figure(figsize=(20, 18))
plt.suptitle('Taskmaster Tasks Analysis by Location', fontsize=24, y=0.98)

# 1. t-SNE plot by specific location with category-based colormaps
ax1 = plt.subplot(2, 2, 1)

# Get unique locations within each category
locations_by_category = {}
for category in location_order:
    locations_by_category[category] = df[df['location_category'] == category]['standardized_location'].unique()

# Create a colormap for each specific location using the category base color
location_colors = {}

# Count points for verification
total_points_plotted = 0

# Plot each location category with a unique colormap
for category_idx, category in enumerate(location_order):
    # Get the base colormap for this category
    cmap_name = category_base_colors[category]
    cmap = plt.colormaps.get_cmap(cmap_name)
    
    # Get list of specific locations in this category
    specific_locations = locations_by_category[category]
    n_locations = len(specific_locations)
    
    # Create evenly spaced colors within the colormap
    for loc_idx, specific_loc in enumerate(specific_locations):
        # Get color from the colormap - use weighted position to avoid too light/dark colors
        position = 0.3 + (0.6 * loc_idx / max(1, n_locations - 1))
        color = cmap(position)
        
        # Store the color for this specific location
        location_colors[specific_loc] = color
        
        # Create mask for this specific location
        mask = df['standardized_location'] == specific_loc
        points_in_location = np.sum(mask)
        
        # Skip if no points (shouldn't happen but just in case)
        if points_in_location == 0:
            continue
            
        # Plot points for this specific location
        scatter = plt.scatter(
            tsne_results[mask, 0], 
            tsne_results[mask, 1], 
            color=color, 
            alpha=0.7, 
            s=40,
            edgecolors='white',
            linewidth=0.3,
            label=f"{specific_loc} ({points_in_location})"
        )
        
        total_points_plotted += points_in_location

print(f"Total points plotted in t-SNE: {total_points_plotted}")

# Set axis limits wider to show more spread
x_percentiles = np.percentile(tsne_results[:, 0], [0.5, 99.5])
y_percentiles = np.percentile(tsne_results[:, 1], [0.5, 99.5])
x_margin = (x_percentiles[1] - x_percentiles[0]) * 0.1
y_margin = (y_percentiles[1] - y_percentiles[0]) * 0.1
plt.xlim(x_percentiles[0] - x_margin, x_percentiles[1] + x_margin)
plt.ylim(y_percentiles[0] - y_margin, y_percentiles[1] + y_margin)

# Add labels for the plot
plt.title(f't-SNE Visualization by Location (n={total_points_plotted})', fontsize=16)

# Create a simplified legend that just shows the main categories with a note about shading
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add simplified legend with category colors
category_handles = []
category_labels = []

for category in location_order:
    count = len(df[df['location_category'] == category])
    cmap = plt.colormaps.get_cmap(category_base_colors[category])
    category_patch = Patch(color=cmap(0.5), label=f"{category} ({count})")
    category_handles.append(category_patch)
    category_labels.append(f"{category} ({count})")

# Create a cleaner legend inside the plot
plt.legend(handles=category_handles, labels=category_labels, 
           title="Location Categories", loc='best', fontsize=9)

# Add a text note explaining the color shading inside the plot
# Calculate a good position inside the plot (lower right corner)
x_range = plt.xlim()[1] - plt.xlim()[0]
y_range = plt.ylim()[1] - plt.ylim()[0]
note_x = plt.xlim()[0] + x_range * 0.02  # 2% from left
note_y = plt.ylim()[0] + y_range * 0.02  # 2% from bottom

plt.annotate("Note: Shading within each color represents different specific locations.", 
            xy=(note_x, note_y), xycoords='data',
            bbox={"boxstyle":"round,pad=0.5", "facecolor":"white", "alpha":0.8},
            fontsize=8)

# 2. Distribution of skill scores (overall, not by location)
ax2 = plt.subplot(2, 2, 2)

# Melt the dataframe to get all skill scores
skill_scores = df.melt(
    value_vars=skill_columns,
    var_name='skill_type',
    value_name='score'
)
# Clean up skill names for display
skill_scores['skill_type'] = skill_scores['skill_type'].str.replace('std_', '').str.replace('_score', '')
skill_scores['skill_type'] = skill_scores['skill_type'].str.replace('_', ' ').str.title()

# Create a violin plot of overall skill distribution with a non-sequential colormap
skill_palette = sns.color_palette("Set3", n_colors=len(skill_columns))

# Use hue for the violinplot to avoid warning
sns.violinplot(
    x='skill_type', 
    y='score', 
    data=skill_scores, 
    inner='quartile',
    hue='skill_type',
    palette=skill_palette,
    legend=False
)
plt.title('Distribution of Skill Scores Across All Tasks', fontsize=16)
plt.xlabel('')
plt.ylabel('Score', fontsize=14)
plt.xticks(rotation=45, ha='right')

# 3. Distribution of task types by location category
ax3 = plt.subplot(2, 2, 3)
# Create a crosstab with locations in the specified order
cross_tab = pd.crosstab(df['location_category'], df['standardized_task_type'])
# Reorder rows to match the desired location order
cross_tab = cross_tab.reindex(location_order)

# Convert to percentages by row
cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

# Use a diverging colormap for the heatmap - better for showing distribution
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Create the heatmap with annotations
sns.heatmap(cross_tab_pct, annot=True, cmap=cmap, fmt='.1f', linewidths=.5)
plt.title('Distribution of Task Types by Location (%)', fontsize=16)
plt.ylabel('Location Category', fontsize=14)
plt.xlabel('Task Type', fontsize=14)

# 4. Task count by location category
ax4 = plt.subplot(2, 2, 4)
# Create a Series with the counts in the specified order
ordered_counts = pd.Series(
    [location_counts.get(loc, 0) for loc in location_order],
    index=location_order
)

# Updated barplot to use categorical colors that match the t-SNE
category_colors = [plt.colormaps.get_cmap(category_base_colors[loc])(0.5) for loc in location_order]

# Use hue for barplot to avoid warning
ordered_counts_df = pd.DataFrame({
    'Location': ordered_counts.index,
    'Count': ordered_counts.values
})

sns.barplot(
    x='Location', 
    y='Count', 
    data=ordered_counts_df,
    hue='Location',
    palette=category_colors,
    legend=False
)
plt.title('Number of Tasks by Location Category', fontsize=16)
plt.xlabel('Location Category', fontsize=14)
plt.ylabel('Number of Tasks', fontsize=14)
plt.xticks(rotation=0)

# Add count labels on bars
for i, count in enumerate(ordered_counts.values):
    plt.text(i, count + 5, str(count), ha='center', fontsize=12)

# Adjust overall layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save the main visualization in multiple formats
base_filename = f'{output_dir}/location_analysis_improved'
save_figure(fig, base_filename)

# Create a separate figure for t-SNE only with more focus
fig2 = plt.figure(figsize=(16, 14))
plt.suptitle('t-SNE Visualization of Taskmaster Tasks by Location', fontsize=20)

# Plot each specific location with its assigned color
for category in location_order:
    # Get specific locations for this category
    specific_locations = locations_by_category[category]
    
    for specific_loc in specific_locations:
        # Get the color assigned to this specific location
        color = location_colors[specific_loc]
        
        # Create mask for this specific location
        mask = df['standardized_location'] == specific_loc
        points_in_location = np.sum(mask)
        
        # Skip if no points
        if points_in_location == 0:
            continue
            
        # Plot points for this specific location
        plt.scatter(
            tsne_results[mask, 0], 
            tsne_results[mask, 1], 
            color=color, 
            alpha=0.7, 
            s=60,  # Larger for better visibility
            edgecolors='white',
            linewidth=0.3,
            label=f"{specific_loc} ({points_in_location})"
        )

# Set the same axis limits as before
plt.xlim(x_percentiles[0] - x_margin, x_percentiles[1] + x_margin)
plt.ylim(y_percentiles[0] - y_margin, y_percentiles[1] + y_margin)

# Create the same custom legend structure
from matplotlib.lines import Line2D

# First, create category headers
category_handles = []
category_labels = []

for category in location_order:
    count = len(df[df['location_category'] == category])
    cmap = plt.colormaps.get_cmap(category_base_colors[category])
    category_patch = Patch(color=cmap(0.5), label=f"{category} ({count})")
    category_handles.append(category_patch)
    category_labels.append(f"{category} ({count})")

# Create a legend with the category headers - inside the plot for this detailed version
plt.legend(handles=category_handles, labels=category_labels, 
           title="Location Categories", loc='best', fontsize=12)

# Add descriptive note inside the plot
plt.annotate("Shading within each color represents different specific locations.", 
            xy=(0.02, 0.02), xycoords='axes fraction',
            bbox={"boxstyle":"round,pad=0.5", "facecolor":"white", "alpha":0.8},
            fontsize=10)

# Add title with proper positioning
plt.title(f't-SNE Visualization by Location (n={total_points_plotted})', fontsize=16, pad=20)

# Save the detailed figure in multiple formats
plt.tight_layout()
base_filename = f'{output_dir}/location_tsne_detailed'
save_figure(fig2, base_filename)

# Create a third visualization focusing on task type distribution across locations
fig3 = plt.figure(figsize=(14, 10))
plt.suptitle('Task Type Distribution by Location Category', fontsize=20)

# Create a heatmap showing raw counts with locations in the specified order
task_type_location_counts = pd.crosstab(df['standardized_task_type'], df['location_category'])
# Reorder columns to match the desired location order
task_type_location_counts = task_type_location_counts[location_order]

# Use a better colormap for the counts
cmap = sns.color_palette("YlGnBu", as_cmap=True)
sns.heatmap(task_type_location_counts, annot=True, fmt='d', cmap=cmap, linewidths=.5)

plt.title('Task Type Distribution by Location (Counts)', fontsize=16)
plt.xlabel('Location Category', fontsize=14)
plt.ylabel('Task Type', fontsize=14)

# Save the count matrix figure in multiple formats
plt.tight_layout()
base_filename = f'{output_dir}/task_type_by_location_counts'
save_figure(fig3, base_filename)

print("All visualizations completed!") 