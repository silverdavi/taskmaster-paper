#!/usr/bin/env python3
"""
Perform PCA on four key statistics for each Taskmaster season:
1. Percentage of 1 ratings
2. Percentage of 10 ratings
3. Mean of ratings 2-9
4. Standard deviation of ratings 2-9

Visualize the results in a 2D scatter plot with sequential colormap.
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects  # Add proper import for path effects
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter

# Configuration
CSV_FILE = "taskmaster_histograms_corrected_fixed.csv"

def extract_season_statistics():
    """
    Extract the four key statistics for each season:
    - Percentage of 1 ratings
    - Percentage of 10 ratings
    - Mean of ratings 2-9
    - Standard deviation of ratings 2-9
    """
    # Initialize dictionaries to store statistics by season
    season_stats = {}
    all_season_data = {}
    
    # Read CSV data
    with open(CSV_FILE, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            season = int(row[0])
            
            if season not in all_season_data:
                all_season_data[season] = {
                    'votes': np.zeros(10),
                    'episodes': 0,
                    'total_votes': 0
                }
            
            # Count episodes
            all_season_data[season]['episodes'] += 1
            
            # Process histogram data
            # Remember: hist1_votes = rating 10, hist2_votes = rating 9, etc.
            for i in range(10):
                rating_idx = 9 - i  # Convert to 0-indexed (rating 1 = index 0)
                votes_count = int(row[7 + i*2])  # hist_votes columns
                all_season_data[season]['votes'][rating_idx] += votes_count
                all_season_data[season]['total_votes'] += votes_count
    
    # Calculate statistics for each season
    for season, data in all_season_data.items():
        # Convert to percentages
        votes_pct = data['votes'] / data['total_votes'] * 100
        
        # Extract ratings 2-9 (indices 1-8)
        middle_ratings = np.arange(2, 10)
        middle_votes = votes_pct[1:9]
        
        # Calculate statistics
        percent_1s = votes_pct[0]
        percent_10s = votes_pct[9]
        
        # Calculate weighted mean and std for middle ratings
        weights = data['votes'][1:9]
        if np.sum(weights) > 0:
            mean_2_9 = np.average(middle_ratings, weights=weights)
            # Weighted std calculation
            variance = np.average((middle_ratings - mean_2_9)**2, weights=weights)
            std_2_9 = np.sqrt(variance)
        else:
            mean_2_9 = 0
            std_2_9 = 0
        
        # Store calculated statistics
        season_stats[season] = {
            'percent_1s': percent_1s,
            'percent_10s': percent_10s,
            'mean_2_9': mean_2_9,
            'std_2_9': std_2_9,
            'episodes': data['episodes'],
            'total_votes': data['total_votes']
        }
    
    return season_stats

def perform_pca_and_plot():
    """
    Perform PCA on the four key statistics and create a 2D scatter plot.
    """
    # Get season statistics
    season_stats = extract_season_statistics()
    
    # Remove debug code now that we've identified all seasons are present correctly
    seasons = sorted(season_stats.keys())
    
    # Prepare the data for PCA
    features = ['percent_1s', 'percent_10s', 'mean_2_9', 'std_2_9']
    data = np.array([[season_stats[s][feature] for feature in features] for s in seasons])
    
    # Standardize the data (mean=0, variance=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create a DataFrame for plotting
    pca_df = pd.DataFrame({
        'season': seasons,
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'episodes': [season_stats[s]['episodes'] for s in seasons],
        'total_votes': [season_stats[s]['total_votes'] for s in seasons],
        'percent_1s': [season_stats[s]['percent_1s'] for s in seasons],
        'percent_10s': [season_stats[s]['percent_10s'] for s in seasons],
        'mean_2_9': [season_stats[s]['mean_2_9'] for s in seasons],
        'std_2_9': [season_stats[s]['std_2_9'] for s in seasons]
    })
    
    # Calculate feature loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # Set up the figure with equal aspect ratio for proper distance perception
    plt.figure(figsize=(14, 14))
    
    # Set style with improved aesthetics
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
    })
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'axes.edgecolor': '0.2',
        'axes.linewidth': 1.0
    })
    
    # Create a colormap (sequential but discrete with better separation)
    n_seasons = len(seasons)
    # Use a colormap with better discrimination for adjacent colors
    cmap = plt.cm.viridis_r  # Reversed viridis has better differentiation for this data
    colors = cmap(np.linspace(0, 0.9, n_seasons))  # Using 0-0.9 range for more vibrant colors
    
    # Determine the scale for proper visualization
    # Calculate the range of PC values to ensure proper scaling
    pc1_range = pca_df['PC1'].max() - pca_df['PC1'].min()
    pc2_range = pca_df['PC2'].max() - pca_df['PC2'].min() 
    max_range = max(pc1_range, pc2_range)
    
    # Set axis limits with proper padding for equal scaling
    margin_factor = 0.25  # 25% margin
    margin = max_range * margin_factor
    
    # Calculate the center of the plot
    pc1_center = (pca_df['PC1'].max() + pca_df['PC1'].min()) / 2
    pc2_center = (pca_df['PC2'].max() + pca_df['PC2'].min()) / 2
    
    # Calculate limits with equal range in both dimensions
    xlim_min = pc1_center - (max_range / 2 + margin)
    xlim_max = pc1_center + (max_range / 2 + margin)
    ylim_min = pc2_center - (max_range / 2 + margin)
    ylim_max = pc2_center + (max_range / 2 + margin)
    
    # Create a background shading grid
    # Number of points in each direction for the shading grid
    grid_density = 100
    x_grid = np.linspace(xlim_min, xlim_max, grid_density)
    y_grid = np.linspace(ylim_min, ylim_max, grid_density)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create a function to evaluate "goodness" at each point
    def calculate_goodness(pc1, pc2):
        # Convert PC coordinates back to original features
        # First combine into 2D array
        pc_coords = np.column_stack((pc1.flatten(), pc2.flatten()))
        
        # Transform back to the original feature space
        # Need to un-rotate (using PC components) and un-standardize
        feature_values = np.zeros((pc_coords.shape[0], 4))
        
        for i in range(pc_coords.shape[0]):
            # Approximate reconstruction of standardized features
            std_features = np.zeros(4)
            # Each principal component is a linear combination of standardized features
            # We use the loadings/components to reverse this
            for j in range(2):  # We have 2 principal components
                for k in range(4):  # 4 features
                    # Use normalized loadings for reconstruction
                    std_features[k] += pc_coords[i, j] * pca.components_[j, k]
            
            # Un-standardize (mean=0, std=1 -> original scale)
            for j in range(4):
                feature_values[i, j] = std_features[j] * scaler.scale_[j] + scaler.mean_[j]
        
        # Now calculate goodness score based on feature values
        # Extract feature values
        percent_1s = feature_values[:, 0]
        percent_10s = feature_values[:, 1]
        mean_2_9 = feature_values[:, 2]
        std_2_9 = feature_values[:, 3]
        
        # Indicators of "goodness":
        # - High percent_10s is good (positive)
        # - High mean_2_9 is good (positive)
        # - High percent_1s is bad (negative)
        # - High std_2_9 is bad (variable ratings, negative)
        
        # Scale each component to roughly [0,1] range based on observed data ranges
        percent_1s_scaled = (percent_1s - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
        percent_10s_scaled = (percent_10s - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))
        mean_2_9_scaled = (mean_2_9 - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))
        std_2_9_scaled = (std_2_9 - np.min(data[:, 3])) / (np.max(data[:, 3]) - np.min(data[:, 3]))
        
        # Some values might be outside observed ranges, so clip to [0,1]
        percent_1s_scaled = np.clip(percent_1s_scaled, 0, 1)
        percent_10s_scaled = np.clip(percent_10s_scaled, 0, 1)
        mean_2_9_scaled = np.clip(mean_2_9_scaled, 0, 1)
        std_2_9_scaled = np.clip(std_2_9_scaled, 0, 1)
        
        # Calculate overall goodness score [-1,1]
        # Positive factors: high 10s, high mean
        positive = (percent_10s_scaled + mean_2_9_scaled) / 2
        # Negative factors: high 1s, high std
        negative = (percent_1s_scaled + std_2_9_scaled) / 2
        
        # Combine into a single score
        goodness = positive - negative
        return goodness.reshape(X.shape)
    
    # Calculate goodness values
    goodness = calculate_goodness(X, Y)
    
    # Create a custom colormap for the background
    # We want green for good areas (high score)
    # Red for bad areas (low score)
    # And neutral/transparent for middle
    
    # Define custom colormap: red -> transparent -> green
    cmap_bg = LinearSegmentedColormap.from_list(
        'RedTransparentGreen', 
        [(0.8, 0.0, 0.0, 0.4),    # Red with alpha
         (1.0, 1.0, 1.0, 0.0),    # Transparent white
         (0.0, 0.7, 0.0, 0.4)]    # Green with alpha
    )
    
    # Draw the background shading
    plt.pcolormesh(X, Y, goodness, cmap=cmap_bg, shading='gouraud', zorder=0)
    
    # Add a simple contour line to show the boundary between good and bad regions
    plt.contour(X, Y, goodness, levels=[0], colors=['#555555'], linewidths=1, linestyles='dashed', alpha=0.5, zorder=1)
    
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    
    # Set axis equal to ensure distances are preserved
    plt.gca().set_aspect('equal')
    
    # Add grid lines first (behind everything)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=1)
    
    # Add origin lines
    plt.axhline(y=0, color='#222222', linestyle='-', alpha=0.4, zorder=1, linewidth=1.5)
    plt.axvline(x=0, color='#222222', linestyle='-', alpha=0.4, zorder=1, linewidth=1.5)
    
    # Use a fixed size for all circles
    circle_size = 180  # Constant size for all points
    
    # Increase zorder for scatter and labels to ensure they're on top
    scatter_zorder = 10
    label_zorder = 11
    
    # Plot each season as a scatter point
    for i, season in enumerate(seasons):
        row = pca_df[pca_df['season'] == season]
        x, y = row['PC1'].values[0], row['PC2'].values[0]
        
        # Plot point with more distinct border
        plt.scatter(
            x, y,
            s=circle_size,
            color=colors[i],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.85,
            zorder=scatter_zorder  # Increased zorder to be on top
        )
        
        # Add season label significantly below the point
        # Calculate label offset (a larger offset to ensure it's below the point)
        label_offset_y = -np.sqrt(circle_size) * 0.8  # Increased from 0.2 to 0.8
        
        plt.annotate(
            f"S{season}",
            xy=(x, y),
            xytext=(0, label_offset_y),  # Offset text more below the point
            textcoords='offset points',
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='top',  # Changed from 'center' to 'top' to align top of text at offset point
            color='black',
            # Add a white outline for better readability on any background
            path_effects=[
                path_effects.withStroke(linewidth=3, foreground='white')
            ],
            zorder=label_zorder  # Increased zorder to be on top of everything
        )
    
    # Add feature vectors (loadings)
    origin = np.zeros(2)
    max_loading = max(np.max(np.abs(loadings[:, 0])), np.max(np.abs(loadings[:, 1])))
    scale_factor = max_range * 0.4 / max_loading  # Scale relative to the data range
    
    for i, feature in enumerate(features):
        # Calculate arrow endpoints
        arrow_x = loadings[i, 0] * scale_factor
        arrow_y = loadings[i, 1] * scale_factor
        
        plt.arrow(
            origin[0], origin[1],
            arrow_x, arrow_y,
            head_width=max_range * 0.03,
            head_length=max_range * 0.04,
            fc='#555555', 
            ec='#555555', 
            alpha=0.75,
            zorder=2,
            linewidth=2
        )
        
        # Add feature labels with better positioning and styling
        # Calculate label position with more space from arrow tip
        label_x = arrow_x * 1.2
        label_y = arrow_y * 1.2
        feature_names = {
            'percent_1s': '% of Rating 1',
            'percent_10s': '% of Rating 10',
            'mean_2_9': 'Mean of Ratings 2-9',
            'std_2_9': 'Std Dev of Ratings 2-9'
        }
        
        # Position labels more intelligently
        ha = 'center'
        if abs(label_x) > 0.2 * max_range:
            ha = 'right' if label_x < 0 else 'left'
            
        va = 'center'
        if abs(label_y) > 0.2 * max_range:
            va = 'bottom' if label_y > 0 else 'top'
            
        # Add feature label with better styling
        plt.text(
            label_x, label_y,
            feature_names[feature],
            fontsize=12,
            ha=ha,
            va=va,
            fontweight='bold',
            color='#333333',
            bbox=dict(
                facecolor='white', 
                alpha=0.9, 
                edgecolor='#888888',
                boxstyle='round,pad=0.4',
                linewidth=1
            ),
            zorder=5
        )
    
    # Add explained variance
    explained_variance = pca.explained_variance_ratio_
    explained_variance_labels = [f'PC{i+1} ({var:.1%})' for i, var in enumerate(explained_variance)]
    
    # Set axis labels with better styling
    plt.xlabel(explained_variance_labels[0], fontsize=15, fontweight='bold', labelpad=15)
    plt.ylabel(explained_variance_labels[1], fontsize=15, fontweight='bold', labelpad=15)
    
    # Create season legend with better styling
    legend_elements = []
    
    # Group the seasons in columns for the legend
    season_groups = []
    group_size = 6  # Adjust for better legend layout
    for i in range(0, len(seasons), group_size):
        season_groups.append(seasons[i:i+group_size])
    
    # Add season entries to legend
    for group in season_groups:
        for season in group:
            i = seasons.index(season)
            legend_elements.append(
                Line2D([0], [0], 
                       marker='o', 
                       color='w', 
                       markerfacecolor=colors[i],
                       markeredgecolor='black',
                       markersize=10, 
                       label=f'Season {season}')
            )
    
    # Add a legend for the background shading
    legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
    legend_elements.append(
        Patch(facecolor=(0.0, 0.7, 0.0, 0.4), 
              edgecolor='#888888', 
              label='Good ratings (high 10s, high mean)')
    )
    legend_elements.append(
        Patch(facecolor=(0.8, 0.0, 0.0, 0.4), 
              edgecolor='#888888', 
              label='Poor ratings (high 1s, high std dev)')
    )
    
    # Add feature vector legend
    legend_elements.append(Line2D([0], [0], color='none', label=''))  # Spacer
    legend_elements.append(
        Line2D([0], [0], 
               color='#555555', 
               lw=2, 
               marker='>',
               markersize=8, 
               label='Feature vector')
    )
    
    # Add legend with improved styling
    plt.legend(
        handles=legend_elements,
        loc='best',
        title="Legend",
        frameon=True,
        framealpha=0.95,
        edgecolor='#888888',
        fontsize=11,
        title_fontsize=13,
        ncol=len(season_groups) + 1  # +1 for the feature vector and shading column
    )
    
    # Add plot title with better styling
    plt.suptitle(
        'PCA Analysis of Taskmaster Seasons',
        fontsize=18, 
        fontweight='bold',
        y=0.98
    )
    plt.title(
        'Based on Rating Distributions (1s, 10s, mean & std of 2-9)',
        fontsize=14,
        pad=20
    )
    
    # Add explanatory annotations with better styling
    # Find extreme seasons in different directions
    max_pc1_idx = np.argmax(pca_result[:, 0])
    min_pc1_idx = np.argmin(pca_result[:, 0])
    max_pc2_idx = np.argmax(pca_result[:, 1])
    min_pc2_idx = np.argmin(pca_result[:, 1])
    
    # Calculate annotation offset based on the data range
    offset_x = max_range * 0.15
    offset_y = max_range * 0.15
    
    # Track which seasons have been annotated to avoid duplicate annotations
    annotated_seasons = set()
    
    # Add annotations for interesting patterns with better styling
    for idx, direction, text_offset in [(max_pc1_idx, "→", (offset_x, 0)), 
                                      (min_pc1_idx, "←", (-offset_x, 0)),
                                      (max_pc2_idx, "↑", (0, offset_y)), 
                                      (min_pc2_idx, "↓", (0, -offset_y))]:
        season = seasons[idx]
        
        # Skip if this season has already been annotated
        if season in annotated_seasons:
            continue
        
        # Mark this season as annotated
        annotated_seasons.add(season)
        
        row = pca_df[pca_df['season'] == season]
        x, y = row['PC1'].values[0], row['PC2'].values[0]
        
        # Create custom annotation with better formatting
        stats = [
            f"% Rating 1: {row['percent_1s'].values[0]:.1f}%",
            f"% Rating 10: {row['percent_10s'].values[0]:.1f}%",
            f"Mean(2-9): {row['mean_2_9'].values[0]:.2f}",
            f"Std(2-9): {row['std_2_9'].values[0]:.2f}"
        ]
        
        annotation_text = f"Season {season}\n" + "\n".join(stats)
        
        plt.annotate(
            annotation_text,
            xy=(x, y),
            xytext=(x + text_offset[0], y + text_offset[1]),
            arrowprops=dict(
                arrowstyle="->", 
                color='#333333', 
                alpha=0.8, 
                connectionstyle="arc3,rad=0.2",
                linewidth=1.5
            ),
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.5", 
                fc="white", 
                ec="#888888", 
                alpha=0.9,
                linewidth=1
            ),
            zorder=5  # Lower than scatter and season labels but above background
        )
    
    # Save the figure with high quality
    plt.tight_layout()
    plt.savefig('taskmaster_pca_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    # Print PCA results
    print("PCA completed successfully!")
    print(f"Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    print("Feature loadings:")
    for i, feature in enumerate(features):
        print(f"  {feature}: PC1={loadings[i, 0]:.3f}, PC2={loadings[i, 1]:.3f}")

if __name__ == "__main__":
    perform_pca_and_plot() 