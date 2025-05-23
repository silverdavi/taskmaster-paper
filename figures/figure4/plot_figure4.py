import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw
import seaborn as sns
from scipy.stats import gaussian_kde
import os
from process_figure4_data import get_uk_density_data, get_international_contestants, get_contestant_counts_by_country, get_project_root

def geo_to_pixel(lat, lon, matrix):
    """Convert latitude and longitude to pixel coordinates"""
    x = matrix[0, 0] * lon + matrix[0, 1] * lat + matrix[0, 2]
    y = matrix[1, 0] * lon + matrix[1, 1] * lat + matrix[1, 2]
    return int(x), int(y)

def is_within_map(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Check if coordinates are within the map bounds"""
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def create_figure4():
    # Get project root path
    project_root = get_project_root()
    
    # Load the background map image
    img_path = os.path.join(project_root, "figures", "figure4", "British_Isles_map_showing_UK,_Republic_of_Ireland,_and_historic_counties.svg.png")
    img = Image.open(img_path).convert("RGBA")
    img_width, img_height = img.size
    
    # Define geographic bounds of the image manually (as provided)
    lat_min, lat_max = 49.5, 61.0
    lon_min, lon_max = -11.0, 2.0
    
    # Create affine matrix for lat/lon to pixel transform
    affine_matrix = np.array([
        [img_width / (lon_max - lon_min), 0, -lon_min * img_width / (lon_max - lon_min)],
        [0, -img_height / (lat_max - lat_min), lat_max * img_height / (lat_max - lat_min)]
    ])
    
    # Get data for UK/Ireland contestants
    uk_data = get_uk_density_data()
    
    # Get data for international contestants
    intl_data = get_international_contestants()
    
    # Get country counts for the legend
    country_counts = get_contestant_counts_by_country()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 15))
    
    # Display the map image
    ax.imshow(img)
    ax.axis('off')
    
    # Extract coordinates for UK/Ireland contestants
    uk_lats = uk_data['Latitude'].values
    uk_lons = uk_data['Longitude'].values
    
    # Convert UK coordinates to pixel values
    uk_pixels = []
    for lat, lon in zip(uk_lats, uk_lons):
        if is_within_map(lat, lon, lat_min, lat_max, lon_min, lon_max):
            x, y = geo_to_pixel(lat, lon, affine_matrix)
            uk_pixels.append((x, y))
    
    uk_x, uk_y = zip(*uk_pixels) if uk_pixels else ([], [])
    
    # Create scatterplot of UK/Ireland contestants
    scatter = ax.scatter(
        uk_x, uk_y, 
        alpha=0.7, 
        c='steelblue', 
        edgecolor='white',
        s=100,
        zorder=3
    )
    
    # Calculate kernel density estimate for heat map
    if len(uk_x) > 1:  # Need at least 2 points for KDE
        # Create a meshgrid covering the map
        x_grid = np.linspace(0, img_width, 100)
        y_grid = np.linspace(0, img_height, 125)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Stack coordinates for density calculation
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([uk_x, uk_y])
        
        # Compute kernel density estimate
        kernel = gaussian_kde(values, bw_method=0.15)  # Adjust bandwidth for smoother/sharper density
        density = kernel(positions).reshape(xx.shape)
        
        # Plot density contour
        contour = ax.contourf(
            xx, yy, density, 
            levels=15, 
            cmap='Blues', 
            alpha=0.4,
            zorder=2
        )
    
    # Create a box for international contestants in top-left corner
    intl_box_width = img_width * 0.3
    intl_box_height = img_height * 0.2
    intl_box_x = img_width * 0.05
    intl_box_y = img_height * 0.05
    
    # Draw a background box for international contestants
    rect = plt.Rectangle(
        (intl_box_x, intl_box_y), 
        intl_box_width, 
        intl_box_height, 
        facecolor='lightgray', 
        alpha=0.8,
        edgecolor='black', 
        zorder=5
    )
    ax.add_patch(rect)
    
    # Add title for international box
    ax.text(
        intl_box_x + intl_box_width/2, 
        intl_box_y + 20, 
        'International Contestants',
        ha='center',
        fontsize=14,
        fontweight='bold',
        zorder=6
    )
    
    # Group international contestants by country
    intl_grouped = intl_data.groupby('Country')
    
    # Place circles for each country with international contestants
    y_offset = 50
    for i, (country, group) in enumerate(intl_grouped):
        # Calculate position in the international box
        x_pos = intl_box_x + intl_box_width * 0.2
        y_pos = intl_box_y + y_offset + i * 30
        
        # Draw circle
        circle = plt.Circle(
            (x_pos, y_pos), 
            10, 
            color='darkgoldenrod',
            alpha=0.8,
            zorder=6
        )
        ax.add_patch(circle)
        
        # Add country name and count
        ax.text(
            x_pos + 20, 
            y_pos, 
            f"{country} ({len(group)})",
            va='center',
            fontsize=12,
            zorder=6
        )
    
    # Add title and footer
    plt.suptitle(
        'Figure 4: Geographic Origins of Taskmaster Contestants',
        fontsize=18,
        fontweight='bold',
        y=0.95
    )
    
    # Add footnote about missing contestants
    footnote = "Note: Map shows birthplaces of contestants. Emma Sidi (USA) has no specific coordinates available."
    plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=10)
    
    # Save the figure
    plt.savefig(os.path.join(project_root, "figures", "figure4", "figure4.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(project_root, "figures", "figure4", "figure4.pdf"), bbox_inches='tight')
    
    plt.close()
    
    print("Figure 4 created successfully.")

if __name__ == "__main__":
    create_figure4() 