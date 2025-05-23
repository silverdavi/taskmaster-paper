import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import os
import sys
from pathlib import Path

# Import the plotting configuration utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.plot_utils import load_config, apply_plot_style

def create_heat_kernel(size, intensity=1.0, sigma=0.2):
    """Create a heat kernel for the heatmap
    
    Args:
        size: Size of the kernel (should be odd)
        intensity: Overall intensity multiplier
        sigma: Controls the spread of the kernel (higher = wider heat spot)
    """
    # Create a radial gradient
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate radial distance from center
    r = np.sqrt(xx**2 + yy**2)
    
    # Create gaussian kernel
    kernel = np.exp(-(r**2) / (2 * sigma**2))
    
    # Normalize and apply intensity
    kernel = kernel / kernel.max() * intensity
    
    return kernel

def get_project_root():
    """Get the absolute path to the project root directory"""
    # Assuming we're in figures/figure4
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def create_figure4():
    """Create Figure 4 using pre-processed data from CSV files"""
    # Get project root path
    project_root = get_project_root()
    figure_dir = os.path.join(project_root, "figures", "figure4")
    
    # Load configuration
    config = load_config()
    
    # Load the pre-processed data
    try:
        # Load the transform info
        transform_info = pd.read_csv(os.path.join(figure_dir, "transform_info.csv"))
        if len(transform_info) == 0:
            raise ValueError("Transform info is empty")
            
        # Load the grid cell data
        cell_data = pd.read_csv(os.path.join(figure_dir, "grid_cell_data.csv"))
        if len(cell_data) == 0:
            raise ValueError("Cell data is empty")
            
        # Load the country counts
        country_counts = pd.read_csv(os.path.join(figure_dir, "country_counts.csv"))
        if len(country_counts) == 0:
            raise ValueError("Country counts are empty")
            
    except Exception as e:
        print(f"Error loading pre-processed data: {e}")
        print("Please run process_figure4_data.py first to generate the required CSV files")
        sys.exit(1)
    
    # Load the background map image
    img_path = os.path.join(figure_dir, "British_Isles_map_showing_UK,_Republic_of_Ireland,_and_historic_counties.svg.png")
    img = Image.open(img_path).convert("RGBA")
    img_width, img_height = img.size
    
    # Create empty heat map layer (transparent)
    heatmap = np.zeros((img_height, img_width, 4), dtype=np.uint8)
    
    # Define base kernel size
    base_kernel_size = 61  # Base size of the heat spot (odd number)
    
    # Define heat colors (yellow to red gradient)
    color_map = plt.cm.YlOrRd(np.linspace(0, 1, 256))
    color_map = (color_map[:, :3] * 255).astype(np.uint8)
    
    # Process each grid cell
    for _, cell in cell_data.iterrows():
        # Get cell center coordinates
        cell_center_x = int(cell['center_x'])
        cell_center_y = int(cell['center_y'])
        count = int(cell['count'])
        
        # Scale kernel intensity based on count (logarithmic scaling)
        intensity_factor = 0.7 + 0.3 * np.log1p(count) / np.log1p(3)
        intensity_factor = min(intensity_factor, 1.5)  # Cap the maximum intensity
        
        # Scale kernel size based on count (logarithmic scaling)
        # Larger counts get larger heat spots
        size_factor = 1.0 + 0.4 * np.log1p(count - 1) / np.log1p(5)  # No increase for count=1
        kernel_size = int(base_kernel_size * size_factor)
        if kernel_size % 2 == 0:  # Ensure odd size
            kernel_size += 1
            
        # Increase sigma (spread) for higher counts
        sigma_factor = 0.2 + 0.1 * np.log1p(count - 1) / np.log1p(5)  # No increase for count=1
        
        # Create kernel with appropriate size and intensity
        kernel = create_heat_kernel(kernel_size, intensity=intensity_factor, sigma=sigma_factor)
        
        # Skip if too close to edge
        half_size = kernel_size // 2
        if (cell_center_x < half_size or cell_center_x >= img_width - half_size or 
            cell_center_y < half_size or cell_center_y >= img_height - half_size):
            continue
            
        # Place heat kernel at this location
        x_min, x_max = cell_center_x - half_size, cell_center_x + half_size + 1
        y_min, y_max = cell_center_y - half_size, cell_center_y + half_size + 1
        
        # Apply kernel to alpha channel
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                if 0 <= i < img_height and 0 <= j < img_width:
                    kernel_val = kernel[i - y_min, j - x_min]
                    
                    # Apply color based on kernel value
                    color_idx = int(min(kernel_val, 1.0) * 255)
                    if color_idx > 0:
                        # Blend colors if already set
                        alpha = int(min(kernel_val * 200, 255))  # Cap at 255 (max for uint8)
                        
                        # Only override if new value is more intense
                        if alpha > heatmap[i, j, 3]:
                            heatmap[i, j, 0] = color_map[color_idx, 0]  # R
                            heatmap[i, j, 1] = color_map[color_idx, 1]  # G
                            heatmap[i, j, 2] = color_map[color_idx, 2]  # B
                            heatmap[i, j, 3] = alpha  # A
    
    # Convert numpy array to PIL Image
    heatmap_img = Image.fromarray(heatmap)
    
    # Apply slight blur for smoother appearance
    heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=1))
    
    # Composite heatmap over background
    result_img = Image.alpha_composite(img, heatmap_img)
    
    # Create figure and axis for matplotlib (for legend and annotations)
    # Use a fixed figure size for this specific visualization
    fig, ax = plt.subplots(figsize=(12, 15))
    
    # Apply the plot style but don't use it for this specific figure
    # We'll just load the config for DPI settings
    
    # Display the composited image
    ax.imshow(result_img)
    ax.axis('off')
    
    # Sort countries by count (descending)
    sorted_countries = country_counts.sort_values('Count', ascending=False)
    
    # Calculate required height for the legend box
    # Each country entry takes about 30 pixels of height, plus margins
    num_countries = len(sorted_countries)
    
    # Create a box for country listing in top-left corner
    countries_box_width = img_width * 0.25
    countries_box_height = 60 + (num_countries * 30)  # Height based on number of countries
    countries_box_x = img_width * 0.05
    countries_box_y = img_height * 0.05
    
    # Draw a background box for countries
    rect = plt.Rectangle(
        (countries_box_x, countries_box_y), 
        countries_box_width, 
        countries_box_height, 
        facecolor='white', 
        alpha=0.8,
        edgecolor='black', 
        zorder=5
    )
    ax.add_patch(rect)
    
    # Add title for countries box with fixed size
    ax.text(
        countries_box_x + countries_box_width/2, 
        countries_box_y + 33, 
        'Countries',
        ha='center',
        fontsize=14,  # Fixed size for this visualization
        fontweight='bold',
        zorder=6
    )
    
    # Place circles for each country
    y_offset = 50
    for i, (_, row) in enumerate(sorted_countries.iterrows()):
        country = row['Country']
        count = row['Count']
        
        # Only include countries with data
        if pd.isna(country) or pd.isna(count):
            continue
            
        # Calculate position in the countries box
        x_pos = countries_box_x + 20
        y_pos = countries_box_y + y_offset + i * 30
        
        # Use different colors for UK/Ireland vs international
        color = 'darkred' if country in ['England', 'Scotland', 'Wales', 'Ireland', 'Northern Ireland'] else 'darkorange'
        
        # Draw circle
        circle = plt.Circle(
            (x_pos, y_pos), 
            8, 
            color=color,
            alpha=0.8,
            zorder=6
        )
        ax.add_patch(circle)
        
        # Add country name and count with fixed font size
        ax.text(
            x_pos + 15, 
            y_pos, 
            f"{country} ({int(count)})",
            va='center',
            fontsize=11,  # Fixed size for this visualization
            zorder=6
        )
    
    # No title - let the figure speak for itself
    
    # Save the figure with configured DPI
    dpi = config['global'].get('dpi', 300)
    plt.savefig(os.path.join(figure_dir, "figure4.png"), dpi=dpi, bbox_inches='tight')
    plt.savefig(os.path.join(figure_dir, "figure4.pdf"), bbox_inches='tight')
    
    plt.close()
    
    print("Figure 4 created successfully.")

if __name__ == "__main__":
    create_figure4() 