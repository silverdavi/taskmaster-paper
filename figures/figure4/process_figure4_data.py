import pandas as pd
import numpy as np
import os
from collections import defaultdict

# Get the absolute path to the project root
def get_project_root():
    """Get the absolute path to the project root directory"""
    # Assuming we're in figures/figure4
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

def geo_to_pixel(lat, lon, matrix):
    """Convert latitude and longitude to pixel coordinates"""
    x = matrix[0, 0] * lon + matrix[0, 1] * lat + matrix[0, 2]
    y = matrix[1, 0] * lon + matrix[1, 1] * lat + matrix[1, 2]
    return int(x), int(y)

def is_within_map(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Check if coordinates are within the map bounds"""
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def load_contestant_data():
    """Load contestant location data from TSV file and contestant info from CSV file"""
    # Get project root path
    project_root = get_project_root()
    
    # Load location data
    locations_df = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'Cont_lon_lat.tsv'), sep='\t')
    
    # Load contestant info
    contestants_df = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'contestants.csv'))
    
    # Merge the dataframes on contestant name
    merged_df = pd.merge(
        locations_df, 
        contestants_df[['name', 'contestant_id', 'series', 'placement']], 
        left_on='Contestant Name', 
        right_on='name',
        how='left'
    )
    
    # Clean up the data
    merged_df = merged_df.dropna(subset=['Longitude', 'Latitude'])
    
    # Convert latitude and longitude to numeric values
    merged_df['Longitude'] = pd.to_numeric(merged_df['Longitude'], errors='coerce')
    merged_df['Latitude'] = pd.to_numeric(merged_df['Latitude'], errors='coerce')
    
    # Drop rows with invalid coordinates
    merged_df = merged_df.dropna(subset=['Longitude', 'Latitude'])
    
    # Create UK/Ireland filter
    uk_ireland = merged_df['Country'].isin(['England', 'Scotland', 'Wales', 'Ireland', 'Northern Ireland'])
    
    # Create 'location_type' column to distinguish UK/Ireland from international contestants
    merged_df['location_type'] = np.where(uk_ireland, 'UK_Ireland', 'International')
    
    return merged_df

def get_contestant_counts_by_country():
    """Get counts of contestants by country"""
    project_root = get_project_root()
    locations_df = pd.read_csv(os.path.join(project_root, 'data', 'raw', 'Cont_lon_lat.tsv'), sep='\t')
    return locations_df['Country'].value_counts()

def get_uk_density_data():
    """Prepare data for density visualization"""
    merged_df = load_contestant_data()
    uk_ireland_df = merged_df[merged_df['location_type'] == 'UK_Ireland']
    
    # Return dataframe with necessary columns for density visualization
    return uk_ireland_df[['Contestant Name', 'Latitude', 'Longitude', 'Country', 'contestant_id', 'series', 'placement']]

def get_international_contestants():
    """Get data for international contestants"""
    merged_df = load_contestant_data()
    intl_df = merged_df[merged_df['location_type'] == 'International']
    
    # Add example coordinates for labels (these won't be used for actual plotting)
    return intl_df[['Contestant Name', 'Latitude', 'Longitude', 'Country', 'contestant_id', 'series', 'placement']]

def process_and_transform_data():
    """Process contestant data and transform coordinates for plotting"""
    # Get project root path
    project_root = get_project_root()
    
    # Load the map image dimensions (assuming 16:9 aspect ratio if image not available)
    try:
        from PIL import Image
        img_path = os.path.join(project_root, "figures", "figure4", "British_Isles_map_showing_UK,_Republic_of_Ireland,_and_historic_counties.svg.png")
        img = Image.open(img_path)
        img_width, img_height = img.size
    except:
        # Default dimensions if image not available
        img_width, img_height = 1600, 1800
    
    # Define geographic bounds of the image manually (as provided)
    lat_min, lat_max = 49.5, 61.0
    lon_min, lon_max = -11.0, 2.0
    
    # Create affine matrix for lat/lon to pixel transform
    affine_matrix = np.array([
        [img_width / (lon_max - lon_min), 0, -lon_min * img_width / (lon_max - lon_min)],
        [0, -img_height / (lat_max - lat_min), lat_max * img_height / (lat_max - lat_min)]
    ])
    
    # Get UK/Ireland contestant data
    uk_data = get_uk_density_data()
    
    # Create a results list to store transformed data
    transformed_data = []
    
    # Process each UK/Ireland contestant
    for _, row in uk_data.iterrows():
        lat, lon = row['Latitude'], row['Longitude']
        
        # Check if within map bounds
        if is_within_map(lat, lon, lat_min, lat_max, lon_min, lon_max):
            # Convert to pixel coordinates
            x, y = geo_to_pixel(lat, lon, affine_matrix)

            affine_matrix_corrected = np.array([
                [1.0196, 0.00493, -40],     # t_x changed from -33.7152 to -34.2
                [-0.03288, 0.99173, 63.0],    # t_y changed from 56.5683 to 57.0
                [0.0, 0.0, 1.0]
            ])
            # Convert to homogeneous coordinate and apply correction
            corrected = affine_matrix_corrected @ np.array([x, y, 1])
            x_corr, y_corr = corrected[0], corrected[1]

            # Add to transformed data
            transformed_data.append({
                'Contestant Name': row['Contestant Name'],
                'Country': row['Country'],
                'contestant_id': row['contestant_id'],
                'series': row['series'],
                'placement': row['placement'],
                'latitude': lat,
                'longitude': lon,
                'pixel_x': x_corr,
                'pixel_y': y_corr
            })
    
    # Convert to DataFrame
    transformed_df = pd.DataFrame(transformed_data)
    
    # Count contestants in grid cells (10x10 pixel cells)
    cell_counts = defaultdict(int)
    for _, row in transformed_df.iterrows():
        grid_cell = (row['pixel_x'] // 10, row['pixel_y'] // 10)
        cell_counts[grid_cell] += 1
    
    # Create a cell data list
    cell_data = []
    processed_cells = set()
    
    # Group by grid cells
    for _, row in transformed_df.iterrows():
        grid_cell = (row['pixel_x'] // 10, row['pixel_y'] // 10)
        
        # Skip if we've already processed this cell
        if grid_cell in processed_cells:
            continue
            
        # Mark as processed
        processed_cells.add(grid_cell)
        
        # Calculate cell center
        cell_center_x = grid_cell[0] * 10 + 5
        cell_center_y = grid_cell[1] * 10 + 5
        
        # Add to cell data
        contestants_in_cell = transformed_df[
            (transformed_df['pixel_x'] // 10 == grid_cell[0]) & 
            (transformed_df['pixel_y'] // 10 == grid_cell[1])
        ]
        
        cell_data.append({
            'grid_cell_x': grid_cell[0],
            'grid_cell_y': grid_cell[1],
            'center_x': cell_center_x,
            'center_y': cell_center_y,
            'count': cell_counts[grid_cell],
            'country': contestants_in_cell.iloc[0]['Country'],  # Using first contestant's country
            'contestant_ids': ','.join(contestants_in_cell['contestant_id'].astype(str).tolist()),
            'contestant_names': ','.join(contestants_in_cell['Contestant Name'].tolist())
        })
    
    # Convert to DataFrame
    cell_df = pd.DataFrame(cell_data)
    
    # Get country counts
    country_counts = uk_data['Country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']
    
    # Get international contestant counts
    intl_data = get_international_contestants()
    intl_counts = intl_data['Country'].value_counts().reset_index()
    intl_counts.columns = ['Country', 'Count']
    
    # Combine counts
    all_country_counts = pd.concat([country_counts, intl_counts])
    
    # Save all processed data to CSV files
    output_dir = os.path.join(project_root, "figures", "figure4")
    
    # Save contestant pixel locations
    transformed_df.to_csv(os.path.join(output_dir, "contestant_pixel_locations.csv"), index=False)
    
    # Save cell data
    cell_df.to_csv(os.path.join(output_dir, "grid_cell_data.csv"), index=False)
    
    # Save country counts
    all_country_counts.to_csv(os.path.join(output_dir, "country_counts.csv"), index=False)
    
    # Save image dimensions and transform info
    transform_info = pd.DataFrame([{
        'img_width': img_width,
        'img_height': img_height,
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max
    }])
    transform_info.to_csv(os.path.join(output_dir, "transform_info.csv"), index=False)
    
    print(f"Processed {len(transformed_df)} contestants into {len(cell_df)} grid cells")
    print(f"Data saved to CSV files in {output_dir}")
    
    return {
        'contestants': transformed_df,
        'cells': cell_df,
        'country_counts': all_country_counts,
        'transform_info': transform_info
    }

if __name__ == "__main__":
    # Process and transform data, saving to CSV
    process_and_transform_data() 