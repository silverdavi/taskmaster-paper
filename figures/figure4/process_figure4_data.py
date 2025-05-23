import pandas as pd
import numpy as np
import os

# Get the absolute path to the project root
def get_project_root():
    """Get the absolute path to the project root directory"""
    # Assuming we're in figures/figure4
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root

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
    
    # Group by country and count
    country_counts = intl_df.groupby('Country').size().reset_index(name='count')
    
    # Add example coordinates for labels (these won't be used for actual plotting)
    return intl_df[['Contestant Name', 'Latitude', 'Longitude', 'Country', 'contestant_id', 'series', 'placement']]

if __name__ == "__main__":
    # Test the functions
    merged_data = load_contestant_data()
    print(f"Total contestants with valid coordinates: {len(merged_data)}")
    
    country_counts = get_contestant_counts_by_country()
    print("\nContestants by country:")
    print(country_counts)
    
    uk_density = get_uk_density_data()
    print(f"\nUK/Ireland contestants: {len(uk_density)}")
    
    intl_contestants = get_international_contestants()
    print(f"\nInternational contestants: {len(intl_contestants)}")
    print(intl_contestants[['Contestant Name', 'Country']]) 