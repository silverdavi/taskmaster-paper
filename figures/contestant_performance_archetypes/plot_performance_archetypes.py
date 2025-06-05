import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle, Rectangle, FancyArrowPatch
import os
import yaml
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

def load_config():
    """Load plot configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              'config', 'plot_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_2d_coordinates():
    """
    Generate idealized 2D coordinates for the five archetypes.
    These positions create a layout with four archetypes in the corners and one in the middle.
    """
    # Define ideal positions for the archetypes
    archetype_positions = {
        'Steady Performer': (0.25, 0.75),  # Top left
        'Late Bloomer': (0.75, 0.75),      # Top right
        'Early Star': (0.25, 0.25),        # Bottom left
        'Chaotic Wildcard': (0.75, 0.25),  # Bottom right
        'Consistent Middle': (0.5, 0.5)    # Center
    }
    
    return archetype_positions

def load_archetype_data(file_path='final_archetypes.csv'):
    """
    Load the archetype assignments from the CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def load_features_data(file_path='contestant_features.csv'):
    """
    Load the contestant features data to identify winners.
    """
    df = pd.read_csv(file_path)
    # DEBUG: Print Series 1 data
    print("Series 1 contestants (from features CSV):")
    print(df[df['Series']==1][['ContestantName', 'ContestantID', 'last_rank']].to_string(index=False))
    return df

def generate_6x3_grid_plot(archetypes_df, features_df, config, output_file='fig7.pdf'):
    """
    Generate a 3×6 grid plot (18 series) with a 2D representation of the five archetypes.
    Each subplot shows the five contestants from one series positioned according to their archetype.
    The winner of each series (last_rank=1) is highlighted with a gold and black circle.
    """
    # Get the idealized 2D coordinates for each archetype
    archetype_positions = generate_2d_coordinates()
    
    # Create a color mapping for archetypes with highly distinct colors
    colors = {
        'Steady Performer': '#3366CC',    # Strong blue
        'Late Bloomer': '#33CC66',        # Vibrant green
        'Early Star': '#FF9933',          # Bright orange
        'Chaotic Wildcard': '#CC3366',    # Magenta/pink
        'Consistent Middle': '#9966CC'    # Purple
    }
    
    # Find winners for each series (contestants with last_rank=1)
    winners = {}
    for _, row in features_df.iterrows():
        if row['last_rank'] == 1:
            winners[row['Series']] = row['ContestantID']
    
    # DEBUG: Print winners dictionary
    print("Winners by series:")
    for series, contestant_id in winners.items():
        winner_name = features_df[(features_df['Series']==series) & (features_df['ContestantID']==contestant_id)]['ContestantName'].values[0]
        print(f"Series {series}: Contestant ID {contestant_id} ({winner_name})")
    
    # Apply font family from config
    plt.rcParams['font.family'] = config['global']['font_family']
    
    # Set up the figure - 3 rows, 6 columns with more vertical space
    fig, axes = plt.subplots(3, 6, figsize=(18, 11))
    axes = axes.flatten()  # Flatten for easier indexing
    
    # For each series, create a subplot
    for series in range(1, 19):
        # Get the contestants for this series
        series_data = archetypes_df[archetypes_df['Series'] == series]
        
        # Get the axis for this subplot
        ax = axes[series - 1]
        
        # Plot each contestant in their archetype position
        for _, contestant in series_data.iterrows():
            archetype = contestant['Archetype']
            name = contestant['ContestantName']
            contestant_id = contestant['ContestantID']
            
            # Get the position for this archetype
            x, y = archetype_positions[archetype]
            
            # Add a small random jitter to avoid overlapping
            np.random.seed(int(contestant_id))  # Use contestant ID as seed for consistent jitter
            x_jitter = np.random.uniform(-0.05, 0.05)
            y_jitter = np.random.uniform(-0.05, 0.05)
            x += x_jitter
            y += y_jitter
            
            # Plot the contestant
            ax.plot(x, y, 'o', color=colors[archetype], markersize=16)
            
            # Add contestant name with white background
            last_name = name.split()[-1]
            text = ax.annotate(last_name, (x, y), 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        ha='center', 
                        fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor='white', 
                                  alpha=0.8, 
                                  edgecolor='none'))
            
            # Highlight the winner with a gold and black circle
            if series in winners and contestant_id == winners[series]:
                # Add a gold circle around the winner (larger to match bigger markers)
                circle_gold = Circle((x, y), radius=0.10, fill=False, edgecolor='#FFD700', linewidth=2.5)
                ax.add_patch(circle_gold)
                # Add a black circle for contrast
                circle_black = Circle((x, y), radius=0.105, fill=False, edgecolor='black', linewidth=1)
                ax.add_patch(circle_black)
                
                # DEBUG: Print when adding winner highlight
                print(f"Adding winner highlight for Series {series}, Contestant {name} (ID: {contestant_id})")
        
        # Set up the plot area
        ax.grid(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Remove tick marks completely
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a thin border to the subplot and make it visible
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color('gray')
        
        # Create a connected title for each subplot
        # Draw a bar at the top of the subplot for the title to sit on
        title_bar = Rectangle((0, 1), 1, 0.05, transform=ax.transAxes, 
                              facecolor='lightgray', alpha=0.3, zorder=0,
                              clip_on=False)
        ax.add_patch(title_bar)
        
        # Add the title on top of the bar
        ax.set_title(f"Series {series}", fontsize=12, pad=2)
        
        # Only show axis labels for the first subplot (inside the box)
        if series == 1:
            # Add x-axis directional label with arrow
            ax.annotate("→ Chaotic Performance", 
                        xy=(0.95, 0.05), 
                        xytext=(0.1, 0.05),
                        textcoords='axes fraction',
                        xycoords='axes fraction',
                        ha='left', va='center',
                        fontsize=12,
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3",
                                        color='black',
                                        alpha=0.7))
            
            # Add y-axis directional label with arrow
            ax.annotate("→ Higher Scores", 
                        xy=(0.05, 0.95), 
                        xytext=(0.05, 0.15),
                        textcoords='axes fraction',
                        xycoords='axes fraction',
                        ha='center', va='bottom',
                        fontsize=12,
                        rotation=90,
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3",
                                        color='black',
                                        alpha=0.7))
        
        # Add subtle archetype labels to the first subplot only
        if series == 1:
            for archetype, (x, y) in archetype_positions.items():
                ax.annotate(archetype, (x, y), 
                            xytext=(0, -15), 
                            textcoords='offset points',
                            ha='center', va='center', 
                            fontsize=9, color='gray',
                            alpha=0.7,
                            bbox=dict(boxstyle="round,pad=0.2", 
                                     facecolor='white', 
                                     alpha=0.7, 
                                     edgecolor='none'))
    
    # Create a legend for the archetypes
    legend_elements = [Patch(facecolor=colors[a], label=a) for a in colors]
    # Add winner to legend
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
                                     markeredgecolor='#FFD700', markeredgewidth=2.5,
                                     markersize=16, label='Series Winner'))
    
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=12)
    
    # Adjust layout with more vertical space between rows
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, hspace=0.4)  # Increased hspace for more vertical spacing
    
    # Save the figure
    plt.savefig(output_file, dpi=config['global']['dpi'], bbox_inches='tight')
    print(f"Figure saved to {output_file}")
    
    # Also save a PNG version
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=config['global']['dpi'], bbox_inches='tight')
    print(f"Figure saved to {output_file.replace('.pdf', '.png')}")
    
    return fig

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Load the archetype data
    archetypes_df = load_archetype_data()
    
    # Load the features data to identify winners
    features_df = load_features_data()
    
    # Generate the 6x3 grid plot
    fig = generate_6x3_grid_plot(archetypes_df, features_df, config)
    
    # Show the plot
    plt.show() 