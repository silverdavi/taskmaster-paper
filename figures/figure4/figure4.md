# Figure 4: Geographic Origins of Taskmaster Contestants

## Overview

Figure 4 visualizes the geographic distribution of Taskmaster contestants' birthplaces, with a focus on the United Kingdom and Ireland where the majority of contestants originated. The visualization includes:

1. A map of the British Isles showing birthplace density for UK and Irish contestants
2. A heatmap and contour overlay highlighting regions with higher concentrations of contestants
3. A comprehensive legend showing contestant counts by country of origin

## Data Sources

The visualization uses two primary data sources:

1. **Contestant Birthplace Data**: From `data/raw/Cont_lon_lat.tsv`, containing:
   - Contestant names
   - Birthplace locations (city and country)
   - Geographic coordinates (latitude and longitude)
   - Verification details and notes

2. **Contestant Information**: From `data/raw/contestants.csv`, containing:
   - Contestant IDs and series participation
   - Placement in their respective series
   - Other demographic and career information

## Implementation Details

### Data Processing (`process_figure4_data.py`)
- Merges contestant location data with contestant information
- Converts latitude and longitude to numeric values
- Separates UK/Ireland contestants from international contestants
- Calculates country-level statistics

### Visualization (`plot_figure4.py`)
- Uses a high-quality map of the British Isles as the background
- Transforms geographic coordinates to pixel positions on the map
- Creates a heatmap to visualize contestant density
- Adds contour lines to highlight concentration patterns
- Uses a smaller bandwidth for the kernel density estimation to ensure individual points remain visible
- Provides a comprehensive "Countries" legend showing all contestant countries of origin
- Saves both PNG and PDF versions of the figure

## Visualization Choices

Several specific design decisions were made for this visualization:

1. **Heatmap Instead of Scatter**: A heatmap with contours was chosen instead of a scatter plot to address projection imperfections that might place points in inaccurate locations (such as in the sea or on beaches).

2. **Density Estimation**: The kernel density estimation uses a smaller bandwidth parameter to ensure that even isolated points create visible "hot spots" in the visualization.

3. **Unified Country Legend**: All countries are represented in a single legend with:
   - Steel blue circles for UK and Ireland
   - Dark golden rod circles for international countries
   - Countries sorted by contestant count (descending)

## Key Findings

The visualization reveals several patterns in the geographic origins of Taskmaster contestants:

1. **London Dominance**: A high concentration of contestants originated from London and the surrounding areas, reflecting the centralized nature of the UK comedy industry.

2. **Regional Distribution**: Beyond London, contestants come from various regions across the UK, with noticeable clusters in certain areas.

3. **International Representation**: While predominantly a UK show, Taskmaster has featured contestants from several other countries, particularly:
   - Australia
   - New Zealand
   - Ireland
   - Canada
   - USA

4. **Urban Centers**: Most contestants were born in or near major urban centers rather than rural areas.

## Interpretation

This geographic distribution provides insight into the diversity of backgrounds represented on Taskmaster, while also highlighting the London-centric nature of British comedy. The international contestants, while fewer in number, demonstrate the show's reach and appeal beyond the UK.

The visualization helps contextualize the cultural references and comedic styles brought by contestants from different regions, potentially explaining similarities and differences in their approaches to tasks. 