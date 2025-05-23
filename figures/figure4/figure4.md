# Figure 4: Geographic Origins of Taskmaster Contestants

## Overview

Figure 4 visualizes the geographic distribution of Taskmaster contestants' birthplaces, with a focus on the United Kingdom and Ireland where the majority of contestants originated. The visualization includes:

1. A map of the British Isles showing precise birthplace locations for UK and Irish contestants
2. A density overlay highlighting regions with higher concentrations of contestants
3. A supplementary box showing international contestants grouped by country

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
- Creates a scatter plot of contestant birthplaces
- Generates a kernel density estimate to show concentration areas
- Adds a box for international contestants with counts by country
- Saves both PNG and PDF versions of the figure

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