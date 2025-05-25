# Contestant Geographic Origins Analysis

## Overview

This figure visualizes the birthplaces of all 90 Taskmaster UK contestants across 18 series using a heat map overlaid on a detailed map of the UK and Ireland. The visualization reveals geographic clustering patterns and highlights the show's predominantly English contestant base while showcasing international diversity.

## Key Results

### Geographic Distribution

| Country | Count | Percentage |
|---------|-------|------------|
| England | 67 | 74.4% |
| Ireland | 4 | 4.4% |
| Scotland | 3 | 3.3% |
| Wales | 3 | 3.3% |
| Australia | 3 | 3.3% |
| USA | 3 | 3.3% |
| Canada | 2 | 2.2% |
| New Zealand | 1 | 1.1% |
| Hong Kong | 1 | 1.1% |
| Malaysia | 1 | 1.1% |
| Pakistan | 1 | 1.1% |
| Japan | 1 | 1.1% |

### Key Findings

1. **London Dominance**: The largest cluster (29 contestants) is centered on London, representing 32.2% of all contestants
   - Notable London-based contestants include Jack Dee, Victoria Coren Mitchell, Jo Brand, Ed Gamble, and many others

2. **UK and Ireland Total**: 77 contestants (85.6%) were born in the British Isles
   - England: 67 (74.4%)
   - Combined Celtic nations: 10 (11.1%)

3. **International Contestants**: 13 contestants (14.4%) were born outside UK/Ireland
   - Reflects the international nature of UK comedy scene
   - Includes established comedians who moved to UK for their careers

### Regional Hotspots in UK/Ireland

The heat map reveals several geographic clusters:

1. **Greater London**: Massive concentration (29 contestants)
2. **Northwest England**: Secondary cluster around Manchester/Liverpool
3. **Yorkshire**: Notable presence around Leeds/York
4. **Scotland**: Concentrated in Glasgow/Edinburgh area
5. **Ireland**: Distributed between Dublin and rural areas

### Notable Geographic Patterns

- **Urban Bias**: Most contestants come from major urban centers
- **Comedy Circuit Geography**: Aligns with UK's major comedy venues and scenes
- **Celtic Representation**: 11.1% from Scotland, Wales, and Ireland combined
- **International Integration**: Foreign-born comedians well-integrated into UK comedy scene

### Contestant Examples by Region

**London Cluster (29)**: Jack Dee, Jo Brand, Victoria Coren Mitchell, Ed Gamble, Josh Widdicombe, Katherine Parkinson, and many others

**Scotland (3)**: Iain Stirling, Fern Brady, Frankie Boyle

**Wales (3)**: Rhod Gilbert, Katy Wix, Sian Gibson

**Ireland (4)**: Aisling Bea, Dara Ó Briain, Ardal O'Hanlon, Joanne McNally

**International**:
- USA: Rich Fulcher, Rose Matafeo, Desiree Burch
- Australia: Sarah Kendall, Sam Campbell, Felicity Ward
- Canada: Katherine Ryan, Mae Martin
- Others: Phil Wang (Malaysia), Nish Kumar (origin), Paul Chowdhry (origin)

## Implementation Details

### Data Processing (`process_geographic_origins_data.py`)

The script:
1. Loads contestant location data from `Cont_lon_lat.tsv`
2. Maps contestants to their birthplace coordinates
3. Handles international contestants by placing them at map edges
4. Creates a grid overlay (10km cells) for heat map generation
5. Calculates contestant density per grid cell
6. Generates output files for visualization

### Plotting (`plot_geographic_origins.py`)

Creates a sophisticated visualization:
1. Uses high-resolution UK/Ireland map as base layer
2. Overlays semi-transparent heat map showing contestant density
3. Uses color gradient from light yellow (low density) to dark red (high density)
4. Includes country labels for readability
5. Lists international contestants separately at bottom
6. Maintains geographic accuracy while ensuring visibility

### Technical Details

- **Grid Resolution**: 10km × 10km cells
- **Heat Map Interpolation**: Gaussian kernel with 30km standard deviation
- **Color Scheme**: Matplotlib 'hot' colormap with custom transparency
- **Map Projection**: Maintains original map projection for accuracy

## Output Files

- `contestant_pixel_locations.csv`: Pixel coordinates for each contestant
- `grid_cell_data.csv`: Aggregated data per grid cell
- `country_counts.csv`: Summary statistics by country
- `transform_info.csv`: Coordinate transformation parameters
- `figure4.pdf/png`: Final heat map visualization

## Insights for Paper

1. **London-Centric Comedy Scene**: The extreme concentration in London (32.2%) reflects the centralization of UK's entertainment industry.

2. **Urban Comedy Pipeline**: The urban bias suggests comedy careers typically develop in cities with established comedy circuits and venues.

3. **Celtic Nations Underrepresented**: Despite making up ~15% of UK population, Scotland, Wales, and Northern Ireland contribute only 11.1% of contestants.

4. **International Success Stories**: The 14.4% international contingent demonstrates the UK comedy scene's openness to international talent.

5. **Regional Diversity Within England**: While London dominates, significant representation from Northern England, Midlands, and other regions shows some geographic diversity.

6. **Comedy Migration Patterns**: Many international contestants (e.g., Katherine Ryan from Canada, Rose Matafeo from New Zealand) represent successful comedy immigrants who built careers in the UK. 