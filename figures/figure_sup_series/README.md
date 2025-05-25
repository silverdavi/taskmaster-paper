# Supplementary Figure: Series Deep Dive Analysis

## Overview

This supplementary figure provides detailed analysis of individual Taskmaster series, showing contestant performance progression and cumulative scoring patterns. Each series gets its own comprehensive visualization with two components:

1. **Top Plot**: Ranking progression across all tasks with episode boundaries
2. **Bottom Plot**: Cumulative scores as line plot

## Data Processing

### Source Data
- **Input**: `data/raw/scores.csv` - Complete scoring data for all series
- **Processing**: Task-by-task analysis with episode boundary detection
- **Output**: Individual JSON files for each series containing processed data

### Key Metrics Calculated
- **Ranking Progression**: Position (1-5) after each task based on cumulative scores
- **Cumulative Scores**: Running total of points earned through each task
- **Episode Boundaries**: Task positions where new episodes begin
- **Final Standings**: Total scores and final rankings for each contestant

## Visualization Features

### Top Plot: Ranking Progression
- **X-axis**: Task number (chronological order)
- **Y-axis**: Ranking position (1 = best, inverted scale)
- **Lines**: Each contestant's ranking journey with distinctive colors
- **Markers**: Large circular "beads" at each task position
- **Shading**: Alternating gray bands to separate episodes
- **Episode Labels**: "Ep 1", "Ep 2", etc. at episode centers
- **Boundaries**: Dashed vertical lines between episodes

### Bottom Plot: Cumulative Scores
- **Full Width**: Traditional line plot showing score accumulation
- **Consistent Colors**: Same color scheme as ranking plot
- **Seaborn Styling**: Clean, professional appearance with grids

## Layout Design

### Grid Structure
- **2 rows × 1 column** with equal height ratios
- **Top plot**: Ranking progression with episode boundaries
- **Bottom plot**: Cumulative scores at full width
- **Figure size**: 16×10 inches for detailed visibility

### Styling Elements
- **Font**: Arial family (consistent with paper style)
- **Colors**: HUSL palette for maximum distinction between contestants
- **Legends**: Positioned outside plots for clarity
- **Grid**: Subtle alpha=0.3 for readability
- **Markers**: White edges for better visibility

## Generated Files

### Data Files
- `series_N_data.json` - Processed data for series N
- `series_summary.json` - Overview statistics across all series

### Visualization Files
- `series_N_deep_dive.png` - High-resolution plot (450 DPI)
- `series_N_deep_dive.pdf` - Vector format for publication

### Scripts
- `process_data_series_deep_dive.py` - Data processing pipeline
- `create_series_plots.py` - Visualization generation

## Usage

### Generate Data
```bash
# Process scoring data for first 3 series
python process_data_series_deep_dive.py
```

### Create Visualizations
```bash
# Generate plots for all processed series
python create_series_plots.py
```

### Extend to More Series
Edit `process_data_series_deep_dive.py` line 203 to process more series:
```python
# Change from [:3] to [:N] or remove slice for all series
for series_num in available_series[:N]:
```

## Key Insights Revealed

### Ranking Dynamics
- **Early Leaders**: Contestants who start strong vs. late bloomers
- **Consistency**: Steady performers vs. volatile rankings
- **Episode Effects**: How episode boundaries affect momentum
- **Final Sprints**: Last-episode comebacks or collapses

### Scoring Patterns
- **Cumulative Growth**: Linear vs. exponential score accumulation
- **Gap Analysis**: When leads become insurmountable
- **Relative Performance**: Individual vs. group dynamics
- **Task Impact**: High-scoring tasks that change rankings

## Technical Notes

### Dependencies
- pandas (data processing)
- numpy (numerical operations)
- matplotlib (plotting framework)
- seaborn (styling and colors)
- json (data serialization)

### Performance
- Memory efficient: Processes one series at a time
- Scalable: Can handle all 18+ series
- Fast execution: ~2-3 seconds per series

### Data Quality
- Handles missing scores gracefully
- Validates episode boundaries
- Ensures consistent contestant ordering
- Robust JSON serialization

## Validation

### Data Integrity
- ✅ All contestants appear in every task
- ✅ Rankings sum correctly (1+2+3+4+5 = 15)
- ✅ Cumulative scores only increase
- ✅ Episode boundaries align with task structure

### Visual Quality
- ✅ Clear distinction between contestants
- ✅ Readable legends and labels
- ✅ Proper episode demarcation
- ✅ Consistent color schemes across plots

## Future Extensions

### Additional Analysis
- Win probability over time
- Task difficulty correlation
- Contestant archetype patterns
- Cross-series comparisons

### Interactive Features
- Hover tooltips with task names
- Clickable episode navigation
- Contestant filtering
- Animation of progression

## Conclusion

This supplementary figure provides comprehensive insight into the dynamics of individual Taskmaster series, revealing patterns of contestant performance that are not visible in aggregate analyses. The multi-panel design efficiently presents both ranking progression and score accumulation, making it easy to identify key moments and trends within each series. 