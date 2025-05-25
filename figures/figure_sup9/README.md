# Figure Supplementary 9: Task Skill Profile Spider Plot

## Overview

This supplementary figure presents spider/radar plots showing the skill requirement profiles of polarized Taskmaster tasks using **continuous skill intensity scores** (0.0 to 1.0 range).

## Data Processing

### Continuous Skill Mapping
The analysis maps existing numerical scores to interpretable skill dimensions:

| Skill Dimension | Source Column | Description |
|------------------|---------------|-------------|
| Creativity | `creativity_required_score` | Creative thinking and innovation |
| Physical Coordination | `physical_demand_score` | Physical dexterity and movement |
| Problem Solving | `technical_difficulty_score` | Analytical and logical thinking |
| Time Pressure | `time_pressure_score` | Working under time constraints |
| Originality | `weirdness_score` | Unusual or novel approaches |
| Entertainment | `entertainment_value_score` | Performance and audience appeal |
| Strategic Planning | `preparation_possible_score` | Advance planning opportunities |
| Adaptability | `luck_factor_score` | Handling unpredictable elements |

### Normalization
- Original scores (1-10 scale) normalized to 0.0-1.0 range
- Formula: `(raw_score - 1) / 9`
- Ensures consistent scaling across all skill dimensions

## Polarized Task Selection

Tasks are selected based on **Euclidean distance** in 8-dimensional skill space to find genuinely different skill profiles. **4 examples were chosen** for optimal visual clarity (avoiding overcrowded spider plots).

### Selected Tasks

1. **"Guess the number on Alex's forearm"** (Series 7)
   - High: Adaptability (1.00), Entertainment (0.78)
   - Low: Most other skills (0.00-0.11)
   - Profile: Pure adaptability/entertainment task

2. **"Get dressed while wearing handcuffs"** (Series 7)
   - High: Time Pressure (1.00), Physical Coordination (0.89), Entertainment (0.89)
   - Profile: Intense physical-temporal challenge

3. **"Recite Pi"** (Series 16)
   - High: Problem Solving (0.78), Time Pressure (0.89)
   - Low: Creativity (0.00), Physical skills (0.00)
   - Profile: Pure mental/memory challenge

4. **"The present that raises the most questions"** (Series 16)
   - High: Strategic Planning (1.00), Creativity (0.89), Entertainment (0.89)
   - Low: Time Pressure (0.00), Problem Solving (0.00)
   - Profile: Creative strategic thinking

## Visualization Features

### Multiline Legend
- **Full task titles** displayed in legend (no truncation)
- **Automatic text wrapping** for readability
- **Positioned to the right** of the plot for clear reference

### Clean Design
- **4 carefully selected** tasks avoid overcrowding
- **Distinct colors** for each task line
- **Larger figure size** (16×12) for better readability
- **Enhanced spacing** and typography

## Files Generated

### Data Files
- `skills_data.json` - Complete skill analysis data
- `radar_plot_data.json` - Formatted data for spider plot
- `tasks_summary.txt` - Detailed text analysis with visual bars

### Visualization Files
- `figure_sup9_spider_plot.png` - Final spider plot (PNG)
- `figure_sup9_spider_plot.pdf` - Final spider plot (PDF)

### Scripts
- `process_data_figure_sup9.py` - Data processing (continuous approach)
- `create_spider_plot.py` - Visualization creation

## Key Insights

### Task Differentiation
The continuous spider plots reveal:
- **Clear polarization**: Tasks have genuinely different skill intensity profiles
- **Smooth gradations**: Skill requirements vary continuously, not binarily
- **Interpretable patterns**: Easy to identify task types (physical vs. mental vs. creative)

### Skill Combinations
- Some tasks are **specialist** (high in 1-2 skills): "Recite Pi", "Guess the number"
- Others are **generalist** (moderate across many): "Get dressed while wearing handcuffs"
- Physical tasks cluster on coordination/time pressure
- Creative tasks cluster on creativity/strategic planning

## Usage

### Generate the Analysis
```bash
# Run continuous data processing
python figures/figure_sup9/process_data_figure_sup9.py

# Create spider plot visualization
python figures/figure_sup9/create_spider_plot.py
```

### Validate Results
```bash
# Check generated files
ls -la figures/figure_sup9/

# View skill profiles
cat figures/figure_sup9/tasks_summary.txt
```

## Technical Notes

### Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (for distance calculations)

### Coordinate System
- Polar coordinates for spider plot
- 8 axes (one per skill dimension)
- Radial scale: 0.0 (center) to 1.0 (edge)

### Color Scheme
- Red (#E31A1C): Task 1
- Blue (#1F78B4): Task 2  
- Green (#33A02C): Task 3
- Orange (#FF7F00): Task 4

## Validation

The continuous approach is validated by:
- ✅ Meaningful visual differences between tasks
- ✅ Smooth, interpretable spider plot shapes
- ✅ Clear correspondence between known task characteristics and skill profiles
- ✅ Proper utilization of the full 0.0-1.0 range
- ✅ Clean, uncluttered visualization with 4 examples
- ✅ Full task titles visible in multiline legend

## Conclusion

**The continuous skill intensity approach successfully creates meaningful spider plots that reveal the multidimensional nature of Taskmaster task requirements.** The 4 polarized examples demonstrate clear differentiation in skill profiles, from specialist tasks requiring specific skills to generalist tasks requiring balanced capabilities across multiple dimensions. The improved visualization design ensures optimal readability and interpretability. 