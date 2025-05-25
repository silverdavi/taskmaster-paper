# Supplementary Figure: Series Deep Dive Analysis

## Title
Taskmaster Series Deep Dive: Contestant Performance Progression and Cumulative Scoring Patterns

## Caption
Multi-panel visualization showing detailed performance analysis for individual Taskmaster series. **Top panel**: Ranking progression across all tasks with episode boundaries marked by alternating gray shading and dashed vertical lines. Each contestant's journey is shown with distinctive colored lines and circular markers ("beads"), with ranking position (1=best) on the y-axis and task number on the x-axis. Episode labels ("Ep 1", "Ep 2", etc.) are positioned at episode centers. **Bottom panel**: Cumulative score progression shown as line plot with the same color scheme, displaying how total points accumulate task by task at full width for clear visualization. Both plots use consistent HUSL color palette for maximum distinction between contestants, with legends positioned outside plot areas for clarity. The visualization reveals ranking dynamics (early leaders vs. late bloomers, consistency vs. volatility), scoring patterns (linear vs. exponential accumulation), and critical moments where episode boundaries affect momentum or leads become insurmountable.

## Methods Note
Data processed from complete scoring records in scores.csv, with task-by-task analysis calculating cumulative scores and rankings after each task. Episode boundaries detected automatically from task-episode mapping. Rankings calculated as position (1-5) based on cumulative scores at each task, with ties handled consistently. Visualization uses 2×1 grid layout with equal height ratios and 16×10 inch figure size for detailed visibility.

## Technical Details
- **Ranking Calculation**: Position after each task based on cumulative score totals
- **Episode Detection**: Automatic boundary identification from task-episode relationships  
- **Color Scheme**: HUSL palette ensuring maximum perceptual distinction between contestants
- **Layout**: Top plot spans both columns, bottom plots split for comparison
- **Styling**: Seaborn whitegrid theme with Arial fonts and 450 DPI resolution

## Data Coverage
Analysis covers Series 1-3 as demonstration (extensible to all 18+ series). Series 1: 6 episodes, 34 tasks, 5 contestants. Series 2: 5 episodes, 28 tasks, 5 contestants. Series 3: 5 episodes, 28 tasks, 5 contestants. Each series processed independently with consistent methodology and visualization standards.

## Key Insights
The deep dive analysis reveals distinct performance patterns: some contestants establish early leads and maintain them (consistent performers), others show volatile rankings with dramatic swings (high-variance players), and some demonstrate late-series surges or collapses. Episode boundaries often coincide with momentum shifts, suggesting psychological or strategic effects of episode structure on contestant performance. Cumulative score plots reveal when leads become mathematically insurmountable and identify high-impact tasks that significantly alter standings. 