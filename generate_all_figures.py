#!/usr/bin/env python3
"""
Master script to generate all figures for the Taskmaster Paper.

This script:
1. Processes data for all figures
2. Generates all figure plots
3. Collects all captions into a single document
"""

import subprocess
import sys
from pathlib import Path
import json
import time
import argparse
import os

def run_script(script_path, description):
    """Run a Python script and report success/failure."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'-'*80}")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ Success ({elapsed_time:.2f}s)")
        print(result.stdout)
        return True
    else:
        print(f"❌ Failed ({elapsed_time:.2f}s)")
        print("Error output:")
        print(result.stderr)
        print("Standard output:")
        print(result.stdout)
        return False

def process_figure(figure_num, process_only=False, plot_only=False):
    """Process and plot a specific figure."""
    figure_dir = Path(__file__).parent / "figures" / f"figure{figure_num}"
    
    success = True
    
    if not plot_only:
        # Process data - use figure-specific process script if available
        process_script = figure_dir / f"process_data_figure{figure_num}.py"
        if not process_script.exists():
            # Fall back to generic process_data.py
            process_script = figure_dir / "process_data.py"
        
        if process_script.exists():
            success = run_script(process_script, f"Processing data for Figure {figure_num}")
        else:
            print(f"⚠️ Warning: No processing script found for Figure {figure_num}")
    
    if success and not process_only:
        # Plot figure - use figure-specific script if available
        plot_script = figure_dir / f"plot_figure{figure_num}.py"
        if not plot_script.exists():
            # Fall back to generic plot_figure.py
            plot_script = figure_dir / "plot_figure.py"
        
        if plot_script.exists():
            success = run_script(plot_script, f"Plotting Figure {figure_num}")
        else:
            print(f"⚠️ Warning: No plotting script found for Figure {figure_num}")
    
    return success

def collect_captions():
    """Collect all figure captions into a single document."""
    figures_dir = Path(__file__).parent / "figures"
    output_file = Path(__file__).parent / "all_figure_captions.md"
    
    captions = []
    
    # Look for each figure directory
    for figure_dir in sorted(figures_dir.glob("figure*")):
        caption_file = figure_dir / "caption.txt"
        if caption_file.exists():
            with open(caption_file, "r") as f:
                caption = f.read().strip()
                captions.append(caption)
    
    # Write all captions to a single file
    with open(output_file, "w") as f:
        f.write("# Figure Captions for Taskmaster Paper\n\n")
        for caption in captions:
            f.write(caption + "\n\n")
    
    print(f"\nCollected {len(captions)} captions into {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate figures for Taskmaster Paper")
    parser.add_argument("--figure", type=int, help="Generate only this figure number")
    parser.add_argument("--process-only", action="store_true", help="Only process data, don't plot")
    parser.add_argument("--plot-only", action="store_true", help="Only plot figures, don't process data")
    parser.add_argument("--captions-only", action="store_true", help="Only collect captions")
    args = parser.parse_args()
    
    if args.captions_only:
        collect_captions()
        return
    
    if args.figure:
        # Process a single figure
        process_figure(args.figure, args.process_only, args.plot_only)
    else:
        # Process all figures
        all_success = True
        
        # Define the figure numbers based on FiguresGuide.MD
        figure_numbers = range(1, 8)  # Figures 1-7
        
        for figure_num in figure_numbers:
            success = process_figure(figure_num, args.process_only, args.plot_only)
            if not success:
                all_success = False
        
        if all_success and not args.process_only and not args.plot_only:
            collect_captions()

if __name__ == "__main__":
    main() 