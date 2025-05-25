#!/usr/bin/env python3
"""
Master script to generate all analyses for the Taskmaster quantitative exploration.

This script:
1. Processes data for all analysis modules
2. Generates all visualizations
3. Collects all metrics and documentation
"""

import subprocess
import sys
from pathlib import Path
import json
import time
import argparse
import os

# Define all analysis modules in the current project
ANALYSIS_MODULES = {
    'series_ratings_analysis': {
        'description': 'Series-Level IMDb Ratings Analysis with Mixture Models',
        'process_script': 'process_series_ratings_data.py',
        'plot_script': 'plot_seaborn_ridgeline_decomposed.py',
        'status': 'implemented'
    },
    'episode_rating_trajectories': {
        'description': 'Episode Rating Trajectory Pattern Analysis',
        'process_script': 'process_episode_trajectories_data.py',
        'plot_script': 'plot_episode_trajectories.py',
        'status': 'implemented'
    },
    'task_characteristics_analysis': {
        'description': 'Task Typology and Demand Analysis',
        'process_script': 'process_task_characteristics_data.py',
        'plot_script': 'plot_task_characteristics.py',
        'status': 'implemented'
    },
    'contestant_geographic_origins': {
        'description': 'Geographic Distribution and Cultural Analysis',
        'process_script': 'process_geographic_origins_data.py',
        'plot_script': 'plot_geographic_origins.py',
        'status': 'implemented'
    },
    'contestant_performance_archetypes': {
        'description': 'Performance-Based Clustering Analysis',
        'process_script': 'extract_features.py',
        'plot_script': 'plot_performance_archetypes.py',
        'status': 'implemented'
    },
    'sentiment_trends_analysis': {
        'description': 'Comedic Sentiment Pattern Analysis',
        'process_script': 'process_data.py',
        'plot_script': 'plot_sentiment_trends.py',
        'status': 'implemented'
    },
    'predictive_modeling_analysis': {
        'description': 'Episode Success Prediction Models',
        'process_script': 'run_all.py',
        'plot_script': None,  # run_all.py handles both processing and plotting
        'status': 'implemented'
    },
    'scoring_pattern_geometry': {
        'description': 'Task Scoring System Analysis',
        'process_script': 'process_data.py',
        'plot_script': 'plot_scoring_patterns.py',
        'status': 'implemented'
    },
    'task_skill_profiles': {
        'description': 'Task-Skill Requirement Mapping',
        'process_script': 'process_skill_profiles_data.py',
        'plot_script': 'create_skill_spider_plot.py',
        'status': 'implemented'
    },
    'individual_series_analysis': {
        'description': 'Series-Specific Deep Dive Analysis',
        'process_script': 'process_series_progression_data.py',
        'plot_script': 'create_series_progression_plots.py',
        'status': 'implemented'
    }
}

def run_script(script_path, description):
    """Run a Python script and report success/failure."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'-'*80}")
    
    start_time = time.time()
    
    # Change to the script's directory before running
    script_dir = script_path.parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(script_dir)
        result = subprocess.run([sys.executable, script_path.name], capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success ({elapsed_time:.2f}s)")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ Failed ({elapsed_time:.2f}s)")
            print("Error output:")
            print(result.stderr)
            if result.stdout.strip():
                print("Standard output:")
                print(result.stdout)
            return False
    finally:
        os.chdir(original_cwd)

def process_module(module_name, process_only=False, plot_only=False):
    """Process and plot a specific analysis module."""
    if module_name not in ANALYSIS_MODULES:
        print(f"❌ Unknown module: {module_name}")
        print(f"Available modules: {', '.join(ANALYSIS_MODULES.keys())}")
        return False
    
    module_info = ANALYSIS_MODULES[module_name]
    module_dir = Path(__file__).parent / "figures" / module_name
    
    if not module_dir.exists():
        print(f"❌ Module directory not found: {module_dir}")
        return False
    
    print(f"\n🔬 Processing module: {module_name}")
    print(f"📝 Description: {module_info['description']}")
    print(f"📊 Status: {module_info['status']}")
    
    success = True
    
    if not plot_only:
        # Process data
        process_script = module_dir / module_info['process_script']
        if process_script.exists():
            success = run_script(process_script, f"Processing data for {module_name}")
        else:
            if module_info['status'] == 'implemented':
                print(f"❌ Expected processing script not found: {process_script}")
                return False
            else:
                print(f"⚠️ Processing script not yet implemented: {process_script}")
                return True  # Not an error for planned modules
    
    if success and not process_only:
        # Generate plots
        if module_info['plot_script'] is None:
            # Some modules (like predictive_modeling_analysis) handle both in one script
            print(f"ℹ️ Module {module_name} handles plotting in the processing script")
        else:
            plot_script = module_dir / module_info['plot_script']
            if plot_script.exists():
                success = run_script(plot_script, f"Generating plots for {module_name}")
            else:
                if module_info['status'] == 'implemented':
                    print(f"❌ Expected plotting script not found: {plot_script}")
                    return False
                else:
                    print(f"⚠️ Plotting script not yet implemented: {plot_script}")
                    return True  # Not an error for planned modules
    
    return success

def collect_documentation():
    """Collect all module documentation and metrics."""
    figures_dir = Path(__file__).parent / "figures"
    output_file = Path(__file__).parent / "analysis_summary.md"
    
    documentation = []
    metrics_summary = {}
    
    # Look for each module directory
    for module_name in ANALYSIS_MODULES.keys():
        module_dir = figures_dir / module_name
        if not module_dir.exists():
            continue
            
        # Check for overview documentation
        overview_file = module_dir / f"{module_name}_overview.md"
        if overview_file.exists():
            with open(overview_file, "r") as f:
                content = f.read().strip()
                documentation.append(f"## {module_name.replace('_', ' ').title()}\n\n{content}")
        
        # Check for metrics
        metrics_file = module_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                metrics_summary[module_name] = metrics
    
    # Write summary documentation
    with open(output_file, "w") as f:
        f.write("# Taskmaster Analysis: Complete Summary\n\n")
        f.write("## Overview\n\n")
        f.write("This document provides a comprehensive summary of all analysis modules ")
        f.write("in the Taskmaster quantitative exploration project.\n\n")
        
        if documentation:
            f.write("## Module Documentation\n\n")
            for doc in documentation:
                f.write(doc + "\n\n")
        
        if metrics_summary:
            f.write("## Key Metrics Summary\n\n")
            f.write("```json\n")
            f.write(json.dumps(metrics_summary, indent=2))
            f.write("\n```\n\n")
    
    print(f"\n📋 Collected documentation for {len(documentation)} modules")
    print(f"📊 Collected metrics for {len(metrics_summary)} modules")
    print(f"📄 Summary written to: {output_file}")

def list_modules():
    """List all available analysis modules with their status."""
    print("\n📚 Available Analysis Modules:\n")
    
    implemented = []
    planned = []
    
    for module_name, info in ANALYSIS_MODULES.items():
        if info['status'] == 'implemented':
            implemented.append((module_name, info['description']))
        else:
            planned.append((module_name, info['description']))
    
    if implemented:
        print("✅ Implemented Modules:")
        for name, desc in implemented:
            print(f"   • {name}: {desc}")
        print()
    
    if planned:
        print("📋 Planned Modules:")
        for name, desc in planned:
            print(f"   • {name}: {desc}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Generate analyses for Taskmaster quantitative exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_figures.py                           # Run all analysis modules
  python generate_all_figures.py --module series_ratings_analysis  # Run specific module
  python generate_all_figures.py --process-only            # Only process data
  python generate_all_figures.py --plot-only               # Only generate plots
  python generate_all_figures.py --list                    # List all modules
  python generate_all_figures.py --collect-docs            # Collect documentation
        """
    )
    
    parser.add_argument("--module", help="Generate only this analysis module")
    parser.add_argument("--process-only", action="store_true", help="Only process data, don't plot")
    parser.add_argument("--plot-only", action="store_true", help="Only plot figures, don't process data")
    parser.add_argument("--collect-docs", action="store_true", help="Collect all documentation and metrics")
    parser.add_argument("--list", action="store_true", help="List all available modules")
    parser.add_argument("--implemented-only", action="store_true", help="Only run implemented modules")
    
    args = parser.parse_args()
    
    if args.list:
        list_modules()
        return
    
    if args.collect_docs:
        collect_documentation()
        return
    
    if args.module:
        # Process a single module
        success = process_module(args.module, args.process_only, args.plot_only)
        if success:
            print(f"\n🎉 Successfully completed: {args.module}")
        else:
            print(f"\n💥 Failed to complete: {args.module}")
            sys.exit(1)
    else:
        # Process modules based on selection criteria
        modules_to_run = []
        
        if args.implemented_only:
            modules_to_run = [name for name, info in ANALYSIS_MODULES.items() 
                            if info['status'] == 'implemented']
        else:
            # Run all modules (but skip planned ones that don't have scripts)
            modules_to_run = list(ANALYSIS_MODULES.keys())
        
        print(f"\n🚀 Running {len(modules_to_run)} analysis modules...")
        
        all_success = True
        completed_modules = []
        failed_modules = []
        
        for module_name in modules_to_run:
            success = process_module(module_name, args.process_only, args.plot_only)
            if success:
                completed_modules.append(module_name)
            else:
                failed_modules.append(module_name)
                all_success = False
        
        # Summary
        print(f"\n{'='*80}")
        print("📊 EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Completed: {len(completed_modules)} modules")
        if completed_modules:
            for module in completed_modules:
                print(f"   • {module}")
        
        if failed_modules:
            print(f"\n❌ Failed: {len(failed_modules)} modules")
            for module in failed_modules:
                print(f"   • {module}")
        
        if all_success and not args.process_only and not args.plot_only:
            collect_documentation()
            print("\n🎉 All analyses completed successfully!")
        elif failed_modules:
            print(f"\n💥 {len(failed_modules)} modules failed. Check output above for details.")
            sys.exit(1)

if __name__ == "__main__":
    main() 