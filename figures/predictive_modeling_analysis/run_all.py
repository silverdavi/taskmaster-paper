#!/usr/bin/env python3
"""
Master Script: Run Complete Figure 8 Analysis Pipeline

This script runs the entire predictive modeling analysis for Figure 8
in the correct order, with error handling and progress reporting.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script with error handling and timing."""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"📝 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        elapsed = time.time() - start_time
        print(f"✅ SUCCESS: {script_name} completed in {elapsed:.1f}s")
        
        # Show last few lines of output
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 3:
                print("📄 Output (last 3 lines):")
                for line in lines[-3:]:
                    print(f"   {line}")
            else:
                print("📄 Output:")
                for line in lines:
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ FAILED: {script_name} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Run the complete Figure 8 analysis pipeline."""
    print("🎯 FIGURE 8 PREDICTIVE MODELING ANALYSIS")
    print("🔬 Complete Pipeline Execution")
    print(f"📁 Working Directory: {Path.cwd()}")
    
    # Define the pipeline
    pipeline = [
        ("1_prepare_episode_data.py", "Data preparation and feature engineering"),
        ("2_feature_selection_episode.py", "Feature selection using mutual information"),
        ("3_model_episode_analysis.py", "Train and evaluate ML models"),
        ("4_plot_figure8a.py", "Generate Figure 8a visualization"),
        ("5_correlation_analysis_figure8b.py", "Generate Figure 8b correlation analysis"),
        ("6_analyze_random_forest_features.py", "Deep dive into Random Forest insights")
    ]
    
    # Track results
    results = []
    total_start = time.time()
    
    # Run each script
    for script_name, description in pipeline:
        success = run_script(script_name, description)
        results.append((script_name, success))
        
        if not success:
            print(f"\n⚠️  Pipeline stopped due to failure in {script_name}")
            print("🔧 Please fix the error and run again")
            break
    
    # Summary
    total_elapsed = time.time() - total_start
    successful = sum(1 for _, success in results if success)
    total_scripts = len(results)
    
    print(f"\n{'='*60}")
    print("📊 PIPELINE EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_elapsed:.1f}s")
    print(f"✅ Successful: {successful}/{total_scripts} scripts")
    
    print(f"\n📋 Script Results:")
    for script_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {status}: {script_name}")
    
    if successful == total_scripts:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"📊 All Figure 8 analysis completed successfully")
        print(f"\n📁 Generated Files:")
        print(f"   • fig10.png/pdf - Episode ML analysis")
        print(f"   • s2_fig.png/pdf - Correlation analysis")
        print(f"   • fig11.png/pdf - Feature insights")
        print(f"   • episode_model_results.pkl - Complete model results")
        print(f"   • episode_data.csv - Processed dataset")
        print(f"\n📖 See README.md for detailed documentation")
        
    else:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"🔧 {total_scripts - successful} scripts failed")
        print(f"💡 Check error messages above and fix issues")
        
    return successful == total_scripts

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 