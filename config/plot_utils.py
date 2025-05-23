#!/usr/bin/env python3
"""
Plotting utilities for Taskmaster Paper figures.

This module provides functions for consistent styling and configuration
across all figures in the paper.
"""

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import cmcrameri.cm as cmc  # Import Fabio Crameri's colormaps

def load_config():
    """Load configuration from plot_config.yaml"""
    config_path = Path(__file__).parent / "plot_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def apply_plot_style(fig=None, ax=None):
    """
    Apply consistent styling to plots.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure, optional
        Figure to apply styling to
    ax : matplotlib.axes.Axes or array of Axes, optional
        Axes to apply styling to
    
    Returns:
    --------
    dict
        The loaded configuration
    """
    config = load_config()
    
    # Set global style
    sns.set_theme(style="whitegrid", font=config["global"]["font_family"])
    
    # If specific figure/axes provided, apply styling
    if fig is not None and ax is not None:
        # Apply font sizes
        if hasattr(ax, '__iter__'):
            # Multiple axes
            for a in ax.flat:
                _style_axis(a, config)
        else:
            # Single axis
            _style_axis(ax, config)
    
    return config

def _style_axis(ax, config):
    """
    Apply styling to a single axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to apply styling to
    config : dict
        Configuration dictionary
    """
    ax.set_title(ax.get_title(), fontsize=config['fonts']['title_size'])
    ax.set_xlabel(ax.get_xlabel(), fontsize=config['fonts']['axis_label_size'])
    ax.set_ylabel(ax.get_ylabel(), fontsize=config['fonts']['axis_label_size'])
    ax.tick_params(labelsize=config['fonts']['tick_label_size'])
    
    if ax.get_legend() is not None:
        ax.legend(fontsize=config['fonts']['legend_size'])

def get_series_colors(num_series=18):
    """
    Get color palette for series (1-18).
    
    Parameters:
    -----------
    num_series : int, optional
        Number of series to get colors for, default is 18
    
    Returns:
    --------
    list
        List of RGB color tuples
    """
    config = load_config()
    return sns.color_palette(config['colors']['series_colormap'], n_colors=num_series)

def get_palette(palette_name, n_colors=None):
    """
    Get a named palette from the configuration.
    
    Parameters:
    -----------
    palette_name : str
        Name of the palette in the config (archetype_palette, task_type_palette)
    n_colors : int, optional
        Number of colors to return, if None will use default
    
    Returns:
    --------
    list
        List of RGB color tuples
    """
    config = load_config()
    return sns.color_palette(config['colors'][palette_name], n_colors=n_colors)

def log_metrics(figure_num, metrics_dict):
    """
    Save metrics for figure captions to a JSON file.
    
    Parameters:
    -----------
    figure_num : int
        Figure number
    metrics_dict : dict
        Dictionary of metrics to log
    """
    output_dir = Path(__file__).parent.parent / "figures" / f"figure{figure_num}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / "metrics.json"
    with open(output_file, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(f"Metrics for Figure {figure_num} saved to {output_file}")

def add_subplot_labels(axes, fontsize=16, fontweight='bold', x_offset=-0.1, y_offset=1.1):
    """
    Add subplot labels (A, B, C, ...) to axes.
    
    Parameters:
    -----------
    axes : matplotlib.axes.Axes or array of Axes
        Axes to add labels to
    fontsize : int, optional
        Font size for labels
    fontweight : str, optional
        Font weight for labels
    x_offset : float, optional
        X offset for label position
    y_offset : float, optional
        Y offset for label position
    """
    if hasattr(axes, '__iter__'):
        # Handle flattened multi-dimensional axes
        if hasattr(axes, 'flat'):
            axes_iter = axes.flat
        else:
            axes_iter = axes
            
        for i, ax in enumerate(axes_iter):
            label = chr(65 + i)  # 65 is ASCII for 'A'
            ax.text(x_offset, y_offset, label, transform=ax.transAxes,
                    fontsize=fontsize, fontweight=fontweight)
    else:
        # Single axis
        axes.text(x_offset, y_offset, 'A', transform=axes.transAxes,
                 fontsize=fontsize, fontweight=fontweight)

def generate_caption(figure_num, title, subplot_descriptions, metrics):
    """
    Generate a figure caption.
    
    Parameters:
    -----------
    figure_num : int
        Figure number
    title : str
        Figure title
    subplot_descriptions : list of str
        List of descriptions for each subplot
    metrics : dict
        Dictionary of metrics to include in the caption
    
    Returns:
    --------
    str
        Formatted caption
    """
    caption = f"Figure {figure_num}: {title}. "
    
    for i, desc in enumerate(subplot_descriptions):
        label = chr(65 + i)  # 65 is ASCII for 'A'
        
        # Format the description, replacing {metric_name} with the actual value
        formatted_desc = desc
        for metric_name, metric_value in metrics.items():
            formatted_desc = formatted_desc.replace(f"{{{metric_name}}}", str(metric_value))
        
        caption += f"({label}) {formatted_desc}. "
    
    return caption.strip()

def save_caption(figure_num, caption):
    """
    Save a caption to a text file.
    
    Parameters:
    -----------
    figure_num : int
        Figure number
    caption : str
        Caption to save
    """
    output_dir = Path(__file__).parent.parent / "figures" / f"figure{figure_num}"
    output_file = output_dir / "caption.txt"
    
    with open(output_file, "w") as f:
        f.write(caption)
    
    print(f"Caption for Figure {figure_num} saved to {output_file}") 