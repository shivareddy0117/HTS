"""
Common utility functions for the HTS Tape Manufacturing Optimization System.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
import logging


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup a logger with file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level (logging level): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def calculate_cv(values, axis=None):
    """
    Calculate coefficient of variation.
    
    Args:
        values (array-like): Input values
        axis (int, optional): Axis along which to calculate CV
        
    Returns:
        float or array: Coefficient of variation
    """
    return np.std(values, axis=axis) / np.mean(values, axis=axis)


def detect_dropouts(critical_current, threshold_factor=1.5, window_size=10):
    """
    Detect dropouts in critical current measurements.
    
    Args:
        critical_current (array-like): Critical current measurements
        threshold_factor (float): Factor of standard deviation below mean to consider as dropout
        window_size (int): Size of moving window for local analysis
        
    Returns:
        tuple: (dropout_indices, dropout_severities)
    """
    # Calculate global threshold
    global_mean = np.mean(critical_current)
    global_std = np.std(critical_current)
    global_threshold = global_mean - threshold_factor * global_std
    
    # Initialize results
    dropout_indices = []
    dropout_severities = []
    
    # Detect dropouts using moving window
    for i in range(len(critical_current) - window_size + 1):
        window = critical_current[i:i+window_size]
        window_mean = np.mean(window)
        
        # Check if window mean is below threshold
        if window_mean < global_threshold:
            # Find minimum value in window
            min_idx = i + np.argmin(window)
            min_value = critical_current[min_idx]
            
            # Calculate severity (percentage below global mean)
            severity = 100 * (1 - min_value / global_mean)
            
            dropout_indices.append(min_idx)
            dropout_severities.append(severity)
    
    return np.array(dropout_indices), np.array(dropout_severities)


def plot_critical_current_heatmap(positions, critical_current, process_params, param_name, 
                                 figsize=(12, 8), cmap='viridis', 
                                 title="Critical Current vs. Position and Process Parameter"):
    """
    Create a heatmap of critical current with respect to position and a process parameter.
    
    Args:
        positions (array-like): Position values
        critical_current (array-like): Critical current values
        process_params (array-like): Process parameter values
        param_name (str): Name of the process parameter
        figsize (tuple): Figure size
        cmap (str or colormap): Colormap to use
        title (str): Plot title
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(
        positions, 
        process_params,
        bins=[50, 30],
        weights=critical_current
    )
    
    # Normalize by count
    count, _, _ = np.histogram2d(
        positions, 
        process_params,
        bins=[xedges, yedges]
    )
    
    # Avoid division by zero
    mask = count > 0
    hist_normalized = np.zeros_like(hist)
    hist_normalized[mask] = hist[mask] / count[mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(
        hist_normalized.T,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        origin='lower',
        aspect='auto',
        cmap=cmap
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Critical Current (A)')
    
    # Labels and title
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel(param_name)
    ax.set_title(title)
    
    return fig


def granger_causality_test(timeseries_x, timeseries_y, max_lag=10):
    """
    Perform Granger causality test to determine if X causes Y.
    
    Args:
        timeseries_x (array-like): Time series X (potential cause)
        timeseries_y (array-like): Time series Y (potential effect)
        max_lag (int): Maximum lag to test
        
    Returns:
        dict: Dictionary with test results
    """
    # Ensure data is stationary
    x_diff = np.diff(timeseries_x)
    y_diff = np.diff(timeseries_y)
    
    min_len = min(len(x_diff), len(y_diff))
    x_diff = x_diff[:min_len]
    y_diff = y_diff[:min_len]
    
    # Prepare data for testing
    data = pd.DataFrame({
        'x': x_diff,
        'y': y_diff
    })
    
    results = {}
    
    for lag in range(1, max_lag + 1):
        # Restricted model (Y depends only on past Y)
        restricted_data = pd.DataFrame()
        
        for l in range(1, lag + 1):
            restricted_data[f'y_lag_{l}'] = data['y'].shift(l)
        
        restricted_data = restricted_data.dropna()
        y_restricted = data['y'].iloc[lag:]
        
        restricted_rss = np.sum((y_restricted - restricted_data.mean(axis=1))**2)
        
        # Unrestricted model (Y depends on past Y and past X)
        unrestricted_data = restricted_data.copy()
        
        for l in range(1, lag + 1):
            unrestricted_data[f'x_lag_{l}'] = data['x'].shift(l)
        
        unrestricted_data = unrestricted_data.dropna()
        y_unrestricted = data['y'].iloc[lag:]
        
        unrestricted_rss = np.sum((y_unrestricted - unrestricted_data.mean(axis=1))**2)
        
        # Calculate F-statistic
        n = len(y_unrestricted)
        df1 = lag
        df2 = n - 2 * lag - 1
        
        if df2 > 0 and restricted_rss > 0:
            f_stat = ((restricted_rss - unrestricted_rss) / df1) / (unrestricted_rss / df2)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
            
            results[lag] = {
                'f_stat': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
    
    return results


def create_custom_colormap(name='hts_cmap', start_color='navy', mid_color='white', end_color='darkred'):
    """
    Create a custom colormap for HTS visualization.
    
    Args:
        name (str): Name of the colormap
        start_color (str): Color for the low end
        mid_color (str): Color for the middle
        end_color (str): Color for the high end
        
    Returns:
        matplotlib.colors.LinearSegmentedColormap: Custom colormap
    """
    colors = [start_color, mid_color, end_color]
    positions = [0, 0.5, 1]
    
    cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
    return cmap


def inverse_pca_transform(pca_components, pca_model, scaler=None):
    """
    Transform PCA components back to original features.
    
    Args:
        pca_components (array-like): PCA components
        pca_model: Fitted PCA model
        scaler: Fitted scaler model (optional)
        
    Returns:
        array-like: Original features
    """
    # Inverse PCA transform
    original_scaled = pca_model.inverse_transform(pca_components)
    
    # Inverse scaling if scaler provided
    if scaler:
        original = scaler.inverse_transform(original_scaled)
    else:
        original = original_scaled
        
    return original


def save_results_to_csv(data, filename, timestamp=True):
    """
    Save results to CSV file.
    
    Args:
        data (dict or DataFrame): Data to save
        filename (str): Output filename
        timestamp (bool): Whether to add timestamp to filename
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data
    
    # Add timestamp to filename if requested
    if timestamp:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    
    # Create directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save data
    df.to_csv(filename, index=False)
    
    return filename 