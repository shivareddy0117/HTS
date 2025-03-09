#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Sample HTS Tape Manufacturing Data

This script generates sample data for testing the HTS Tape Manufacturing Optimization System.
It creates synthetic critical current measurements and process parameters with realistic
correlations and patterns.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime


def generate_process_parameters(length, step_size=2, noise_level=0.1):
    """
    Generate synthetic process parameters for HTS tape manufacturing.
    
    Args:
        length (float): Length of the tape in cm
        step_size (float): Step size in cm
        noise_level (float): Level of noise to add to parameters
        
    Returns:
        pandas.DataFrame: DataFrame with process parameters
    """
    # Generate position values
    num_points = int(length / step_size) + 1
    positions = np.arange(0, num_points * step_size, step_size)
    
    # Base process parameters with sine waves of different frequencies
    substrate_temp_1 = 800 + 15 * np.sin(positions / 120) + np.random.normal(0, 5 * noise_level, len(positions))
    substrate_temp_2 = 820 + 10 * np.sin(positions / 150 + 1) + np.random.normal(0, 3 * noise_level, len(positions))
    substrate_temp_3 = 810 + 12 * np.sin(positions / 130 + 0.5) + np.random.normal(0, 4 * noise_level, len(positions))
    
    reaction_zone_pressure = 15 + 0.5 * np.sin(positions / 80) + np.random.normal(0, 0.1 * noise_level, len(positions))
    reaction_zone_temp = 850 + 8 * np.sin(positions / 100 + 0.3) + np.random.normal(0, 3 * noise_level, len(positions))
    
    deposition_voltage = 10 + 0.2 * np.sin(positions / 200) + np.random.normal(0, 0.05 * noise_level, len(positions))
    deposition_current = 120 + 5 * np.sin(positions / 180 + 0.7) + np.random.normal(0, 1 * noise_level, len(positions))
    
    oxygen_flow = 5 + 0.3 * np.cos(positions / 100) + np.random.normal(0, 0.1 * noise_level, len(positions))
    tension_1 = 2 + 0.1 * np.sin(positions / 90 + 0.2) + np.random.normal(0, 0.02 * noise_level, len(positions))
    tension_2 = 2.2 + 0.15 * np.sin(positions / 85 + 0.4) + np.random.normal(0, 0.03 * noise_level, len(positions))
    
    evaporator_temp = 500 + 10 * np.sin(positions / 110 + 0.6) + np.random.normal(0, 2 * noise_level, len(positions))
    throttle_valve = 45 + 5 * np.sin(positions / 95 + 0.8) + np.random.normal(0, 1 * noise_level, len(positions))
    
    # Combine all parameters into a DataFrame
    process_params = pd.DataFrame({
        'Position': positions,
        'Substrate_Temp_1': substrate_temp_1,
        'Substrate_Temp_2': substrate_temp_2,
        'Substrate_Temp_3': substrate_temp_3,
        'Reaction_Zone_Pressure': reaction_zone_pressure,
        'Reaction_Zone_Temp': reaction_zone_temp,
        'Deposition_Voltage': deposition_voltage,
        'Deposition_Current': deposition_current,
        'Oxygen_Flow': oxygen_flow,
        'Tension_1': tension_1,
        'Tension_2': tension_2,
        'Evaporator_Temp': evaporator_temp,
        'Throttle_Valve': throttle_valve
    })
    
    return process_params


def generate_critical_current(process_params, base_ic=300, dropout_count=5, window_size=8):
    """
    Generate synthetic critical current based on process parameters.
    
    Args:
        process_params (pandas.DataFrame): Process parameters
        base_ic (float): Base critical current value in Amperes
        dropout_count (int): Number of dropout regions to generate
        window_size (int): Size of window for calculation and correlation
        
    Returns:
        numpy.ndarray: Critical current values
    """
    positions = process_params['Position'].values
    
    # Generate base critical current with a sinusoidal pattern
    critical_current = base_ic + 30 * np.sin(positions / 100) + np.random.normal(0, 10, len(positions))
    
    # Add correlation with process parameters
    critical_current += 0.1 * (process_params['Substrate_Temp_1'] - 800)
    critical_current -= 0.2 * (process_params['Substrate_Temp_2'] - 820)
    critical_current += 5 * (process_params['Reaction_Zone_Pressure'] - 15)
    critical_current += 0.05 * (process_params['Reaction_Zone_Temp'] - 850)
    critical_current += 20 * (process_params['Deposition_Voltage'] - 10)
    critical_current -= 0.1 * (process_params['Deposition_Current'] - 120)
    critical_current += 10 * (process_params['Oxygen_Flow'] - 5)
    critical_current -= 50 * (process_params['Tension_1'] - 2)
    critical_current += 0.2 * (process_params['Evaporator_Temp'] - 500)
    
    # Add some dropouts at random positions
    dropout_indices = np.random.choice(range(len(positions)), size=dropout_count, replace=False)
    
    for idx in dropout_indices:
        # Create a dropout region centered at idx
        start = max(0, idx - window_size // 2)
        end = min(len(positions), idx + window_size // 2)
        
        # Apply a Gaussian-shaped dropout
        x = np.arange(start, end)
        center = idx
        sigma = window_size / 4
        dropout_factor = 0.4 + 0.2 * np.random.random()  # Random factor between 0.4 and 0.6
        gaussian = dropout_factor * np.exp(-0.5 * ((x - center) / sigma) ** 2)
        
        critical_current[start:end] *= (1 - gaussian)
    
    return critical_current


def calculate_cv(values, window_size, step_size):
    """
    Calculate coefficient of variation using a moving window.
    
    Args:
        values (array-like): Input values (critical current)
        window_size (int): Size of moving window in points
        step_size (int): Step size in points
        
    Returns:
        tuple: (window_positions, cv_values)
    """
    positions = np.arange(0, len(values))
    cv_positions = []
    cv_values = []
    
    for i in range(0, len(values) - window_size, step_size):
        end_idx = i + window_size
        window = values[i:end_idx]
        
        std_dev = np.std(window)
        mean_val = np.mean(window)
        
        if mean_val > 0:  # Avoid division by zero
            cv = std_dev / mean_val
            cv_values.append(cv)
            cv_positions.append(positions[i])
    
    return np.array(cv_positions), np.array(cv_values)


def generate_sample_data(length=1000, step_size=2, window_size=8, dropout_count=5, 
                        base_ic=300, noise_level=0.1, output_file=None):
    """
    Generate complete sample dataset for HTS tape manufacturing.
    
    Args:
        length (float): Length of the tape in cm
        step_size (float): Step size in cm
        window_size (int): Size of window for CV calculation in points
        dropout_count (int): Number of dropout regions to generate
        base_ic (float): Base critical current value in Amperes
        noise_level (float): Level of noise to add to parameters
        output_file (str, optional): Path to save the generated data
        
    Returns:
        tuple: (data, cv_positions, cv_values)
    """
    # Generate process parameters
    process_params = generate_process_parameters(length, step_size, noise_level)
    
    # Generate critical current
    critical_current = generate_critical_current(
        process_params, base_ic, dropout_count, window_size)
    
    # Calculate CV
    cv_positions, cv_values = calculate_cv(
        critical_current, window_size, step_size)
    
    # Combine into a single DataFrame
    data = process_params.copy()
    data['Ic'] = critical_current
    
    # Save to file if specified
    if output_file:
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        data.to_csv(output_file, index=False)
        print(f"Sample data saved to {output_file}")
    
    return data, cv_positions, cv_values


def plot_sample_data(data, cv_positions, cv_values, figsize=(12, 8)):
    """
    Plot the generated sample data.
    
    Args:
        data (pandas.DataFrame): Generated data
        cv_positions (array-like): Positions for CV values
        cv_values (array-like): CV values
        figsize (tuple): Figure size
    """
    positions = data['Position'].values
    critical_current = data['Ic'].values
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot critical current
    axes[0].plot(positions, critical_current, 'b-')
    axes[0].set_ylabel('Critical Current (A)')
    axes[0].set_title('Critical Current vs. Position')
    axes[0].grid(True)
    
    # Plot CV
    axes[1].plot(positions[cv_positions.astype(int)], cv_values, 'r-')
    axes[1].set_ylabel('Coefficient of Variation')
    axes[1].set_title('CV vs. Position')
    axes[1].grid(True)
    
    # Plot a selected process parameter
    param_name = 'Substrate_Temp_1'
    axes[2].plot(positions, data[param_name], 'g-')
    axes[2].set_xlabel('Position (cm)')
    axes[2].set_ylabel(param_name)
    axes[2].set_title(f'{param_name} vs. Position')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function for generating sample data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate sample HTS tape manufacturing data')
    
    parser.add_argument('--length', type=float, default=1000,
                        help='Length of the tape in cm')
    parser.add_argument('--step-size', type=float, default=2,
                        help='Step size in cm')
    parser.add_argument('--window-size', type=int, default=8,
                        help='Size of window for CV calculation in points')
    parser.add_argument('--dropout-count', type=int, default=5,
                        help='Number of dropout regions to generate')
    parser.add_argument('--base-ic', type=float, default=300,
                        help='Base critical current value in Amperes')
    parser.add_argument('--noise-level', type=float, default=0.1,
                        help='Level of noise to add to parameters')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the generated data')
    parser.add_argument('--plot', action='store_true',
                        help='Plot the generated data')
    
    args = parser.parse_args()
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"data/sample_hts_data_{timestamp}.csv"
    
    # Generate the data
    data, cv_positions, cv_values = generate_sample_data(
        length=args.length,
        step_size=args.step_size,
        window_size=args.window_size,
        dropout_count=args.dropout_count,
        base_ic=args.base_ic,
        noise_level=args.noise_level,
        output_file=args.output
    )
    
    print(f"Generated sample data with {len(data)} points")
    print(f"Length: {args.length} cm")
    print(f"Critical current range: {data['Ic'].min():.2f} - {data['Ic'].max():.2f} A")
    print(f"Average CV: {np.mean(cv_values):.4f}")
    
    # Plot the data if requested
    if args.plot:
        plot_sample_data(data, cv_positions, cv_values)


if __name__ == "__main__":
    main() 