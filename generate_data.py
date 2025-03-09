#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate Sample Data for HTS Tape Manufacturing Optimization System.

This script provides a simple way to generate sample data for testing
and demonstrating the HTS Tape Manufacturing Optimization System.
"""

import os
import sys
import argparse
from datetime import datetime

# Check for required dependencies before importing
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"ERROR: Missing required dependency - {e}")
    print("\nPlease install the required packages using:")
    print("pip install -r requirements.txt")
    print("\nOr install the specific missing package:")
    print(f"pip install {str(e).split()[-1]}")
    sys.exit(1)

try:
    from src.data.generate_sample_data import generate_sample_data, plot_sample_data
except ImportError as e:
    print(f"ERROR: Could not import generate_sample_data module - {e}")
    print("Make sure you're running the script from the project root directory.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate sample data for HTS Tape Manufacturing Optimization System')
    
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
    
    return parser.parse_args()


def ensure_directories_exist():
    """Ensure necessary directories exist."""
    for directory in ['data', 'models', 'plots']:
        if not os.path.exists(directory):
            print(f"Creating {directory} directory...")
            os.makedirs(directory)


def main():
    """Main function to generate sample data."""
    args = parse_arguments()
    
    # Ensure necessary directories exist
    ensure_directories_exist()
    
    print("=" * 80)
    print("HTS Tape Manufacturing Optimization System - Sample Data Generator")
    print("=" * 80)
    
    # Generate default output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"data/sample_hts_data_{timestamp}.csv"
    
    print(f"Generating sample data with the following parameters:")
    print(f"  Length: {args.length} cm")
    print(f"  Step size: {args.step_size} cm")
    print(f"  Window size: {args.window_size} points")
    print(f"  Dropout count: {args.dropout_count}")
    print(f"  Base critical current: {args.base_ic} A")
    print(f"  Noise level: {args.noise_level}")
    print(f"  Output file: {args.output}")
    
    try:
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
        
        print(f"\nGenerated sample data with {len(data)} points")
        print(f"Critical current range: {data['Ic'].min():.2f} - {data['Ic'].max():.2f} A")
        print(f"Average CV: {cv_values.mean():.4f}")
        print(f"Data saved to {args.output}")
        
        # Plot the data if requested
        if args.plot:
            print("\nPlotting generated data...")
            try:
                plot_sample_data(data, cv_positions, cv_values)
            except Exception as e:
                print(f"WARNING: Could not plot data: {e}")
                print("Continuing without plotting.")
                
    except Exception as e:
        print(f"\nERROR: Failed to generate sample data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 