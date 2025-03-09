#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the HTS Tape Manufacturing Optimization System Dashboard.

This script provides a simple way to start the dashboard for monitoring
and optimizing HTS tape manufacturing processes.
"""

import os
import sys
import argparse

# Check for required dependencies before importing
try:
    import dash
    import dash_bootstrap_components as dbc
    import plotly
    import numpy
    import pandas
except ImportError as e:
    print(f"ERROR: Missing required dependency - {e}")
    print("\nPlease install the required packages using:")
    print("pip install -r requirements.txt")
    print("\nOr install the specific missing package:")
    print(f"pip install {str(e).split()[-1]}")
    sys.exit(1)

try:
    from src.ui.dashboard import HTSMonitoringDashboard
    from src.data.data_processor import DataProcessor
    from src.models.nfq_controller import NFQController
except ImportError as e:
    print(f"ERROR: Could not import HTS system modules - {e}")
    print("Make sure you're running the script from the project root directory.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the HTS Tape Manufacturing Optimization System Dashboard')
    
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the HTS data file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a pre-trained model')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for the dashboard server')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    return parser.parse_args()


def ensure_directories_exist():
    """Ensure necessary directories exist."""
    for directory in ['data', 'models', 'plots']:
        if not os.path.exists(directory):
            print(f"Creating {directory} directory...")
            os.makedirs(directory)


def main():
    """Main function to run the dashboard."""
    args = parse_arguments()
    
    # Ensure necessary directories exist
    ensure_directories_exist()
    
    print("=" * 80)
    print("HTS Tape Manufacturing Optimization System Dashboard")
    print("=" * 80)
    
    # Setup data processor if data file provided
    data_processor = None
    if args.data:
        if not os.path.exists(args.data):
            print(f"ERROR: Data file not found: {args.data}")
            print("Please generate sample data first using generate_data.py")
            sys.exit(1)
            
        print(f"Loading data from {args.data}...")
        data_processor = DataProcessor(data_path=args.data)
        try:
            data_processor.process()
        except Exception as e:
            print(f"ERROR: Could not process data: {e}")
            sys.exit(1)
    
    # Setup controller if model file provided
    controller = None
    if args.model:
        if not os.path.exists(args.model):
            print(f"WARNING: Model file not found: {args.model}")
            print("Continuing without pre-trained model...")
        else:
            print(f"Loading model from {args.model}...")
            try:
                controller = NFQController(model_path=args.model)
            except Exception as e:
                print(f"ERROR: Could not load model: {e}")
                print("Continuing without pre-trained model...")
    
    # Create and run dashboard
    print(f"\nStarting dashboard on port {args.port}...")
    print(f"Dashboard will be available at http://localhost:{args.port}")
    print("\nPress Ctrl+C to stop the server.")
    
    try:
        dashboard = HTSMonitoringDashboard(
            data_processor=data_processor,
            controller=controller
        )
        
        dashboard.run_server(debug=args.debug, port=args.port)
    except Exception as e:
        print(f"\nERROR: Could not start dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 