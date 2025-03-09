#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the complete HTS Tape Manufacturing Optimization System.

This script provides a simple way to run the complete HTS Tape Manufacturing
Optimization System, including data processing, model training, optimization,
and the dashboard.
"""

import os
import sys
import argparse
import threading
import time
from datetime import datetime

# Check for required dependencies before importing
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import dash
    import dash_bootstrap_components as dbc
except ImportError as e:
    print(f"ERROR: Missing required dependency - {e}")
    print("\nPlease install the required packages using:")
    print("pip install -r requirements.txt")
    print("\nOr install the specific missing package:")
    print(f"pip install {str(e).split()[-1]}")
    sys.exit(1)

try:
    from src.data.data_processor import DataProcessor
    from src.models.nfq_controller import NFQController
    from src.ui.dashboard import HTSMonitoringDashboard
    from src.data.generate_sample_data import generate_sample_data
except ImportError as e:
    print(f"ERROR: Could not import HTS system modules - {e}")
    print("Make sure you're running the script from the project root directory.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the HTS Tape Manufacturing Optimization System')
    
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the HTS data file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a pre-trained model')
    parser.add_argument('--generate-data', action='store_true',
                        help='Generate sample data if no data file is provided')
    parser.add_argument('--train', action='store_true',
                        help='Train a new model')
    parser.add_argument('--optimize', action='store_true',
                        help='Run parameter optimization')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=800,
                        help='Batch size for training')
    parser.add_argument('--window-size', type=int, default=8,
                        help='Window size in cm for calculating CV')
    parser.add_argument('--step-size', type=int, default=2,
                        help='Step size in cm for moving the window')
    parser.add_argument('--ui', action='store_true',
                        help='Run the UI dashboard')
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


def run_dashboard(data_processor, controller, port, debug):
    """Run the dashboard in a separate thread."""
    try:
        dashboard = HTSMonitoringDashboard(
            data_processor=data_processor,
            controller=controller
        )
        
        dashboard.run_server(debug=debug, port=port)
    except Exception as e:
        print(f"ERROR: Dashboard error: {e}")
        sys.exit(1)


def main():
    """Main function to run the system."""
    args = parse_arguments()
    
    # Ensure necessary directories exist
    ensure_directories_exist()
    
    print("=" * 80)
    print("HTS Tape Manufacturing Optimization System")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generate sample data if requested and no data file provided
    if args.generate_data and not args.data:
        print("\nGenerating sample data...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.data = f"data/sample_hts_data_{timestamp}.csv"
        
        try:
            # Generate the data
            data, _, _ = generate_sample_data(
                length=1000,
                step_size=args.step_size,
                window_size=args.window_size,
                dropout_count=5,
                base_ic=300,
                noise_level=0.1,
                output_file=args.data
            )
            
            print(f"Sample data generated and saved to {args.data}")
        except Exception as e:
            print(f"ERROR: Could not generate sample data: {e}")
            sys.exit(1)
    
    # Setup data processor
    data_processor = DataProcessor(
        data_path=args.data,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    # Load data if path is provided
    processed_data = None
    if args.data:
        if not os.path.exists(args.data):
            print(f"ERROR: Data file not found: {args.data}")
            print("Please generate sample data first with --generate-data flag")
            sys.exit(1)
            
        print(f"\nLoading data from {args.data}...")
        try:
            processed_data = data_processor.process()
            if processed_data is None:
                print("ERROR: Could not process data.")
                sys.exit(1)
            
            print(f"Data processed successfully.")
            print(f"Positions shape: {processed_data['positions'].shape}")
            print(f"CV values shape: {processed_data['cv_values'].shape}")
            print(f"PCA components shape: {processed_data['pca_result'].shape}")
            print(f"Training data shape: X={processed_data['X'].shape}, y={processed_data['y'].shape}")
        except Exception as e:
            print(f"ERROR: Data processing failed: {e}")
            sys.exit(1)
    else:
        print("\nNo data file provided. Using sample data for demonstration.")
        if args.train or args.optimize:
            print("ERROR: Cannot train or optimize without data.")
            print("Please provide a data file with --data or generate sample data with --generate-data")
            sys.exit(1)
    
    # Setup controller
    controller = None
    if args.model:
        if not os.path.exists(args.model):
            print(f"WARNING: Model file not found: {args.model}")
            if args.train:
                print("Will train a new model.")
            else:
                print("Continuing without pre-trained model...")
        else:
            try:
                print(f"\nLoading model from {args.model}...")
                controller = NFQController(model_path=args.model)
            except Exception as e:
                print(f"ERROR: Could not load model: {e}")
                if args.train:
                    print("Will train a new model.")
                    controller = NFQController()
                else:
                    print("Continuing without pre-trained model...")
    else:
        controller = NFQController()
    
    # Train model if requested
    if args.train and processed_data:
        print("\nTraining NFQ controller...")
        try:
            history = controller.train(
                processed_data=processed_data,
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            
            print(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join('models', f'nfq_controller_{timestamp}.h5')
            controller.save_model(model_path)
            print(f"Model saved to {model_path}")
            
            # Plot training history
            plt.figure(figsize=(10, 6))
            plt.plot(history['loss'])
            plt.title('NFQ Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join('plots', f'training_history_{timestamp}.png')
            plt.savefig(plot_path)
            print(f"Training history plot saved to {plot_path}")
        except Exception as e:
            print(f"ERROR: Training failed: {e}")
            if args.optimize or args.ui:
                print("Continuing with untrained model...")
            else:
                sys.exit(1)
    
    # Run optimization if requested
    if args.optimize and processed_data and controller:
        print("\nRunning parameter optimization...")
        try:
            optimization_results = controller.optimize_parameters(processed_data)
            
            original_cv = optimization_results['original_cv']
            optimal_cv = optimization_results['optimal_cv']
            improvement = optimization_results['improvement']
            percentage = (improvement / np.mean(original_cv)) * 100
            
            print(f"Optimization completed.")
            print(f"Original average CV: {np.mean(original_cv):.4f}")
            print(f"Optimized average CV: {np.mean(optimal_cv):.4f}")
            print(f"Improvement: {improvement:.4f} ({percentage:.2f}%)")
            
            # Plot optimization results
            plt.figure(figsize=(12, 6))
            plt.plot(original_cv, 'b-', label='Original CV')
            plt.plot(optimal_cv, 'r-', label='Optimized CV')
            plt.xlabel('Position Index')
            plt.ylabel('Coefficient of Variation (CV)')
            plt.title('CV Comparison: Original vs. Optimized')
            plt.legend()
            plt.grid(True)
            
            # Add average lines
            avg_original = np.mean(original_cv)
            avg_optimal = np.mean(optimal_cv)
            plt.axhline(y=avg_original, color='b', linestyle='--', 
                       label=f'Avg Original: {avg_original:.4f}')
            plt.axhline(y=avg_optimal, color='r', linestyle='--', 
                       label=f'Avg Optimized: {avg_optimal:.4f}')
            
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join('plots', f'optimization_results_{timestamp}.png')
            plt.savefig(plot_path)
            print(f"Optimization results plot saved to {plot_path}")
        except Exception as e:
            print(f"ERROR: Optimization failed: {e}")
            if args.ui:
                print("Continuing without optimization results...")
            else:
                sys.exit(1)
    
    # Run UI dashboard if requested
    if args.ui:
        print(f"\nStarting dashboard on port {args.port}...")
        print(f"Dashboard will be available at http://localhost:{args.port}")
        print("\nPress Ctrl+C to stop the server.")
        
        # Run dashboard in main thread
        run_dashboard(data_processor, controller, args.port, args.debug)
    
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 