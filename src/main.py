#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HTS Tape Manufacturing Optimization System - Main Script

This script integrates the data processing, model training, and UI components
to provide a complete system for optimizing HTS tape manufacturing processes.
"""

import os
import argparse
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to path for importing modules
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data.data_processor import DataProcessor
from models.nfq_controller import NFQController
from ui.dashboard import HTSMonitoringDashboard


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='HTS Tape Manufacturing Optimization System')
    
    parser.add_argument('--data', type=str, default=None,
                        help='Path to the HTS data file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to a pre-trained model')
    parser.add_argument('--train', action='store_true',
                        help='Train a new model')
    parser.add_argument('--optimize', action='store_true',
                        help='Run parameter optimization')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=800,
                        help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size in cm for calculating CV')
    parser.add_argument('--step_size', type=int, default=2,
                        help='Step size in cm for moving the window')
    parser.add_argument('--ui', action='store_true',
                        help='Run the UI dashboard')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for the dashboard server')
    
    return parser.parse_args()


def main():
    """Main function for the HTS Tape Manufacturing Optimization System."""
    args = parse_arguments()
    
    print("=" * 80)
    print("HTS Tape Manufacturing Optimization System")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup data processor
    data_processor = DataProcessor(
        data_path=args.data,
        window_size=args.window_size,
        step_size=args.step_size
    )
    
    # Load data if path is provided
    if args.data:
        print(f"\nLoading data from {args.data}...")
        data = data_processor.process()
        if data is None:
            print("Error: Could not load or process data.")
            sys.exit(1)
        
        print(f"Data processed successfully.")
        print(f"Positions shape: {data['positions'].shape}")
        print(f"CV values shape: {data['cv_values'].shape}")
        print(f"PCA components shape: {data['pca_result'].shape}")
        print(f"Training data shape: X={data['X'].shape}, y={data['y'].shape}")
    else:
        data = None
        print("\nNo data file provided. Using sample data for demonstration.")
    
    # Setup controller
    controller = NFQController(model_path=args.model)
    
    # Train model if requested
    if args.train and data:
        print("\nTraining NFQ controller...")
        history = controller.train(
            processed_data=data,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print(f"Training completed. Final loss: {history['loss'][-1]:.6f}")
        
        # Save model
        model_dir = 'models'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, f'nfq_controller_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
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
        
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        history_plot_path = os.path.join(plot_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(history_plot_path)
        print(f"Training history plot saved to {history_plot_path}")
    
    # Run optimization if requested
    if args.optimize and data:
        print("\nRunning parameter optimization...")
        optimization_results = controller.optimize_parameters(data)
        
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
        
        plot_dir = 'plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        optimization_plot_path = os.path.join(plot_dir, f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(optimization_plot_path)
        print(f"Optimization results plot saved to {optimization_plot_path}")
    
    # Run UI dashboard if requested
    if args.ui:
        print("\nStarting UI dashboard...")
        dashboard = HTSMonitoringDashboard(
            data_processor=data_processor,
            controller=controller
        )
        
        print(f"Dashboard available at http://localhost:{args.port}")
        dashboard.run_server(debug=True, port=args.port)
    
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 