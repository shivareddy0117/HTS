#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the DataProcessor class.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_processor import DataProcessor
from src.data.generate_sample_data import generate_sample_data


class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Generate sample data
        self.data, self.cv_positions, self.cv_values = generate_sample_data(
            length=100,  # Short tape for testing
            step_size=2,
            window_size=8,
            dropout_count=2,
            base_ic=300,
            noise_level=0.1,
            output_file=None  # Don't save to file
        )
        
        # Create a temporary CSV file
        self.temp_file = 'tests/temp_data.csv'
        self.data.to_csv(self.temp_file, index=False)
        
        # Create DataProcessor instance
        self.processor = DataProcessor(
            data_path=self.temp_file,
            window_size=8,
            step_size=2
        )
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
    
    def test_load_data(self):
        """Test loading data from file."""
        data = self.processor.load_data()
        
        self.assertIsNotNone(data)
        self.assertEqual(len(data), len(self.data))
        self.assertIn('Ic', data.columns)
        self.assertIn('Position', data.columns)
    
    def test_calculate_dynamic_cv(self):
        """Test calculation of dynamic CV."""
        # Load data first
        self.processor.load_data()
        
        # Calculate CV
        positions, cv_values = self.processor.calculate_dynamic_cv()
        
        self.assertIsNotNone(positions)
        self.assertIsNotNone(cv_values)
        self.assertTrue(len(positions) > 0)
        self.assertEqual(len(positions), len(cv_values))
        
        # Check CV values are reasonable
        self.assertTrue(np.all(cv_values >= 0))  # CV should be non-negative
        self.assertTrue(np.all(cv_values < 1))   # CV should be less than 1 for our test data
    
    def test_apply_pca(self):
        """Test PCA application to process parameters."""
        # Load data first
        self.processor.load_data()
        
        # Apply PCA
        pca_result, pca_model, explained_variance = self.processor.apply_pca()
        
        self.assertIsNotNone(pca_result)
        self.assertIsNotNone(pca_model)
        self.assertIsNotNone(explained_variance)
        
        # Check dimensions
        self.assertEqual(pca_result.shape[0], len(self.data))
        self.assertTrue(pca_result.shape[1] <= self.processor.process_parameters.shape[1])
        
        # Check explained variance
        self.assertTrue(np.sum(explained_variance) <= 1.0)
        self.assertTrue(np.sum(explained_variance) > 0.9)  # Should explain at least 90% of variance
    
    def test_prepare_training_data(self):
        """Test preparation of training data."""
        # Load data first
        self.processor.load_data()
        
        # Calculate CV
        positions, cv_values = self.processor.calculate_dynamic_cv()
        
        # Apply PCA
        pca_result, _, _ = self.processor.apply_pca()
        
        # Prepare training data
        X, y = self.processor.prepare_training_data(pca_result, cv_values)
        
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertTrue(len(X) > 0)
        self.assertEqual(len(X), len(y))
        
        # Check dimensions
        self.assertTrue(X.shape[1] > pca_result.shape[1])  # Should include previous time steps
    
    def test_process(self):
        """Test the complete processing pipeline."""
        processed_data = self.processor.process()
        
        self.assertIsNotNone(processed_data)
        self.assertIn('positions', processed_data)
        self.assertIn('cv_values', processed_data)
        self.assertIn('pca_result', processed_data)
        self.assertIn('pca_model', processed_data)
        self.assertIn('X', processed_data)
        self.assertIn('y', processed_data)
        
        # Check data shapes
        self.assertTrue(len(processed_data['positions']) > 0)
        self.assertEqual(len(processed_data['positions']), len(processed_data['cv_values']))
        self.assertEqual(processed_data['pca_result'].shape[0], len(self.data))
        self.assertTrue(len(processed_data['X']) > 0)
        self.assertEqual(len(processed_data['X']), len(processed_data['y']))


if __name__ == '__main__':
    unittest.main() 