import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os


class DataProcessor:
    """
    Process and prepare HTS tape manufacturing data for analysis and modeling.
    
    This class handles data loading, preprocessing, feature extraction, and visualization
    of high-temperature superconductor (HTS) tape manufacturing data.
    """
    
    def __init__(self, data_path=None, window_size=8, step_size=2):
        """
        Initialize the DataProcessor.
        
        Args:
            data_path (str): Path to the HTS data files
            window_size (int): Size of moving window in cm for calculating CV_Ic
            step_size (int): Step size in cm for moving the window
        """
        self.data_path = data_path
        self.window_size = window_size
        self.step_size = step_size
        self.data = None
        self.process_parameters = None
        self.critical_current = None
        self.scaler = StandardScaler()
        
    def load_data(self, data_path=None):
        """
        Load HTS tape manufacturing data from file.
        
        Args:
            data_path (str, optional): Path to the data file. If not provided, uses the
                                      path specified during initialization.
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if data_path is not None:
            self.data_path = data_path
            
        if self.data_path is None:
            print("Error: No data path provided.")
            return None
            
        # Verify file exists
        if not os.path.exists(self.data_path):
            print(f"Error: Data file not found: {self.data_path}")
            return None
            
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            
            # Verify required columns exist
            required_columns = ['Position', 'Ic']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            if missing_columns:
                print(f"Warning: Missing required columns: {missing_columns}")
                print("Data may not be processed correctly.")
            
            # Extract process parameters and critical current
            self.process_parameters = self.data.drop(['Position', 'Ic'], axis=1, errors='ignore')
            if 'Ic' in self.data.columns:
                self.critical_current = self.data['Ic'].values
            else:
                print("Warning: 'Ic' column not found in data.")
                
            return self.data
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def calculate_dynamic_cv(self, critical_current=None, position=None):
        """
        Calculate dynamic coefficient of variation (CV) of critical current using moving window.
        
        Args:
            critical_current (array-like, optional): Critical current measurements
            position (array-like, optional): Position measurements in cm
            
        Returns:
            tuple: (positions, cv_values) - Positions and corresponding CV values
        """
        if critical_current is None:
            critical_current = self.critical_current
            
        if position is None:
            if 'Position' in self.data.columns:
                position = self.data['Position'].values
            else:
                position = np.arange(0, len(critical_current) * self.step_size, self.step_size)
        
        # Calculate CV in moving windows
        cv_values = []
        window_positions = []
        
        for i in range(0, len(position) - int(self.window_size / self.step_size)):
            start_idx = i
            end_idx = i + int(self.window_size / self.step_size)
            
            # Get window slice
            window_slice = critical_current[start_idx:end_idx]
            
            # Calculate CV (standard deviation / mean)
            std_dev = np.std(window_slice)
            mean_val = np.mean(window_slice)
            
            if mean_val > 0:  # Avoid division by zero
                cv = std_dev / mean_val
                cv_values.append(cv)
                window_positions.append(position[start_idx])
        
        return np.array(window_positions), np.array(cv_values)
    
    def apply_pca(self, data=None, n_components=10, variance_threshold=0.999):
        """
        Apply Principal Component Analysis to process parameters.
        
        Args:
            data (DataFrame, optional): Process parameter data
            n_components (int): Maximum number of PCA components
            variance_threshold (float): Minimum variance to explain (0.0-1.0)
            
        Returns:
            tuple: (pca_result, pca_model, explained_variance_ratio)
        """
        if data is None:
            data = self.process_parameters
            
        # Standardize the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_data)
        
        # Determine number of components based on variance threshold
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        print(f"Selected {n_components_threshold} PCA components explaining {cumulative_variance[n_components_threshold-1]*100:.2f}% of variance")
        
        # Return only the necessary components
        return pca_result[:, :n_components_threshold], pca, explained_variance
    
    def align_data(self, process_params, critical_current, lag=None):
        """
        Align process parameters with critical current measurements by correcting time lag.
        
        Args:
            process_params (DataFrame): Process parameter data
            critical_current (Series): Critical current measurements
            lag (int, optional): Known time lag value. If None, computes optimal lag.
            
        Returns:
            tuple: (aligned_process_params, aligned_critical_current)
        """
        if lag is None:
            # Compute optimal lag using cross-correlation
            max_lag = min(len(process_params) // 4, 100)  # Limit max lag search
            best_lag = 0
            best_corr = 0
            
            for l in range(-max_lag, max_lag):
                if l < 0:
                    p1 = process_params.iloc[:l].values.mean(axis=1)
                    c1 = critical_current.iloc[-l:].values
                else:
                    p1 = process_params.iloc[l:].values.mean(axis=1)
                    c1 = critical_current.iloc[:-l if l > 0 else None].values
                
                min_len = min(len(p1), len(c1))
                if min_len > 10:  # Ensure enough data points
                    correlation = np.corrcoef(p1[:min_len], c1[:min_len])[0, 1]
                    if not np.isnan(correlation) and abs(correlation) > abs(best_corr):
                        best_corr = correlation
                        best_lag = l
            
            lag = best_lag
            print(f"Optimal lag determined: {lag} with correlation: {best_corr:.3f}")
        
        # Apply lag correction
        if lag < 0:
            aligned_pp = process_params.iloc[:lag].reset_index(drop=True)
            aligned_ic = critical_current.iloc[-lag:].reset_index(drop=True)
        else:
            aligned_pp = process_params.iloc[lag:].reset_index(drop=True)
            aligned_ic = critical_current.iloc[:-lag if lag > 0 else None].reset_index(drop=True)
        
        # Ensure same length
        min_len = min(len(aligned_pp), len(aligned_ic))
        aligned_pp = aligned_pp.iloc[:min_len]
        aligned_ic = aligned_ic.iloc[:min_len]
        
        return aligned_pp, aligned_ic
    
    def prepare_training_data(self, process_params, cv_values, window_size=1):
        """
        Prepare training data for the NFQ model with current and previous time steps.
        
        Args:
            process_params (array-like): Process parameter data (PCA components)
            cv_values (array-like): CV of critical current
            window_size (int): Number of previous time steps to include
            
        Returns:
            tuple: (X, y) - Input features and target values
        """
        X = []
        y = []
        
        for i in range(window_size, len(cv_values)):
            # Current and previous process parameters
            features = []
            for j in range(window_size, 0, -1):
                features.extend(process_params[i-j])
            
            # Current process parameters
            features.extend(process_params[i])
            
            # Previous CV
            features.append(cv_values[i-1])
            
            X.append(features)
            y.append(cv_values[i])
            
        return np.array(X), np.array(y)
    
    def visualize_cv(self, positions, cv_values, title="Dynamic CV of Critical Current"):
        """
        Visualize the dynamic coefficient of variation (CV) of critical current.
        
        Args:
            positions (array-like): Positions along the tape
            cv_values (array-like): CV values
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(positions, cv_values, 'b-')
        plt.xlabel('Position (cm)')
        plt.ylabel('Coefficient of Variation (CV)')
        plt.title(title)
        plt.grid(True)
        
        # Add average CV line
        avg_cv = np.mean(cv_values)
        plt.axhline(y=avg_cv, color='r', linestyle='--', label=f'Average CV: {avg_cv:.4f}')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def process(self, data_path=None):
        """
        Complete data processing pipeline.
        
        Args:
            data_path (str, optional): Path to the data file
            
        Returns:
            dict: Processed data including PCA components and CV values
        """
        # Load data
        if self.data is None:
            self.load_data(data_path)
        
        if self.data is None:
            return None
        
        # Calculate dynamic CV
        positions, cv_values = self.calculate_dynamic_cv()
        
        # Apply PCA to process parameters
        pca_result, pca_model, explained_variance = self.apply_pca()
        
        # Prepare data for modeling
        X, y = self.prepare_training_data(pca_result, cv_values)
        
        processed_data = {
            'positions': positions,
            'cv_values': cv_values,
            'pca_result': pca_result,
            'pca_model': pca_model,
            'X': X,
            'y': y,
            'original_data': self.data,
            'original_process_params': self.process_parameters,
            'original_critical_current': self.critical_current
        }
        
        return processed_data

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor("../../data/sample_hts_data.csv")
    processed_data = processor.process()
    
    if processed_data:
        # Visualize CV
        processor.visualize_cv(processed_data['positions'], processed_data['cv_values'])
        
        print(f"Processed data shape: X={processed_data['X'].shape}, y={processed_data['y'].shape}")
        
        # Show PCA explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(processed_data['pca_model'].explained_variance_ratio_) + 1), 
                processed_data['pca_model'].explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.tight_layout()
        plt.show() 