#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check Environment for HTS Tape Manufacturing Optimization System.

This script checks if all required dependencies are installed and
the environment is properly set up to run the HTS system.
"""

import os
import sys
import importlib
import platform
from datetime import datetime

# List of required packages
REQUIRED_PACKAGES = [
    # Data processing and analysis
    'numpy', 'pandas', 'scipy', 'sklearn', 'statsmodels',
    
    # Deep Learning
    'tensorflow', 'torch', 'pyspark',
    
    # Visualization and UI
    'matplotlib', 'seaborn', 'plotly', 'dash', 'dash_bootstrap_components',
    
    # Reinforcement Learning
    'gymnasium', 'stable_baselines3',
    
    # R integration
    'rpy2',
    
    # Utilities
    'tqdm', 'joblib', 'pyarrow'
]

# Directories that should exist
REQUIRED_DIRECTORIES = [
    'data', 'models', 'plots',
    'src', 'src/data', 'src/models', 'src/ui', 'src/utils',
    'tests', 'docs'
]

# Files that should exist
REQUIRED_FILES = [
    'requirements.txt', 'README.md',
    'src/__init__.py', 'src/data/__init__.py', 
    'src/models/__init__.py', 'src/ui/__init__.py', 
    'src/utils/__init__.py',
    'src/data/data_processor.py', 'src/models/nfq_controller.py',
    'src/ui/dashboard.py', 'src/utils/common.py',
    'src/data/generate_sample_data.py', 'src/main.py'
]

def check_package(package_name):
    """Check if a package is installed."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_directory(directory):
    """Check if a directory exists."""
    return os.path.isdir(directory)

def check_file(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False

def format_check_result(name, result):
    """Format the result of a check."""
    if result:
        return f"✅ {name}"
    else:
        return f"❌ {name}"

def main():
    """Main function to check the environment."""
    print("=" * 80)
    print("HTS Tape Manufacturing Optimization System - Environment Check")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {platform.python_version()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print("=" * 80)
    
    # Check packages
    print("\nChecking required packages:")
    missing_packages = []
    for package in REQUIRED_PACKAGES:
        result = check_package(package)
        print(f"  {format_check_result(package, result)}")
        if not result:
            missing_packages.append(package)
    
    # Check directories
    print("\nChecking required directories:")
    missing_directories = []
    for directory in REQUIRED_DIRECTORIES:
        result = check_directory(directory)
        print(f"  {format_check_result(directory, result)}")
        if not result:
            missing_directories.append(directory)
    
    # Check files
    print("\nChecking required files:")
    missing_files = []
    for file_path in REQUIRED_FILES:
        result = check_file(file_path)
        print(f"  {format_check_result(file_path, result)}")
        if not result:
            missing_files.append(file_path)
    
    # Summary
    print("\nSummary:")
    if missing_packages:
        print(f"  ❌ Missing {len(missing_packages)} package(s):")
        for package in missing_packages:
            print(f"    - {package}")
        print("\n  To install missing packages, run:")
        print("  pip install -r requirements.txt")
        print("\n  Or install them individually:")
        print(f"  pip install {' '.join(missing_packages)}")
    else:
        print("  ✅ All required packages are installed.")
    
    if missing_directories:
        print(f"\n  ❌ Missing {len(missing_directories)} directory(ies):")
        for directory in missing_directories:
            print(f"    - {directory}")
        
        print("\n  Do you want to create the missing directories? (y/n)")
        response = input("  > ")
        if response.lower() == 'y':
            for directory in missing_directories:
                if create_directory(directory):
                    print(f"    ✅ Created {directory}")
                else:
                    print(f"    ❌ Failed to create {directory}")
    else:
        print("  ✅ All required directories exist.")
    
    if missing_files:
        print(f"\n  ❌ Missing {len(missing_files)} file(s):")
        for file_path in missing_files:
            print(f"    - {file_path}")
        print("\n  Please recreate or download these files.")
    else:
        print("  ✅ All required files exist.")
    
    # Overall status
    if not missing_packages and not missing_directories and not missing_files:
        print("\n✅ Environment is ready for running the HTS system.")
    else:
        print("\n❌ Environment issues need to be resolved before running the HTS system.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCheck interrupted by user.")
    except Exception as e:
        print(f"\nError during environment check: {e}")
        sys.exit(1) 