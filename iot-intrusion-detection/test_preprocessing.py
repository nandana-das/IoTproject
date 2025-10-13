#!/usr/bin/env python3
"""
Test preprocessing pipeline.
"""

import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import IoTDataLoader
from preprocessor import IoTPreprocessor

def main():
    print("Testing Preprocessing Pipeline")
    print("=" * 40)
    
    # Load data
    print("1. Loading data...")
    loader = IoTDataLoader("data/raw")
    data = loader.load_data()
    print(f"   Dataset shape: {data.shape}")
    
    # Test preprocessing
    print("2. Testing preprocessing...")
    preprocessor = IoTPreprocessor()
    X, y = preprocessor.prepare_data(data)
    
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Show class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"   Class distribution:")
    for class_name, count in zip(unique_classes, counts):
        percentage = (count / len(y)) * 100
        print(f"     {class_name:15s}: {count:8,} ({percentage:5.1f}%)")
    
    print("\nPreprocessing test completed successfully!")
    print("Ready to run the complete pipeline!")

if __name__ == "__main__":
    main()
