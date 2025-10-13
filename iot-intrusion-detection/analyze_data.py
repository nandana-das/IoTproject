#!/usr/bin/env python3
"""
Quick data analysis script for the BoT-IoT dataset.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import IoTDataLoader

def main():
    print("Analyzing BoT-IoT Dataset")
    print("=" * 40)
    
    # Load data
    loader = IoTDataLoader("data/raw")
    data = loader.load_data()
    
    print(f"\nDataset Information:")
    print(f"Shape: {data.shape}")
    print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nColumns ({len(data.columns)}):")
    for i, col in enumerate(data.columns):
        print(f"{i+1:2d}. {col}")
    
    print(f"\nTarget Variable (category) Distribution:")
    category_counts = data['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {category:15s}: {count:8,} ({percentage:5.1f}%)")
    
    print(f"\nData Types:")
    print(data.dtypes)
    
    print(f"\nSample Data:")
    print(data.head())
    
    print(f"\nStatistical Summary:")
    print(data.describe())
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print(f"\nNo missing values found")
    
    # Identify numeric features
    numeric_features = data.select_dtypes(include=['number']).columns.tolist()
    print(f"\nNumeric Features ({len(numeric_features)}):")
    for i, feature in enumerate(numeric_features):
        print(f"{i+1:2d}. {feature}")
    
    print(f"\nData analysis completed!")

if __name__ == "__main__":
    main()
