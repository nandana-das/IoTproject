"""
Data loading utilities for IoT intrusion detection system.
Handles loading and combining BoT-IoT dataset files.
"""

import pandas as pd
import glob
import os
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTDataLoader:
    """Data loader for BoT-IoT dataset."""
    
    def __init__(self, data_path: str, file_pattern: str = "UNSW_2018_IoT_Botnet_Final_10_Best.csv"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to directory containing CSV files
            file_pattern: Pattern to match CSV files
        """
        self.data_path = data_path
        self.file_pattern = file_pattern
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and combine all CSV files matching the pattern.
        
        Returns:
            Combined DataFrame with all data
        """
        logger.info(f"Loading data from {self.data_path} with pattern '{self.file_pattern}'")
        
        # Find matching CSV files
        csv_files = glob.glob(os.path.join(self.data_path, self.file_pattern))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching pattern '{self.file_pattern}' in {self.data_path}")
        
        logger.info(f"Found {len(csv_files)} CSV files: {', '.join(map(os.path.basename, csv_files))}")
        
        # Load and combine all files
        dataframes = []
        for file in csv_files:
            logger.info(f"Loading {file}")
            df = pd.read_csv(file, sep=';', on_bad_lines='skip')
            dataframes.append(df)
            logger.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
        
        # Combine all dataframes
        if dataframes:
            self.data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined dataset shape: {self.data.shape}")
        else:
            self.data = pd.DataFrame()
            logger.warning("No dataframes to combine.")
        
        return self.data
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the loaded dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'total_records': len(self.data),
            'total_features': len(self.data.columns),
            'feature_names': list(self.data.columns),
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
        
        # Check if target column exists
        if 'category' in self.data.columns:
            info['class_distribution'] = self.data['category'].value_counts().to_dict()
            info['num_classes'] = len(self.data['category'].unique())
        
        return info
    
    def get_sample_data(self, n_samples: int = 5) -> pd.DataFrame:
        """
        Get a sample of the data for inspection.
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Sample DataFrame
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data.sample(n=n_samples, random_state=42)
    
    def save_data_info(self, output_path: str):
        """
        Save dataset information to a text file.
        
        Args:
            output_path: Path to save the info file
        """
        if self.data is None or self.data.empty:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = self.get_data_info()
        
        with open(output_path, 'w') as f:
            f.write("IoT Intrusion Detection Dataset Information\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Records: {info['total_records']:,}\n")
            f.write(f"Total Features: {info['total_features']}\n")
            f.write(f"Memory Usage: {info['memory_usage']:.2f} MB\n\n")
            
            f.write("Feature Names:\n")
            for i, feature in enumerate(info['feature_names'], 1):
                f.write(f"{i:2d}. {feature}\n")
            
            f.write(f"\nData Types:\n")
            for feature, dtype in info['data_types'].items():
                f.write(f"  {feature}: {dtype}\n")
            
            f.write(f"\nMissing Values:\n")
            for feature, missing in info['missing_values'].items():
                f.write(f"  {feature}: {missing}\n")
            
            if 'class_distribution' in info:
                f.write(f"\nClass Distribution:\n")
                for class_name, count in info['class_distribution'].items():
                    percentage = (count / info['total_records']) * 100
                    f.write(f"  {class_name}: {count:,} ({percentage:.2f}%)\n")
        
        logger.info(f"Dataset information saved to {output_path}")


def load_iot_dataset(data_path: str, file_pattern: str = "UNSW_2018_IoT_Botnet_Final_10_Best.csv") -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to load IoT dataset and return data with info.
    
    Args:
        data_path: Path to directory containing CSV files
        file_pattern: Pattern to match CSV files
        
    Returns:
        Tuple of (combined_dataframe, dataset_info)
    """
    loader = IoTDataLoader(data_path, file_pattern)
    data = loader.load_data()
    info = loader.get_data_info()
    
    return data, info


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw"
    # Set the correct file pattern for your dataset
    loader = IoTDataLoader(data_path, file_pattern="UNSW_2018_IoT_Botnet_Final_10_Best.csv")
    
    try:
        data = loader.load_data()
        info = loader.get_data_info()
        
        print("Dataset loaded successfully!")
        print(f"Shape: {data.shape}")
        print(f"Features: {info['feature_names']}")
        print(f"Classes: {info.get('num_classes', 'Unknown')}")
        
        # Save info to file
        loader.save_data_info("data_info.txt")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV files are in the data/raw directory.")