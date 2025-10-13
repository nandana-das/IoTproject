"""
Data preprocessing utilities for IoT intrusion detection system.
Handles feature scaling, label encoding, train-test splitting, and sequence generation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import logging
from typing import Tuple, Dict, Any
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTPreprocessor:
    """Preprocessor for IoT intrusion detection data."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = None
        self.label_encoder = None
        self.class_weights = None
        
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data by separating features and labels.
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Tuple of (features, labels)
        """
        logger.info("Preparing data: separating features and labels")
        
        # Check if 'category' column exists (target variable)
        if 'category' not in data.columns:
            raise ValueError("'category' column not found in data. Please check your dataset.")
        
        # Select 10 best numeric features (excluding index, attack, and categorical columns)
        exclude_columns = ['Unnamed: 0', 'pkSeqID', 'attack', 'category', 'subcategory', 
                          'proto', 'saddr', 'sport', 'daddr', 'dport']
        
        # Get numeric columns and remove excluded ones
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Take first 10 features (they are already ordered by importance in the dataset)
        feature_columns = feature_columns[:10]
        
        logger.info(f"Selected 10 best features: {feature_columns}")
        
        # Separate features and target
        X = data[feature_columns].values
        y = data['category'].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Labels shape: {y.shape}")
        logger.info(f"Feature columns: {feature_columns}")
        
        return X, y
    
    def handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """
        Handle missing values in features.
        
        Args:
            X: Feature array
            
        Returns:
            Feature array with missing values handled
        """
        logger.info("Handling missing values")
        
        # Check for missing values
        missing_mask = np.isnan(X)
        missing_count = np.sum(missing_mask)
        
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values. Using median imputation.")
            
            # Replace NaN with median for each feature
            for i in range(X.shape[1]):
                feature_median = np.nanmedian(X[:, i])
                X[missing_mask[:, i], i] = feature_median
        else:
            logger.info("No missing values found.")
        
        return X
    
    def encode_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Encode string labels to integers.
        
        Args:
            y: String labels
            
        Returns:
            Integer encoded labels
        """
        logger.info("Encoding labels")
        
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Store class names for reference
        self.class_names = self.label_encoder.classes_
        logger.info(f"Class names: {self.class_names}")
        logger.info(f"Class mapping: {dict(zip(self.class_names, range(len(self.class_names))))}")
        
        return y_encoded
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/validation/test sets")
        
        test_size = self.config['preprocessing']['test_size']
        val_size = self.config['preprocessing']['val_size']
        random_state = self.config['preprocessing']['random_state']
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for the first split
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size_adjusted, 
            random_state=random_state, 
            stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]:,} samples")
        logger.info(f"Validation set: {X_val.shape[0]:,} samples")
        logger.info(f"Test set: {X_test.shape[0]:,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled features (X_train, X_val, X_test)
        """
        logger.info("Scaling features using StandardScaler")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: 2D feature array (samples, features)
            y: 1D label array (samples,)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        logger.info("Creating sequences for LSTM input")
        
        sequence_length = self.config['sequences']['sequence_length']
        stride = self.config['sequences']['stride']
        
        X_sequences = []
        y_sequences = []
        
        # Create sliding window sequences
        for i in range(0, len(X) - sequence_length + 1, stride):
            X_sequences.append(X[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length - 1])  # Use label of last timestep
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        logger.info(f"Created sequences: {X_sequences.shape}")
        logger.info(f"Sequence shape: (batch_size, {sequence_length}, {X.shape[1]})")
        
        return X_sequences, y_sequences
    
    def one_hot_encode_labels(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert integer labels to one-hot vectors.
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
            
        Returns:
            Tuple of one-hot encoded labels
        """
        logger.info("Converting labels to one-hot encoding")
        
        num_classes = self.config['data']['num_classes']
        
        y_train_onehot = np.eye(num_classes)[y_train]
        y_val_onehot = np.eye(num_classes)[y_val]
        y_test_onehot = np.eye(num_classes)[y_test]
        
        logger.info(f"One-hot encoded labels shape: {y_train_onehot.shape}")
        
        return y_train_onehot, y_val_onehot, y_test_onehot
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for handling class imbalance.
        
        Args:
            y_train: Training labels
            
        Returns:
            Dictionary of class weights
        """
        logger.info("Computing class weights")
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        self.class_weights = dict(zip(np.unique(y_train), class_weights))
        logger.info(f"Class weights: {self.class_weights}")
        
        return self.class_weights
    
    def save_preprocessing_objects(self, output_path: str):
        """
        Save scaler and label encoder for later use.
        
        Args:
            output_path: Path to save objects
        """
        logger.info(f"Saving preprocessing objects to {output_path}")
        
        if self.scaler is not None:
            scaler_path = os.path.join(output_path, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")
        
        if self.label_encoder is not None:
            encoder_path = os.path.join(output_path, 'label_encoder.pkl')
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            logger.info(f"Label encoder saved to {encoder_path}")
        
        if self.class_weights is not None:
            weights_path = os.path.join(output_path, 'class_weights.pkl')
            with open(weights_path, 'wb') as f:
                pickle.dump(self.class_weights, f)
            logger.info(f"Class weights saved to {weights_path}")
    
    def save_processed_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                          y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                          output_path: str):
        """
        Save processed data as numpy arrays.
        
        Args:
            X_train, X_val, X_test: Feature arrays
            y_train, y_val, y_test: Label arrays
            output_path: Path to save arrays
        """
        logger.info(f"Saving processed data to {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save feature arrays
        np.save(os.path.join(output_path, 'X_train.npy'), X_train)
        np.save(os.path.join(output_path, 'X_val.npy'), X_val)
        np.save(os.path.join(output_path, 'X_test.npy'), X_test)
        
        # Save label arrays
        np.save(os.path.join(output_path, 'y_train.npy'), y_train)
        np.save(os.path.join(output_path, 'y_val.npy'), y_val)
        np.save(os.path.join(output_path, 'y_test.npy'), y_test)
        
        logger.info("Processed data saved successfully")
    
    def process_full_pipeline(self, data: pd.DataFrame, output_path: str) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            data: Raw DataFrame
            output_path: Path to save processed data and objects
            
        Returns:
            Dictionary with processed data and metadata
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Step 1: Prepare data
        X, y = self.prepare_data(data)
        
        # Step 2: Handle missing values
        X = self.handle_missing_values(X)
        
        # Step 3: Encode labels
        y = self.encode_labels(y)
        
        # Step 4: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 5: Scale features
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)
        
        # Step 6: Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
        
        # Step 7: One-hot encode labels
        y_train_onehot, y_val_onehot, y_test_onehot = self.one_hot_encode_labels(
            y_train_seq, y_val_seq, y_test_seq
        )
        
        # Step 8: Compute class weights
        self.compute_class_weights(y_train_seq)
        
        # Step 9: Save everything
        self.save_processed_data(
            X_train_seq, X_val_seq, X_test_seq,
            y_train_onehot, y_val_onehot, y_test_onehot,
            output_path
        )
        
        self.save_preprocessing_objects(output_path)
        
        # Prepare metadata
        metadata = {
            'train_samples': len(X_train_seq),
            'val_samples': len(X_val_seq),
            'test_samples': len(X_test_seq),
            'sequence_length': self.config['sequences']['sequence_length'],
            'num_features': X.shape[1],
            'num_classes': len(self.class_names),
            'class_names': list(self.class_names),
            'class_weights': self.class_weights
        }
        
        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Final data shapes:")
        logger.info(f"  Train: {X_train_seq.shape} -> {y_train_onehot.shape}")
        logger.info(f"  Validation: {X_val_seq.shape} -> {y_val_onehot.shape}")
        logger.info(f"  Test: {X_test_seq.shape} -> {y_test_onehot.shape}")
        
        return metadata


if __name__ == "__main__":
    # Example usage
    from data_loader import IoTDataLoader
    
    # Load data
    loader = IoTDataLoader("data/raw")
    data = loader.load_data()
    
    # Process data
    preprocessor = IoTPreprocessor()
    metadata = preprocessor.process_full_pipeline(data, "data/processed")
    
    print("Preprocessing completed!")
    print(f"Metadata: {metadata}")
