#!/usr/bin/env python3
"""
Training script for LSTM-CNN model.
Trains the LSTM-CNN architecture and saves results.
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import IoTDataLoader
from preprocessor import IoTPreprocessor
from lstm_cnn_model import LSTMCnnModel
from train import IoTModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function for LSTM-CNN model."""
    logger.info("Starting LSTM-CNN model training")
    
    try:
        # Step 1: Load raw data
        logger.info("Step 1: Loading raw data")
        data_loader = IoTDataLoader("data/raw")
        data = data_loader.load_data()
        
        # Get dataset info
        info = data_loader.get_data_info()
        logger.info(f"Dataset loaded: {info['total_records']:,} records, {info['total_features']} features")
        
        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data")
        preprocessor = IoTPreprocessor()
        metadata = preprocessor.process_full_pipeline(data, "data/processed")
        logger.info(f"Preprocessing completed: {metadata}")
        
        # Step 3: Load processed data
        logger.info("Step 3: Loading processed data")
        X_train = np.load("data/processed/X_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_val = np.load("data/processed/y_val.npy")
        
        logger.info(f"Processed data loaded:")
        logger.info(f"  Train: {X_train.shape} -> {y_train.shape}")
        logger.info(f"  Validation: {X_val.shape} -> {y_val.shape}")
        
        # Step 4: Build model
        logger.info("Step 4: Building LSTM-CNN model")
        model_builder = LSTMCnnModel()
        model = model_builder.build_model(X_train.shape[1:], y_train.shape[1])
        
        # Print model info
        model_info = model_builder.get_model_info()
        logger.info(f"Model built: {model_info['name']}")
        logger.info(f"Total parameters: {model_info['total_params']:,}")
        logger.info(f"Input shape: {model_info['input_shape']}")
        logger.info(f"Output shape: {model_info['output_shape']}")
        
        # Step 5: Train model
        logger.info("Step 5: Training LSTM-CNN model")
        trainer = IoTModelTrainer()
        results = trainer.train_lstm_cnn(X_train, y_train, X_val, y_val)
        
        # Step 6: Print results
        logger.info("Step 6: Training completed successfully!")
        logger.info("="*50)
        logger.info("LSTM-CNN TRAINING RESULTS")
        logger.info("="*50)
        logger.info(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
        logger.info(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        logger.info(f"Training Time: {results['training_time_minutes']:.1f} minutes")
        logger.info(f"Total Epochs: {results['total_epochs']}")
        logger.info(f"Best Epoch: {results['best_epoch']}")
        
        # Save model summary
        model_summary = model_builder.get_model_summary()
        with open("results/reports/lstm_cnn_model_summary.txt", "w") as f:
            f.write("LSTM-CNN Model Summary\n")
            f.write("="*30 + "\n\n")
            f.write(model_summary)
            f.write(f"\n\nModel Information:\n")
            f.write(f"Total Parameters: {model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {model_info['trainable_params']:,}\n")
            f.write(f"Non-trainable Parameters: {model_info['non_trainable_params']:,}\n")
            f.write(f"Input Shape: {model_info['input_shape']}\n")
            f.write(f"Output Shape: {model_info['output_shape']}\n")
        
        logger.info("LSTM-CNN training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
