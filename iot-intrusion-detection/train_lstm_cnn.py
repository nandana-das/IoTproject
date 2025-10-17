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
from src import resolve_config_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function for LSTM-CNN model."""
    logger.info("Starting LSTM-CNN model training")
    
    try:
        # Step 1 & 2: Load + Preprocess raw data (skipped if processed files exist or SKIP_PREPROCESS=1)
        processed_dir = Path("data/processed")
        processed_exists = all((processed_dir / f).exists() for f in [
            "X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"
        ])
        skip_pre = os.environ.get("SKIP_PREPROCESS", "0") == "1"
        if processed_exists or skip_pre:
            logger.info("Skipping raw data load and preprocessing (using existing processed data)")
        else:
            logger.info("Step 1: Loading raw data")
            data_loader = IoTDataLoader("data/raw")
            data = data_loader.load_data()
            info = data_loader.get_data_info()
            logger.info(f"Dataset loaded: {info['total_records']:,} records, {info['total_features']} features")
            
            logger.info("Step 2: Preprocessing data")
            preprocessor = IoTPreprocessor(resolve_config_path())
            metadata = preprocessor.process_full_pipeline(data, "data/processed")
            logger.info("Preprocessing completed")
        
        # Step 3: Load processed data
        logger.info("Step 3: Loading processed data")
        X_train = np.load("data/processed/X_train.npy")
        X_val = np.load("data/processed/X_val.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_val = np.load("data/processed/y_val.npy")

        # Optional: subsample to fit memory constraints
        train_limit = int(os.environ.get("TRAIN_LIMIT", "500000"))
        val_limit = int(os.environ.get("VAL_LIMIT", "100000"))
        if X_train.shape[0] > train_limit:
            logger.info(f"Subsampling training set from {X_train.shape[0]:,} to {train_limit:,}")
            X_train = X_train[:train_limit]
            y_train = y_train[:train_limit]
        if X_val.shape[0] > val_limit:
            logger.info(f"Subsampling validation set from {X_val.shape[0]:,} to {val_limit:,}")
            X_val = X_val[:val_limit]
            y_val = y_val[:val_limit]
        
        logger.info(f"Processed data loaded:")
        logger.info(f"  Train: {X_train.shape} -> {y_train.shape}")
        logger.info(f"  Validation: {X_val.shape} -> {y_val.shape}")
        
        # Step 4: Build model
        logger.info("Step 4: Building LSTM-CNN model")
        model = LSTMCnnModel(X_train.shape[1:], y_train.shape[1])
        
        # Print model info
        model_info = model.get_model_info()
        logger.info(f"Model built: {model_info['name']}")
        logger.info(f"Total parameters: {model_info['total_params']:,}")
        logger.info(f"Input shape: {model_info['input_shape']}")
        logger.info(f"Output shape: {model_info['output_shape']}")
        
        # Step 5: Train model
        logger.info("Step 5: Training LSTM-CNN model")
        trainer = IoTModelTrainer(resolve_config_path())
        # Optionally lower batch size to reduce memory usage
        try:
            desired_bs = int(os.environ.get("BATCH_SIZE", "64"))
            if desired_bs < trainer.training_config['batch_size']:
                logger.info(f"Reducing batch size from {trainer.training_config['batch_size']} to {desired_bs}")
                trainer.training_config['batch_size'] = desired_bs
        except Exception:
            pass
        # Allow overriding epochs for quick runs
        try:
            desired_epochs = int(os.environ.get("EPOCHS", "0"))
            if desired_epochs:
                logger.info(f"Overriding epochs from {trainer.training_config['epochs']} to {desired_epochs}")
                trainer.training_config['epochs'] = desired_epochs
        except Exception:
            pass
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
        model_summary = model.get_model_summary()
        with open("results/reports/lstm_cnn_model_summary.txt", "w", encoding="utf-8") as f:
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
