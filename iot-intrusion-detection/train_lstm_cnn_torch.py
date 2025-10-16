#!/usr/bin/env python3
"""
PyTorch training script for LSTM-CNN model to mirror Keras pipeline outputs.
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
from src import resolve_config_path
from torch_models import build_torch_model
from torch_train import train_torch_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting PyTorch LSTM-CNN training")

    # Step 1 & 2: Load + Preprocess (skip if processed exists or SKIP_PREPROCESS=1)
    processed_dir = Path("data/processed")
    processed_exists = all((processed_dir / f).exists() for f in [
        "X_train.npy", "X_val.npy", "y_train.npy", "y_val.npy"
    ])
    skip_pre = os.environ.get("SKIP_PREPROCESS", "0") == "1"

    cfg_path = resolve_config_path()

    if processed_exists or skip_pre:
        logger.info("Skipping raw data load and preprocessing (using existing processed data)")
    else:
        logger.info("Step 1: Loading raw data")
        data_loader = IoTDataLoader("data/raw")
        data = data_loader.load_data()
        info = data_loader.get_data_info()
        logger.info(f"Dataset loaded: {info['total_records']:,} records, {info['total_features']} features")

        logger.info("Step 2: Preprocessing data")
        preprocessor = IoTPreprocessor(cfg_path)
        _metadata = preprocessor.process_full_pipeline(data, "data/processed")
        logger.info("Preprocessing completed")

    # Step 3: Load processed data
    logger.info("Step 3: Loading processed data")
    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Optional subsampling
    train_limit = int(os.environ.get("TRAIN_LIMIT", "0"))
    val_limit = int(os.environ.get("VAL_LIMIT", "0"))
    if train_limit and X_train.shape[0] > train_limit:
        logger.info(f"Subsampling training set from {X_train.shape[0]:,} to {train_limit:,}")
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
    if val_limit and X_val.shape[0] > val_limit:
        logger.info(f"Subsampling validation set from {X_val.shape[0]:,} to {val_limit:,}")
        X_val = X_val[:val_limit]
        y_val = y_val[:val_limit]

    logger.info(f"Processed data loaded: Train {X_train.shape} -> {y_train.shape}, Val {X_val.shape} -> {y_val.shape}")

    # Step 4: Build model
    input_shape = X_train.shape[1:]  # (T, F)
    num_classes = y_train.shape[1] if y_train.ndim == 2 else int(np.max(y_train) + 1)
    model = build_torch_model("lstm_cnn", input_shape, num_classes, cfg_path)

    # Step 5: Train model
    results = train_torch_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        model_name="LSTM-CNN",
        config_path=cfg_path,
    )

    logger.info("Training completed")
    logger.info(f"Best Val Acc: {results['best_val_accuracy']:.4f} at epoch {results['best_epoch']}")


if __name__ == "__main__":
    main()
