#!/usr/bin/env python3
"""
Complete pipeline (PyTorch): preprocess -> train (both) -> evaluate (both) -> save results.
"""
import os
import sys
import logging
from pathlib import Path
import numpy as np

# Add src directory to path for module imports
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import IoTDataLoader
from preprocessor import IoTPreprocessor
from src import resolve_config_path
from torch_models import build_torch_model
from torch_train import train_torch_model
from torch_evaluate import IoTEvaluatorTorch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_pipeline_torch():
    logger.info("Starting Complete IoT Pipeline (PyTorch)")

    cfg_path = resolve_config_path()

    # Step 1 & 2: Load + Preprocess (skip if processed exists or SKIP_PREPROCESS=1)
    processed_dir = Path("data/processed")
    processed_exists = processed_dir.exists() and all(
        (processed_dir / f).exists() for f in [
            'X_train.npy', 'X_val.npy', 'y_train.npy', 'y_val.npy'
        ]
    )
    skip_pre = os.environ.get('SKIP_PREPROCESS', '0') == '1'

    if processed_exists or skip_pre:
        logger.info("Skipping raw data load and preprocessing (using existing processed data)")
    else:
        logger.info("Step 1: Loading BoT-IoT Dataset")
        data_loader = IoTDataLoader("data/raw")
        data = data_loader.load_data()
        info = data_loader.get_data_info()
        logger.info(f"Dataset loaded: {info['total_records']:,} records")

        logger.info("Step 2: Preprocessing Data")
        preprocessor = IoTPreprocessor(cfg_path)
        _metadata = preprocessor.process_full_pipeline(data, "data/processed")
        logger.info("Preprocessing completed")

    # Step 3: Training (both models)
    logger.info("Step 3: Training Both Models (PyTorch)")

    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")

    # Optional subsampling via env
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

    input_shape = X_train.shape[1:]
    num_classes = y_train.shape[1] if y_train.ndim == 2 else int(np.max(y_train) + 1)

    # LSTM-CNN
    logger.info("Training LSTM-CNN (PyTorch)")
    model_lstm_cnn = build_torch_model("lstm_cnn", input_shape, num_classes, cfg_path)
    lstm_cnn_results = train_torch_model(model_lstm_cnn, X_train, y_train, X_val, y_val, "LSTM-CNN", cfg_path)
    logger.info(f"LSTM-CNN Best Val Acc: {lstm_cnn_results['best_val_accuracy']:.4f}")

    # CNN-LSTM
    logger.info("Training CNN-LSTM (PyTorch)")
    model_cnn_lstm = build_torch_model("cnn_lstm", input_shape, num_classes, cfg_path)
    cnn_lstm_results = train_torch_model(model_cnn_lstm, X_train, y_train, X_val, y_val, "CNN-LSTM", cfg_path)
    logger.info(f"CNN-LSTM Best Val Acc: {cnn_lstm_results['best_val_accuracy']:.4f}")

    # Step 4: Evaluation (both models)
    logger.info("Step 4: Evaluating Both Models (PyTorch)")
    evaluator = IoTEvaluatorTorch(cfg_path)
    lstm_cnn_metrics, cnn_lstm_metrics = evaluator.evaluate_both_models("data/processed")

    logger.info("PIPELINE COMPLETED SUCCESSFULLY (PyTorch)!")
    logger.info(f"  LSTM-CNN Test Accuracy: {lstm_cnn_metrics['overall']['accuracy']:.4f}")
    logger.info(f"  CNN-LSTM Test Accuracy: {cnn_lstm_metrics['overall']['accuracy']:.4f}")


def main():
    print("IoT Intrusion Detection (PyTorch)")
    print("LSTM-CNN vs CNN-LSTM - Complete Pipeline")
    print("=" * 60)

    # Basic dataset checks
    if not Path("data/raw").exists():
        print("Error: data/raw directory not found! Place the dataset CSV there.")
        return

    csv_files = list(Path("data/raw").glob("UNSW_2018_IoT_Botnet_Final_10_Best.csv"))
    if not csv_files and os.environ.get('SKIP_PREPROCESS', '0') != '1':
        print("Error: No expected CSV files found in data/raw/")
        return

    run_complete_pipeline_torch()


if __name__ == "__main__":
    main()
