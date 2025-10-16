#!/usr/bin/env python3
"""
Evaluation script for PyTorch models (LSTM-CNN and CNN-LSTM).
Generates the same outputs as the Keras evaluator, but uses .pt checkpoints.
"""
import logging
from pathlib import Path
import numpy as np
import os

# Add src directory to path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from torch_evaluate import IoTEvaluatorTorch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting PyTorch evaluation for both models")

    evaluator = IoTEvaluatorTorch()

    # Load processed test data
    X_test, y_test = evaluator.load_test_data("data/processed")

    # Evaluate LSTM-CNN
    lstm_cnn_path = os.path.join(evaluator.config['paths']['models'], 'lstm_cnn_best.pt')
    lstm_cnn_metrics = evaluator.evaluate_model(lstm_cnn_path, "LSTM-CNN", X_test, y_test)

    # Evaluate CNN-LSTM
    cnn_lstm_path = os.path.join(evaluator.config['paths']['models'], 'cnn_lstm_best.pt')
    cnn_lstm_metrics = evaluator.evaluate_model(cnn_lstm_path, "CNN-LSTM", X_test, y_test)

    logger.info("PyTorch evaluation completed")
    logger.info(f"LSTM-CNN Accuracy: {lstm_cnn_metrics['overall']['accuracy']:.4f}")
    logger.info(f"CNN-LSTM Accuracy: {cnn_lstm_metrics['overall']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
