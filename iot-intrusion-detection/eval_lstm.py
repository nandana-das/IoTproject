#!/usr/bin/env python3
"""
Evaluate the existing LSTM-CNN model on the saved test set and print accuracy.
"""

import sys
from pathlib import Path

# Ensure src is on path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluate import IoTEvaluator  # noqa: E402
from src import resolve_config_path


def main() -> None:
    config_path = Path(resolve_config_path())
    data_path = PROJECT_ROOT / "data" / "processed"
    model_path = PROJECT_ROOT / "models" / "lstm_cnn_best.pth"

    evaluator = IoTEvaluator(str(config_path))
    X_test, y_test = evaluator.load_test_data(str(data_path))
    metrics = evaluator.evaluate_model(str(model_path), "LSTM-CNN", X_test, y_test)
    print(f"accuracy={metrics['overall']['accuracy']:.4f}")


if __name__ == "__main__":
    main()


