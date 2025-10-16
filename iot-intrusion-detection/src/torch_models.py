"""
PyTorch model architectures mirroring the TensorFlow/Keras LSTM-CNN and CNN-LSTM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import yaml


class LSTMCNNTorch(nn.Module):
    """LSTM layers followed by 1D CNN layers, then dense classifier."""

    def __init__(self, input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]):
        super().__init__()
        sequence_length, num_features = input_shape
        cfg = config["lstm_cnn"]

        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=int(cfg["lstm_units_1"]),
            batch_first=True,
            dropout=float(cfg["lstm_dropout"]),
            bidirectional=False,
        )
        self.lstm2 = nn.LSTM(
            input_size=int(cfg["lstm_units_1"]),
            hidden_size=int(cfg["lstm_units_2"]),
            batch_first=True,
            dropout=float(cfg["lstm_dropout"]),
            bidirectional=False,
        )

        conv_filters_1 = int(cfg["conv_filters_1"]) 
        conv_filters_2 = int(cfg["conv_filters_2"]) 
        kernel_size = int(cfg["kernel_size"]) 

        # After LSTMs we have output of shape [B, T, H]; Conv1d expects [B, C, T]
        self.conv1 = nn.Conv1d(in_channels=int(cfg["lstm_units_2"]), out_channels=conv_filters_1, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(conv_filters_1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters_1, out_channels=conv_filters_2, kernel_size=kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(conv_filters_2)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        dense_units = int(cfg["dense_units"]) 
        self.fc = nn.Linear(conv_filters_2, dense_units)
        self.dropout_dense = nn.Dropout(float(cfg["dense_dropout"]))
        self.out = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # To Conv1d: [B, C, T]
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # Global max pool over time -> [B, C, 1] -> [B, C]
        x = self.global_max_pool(x).squeeze(-1)
        x = F.relu(self.fc(x))
        x = self.dropout_dense(x)
        logits = self.out(x)
        return logits


class CNNLSTMTorch(nn.Module):
    """Conv1d blocks followed by LSTM, then dense classifier."""

    def __init__(self, input_shape: Tuple[int, int], num_classes: int, config: Dict[str, Any]):
        super().__init__()
        sequence_length, num_features = input_shape
        cfg = config["cnn_lstm"]

        conv_filters_1 = int(cfg["conv_filters_1"]) 
        conv_filters_2 = int(cfg["conv_filters_2"]) 
        kernel_size = int(cfg["kernel_size"]) 
        pool_size = int(cfg["pool_size"]) 

        # Input [B, T, F] -> transpose to [B, F, T] for Conv1d
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=conv_filters_1, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(conv_filters_1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)
        self.drop1 = nn.Dropout(float(cfg["conv_dropout"]))

        self.conv2 = nn.Conv1d(in_channels=conv_filters_1, out_channels=conv_filters_2, kernel_size=kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(conv_filters_2)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size)
        self.drop2 = nn.Dropout(float(cfg["conv_dropout"]))

        # LSTM expects [B, T, C]
        self.lstm = nn.LSTM(
            input_size=conv_filters_2,
            hidden_size=int(cfg["lstm_units"]),
            batch_first=True,
            dropout=float(cfg["lstm_dropout"]),
            bidirectional=False,
        )
        self.drop_lstm = nn.Dropout(float(cfg["lstm_dropout"]))

        dense_units = int(cfg["dense_units"]) 
        self.fc = nn.Linear(int(cfg["lstm_units"]), dense_units)
        self.drop_dense = nn.Dropout(float(cfg["dense_dropout"]))
        self.out = nn.Linear(dense_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        # Back to [B, T, C]
        x = x.transpose(1, 2)
        x, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last = h_n[-1]
        x = self.drop_lstm(last)
        x = F.relu(self.fc(x))
        x = self.drop_dense(x)
        logits = self.out(x)
        return logits


def build_torch_model(model_name: str, input_shape: Tuple[int, int], num_classes: int, config_path: str = "config.yaml") -> nn.Module:
    """Factory to build a torch model by name using YAML config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = model_name.lower()
    if model_name in ("lstm-cnn", "lstm_cnn"):
        return LSTMCNNTorch(input_shape, num_classes, config)
    if model_name in ("cnn-lstm", "cnn_lstm"):
        return CNNLSTMTorch(input_shape, num_classes, config)
    raise ValueError(f"Unknown model_name: {model_name}")
