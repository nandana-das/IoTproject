"""
LSTM-CNN model architecture for IoT intrusion detection.
Based on the base paper architecture with LSTM layers followed by CNN layers.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMCnnModel(nn.Module):
    """LSTM-CNN model for IoT intrusion detection."""
    
    def __init__(self, input_shape: tuple, num_classes: int, config_path: str = "config.yaml"):
        """
        Initialize LSTM-CNN model.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes
            config_path: Path to configuration file
        """
        super(LSTMCnnModel, self).__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['lstm_cnn']
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Extract dimensions
        sequence_length, num_features = input_shape
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=self.model_config['lstm_units_1'],
            batch_first=True,
            dropout=self.model_config['lstm_dropout'] if self.model_config['lstm_dropout'] > 0 else 0
        )
        self.lstm1_dropout = nn.Dropout(self.model_config['lstm_dropout'])
        
        self.lstm2 = nn.LSTM(
            input_size=self.model_config['lstm_units_1'],
            hidden_size=self.model_config['lstm_units_2'],
            batch_first=True,
            dropout=self.model_config['lstm_dropout'] if self.model_config['lstm_dropout'] > 0 else 0
        )
        self.lstm2_dropout = nn.Dropout(self.model_config['lstm_dropout'])
        
        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=self.model_config['lstm_units_2'],
            out_channels=self.model_config['conv_filters_1'],
            kernel_size=self.model_config['kernel_size'],
            padding='same'
        )
        self.bn1 = nn.BatchNorm1d(self.model_config['conv_filters_1'])
        
        self.conv2 = nn.Conv1d(
            in_channels=self.model_config['conv_filters_1'],
            out_channels=self.model_config['conv_filters_2'],
            kernel_size=self.model_config['kernel_size'],
            padding='same'
        )
        self.bn2 = nn.BatchNorm1d(self.model_config['conv_filters_2'])
        
        # Global max pooling
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Dense layers
        self.dense = nn.Linear(self.model_config['conv_filters_2'], self.model_config['dense_units'])
        self.dense_dropout = nn.Dropout(self.model_config['dense_dropout'])
        
        # Output layer
        self.output = nn.Linear(self.model_config['dense_units'], num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First LSTM layer
        x, _ = self.lstm1(x)
        x = self.lstm1_dropout(x)
        
        # Second LSTM layer
        x, _ = self.lstm2(x)
        x = self.lstm2_dropout(x)
        
        # Conv1D expects: (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        
        # First CNN block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second CNN block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Global max pooling
        x = self.global_max_pool(x)
        x = x.squeeze(-1)
        
        # Dense layers
        x = self.dense(x)
        x = self.relu(x)
        x = self.dense_dropout(x)
        
        # Output layer
        x = self.output(x)
        
        return x
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string.
        
        Returns:
            Model summary string
        """
        from torchinfo import summary
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        summary(self, input_size=(1, *self.input_shape))
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def get_model_info(self) -> dict:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        info = {
            'name': self.model_config['name'],
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'input_shape': self.input_shape,
            'output_shape': (self.num_classes,),
            'config': self.model_config
        }
        
        return info
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        logger.info(f"Saving LSTM-CNN model to {filepath}")
        
        # Change .h5 to .pth if needed
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.pth')
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'config': self.model_config
        }, filepath)
        logger.info("Model saved successfully")
    
    @classmethod
    def load_model(cls, filepath: str, config_path: str = "config.yaml"):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
            config_path: Path to configuration file
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading LSTM-CNN model from {filepath}")
        
        # Change .h5 to .pth if needed
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.pth')
        
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = cls(
            input_shape=checkpoint['input_shape'],
            num_classes=checkpoint['num_classes'],
            config_path=config_path
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
        return model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions array
        """
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            outputs = self.forward(X)
            # Apply softmax to get probabilities
            probabilities = self.softmax(outputs)
            return probabilities.cpu().numpy()
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Args:
            X: Input data
            
        Returns:
            Class predictions
        """
        predictions = self.predict(X)
        return np.argmax(predictions, axis=1)


def create_lstm_cnn_model(input_shape: tuple, num_classes: int, config_path: str = "config.yaml") -> LSTMCnnModel:
    """
    Convenience function to create LSTM-CNN model.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classes
        config_path: Path to config file
        
    Returns:
        PyTorch LSTM-CNN model
    """
    model = LSTMCnnModel(input_shape, num_classes, config_path)
    logger.info(f"Built LSTM-CNN model with input shape: {input_shape}")
    return model


if __name__ == "__main__":
    # Example usage
    input_shape = (15, 10)  # sequence_length, num_features
    num_classes = 5
    
    # Create model
    lstm_cnn = LSTMCnnModel(input_shape, num_classes)
    
    # Print model info
    info = lstm_cnn.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Input shape: {info['input_shape']}")
    print(f"Output shape: {info['output_shape']}")
    
    # Test forward pass
    dummy_input = torch.randn(2, *input_shape)
    output = lstm_cnn(dummy_input)
    print(f"\nTest forward pass output shape: {output.shape}")
