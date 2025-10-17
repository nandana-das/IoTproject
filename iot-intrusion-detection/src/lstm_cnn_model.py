"""
LSTM-CNN model architecture for IoT intrusion detection.
Based on the base paper architecture with LSTM layers followed by CNN layers.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMCnnModel:
    """LSTM-CNN model for IoT intrusion detection."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize LSTM-CNN model.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.model_config = self.config['lstm_cnn']
    
    def build_model(self, input_shape: tuple, num_classes: int) -> keras.Model:
        """
        Build LSTM-CNN model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM-CNN model with input shape: {input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='input')
        
        # LSTM layers
        # First LSTM layer
        lstm1 = layers.LSTM(
            units=self.model_config['lstm_units_1'],
            return_sequences=self.model_config['return_sequences'],
            dropout=self.model_config['lstm_dropout'],
            name='lstm_1'
        )(inputs)
        
        # Dropout after first LSTM
        lstm1_dropout = layers.Dropout(
            rate=self.model_config['lstm_dropout'],
            name='lstm_1_dropout'
        )(lstm1)
        
        # Second LSTM layer
        lstm2 = layers.LSTM(
            units=self.model_config['lstm_units_2'],
            return_sequences=self.model_config['return_sequences'],
            dropout=self.model_config['lstm_dropout'],
            name='lstm_2'
        )(lstm1_dropout)
        
        # Dropout after second LSTM
        lstm2_dropout = layers.Dropout(
            rate=self.model_config['lstm_dropout'],
            name='lstm_2_dropout'
        )(lstm2)
        
        # CNN layers
        # First Conv1D layer
        conv1 = layers.Conv1D(
            filters=self.model_config['conv_filters_1'],
            kernel_size=self.model_config['kernel_size'],
            activation='relu',
            padding='same',
            name='conv1d_1'
        )(lstm2_dropout)
        
        # Batch normalization after first conv
        conv1_bn = layers.BatchNormalization(name='conv1d_1_bn')(conv1)
        
        # Second Conv1D layer
        conv2 = layers.Conv1D(
            filters=self.model_config['conv_filters_2'],
            kernel_size=self.model_config['kernel_size'],
            activation='relu',
            padding='same',
            name='conv1d_2'
        )(conv1_bn)
        
        # Batch normalization after second conv
        conv2_bn = layers.BatchNormalization(name='conv1d_2_bn')(conv2)
        
        # Global max pooling
        global_pool = layers.GlobalMaxPooling1D(name='global_max_pool')(conv2_bn)
        
        # Dense layers
        dense = layers.Dense(
            units=self.model_config['dense_units'],
            activation='relu',
            name='dense'
        )(global_pool)
        
        # Dropout after dense layer
        dense_dropout = layers.Dropout(
            rate=self.model_config['dense_dropout'],
            name='dense_dropout'
        )(dense)
        
        # Output layer
        outputs = layers.Dense(
            units=num_classes,
            activation=self.model_config['output_activation'],
            name='output'
        )(dense_dropout)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.model_config['name'])
        
        # Compile model
        self._compile_model()
        
        return self.model
    
    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        logger.info("Compiling LSTM-CNN model")
        
        optimizer = keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate']
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=self.config['training']['loss'],
            metrics=self.config['training']['metrics']
        )
        
        logger.info("Model compiled successfully")
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def get_model_info(self) -> dict:
        """
        Get detailed model information.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        info = {
            'name': self.model_config['name'],
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'layers_count': len(self.model.layers),
            'config': self.model_config
        }
        
        return info
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        logger.info(f"Saving LSTM-CNN model to {filepath}")
        self.model.save(filepath)
        logger.info("Model saved successfully")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        logger.info(f"Loading LSTM-CNN model from {filepath}")
        self.model = keras.models.load_model(filepath)
        logger.info("Model loaded successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded yet. Call load_model() or build_model() first.")
        
        return self.model.predict(X)
    
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


def create_lstm_cnn_model(input_shape: tuple, num_classes: int, config_path: str = "config.yaml") -> keras.Model:
    """
    Convenience function to create LSTM-CNN model.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of classes
        config_path: Path to config file
        
    Returns:
        Compiled Keras model
    """
    model_builder = LSTMCnnModel(config_path)
    model = model_builder.build_model(input_shape, num_classes)
    return model


if __name__ == "__main__":
    # Example usage
    input_shape = (15, 10)  # sequence_length, num_features
    num_classes = 5
    
    # Create model
    lstm_cnn = LSTMCnnModel()
    model = lstm_cnn.build_model(input_shape, num_classes)
    
    # Print model info
    info = lstm_cnn.get_model_info()
    print(f"Model: {info['name']}")
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Input shape: {info['input_shape']}")
    print(f"Output shape: {info['output_shape']}")
    
    # Print model summary
    print("\nModel Summary:")
    print(lstm_cnn.get_model_summary())
