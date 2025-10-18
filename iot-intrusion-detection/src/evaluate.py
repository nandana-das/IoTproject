"""
Evaluation utilities for IoT intrusion detection models.
Handles comprehensive evaluation including metrics, confusion matrix, ROC curves, and reports.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import pickle
import yaml
import logging
from typing import Dict, Any, Tuple, List
from itertools import cycle
try:
    # Package import when importing as src.evaluate
    from src import resolve_config_path  # type: ignore
except Exception:
    try:
        from __init__ import resolve_config_path  # type: ignore
    except Exception:
        def resolve_config_path(default: str = "config.yaml") -> str:  # type: ignore
            return default

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTEvaluator:
    """Evaluator for IoT intrusion detection models."""
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize evaluator.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = resolve_config_path()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.paths_config = self.config['paths']
        self.eval_config = self.config['evaluation']
        self.class_names = None
        self.label_encoder = None
        
    def load_test_data(self, data_path: str = "data/processed") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data and preprocessing objects.
        
        Args:
            data_path: Path to processed data
            
        Returns:
            Tuple of (X_test, y_test)
        """
        logger.info("Loading test data and preprocessing objects")
        
        # Load test data
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        # Load label encoder
        with open(os.path.join(data_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.class_names = self.label_encoder.classes_
        
        logger.info(f"Loaded test data: {X_test.shape} -> {y_test.shape}")
        logger.info(f"Class names: {list(self.class_names)}")
        
        return X_test, y_test
    
    def load_model(self, model_path: str) -> keras.Model:
        """
        Load a trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded Keras model
        """
        logger.info(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    
    def make_predictions(self, model: keras.Model, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Make predictions on test data and measure inference time.
        
        Args:
            model: Trained model
            X_test: Test features
            
        Returns:
            Tuple of (predictions_prob, predictions_class, inference_time_ms)
        """
        logger.info("Making predictions on test data")
        
        # Measure inference time
        start_time = time.time()
        predictions_prob = model.predict(X_test, verbose=0)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        inference_time_per_sample = inference_time / len(X_test)
        
        # Get class predictions
        predictions_class = np.argmax(predictions_prob, axis=1)
        
        logger.info(f"Inference time: {inference_time_per_sample:.3f} ms per sample")
        
        return predictions_prob, predictions_class, inference_time_per_sample
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted labels (class indices)
            y_pred_prob: Predicted probabilities
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating evaluation metrics")
        
        # Convert one-hot to class indices
        y_true_class = np.argmax(y_true, axis=1)
        
        # Overall metrics
        accuracy = accuracy_score(y_true_class, y_pred)
        precision_macro = precision_score(y_true_class, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_class, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_class, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_true_class, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_class, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_class, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true_class, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true_class, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_class, y_pred, average=None, zero_division=0)
        
        # ROC-AUC scores
        roc_auc_scores = []
        for i in range(len(self.class_names)):
            if len(np.unique(y_true[:, i])) > 1:  # Check if class exists in test set
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                roc_auc_scores.append(roc_auc)
            else:
                roc_auc_scores.append(0.0)
        
        roc_auc_macro = np.mean(roc_auc_scores)
        
        # Prepare metrics dictionary
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'roc_auc_macro': float(roc_auc_macro)
            },
            'per_class': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1_score': f1_per_class.tolist(),
                'roc_auc': roc_auc_scores
            },
            'class_names': list(self.class_names)
        }
        
        logger.info(f"Overall accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1-score: {f1_macro:.4f}")
        logger.info(f"Macro ROC-AUC: {roc_auc_macro:.4f}")
        
        return metrics
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, output_path: str):
        """
        Create and save confusion matrix visualization.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted labels (class indices)
            model_name: Name of the model
            output_path: Path to save the plot
        """
        logger.info("Creating confusion matrix")
        
        # Convert one-hot to class indices
        y_true_class = np.argmax(y_true, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_class, y_pred)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        # Add percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Calculate and display accuracy
        accuracy = accuracy_score(y_true_class, y_pred)
        plt.figtext(0.02, 0.02, f'Overall Accuracy: {accuracy:.4f}', 
                   fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def create_roc_curves(self, y_true: np.ndarray, y_pred_prob: np.ndarray,
                         model_name: str, output_path: str):
        """
        Create and save ROC curves visualization.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred_prob: Predicted probabilities
            model_name: Name of the model
            output_path: Path to save the plot
        """
        logger.info("Creating ROC curves")
        
        plt.figure(figsize=(10, 8))
        
        # Colors for different classes
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
        
        roc_auc_scores = []
        
        # Plot ROC curve for each class
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if len(np.unique(y_true[:, i])) > 1:  # Check if class exists in test set
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                roc_auc_scores.append(roc_auc)
                
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        
        # Calculate macro-average ROC-AUC
        macro_auc = np.mean(roc_auc_scores)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curves (Macro AUC = {macro_auc:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_path}")
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     model_name: str, output_path: str):
        """
        Generate and save classification report.
        
        Args:
            y_true: True labels (one-hot encoded)
            y_pred: Predicted labels (class indices)
            model_name: Name of the model
            output_path: Path to save the report
        """
        logger.info("Generating classification report")
        
        # Convert one-hot to class indices
        y_true_class = np.argmax(y_true, axis=1)
        
        # Generate report
        report = classification_report(y_true_class, y_pred, 
                                     target_names=self.class_names,
                                     digits=4)
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"{model_name} - Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
            f.write(f"\n\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info(f"Classification report saved to {output_path}")
    
    def save_metrics(self, metrics: Dict[str, Any], model_name: str, 
                    inference_time: float, output_path: str):
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Evaluation metrics dictionary
            model_name: Name of the model
            inference_time: Inference time per sample in ms
            output_path: Path to save metrics
        """
        # Add inference time to metrics
        metrics['inference_time_ms_per_sample'] = float(inference_time)
        metrics['model_name'] = model_name
        
        # Save to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def evaluate_model(self, model_path: str, model_name: str, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model_path: Path to saved model
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name}")
        
        # Load model
        model = self.load_model(model_path)
        
        # Make predictions
        y_pred_prob, y_pred_class, inference_time = self.make_predictions(model, X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred_class, y_pred_prob)
        
        # Create visualizations
        self.create_confusion_matrix(
            y_test, y_pred_class, model_name,
            os.path.join(self.paths_config['plots'], f'confusion_matrix_{model_name.lower().replace("-", "_")}.png')
        )
        
        self.create_roc_curves(
            y_test, y_pred_prob, model_name,
            os.path.join(self.paths_config['plots'], f'roc_curves_{model_name.lower().replace("-", "_")}.png')
        )
        
        # Generate classification report
        self.generate_classification_report(
            y_test, y_pred_class, model_name,
            os.path.join(self.paths_config['reports'], f'classification_report_{model_name.lower().replace("-", "_")}.txt')
        )
        
        # Save metrics
        metrics_path = os.path.join(self.paths_config['metrics'], f'{model_name.lower().replace("-", "_")}_metrics.json')
        self.save_metrics(metrics, model_name, inference_time, metrics_path)
        
        logger.info(f"Evaluation completed for {model_name}")
        
        return metrics
    
    def evaluate_both_models(self, data_path: str = "data/processed") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate both LSTM-CNN and CNN-LSTM models.
        
        Args:
            data_path: Path to processed data
            
        Returns:
            Tuple of (lstm_cnn_metrics, cnn_lstm_metrics)
        """
        logger.info("Starting evaluation of both models")
        
        # Load test data
        X_test, y_test = self.load_test_data(data_path)
        
        # Evaluate LSTM-CNN model
        lstm_cnn_path = os.path.join(self.paths_config['models'], 'lstm_cnn_best.h5')
        lstm_cnn_metrics = self.evaluate_model(lstm_cnn_path, "LSTM-CNN", X_test, y_test)
        
        # Evaluate CNN-LSTM model
        cnn_lstm_path = os.path.join(self.paths_config['models'], 'cnn_lstm_best.h5')
        cnn_lstm_metrics = self.evaluate_model(cnn_lstm_path, "CNN-LSTM", X_test, y_test)
        
        logger.info("Evaluation of both models completed")
        
        return lstm_cnn_metrics, cnn_lstm_metrics


if __name__ == "__main__":
    # Example usage
    evaluator = IoTEvaluator()
    
    try:
        lstm_cnn_metrics, cnn_lstm_metrics = evaluator.evaluate_both_models()
        
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nLSTM-CNN Performance:")
        print(f"  Accuracy: {lstm_cnn_metrics['overall']['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {lstm_cnn_metrics['overall']['f1_macro']:.4f}")
        print(f"  ROC-AUC (Macro): {lstm_cnn_metrics['overall']['roc_auc_macro']:.4f}")
        print(f"  Inference Time: {lstm_cnn_metrics['inference_time_ms_per_sample']:.3f} ms/sample")
        
        print(f"\nCNN-LSTM Performance:")
        print(f"  Accuracy: {cnn_lstm_metrics['overall']['accuracy']:.4f}")
        print(f"  F1-Score (Macro): {cnn_lstm_metrics['overall']['f1_macro']:.4f}")
        print(f"  ROC-AUC (Macro): {cnn_lstm_metrics['overall']['roc_auc_macro']:.4f}")
        print(f"  Inference Time: {cnn_lstm_metrics['inference_time_ms_per_sample']:.3f} ms/sample")
        
        # Determine winner
        lstm_cnn_score = lstm_cnn_metrics['overall']['accuracy']
        cnn_lstm_score = cnn_lstm_metrics['overall']['accuracy']
        
        winner = "LSTM-CNN" if lstm_cnn_score > cnn_lstm_score else "CNN-LSTM"
        difference = abs(lstm_cnn_score - cnn_lstm_score)
        
        print(f"\nWinner: {winner} (by {difference:.4f} accuracy difference)")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        print("Please ensure the trained models exist in models/ directory")
