"""
PyTorch evaluation utilities mirroring the Keras-based IoTEvaluator API.
Generates identical metrics JSON, confusion matrix, ROC curves, and classification report.
"""
from __future__ import annotations

import os
import time
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging

from torch_models import build_torch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        if y.ndim == 2:
            self.y_idx = torch.from_numpy(np.argmax(y, axis=1)).long()
        else:
            self.y_idx = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y_idx[idx]


class IoTEvaluatorTorch:
    """Evaluator for PyTorch models compatible with existing output files."""

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            # Lazy resolve configs/config.yaml if exists
            from pathlib import Path
            root = Path(__file__).parent.parent
            candidates = [root / "configs" / "config.yaml", root / "config.yaml"]
            for c in candidates:
                if c.exists():
                    config_path = str(c)
                    break
            if config_path is None:
                config_path = "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.paths_config = self.config['paths']
        self.eval_config = self.config.get('evaluation', {})
        self.class_names: List[str] | None = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_test_data(self, data_path: str = "data/processed") -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Loading test data and preprocessing objects")
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        # Load class names from label_encoder if available
        import pickle
        try:
            with open(os.path.join(data_path, 'label_encoder.pkl'), 'rb') as f:
                le = pickle.load(f)
                self.class_names = list(le.classes_)
        except Exception:
            # fallback to generic names
            self.class_names = [f"class_{i}" for i in range(y_test.shape[1] if y_test.ndim == 2 else int(np.max(y_test) + 1))]
        logger.info(f"Loaded test data: {X_test.shape} -> {y_test.shape}")
        return X_test, y_test

    def load_model(self, model_path: str, model_name: str, input_shape: tuple, num_classes: int) -> torch.nn.Module:
        logger.info(f"Loading PyTorch model {model_name} from {model_path}")
        model = build_torch_model(model_name, input_shape, num_classes)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def make_predictions(self, model: torch.nn.Module, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        logger.info("Making predictions on test data (PyTorch)")
        ds = _NpyDataset(X_test, np.zeros((X_test.shape[0],), dtype=np.int64))
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        all_probs: List[np.ndarray] = []
        start = time.time()
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                logits = model(xb)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
        end = time.time()
        probs = np.concatenate(all_probs, axis=0)
        pred_class = np.argmax(probs, axis=1)
        elapsed_ms_per_sample = ((end - start) * 1000.0) / max(1, X_test.shape[0])
        logger.info(f"Inference time: {elapsed_ms_per_sample:.3f} ms per sample")
        return probs, pred_class, elapsed_ms_per_sample

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, Any]:
        logger.info("Calculating evaluation metrics (PyTorch)")
        if y_true.ndim == 2:
            y_true_class = np.argmax(y_true, axis=1)
        else:
            y_true_class = y_true

        accuracy = accuracy_score(y_true_class, y_pred)
        precision_macro = precision_score(y_true_class, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_class, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true_class, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true_class, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true_class, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true_class, y_pred, average='weighted', zero_division=0)

        # ROC-AUC per class (guard missing classes)
        num_classes = y_pred_prob.shape[1]
        if y_true.ndim != 2:
            y_true_onehot = np.eye(num_classes)[y_true_class]
        else:
            y_true_onehot = y_true

        roc_auc_scores = []
        for i in range(num_classes):
            if len(np.unique(y_true_onehot[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
                roc_auc_scores.append(auc(fpr, tpr))
            else:
                roc_auc_scores.append(0.0)

        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'roc_auc_macro': float(np.mean(roc_auc_scores)),
            },
            'per_class': {
                'precision': precision_score(y_true_class, y_pred, average=None, zero_division=0).tolist(),
                'recall': recall_score(y_true_class, y_pred, average=None, zero_division=0).tolist(),
                'f1_score': f1_score(y_true_class, y_pred, average=None, zero_division=0).tolist(),
                'roc_auc': roc_auc_scores,
            },
            'class_names': list(self.class_names or [str(i) for i in range(num_classes)]),
        }
        return metrics

    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: str):
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_roc_curves(self, y_true: np.ndarray, y_pred_prob: np.ndarray, model_name: str, output_path: str):
        if y_true.ndim != 2:
            y_true_onehot = np.eye(y_pred_prob.shape[1])[y_true]
        else:
            y_true_onehot = y_true
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, class_name in enumerate(self.class_names or [str(i) for i in range(y_pred_prob.shape[1])]):
            if len(np.unique(y_true_onehot[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC={roc_auc:.3f})', color=colors[i % len(colors)])
        plt.plot([0,1], [0,1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_path: str):
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        report = classification_report(y_true, y_pred, target_names=self.class_names, digits=4)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(f"{model_name} - Classification Report\n")
            f.write("="*50 + "\n\n")
            f.write(report)
            f.write("\n")

    def save_metrics(self, metrics: Dict[str, Any], model_name: str, inference_time: float, output_path: str):
        metrics['inference_time_ms_per_sample'] = float(inference_time)
        metrics['model_name'] = model_name
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def evaluate_model(self, model_path: str, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        logger.info(f"Evaluating {model_name} (PyTorch)")
        input_shape = X_test.shape[1:]
        num_classes = y_test.shape[1] if y_test.ndim == 2 else int(np.max(y_test) + 1)
        model = self.load_model(model_path, model_name, input_shape, num_classes)
        y_pred_prob, y_pred_class, inf_ms = self.make_predictions(model, X_test)
        metrics = self.calculate_metrics(y_test, y_pred_class, y_pred_prob)
        # Plots and reports
        self.create_confusion_matrix(
            y_test, y_pred_class, model_name,
            os.path.join(self.config['paths']['plots'], f'confusion_matrix_{model_name.lower().replace("-","_")}.png')
        )
        self.create_roc_curves(
            y_test, y_pred_prob, model_name,
            os.path.join(self.config['paths']['plots'], f'roc_curves_{model_name.lower().replace("-","_")}.png')
        )
        self.generate_classification_report(
            y_test, y_pred_class, model_name,
            os.path.join(self.config['paths']['reports'], f'classification_report_{model_name.lower().replace("-","_")}.txt')
        )
        # Save metrics
        out_json = os.path.join(self.config['paths']['metrics'], f'{model_name.lower().replace("-","_")}_metrics.json')
        self.save_metrics(metrics, model_name, inf_ms, out_json)
        return metrics

    def evaluate_both_models(self, data_path: str = "data/processed") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logger.info("Starting evaluation of both models (PyTorch)")
        X_test, y_test = self.load_test_data(data_path)
        lstm_cnn_path = os.path.join(self.config['paths']['models'], 'lstm_cnn_best.pt')
        cnn_lstm_path = os.path.join(self.config['paths']['models'], 'cnn_lstm_best.pt')
        lstm_cnn_metrics = self.evaluate_model(lstm_cnn_path, "LSTM-CNN", X_test, y_test)
        cnn_lstm_metrics = self.evaluate_model(cnn_lstm_path, "CNN-LSTM", X_test, y_test)
        logger.info("Evaluation of both models completed (PyTorch)")
        return lstm_cnn_metrics, cnn_lstm_metrics
