"""
Training utilities for IoT intrusion detection models.
Handles training both LSTM-CNN and CNN-LSTM models with callbacks and monitoring.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import os
import time
import yaml
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
try:
    # Package import when importing as src.train
    from src import resolve_config_path  # type: ignore
except Exception:
    try:
        # Direct module import when src added to sys.path
        from __init__ import resolve_config_path  # type: ignore
    except Exception:
        def resolve_config_path(default: str = "config.yaml") -> str:  # type: ignore
            return default

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTModelTrainer:
    """Trainer for IoT intrusion detection models."""
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = resolve_config_path()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.callbacks_config = self.config['callbacks']
        self.paths_config = self.config['paths']
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['preprocessing']['random_state'])
        torch.manual_seed(self.config['preprocessing']['random_state'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['preprocessing']['random_state'])
    
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           sample_weight: np.ndarray | None = None,
                           val_sample_weight: np.ndarray | None = None) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch data loaders.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            sample_weight: Training sample weights
            val_sample_weight: Validation sample weights
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create datasets
        if sample_weight is not None:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.FloatTensor(sample_weight))
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        if val_sample_weight is not None:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor, torch.FloatTensor(val_sample_weight))
        else:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        # Set num_workers=0 and pin_memory=False for Windows compatibility
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray, model_name: str,
                   sample_weight: np.ndarray | None = None,
                   val_sample_weight: np.ndarray | None = None) -> Dict[str, Any]:
        """
        Train a model.
        
        Args:
            model: PyTorch model to train
            X_train, y_train: Training data
            X_val, y_val: Validation data
            model_name: Name of the model
            sample_weight: Training sample weights
            val_sample_weight: Validation sample weights
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Starting training for {model_name}")
        logger.info(f"Training data shape: {X_train.shape} -> {y_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape} -> {y_val.shape}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val, sample_weight, val_sample_weight
        )
        
        # Setup optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.training_config['learning_rate']
        )
        
        # Setup loss function
        criterion = nn.CrossEntropyLoss(reduction='none')  # We'll handle weights manually
        
        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.callbacks_config['reduce_lr']['factor'],
            patience=self.callbacks_config['reduce_lr']['patience'],
            min_lr=self.callbacks_config['reduce_lr']['min_lr'],
            verbose=True
        )
        
        # Setup TensorBoard
        tensorboard_path = os.path.join("logs", model_name.lower().replace('-', '_'))
        writer = SummaryWriter(tensorboard_path)
        
        # Model checkpoint path
        checkpoint_path = os.path.join(
            self.paths_config['models'], 
            f"{model_name.lower().replace('-', '_')}_best.pth"
        )
        
        # Training history
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Record start time
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.training_config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Progress tracking
            total_batches = len(train_loader)
            log_interval = max(1, total_batches // 10)  # Log every 10% of batches
            
            for batch_idx, batch_data in enumerate(train_loader):
                if len(batch_data) == 3:
                    inputs, targets, weights = batch_data
                    weights = weights.to(self.device)
                else:
                    inputs, targets = batch_data
                    weights = None
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss_per_sample = criterion(outputs, targets.argmax(dim=1))
                if weights is not None:
                    loss = (loss_per_sample * weights).mean()
                else:
                    loss = loss_per_sample.mean()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                _, target_labels = targets.max(1)
                train_correct += predicted.eq(target_labels).sum().item()
                train_total += targets.size(0)
                
                # Log progress periodically
                if (batch_idx + 1) % log_interval == 0:
                    progress = (batch_idx + 1) / total_batches * 100
                    current_loss = train_loss / train_total
                    current_acc = train_correct / train_total
                    logger.info(f"Epoch {epoch + 1}/{self.training_config['epochs']} - "
                               f"Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - "
                               f"Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")
            
            # Calculate training metrics
            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data in val_loader:
                    if len(batch_data) == 3:
                        inputs, targets, weights = batch_data
                        weights = weights.to(self.device)
                    else:
                        inputs, targets = batch_data
                        weights = None
                    
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss_per_sample = criterion(outputs, targets.argmax(dim=1))
                    if weights is not None:
                        loss = (loss_per_sample * weights).mean()
                    else:
                        loss = loss_per_sample.mean()
                    
                    # Statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    _, target_labels = targets.max(1)
                    val_correct += predicted.eq(target_labels).sum().item()
                    val_total += targets.size(0)
            
            # Calculate validation metrics
            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total
            
            # Update history
            history['loss'].append(epoch_train_loss)
            history['accuracy'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_accuracy'].append(epoch_val_acc)
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_train_acc, epoch)
            writer.add_scalar('Accuracy/val', epoch_val_acc, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            logger.info(f"Epoch {epoch + 1}/{self.training_config['epochs']}: "
                       f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
                       f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
            
            # Learning rate scheduling
            scheduler.step(epoch_val_loss)
            
            # Model checkpoint (save best model)
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_accuracy': epoch_val_acc,
                    'input_shape': model.input_shape,
                    'num_classes': model.num_classes,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.callbacks_config['early_stopping']['patience']:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Close TensorBoard writer
        writer.close()
        
        # Load best model
        if self.callbacks_config['early_stopping']['restore_best_weights']:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Restored best model weights")
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Prepare results
        results = {
            'model_name': model_name,
            'training_time_minutes': training_time / 60,
            'total_epochs': len(history['loss']),
            'best_epoch': best_epoch + 1,
            'best_val_loss': best_val_loss,
            'best_val_accuracy': history['val_accuracy'][best_epoch],
            'final_train_loss': history['loss'][-1],
            'final_train_accuracy': history['accuracy'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'history': history
        }
        
        logger.info(f"Training completed for {model_name}")
        logger.info(f"Training time: {training_time/60:.2f} minutes")
        logger.info(f"Best validation accuracy: {history['val_accuracy'][best_epoch]:.4f} at epoch {best_epoch + 1}")
        
        return results
    
    def save_training_results(self, results: Dict[str, Any], output_path: str):
        """
        Save training results to JSON file.
        
        Args:
            results: Training results dictionary
            output_path: Path to save results
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare data for JSON serialization by converting numpy types recursively
        def to_serializable(obj: Any):
            # Scalars
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            # Arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Lists/Tuples
            if isinstance(obj, (list, tuple)):
                return [to_serializable(v) for v in obj]
            # Dicts
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            return obj

        json_results = to_serializable(results)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Training results saved to {output_path}")
    
    def plot_training_curves(self, results: Dict[str, Any], output_path: str):
        """
        Plot and save training curves.
        
        Args:
            results: Training results dictionary
            output_path: Path to save the plot
        """
        history = results['history']
        model_name = results['model_name']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        ax2.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax2.set_title(f'{model_name} - Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add best epoch markers
        best_epoch = results['best_epoch'] - 1  # Convert to 0-based index
        ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({results["best_epoch"]})')
        ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({results["best_epoch"]})')
        
        # Update legends
        ax1.legend(fontsize=10)
        ax2.legend(fontsize=10)
        
        # Add performance metrics as text
        metrics_text = f"Best Val Accuracy: {results['best_val_accuracy']:.4f}\n"
        metrics_text += f"Best Val Loss: {results['best_val_loss']:.4f}\n"
        metrics_text += f"Training Time: {results['training_time_minutes']:.1f} min"
        
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {output_path}")
    
    def train_lstm_cnn(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray,
                      sample_weight: np.ndarray | None = None,
                      val_sample_weight: np.ndarray | None = None) -> Dict[str, Any]:
        """
        Train LSTM-CNN model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            sample_weight: Training sample weights
            val_sample_weight: Validation sample weights
            
        Returns:
            Training results
        """
        from lstm_cnn_model import LSTMCnnModel
        
        # Build model
        model = LSTMCnnModel(X_train.shape[1:], y_train.shape[1])
        
        # Train model
        results = self.train_model(model, X_train, y_train, X_val, y_val, "LSTM-CNN",
                                  sample_weight, val_sample_weight)
        
        # Save results
        self.save_training_results(
            results, 
            os.path.join(self.paths_config['metrics'], 'lstm_cnn_training_results.json')
        )
        
        # Plot training curves
        self.plot_training_curves(
            results, 
            os.path.join(self.paths_config['plots'], 'training_curves_lstm_cnn.png')
        )
        
        return results
    
    def train_cnn_lstm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray,
                      sample_weight: np.ndarray | None = None,
                      val_sample_weight: np.ndarray | None = None) -> Dict[str, Any]:
        """
        Train CNN-LSTM model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            sample_weight: Training sample weights
            val_sample_weight: Validation sample weights
            
        Returns:
            Training results
        """
        from cnn_lstm_model import CnnLstmModel
        
        # Build model
        model = CnnLstmModel(X_train.shape[1:], y_train.shape[1])
        
        # Train model
        results = self.train_model(model, X_train, y_train, X_val, y_val, "CNN-LSTM",
                                  sample_weight, val_sample_weight)
        
        # Save results
        self.save_training_results(
            results, 
            os.path.join(self.paths_config['metrics'], 'cnn_lstm_training_results.json')
        )
        
        # Plot training curves
        self.plot_training_curves(
            results, 
            os.path.join(self.paths_config['plots'], 'training_curves_cnn_lstm.png')
        )
        
        return results


def train_both_models(data_path: str = "data/processed") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train both LSTM-CNN and CNN-LSTM models.
    
    Args:
        data_path: Path to processed data
        
    Returns:
        Tuple of (lstm_cnn_results, cnn_lstm_results)
    """
    logger.info("Loading processed data")
    
    # Load processed data
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    X_val = np.load(os.path.join(data_path, 'X_val.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    
    # Optional: load class weights for sample-weighted training
    class_weights_path = os.path.join(data_path, 'class_weights.pkl')
    loaded_class_weights = None
    if os.path.exists(class_weights_path):
        import pickle as _pickle
        with open(class_weights_path, 'rb') as _f:
            loaded_class_weights = _pickle.load(_f)

    # Optional subsampling via env vars for memory-constrained training
    import os as _os
    train_limit = int(_os.environ.get("TRAIN_LIMIT", "0"))
    val_limit = int(_os.environ.get("VAL_LIMIT", "0"))
    if train_limit and X_train.shape[0] > train_limit:
        logger.info(f"Subsampling training set from {X_train.shape[0]:,} to {train_limit:,}")
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
    if val_limit and X_val.shape[0] > val_limit:
        logger.info(f"Subsampling validation set from {X_val.shape[0]:,} to {val_limit:,}")
        X_val = X_val[:val_limit]
        y_val = y_val[:val_limit]
    
    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Initialize trainer
    trainer = IoTModelTrainer()
    
    # Optionally reduce batch size via env var
    try:
        desired_bs = int(_os.environ.get("BATCH_SIZE", "0"))
        if desired_bs and desired_bs < trainer.training_config['batch_size']:
            logger.info(f"Reducing batch size from {trainer.training_config['batch_size']} to {desired_bs}")
            trainer.training_config['batch_size'] = desired_bs
    except Exception:
        pass

    # Build optional sample weights to handle class imbalance
    def _build_sample_weights(y_onehot: np.ndarray, class_weights_dict: dict | None) -> np.ndarray | None:
        if class_weights_dict is None:
            return None
        # Convert one-hot labels to indices
        y_idx = np.argmax(y_onehot, axis=1)
        # Create vectorized lookup
        max_class = int(max(class_weights_dict.keys()))
        table = np.ones(max_class + 1, dtype=np.float32)
        for k, v in class_weights_dict.items():
            table[int(k)] = float(v)
        return table[y_idx]

    use_class_weights = os.environ.get('USE_CLASS_WEIGHTS', '1') == '1'
    train_sw = _build_sample_weights(y_train, loaded_class_weights) if use_class_weights else None
    val_sw = _build_sample_weights(y_val, loaded_class_weights) if use_class_weights else None

    # Train both models
    logger.info("Training LSTM-CNN model")
    lstm_cnn_results = trainer.train_lstm_cnn(X_train, y_train, X_val, y_val, train_sw, val_sw)
    
    logger.info("Training CNN-LSTM model")
    cnn_lstm_results = trainer.train_cnn_lstm(X_train, y_train, X_val, y_val, train_sw, val_sw)
    
    return lstm_cnn_results, cnn_lstm_results


if __name__ == "__main__":
    # Example usage
    try:
        lstm_cnn_results, cnn_lstm_results = train_both_models()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        
        print(f"\nLSTM-CNN Results:")
        print(f"  Best Validation Accuracy: {lstm_cnn_results['best_val_accuracy']:.4f}")
        print(f"  Training Time: {lstm_cnn_results['training_time_minutes']:.1f} minutes")
        print(f"  Total Epochs: {lstm_cnn_results['total_epochs']}")
        
        print(f"\nCNN-LSTM Results:")
        print(f"  Best Validation Accuracy: {cnn_lstm_results['best_val_accuracy']:.4f}")
        print(f"  Training Time: {cnn_lstm_results['training_time_minutes']:.1f} minutes")
        print(f"  Total Epochs: {cnn_lstm_results['total_epochs']}")
        
        print(f"\nWinner: {'LSTM-CNN' if lstm_cnn_results['best_val_accuracy'] > cnn_lstm_results['best_val_accuracy'] else 'CNN-LSTM'}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"Error: {e}")
        print("Please ensure the processed data exists in data/processed/")
