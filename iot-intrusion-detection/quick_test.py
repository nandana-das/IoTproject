#!/usr/bin/env python3
"""
Quick test of the complete pipeline with a smaller sample of data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import IoTDataLoader
from preprocessor import IoTPreprocessor
from train import IoTModelTrainer
from evaluate import IoTEvaluator

def main():
    print("Quick Test of IoT Intrusion Detection Pipeline")
    print("=" * 50)
    
    # Load data
    print("1. Loading data...")
    loader = IoTDataLoader("data/raw")
    data = loader.load_data()
    print(f"   Original dataset shape: {data.shape}")
    
    # Take a smaller sample for quick testing (10% of data)
    sample_size = min(100000, len(data))
    data_sample = data.sample(n=sample_size, random_state=42)
    print(f"   Using sample size: {len(data_sample)}")
    
    # Show sample distribution
    category_counts = data_sample['category'].value_counts()
    print("   Sample distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(data_sample)) * 100
        print(f"     {category:15s}: {count:6,} ({percentage:5.1f}%)")
    
    # Preprocess sample
    print("\n2. Preprocessing sample...")
    preprocessor = IoTPreprocessor()
    
    # Prepare data
    X, y = preprocessor.prepare_data(data_sample)
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Handle missing values
    X = preprocessor.handle_missing_values(X)
    
    # Encode labels
    y = preprocessor.encode_labels(y)
    
    # Split data (smaller splits for quick test)
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    print(f"   Train: {X_train.shape[0]:,}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")
    
    # Scale features
    X_train, X_val, X_test = preprocessor.scale_features(X_train, X_val, X_test)
    
    # Create sequences (shorter sequences for quick test)
    preprocessor.config['sequences']['sequence_length'] = 5  # Shorter for quick test
    X_train_seq, y_train_seq = preprocessor.create_sequences(X_train, y_train)
    X_val_seq, y_val_seq = preprocessor.create_sequences(X_val, y_val)
    X_test_seq, y_test_seq = preprocessor.create_sequences(X_test, y_test)
    
    # One-hot encode labels
    y_train_onehot, y_val_onehot, y_test_onehot = preprocessor.one_hot_encode_labels(
        y_train_seq, y_val_seq, y_test_seq
    )
    
    print(f"   Final shapes - Train: {X_train_seq.shape} -> {y_train_onehot.shape}")
    
    # Quick model test (just build models, don't train)
    print("\n3. Testing model architectures...")
    
    # Test LSTM-CNN model
    from lstm_cnn_model import LSTMCnnModel
    lstm_cnn_model = LSTMCnnModel(X_train_seq.shape[1:], y_train_onehot.shape[1])
    print(f"   LSTM-CNN model built successfully")
    
    # Test CNN-LSTM model
    from cnn_lstm_model import CnnLstmModel
    cnn_lstm_model = CnnLstmModel(X_train_seq.shape[1:], y_train_onehot.shape[1])
    print(f"   CNN-LSTM model built successfully")
    
    # Quick training test (just 1 epoch)
    print("\n4. Quick training test (1 epoch)...")
    trainer = IoTModelTrainer()
    
    # Override epochs for quick test
    trainer.training_config['epochs'] = 1
    trainer.training_config['batch_size'] = 64
    
    # Train LSTM-CNN for 1 epoch
    print("   Training LSTM-CNN...")
    lstm_cnn_results = trainer.train_model(lstm_cnn_model, X_train_seq, y_train_onehot, 
                                          X_val_seq, y_val_onehot, "LSTM-CNN")
    
    # Train CNN-LSTM for 1 epoch
    print("   Training CNN-LSTM...")
    cnn_lstm_results = trainer.train_model(cnn_lstm_model, X_train_seq, y_train_onehot,
                                          X_val_seq, y_val_onehot, "CNN-LSTM")
    
    print("\n5. Quick evaluation test...")
    
    # Test predictions
    lstm_cnn_pred = lstm_cnn_model.predict(X_test_seq[:100])  # Test on 100 samples
    cnn_lstm_pred = cnn_lstm_model.predict(X_test_seq[:100])
    
    lstm_cnn_acc = np.mean(np.argmax(lstm_cnn_pred, axis=1) == np.argmax(y_test_onehot[:100], axis=1))
    cnn_lstm_acc = np.mean(np.argmax(cnn_lstm_pred, axis=1) == np.argmax(y_test_onehot[:100], axis=1))
    
    print(f"   LSTM-CNN accuracy (100 samples): {lstm_cnn_acc:.4f}")
    print(f"   CNN-LSTM accuracy (100 samples): {cnn_lstm_acc:.4f}")
    
    print("\n" + "=" * 50)
    print("Quick test completed successfully!")
    print("Both models are working correctly.")
    print("Ready to run the full pipeline!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nNext steps:")
            print("1. Run: python train_lstm_cnn.py")
            print("2. Run: python train_cnn_lstm.py")
            print("3. Run: python compare_models.py")
            print("Or run: python run_complete_pipeline.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
