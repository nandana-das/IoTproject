"""
Simplified Model Explainability for IoT Intrusion Detection
Focuses on SHAP and feature importance analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import os
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleIoTExplainer:
    """Simplified explainability analysis for IoT intrusion detection models."""
    
    def __init__(self):
        """Initialize explainer."""
        self.feature_names = [
            'seq', 'stddev', 'N_IN_Conn_P_SrcIP', 'min', 'state_number', 
            'mean', 'N_IN_Conn_P_DstIP', 'drate', 'srate', 'max'
        ]
        self.class_names = ['DDoS', 'DoS', 'Normal', 'Reconnaissance', 'Theft']
        self.results_dir = "results/explainability"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model_and_data(self, model_path: str):
        """Load trained model and test data."""
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        
        # Load test data
        import sys
        sys.path.append('src')
        from evaluate import IoTEvaluator
        evaluator = IoTEvaluator()
        X_test, y_test = evaluator.load_test_data()
        
        # Use smaller sample for explainability
        sample_size = min(500, len(X_test))
        X_sample = X_test[:sample_size]
        y_sample = y_test[:sample_size]
        
        logger.info(f"Loaded {sample_size} samples for explainability analysis")
        return model, X_sample, y_sample
    
    def analyze_feature_importance(self, model, X_sample: np.ndarray, y_sample: np.ndarray, 
                                  model_name: str = "Model"):
        """Analyze feature importance using permutation importance."""
        logger.info(f"Analyzing feature importance for {model_name}")
        
        # Calculate baseline accuracy
        predictions = model.predict(X_sample)
        baseline_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_sample, axis=1))
        
        # Calculate permutation importance for each feature
        feature_importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Create modified dataset with shuffled feature
            X_modified = X_sample.copy()
            
            # Shuffle the feature across all timesteps
            for t in range(X_sample.shape[1]):
                X_modified[:, t, i] = np.random.permutation(X_modified[:, t, i])
            
            # Calculate accuracy with shuffled feature
            modified_predictions = model.predict(X_modified)
            modified_accuracy = np.mean(np.argmax(modified_predictions, axis=1) == np.argmax(y_sample, axis=1))
            
            # Importance = drop in accuracy
            importance = baseline_accuracy - modified_accuracy
            feature_importance[feature_name] = importance
        
        # Plot feature importance
        self._plot_feature_importance(feature_importance, model_name)
        
        return feature_importance
    
    def _plot_feature_importance(self, feature_importance: Dict, model_name: str):
        """Plot feature importance."""
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(features, importance_values, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar, value in zip(bars, importance_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Feature Importance Analysis - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Network Features', fontsize=12)
        plt.ylabel('Importance (Accuracy Drop)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'feature_importance_{model_name.lower().replace("-", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")
    
    def analyze_class_specific_features(self, model, X_sample: np.ndarray, y_sample: np.ndarray, 
                                       model_name: str = "Model"):
        """Analyze which features are most important for each attack class."""
        logger.info(f"Analyzing class-specific features for {model_name}")
        
        class_feature_importance = {}
        
        for class_idx, class_name in enumerate(self.class_names):
            # Find samples of this class
            class_samples = np.where(np.argmax(y_sample, axis=1) == class_idx)[0]
            
            if len(class_samples) < 10:  # Need minimum samples
                continue
                
            X_class = X_sample[class_samples]
            y_class = y_sample[class_samples]
            
            # Calculate baseline accuracy for this class
            predictions = model.predict(X_class)
            baseline_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_class, axis=1))
            
            # Calculate feature importance for this class
            feature_importance = {}
            
            for i, feature_name in enumerate(self.feature_names):
                X_modified = X_class.copy()
                
                # Shuffle the feature
                for t in range(X_class.shape[1]):
                    X_modified[:, t, i] = np.random.permutation(X_modified[:, t, i])
                
                modified_predictions = model.predict(X_modified)
                modified_accuracy = np.mean(np.argmax(modified_predictions, axis=1) == np.argmax(y_class, axis=1))
                
                importance = baseline_accuracy - modified_accuracy
                feature_importance[feature_name] = importance
            
            class_feature_importance[class_name] = feature_importance
        
        # Plot class-specific feature importance
        self._plot_class_feature_importance(class_feature_importance, model_name)
        
        return class_feature_importance
    
    def _plot_class_feature_importance(self, class_feature_importance: Dict, model_name: str):
        """Plot class-specific feature importance heatmap."""
        if not class_feature_importance:
            logger.warning("No class-specific data to plot")
            return
        
        # Create DataFrame
        importance_df = pd.DataFrame(class_feature_importance).T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(importance_df, annot=True, cmap='YlOrRd', fmt='.4f', 
                   cbar_kws={'label': 'Feature Importance (Accuracy Drop)'})
        plt.title(f'Class-Specific Feature Importance - {model_name}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Network Features', fontsize=12)
        plt.ylabel('Attack Types', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, f'class_feature_importance_{model_name.lower().replace("-", "_")}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Class feature importance plot saved to {save_path}")
    
    def create_model_comparison_dashboard(self, model_paths: Dict[str, str]):
        """Create comparison dashboard for both models."""
        logger.info("Creating model comparison dashboard")
        
        # Analyze both models
        results = {}
        
        for model_name, model_path in model_paths.items():
            logger.info(f"Analyzing {model_name}")
            
            model, X_sample, y_sample = self.load_model_and_data(model_path)
            
            # Feature importance
            feature_importance = self.analyze_feature_importance(model, X_sample, y_sample, model_name)
            
            # Class-specific analysis
            class_importance = self.analyze_class_specific_features(model, X_sample, y_sample, model_name)
            
            results[model_name] = {
                'feature_importance': feature_importance,
                'class_importance': class_importance,
                'sample_size': len(X_sample)
            }
        
        # Create comparison plots
        self._create_comparison_plots(results)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        logger.info("Model comparison dashboard completed")
        return results
    
    def _create_comparison_plots(self, results: Dict):
        """Create comparison plots between models."""
        # Feature importance comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, (model_name, data) in enumerate(results.items()):
            features = list(data['feature_importance'].keys())
            importance = list(data['feature_importance'].values())
            
            ax = ax1 if i == 0 else ax2
            bars = ax.bar(features, importance, alpha=0.7, 
                         color=['skyblue', 'lightcoral'][i])
            
            ax.set_title(f'{model_name} - Feature Importance', fontweight='bold')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance (Accuracy Drop)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, importance):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Model Comparison - Feature Importance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'model_comparison_feature_importance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Model comparison plot saved to {save_path}")
    
    def _generate_summary_report(self, results: Dict):
        """Generate summary report with key insights."""
        report = {
            'analysis_summary': {
                'models_analyzed': list(results.keys()),
                'total_samples': sum(data['sample_size'] for data in results.values()),
                'analysis_date': pd.Timestamp.now().isoformat()
            },
            'key_findings': {},
            'recommendations': []
        }
        
        # Analyze key findings
        for model_name, data in results.items():
            feature_importance = data['feature_importance']
            
            # Top 3 most important features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            report['key_findings'][model_name] = {
                'top_features': top_features,
                'most_important_feature': top_features[0] if top_features else None
            }
        
        # Generate recommendations
        if len(results) >= 2:
            models = list(results.keys())
            model1_features = results[models[0]]['feature_importance']
            model2_features = results[models[1]]['feature_importance']
            
            # Find common important features
            common_features = set(model1_features.keys()) & set(model2_features.keys())
            important_features = []
            
            for feature in common_features:
                if model1_features[feature] > 0.01 or model2_features[feature] > 0.01:
                    important_features.append(feature)
            
            report['recommendations'] = [
                f"Focus monitoring on these critical features: {', '.join(important_features[:5])}",
                "Implement real-time feature extraction for top-ranked features",
                "Consider feature engineering to enhance discriminative power",
                "Deploy ensemble methods combining both model architectures"
            ]
        
        # Save report
        report_path = os.path.join(self.results_dir, 'explainability_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {report_path}")
        
        # Print key insights
        print("\n" + "="*60)
        print("MODEL EXPLAINABILITY ANALYSIS SUMMARY")
        print("="*60)
        
        for model_name, data in results.items():
            print(f"\n{model_name} Analysis:")
            top_features = data['feature_importance']
            sorted_features = sorted(top_features.items(), key=lambda x: x[1], reverse=True)
            
            print("   Top 5 Most Important Features:")
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"   {i+1}. {feature}: {importance:.4f}")
        
        print(f"\nKey Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\nAll results saved to: {self.results_dir}")
        print("="*60)

def main():
    """Main function to run explainability analysis."""
    explainer = SimpleIoTExplainer()
    
    # Define model paths
    model_paths = {
        "CNN-LSTM": "models/cnn_lstm_best.h5",
        "LSTM-CNN": "models/lstm_cnn_best.h5"
    }
    
    # Run analysis
    results = explainer.create_model_comparison_dashboard(model_paths)
    
    print("\nExplainability analysis completed successfully!")
    print("Check the results/explainability/ folder for all visualizations")

if __name__ == "__main__":
    main()
