"""Visualization utilities"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Visualizer:
    """Create visualizations for analysis and results"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_feature_importance(feature_importance, feature_names, top_n=20):
        """Plot feature importance"""
        # Sort and get top N
        indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(top_features, top_importance)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_distribution(data, title="Distribution"):
        """Plot distribution of data"""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, title="Predictions vs Actual"):
        """Plot predictions vs actual values (for regression)"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        return plt


if __name__ == "__main__":
    pass
