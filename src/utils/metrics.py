"""Evaluation metrics"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import numpy as np


class ModelMetrics:
    """Compute and display model metrics"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None):
        """
        Compute classification metrics
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except:
                pass
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        Compute regression metrics
        
        Returns:
            Dictionary with metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def print_classification_report(y_true, y_pred):
        """Print detailed classification report"""
        print(classification_report(y_true, y_pred))
    
    @staticmethod
    def get_confusion_matrix(y_true, y_pred):
        """Get confusion matrix"""
        return confusion_matrix(y_true, y_pred)


if __name__ == "__main__":
    pass
