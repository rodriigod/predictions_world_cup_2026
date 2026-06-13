"""
Model evaluation script

Ejecutar: python evaluate.py
"""

import pandas as pd
import sys
import joblib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.utils import ModelMetrics, Visualizer
from scripts.config import MODELS_PATH, OUTPUT_PATH


def evaluate_model():
    """Evaluate trained model"""
    
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading model...")
    model_path = MODELS_PATH / "model_baseline.pkl"
    if not model_path.exists():
        print(f"   ✗ Model not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    print("   ✓ Model loaded")
    
    # Load test data
    print("\n2. Loading test data...")
    loader = DataLoader(OUTPUT_PATH)
    try:
        df = loader.load_csv("test_results.csv")
    except:
        print("   ✗ test_results.csv not found")
        print("   Make sure to save test results from train.py")
        return
    
    # Extract predictions and actual
    y_true = df['actual']
    y_pred = df['predicted']
    
    # Compute metrics
    print("\n3. Computing metrics...")
    metrics = ModelMetrics.classification_metrics(y_true, y_pred)
    
    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:.<30} {value:.4f}")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    evaluate_model()
