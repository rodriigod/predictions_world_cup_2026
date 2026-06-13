"""
Model training script

Ejecutar: python train.py
"""

import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader
from src.models import BaselineModel, AdvancedModels
from src.utils import ModelMetrics
from scripts.config import CONFIG, OUTPUT_PATH, MODELS_PATH, RANDOM_SEED


def load_and_prepare_data():
    """Load and prepare data for training"""
    
    print("Loading data...")
    loader = DataLoader(OUTPUT_PATH)
    df = loader.load_csv("processed_data.csv")
    
    # Separate features and target
    target_col = CONFIG['data']['target_column']
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['data']['test_size'],
        random_state=RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test


def train_model():
    """Train ML model"""
    
    print("=" * 50)
    print("MODEL TRAINING PIPELINE")
    print("=" * 50)
    
    try:
        X_train, X_test, y_train, y_test = load_and_prepare_data()
    except FileNotFoundError:
        print("\n✗ processed_data.csv not found.")
        print("   Run: python scripts/preprocess.py")
        return
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train baseline model
    print("\n1. Training baseline model (Random Forest)...")
    model = BaselineModel.random_forest_classifier(
        X_train, y_train,
        n_estimators=CONFIG['model']['random_forest']['n_estimators'],
        max_depth=CONFIG['model']['random_forest']['max_depth']
    )
    print("   ✓ Model trained")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("\n2. Evaluating model...")
    metrics = ModelMetrics.classification_metrics(y_test, y_pred)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    # Save model
    print("\n3. Saving model...")
    model_path = MODELS_PATH / "model_baseline.pkl"
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved: {model_path}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    train_model()
