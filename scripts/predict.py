"""
Model prediction script

Ejecutar: python predict.py --input <file> --output <file>
"""

import pandas as pd
import sys
import joblib
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import DataLoader
from scripts.config import MODELS_PATH, OUTPUT_PATH


def predict(input_file, output_file):
    """Make predictions on new data"""
    
    print("=" * 50)
    print("MODEL PREDICTION")
    print("=" * 50)
    
    # Load model
    print("\n1. Loading model...")
    model_path = MODELS_PATH / "model_baseline.pkl"
    if not model_path.exists():
        print(f"   ✗ Model not found: {model_path}")
        print("   Run: python scripts/train.py")
        return
    
    model = joblib.load(model_path)
    print("   ✓ Model loaded")
    
    # Load data
    print("\n2. Loading data...")
    loader = DataLoader(str(input_file.parent))
    X = loader.load_csv(input_file.name)
    print(f"   ✓ Loaded {len(X)} samples")
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = model.predict(X)
    print(f"   ✓ Predictions made")
    
    # Save predictions
    print("\n4. Saving predictions...")
    output_df = X.copy()
    output_df['prediction'] = y_pred
    DataLoader.save_csv(output_df, output_file)
    print(f"   ✓ Saved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('--input', type=str, default='files/f1_input/data.csv',
                       help='Input data file')
    parser.add_argument('--output', type=str, default='files/f3_output/predictions.csv',
                       help='Output predictions file')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    predict(input_path, output_path)
