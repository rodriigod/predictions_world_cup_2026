"""
Data preprocessing script

Ejecutar: python preprocess.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import DataLoader, DataPreprocessor
from scripts.config import CONFIG, DATA_PATH, OUTPUT_PATH


def preprocess_data():
    """Main preprocessing pipeline"""
    
    print("=" * 50)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading raw data...")
    loader = DataLoader(DATA_PATH)
    try:
        df = loader.load_csv("data.csv")
        print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print("   ✗ data.csv not found in files/f0_raw/")
        print("   Please add your raw data file first.")
        return
    
    # 2. Handle missing values
    print("\n2. Handling missing values...")
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(
        df,
        method=CONFIG['data']['handle_missing']['method'],
        threshold=CONFIG['data']['handle_missing']['threshold']
    )
    print(f"   ✓ Missing values handled")
    
    # 3. Remove duplicates
    print("\n3. Removing duplicates...")
    initial_rows = len(df)
    df = preprocessor.remove_duplicates(df)
    print(f"   ✓ Removed {initial_rows - len(df)} duplicates")
    
    # 4. Save processed data
    output_file = OUTPUT_PATH / "processed_data.csv"
    DataLoader.save_csv(df, output_file)
    print(f"\n✓ Preprocessing complete!")
    print(f"   Output: {output_file}")


if __name__ == "__main__":
    preprocess_data()
