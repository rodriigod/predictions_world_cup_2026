"""Data loading utilities"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Tuple


class DataLoader:
    """Load data from various formats"""
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataLoader
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
    
    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV file"""
        file_path = self.data_path / filename
        return pd.read_csv(file_path)
    
    def load_excel(self, filename: str, sheet_name: str = 0) -> pd.DataFrame:
        """Load Excel file"""
        file_path = self.data_path / filename
        return pd.read_excel(file_path, sheet_name=sheet_name)
    
    def load_parquet(self, filename: str) -> pd.DataFrame:
        """Load Parquet file"""
        file_path = self.data_path / filename
        return pd.read_parquet(file_path)
    
    def load_multiple(self, filenames: list) -> pd.DataFrame:
        """Load and concatenate multiple CSV files"""
        dfs = [self.load_csv(f) for f in filenames]
        return pd.concat(dfs, ignore_index=True)
    
    @staticmethod
    def save_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False):
        """Save DataFrame to CSV"""
        df.to_csv(path, index=index)
    
    @staticmethod
    def save_parquet(df: pd.DataFrame, path: Union[str, Path]):
        """Save DataFrame to Parquet"""
        df.to_parquet(path)


if __name__ == "__main__":
    # Ejemplo de uso
    loader = DataLoader("./files/f0_raw")
    # df = loader.load_csv("data.csv")
