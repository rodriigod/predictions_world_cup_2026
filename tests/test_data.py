"""
Unit tests for data module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataPreprocessor


class TestDataPreprocessor:
    """Test data preprocessing functions"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 1]
        })
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling"""
        preprocessor = DataPreprocessor()
        result = preprocessor.handle_missing_values(sample_data, method='mean')
        
        assert result.isnull().sum().sum() == 0, "Missing values still present"
        assert len(result) == len(sample_data), "Data length changed"
    
    def test_remove_duplicates(self, sample_data):
        """Test duplicate removal"""
        df_with_dupes = pd.concat([sample_data, sample_data.iloc[[0]]], ignore_index=True)
        
        preprocessor = DataPreprocessor()
        result = preprocessor.remove_duplicates(df_with_dupes)
        
        assert len(result) < len(df_with_dupes), "Duplicates not removed"
        assert result.duplicated().sum() == 0, "Duplicates still present"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
