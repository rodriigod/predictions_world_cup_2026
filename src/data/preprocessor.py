"""Data preprocessing utilities"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from typing import Union, List, Tuple


class DataPreprocessor:
    """Preprocess and clean data"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = None
        self.encoders = {}
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = "mean",
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input DataFrame
            method: 'mean', 'median', 'forward_fill', 'drop'
            threshold: Drop columns with missing % > threshold
        
        Returns:
            DataFrame with missing values handled
        """
        # Drop columns with too many missing values
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        df = df.drop(columns=cols_to_drop)
        
        # Fill remaining missing values
        if method == "mean":
            return df.fillna(df.mean())
        elif method == "median":
            return df.fillna(df.median())
        elif method == "forward_fill":
            return df.fillna(method="ffill").fillna(method="bfill")
        elif method == "drop":
            return df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def remove_outliers(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: IQR multiplier or z-score threshold
        
        Returns:
            DataFrame without outliers
        """
        df = df.copy()
        
        if method == "iqr":
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        elif method == "zscore":
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = "standard",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            method: 'standard' or 'minmax'
            fit: Whether to fit the scaler
        
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if fit:
            if method == "standard":
                self.scaler = StandardScaler()
            elif method == "minmax":
                self.scaler = MinMaxScaler()
            df[columns] = self.scaler.fit_transform(df[columns])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Set fit=True first.")
            df[columns] = self.scaler.transform(df[columns])
        
        return df
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        columns: List[str],
        method: str = "onehot",
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            columns: Columns to encode
            method: 'onehot' or 'label'
            fit: Whether to fit the encoder
        
        Returns:
            DataFrame with encoded features
        """
        df = df.copy()
        
        if method == "onehot":
            if fit:
                self.encoders = {col: OneHotEncoder(sparse_output=False) 
                                for col in columns}
                for col in columns:
                    encoded = self.encoders[col].fit_transform(df[[col]])
                    encoded_df = pd.DataFrame(
                        encoded, 
                        columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                    )
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                for col in columns:
                    encoded = self.encoders[col].transform(df[[col]])
                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=[f"{col}_{cat}" for cat in self.encoders[col].categories_[0]]
                    )
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        
        elif method == "label":
            if fit:
                self.encoders = {col: LabelEncoder() for col in columns}
                for col in columns:
                    df[col] = self.encoders[col].fit_transform(df[col])
            else:
                for col in columns:
                    df[col] = self.encoders[col].transform(df[col])
        
        return df


if __name__ == "__main__":
    # Ejemplo de uso
    preprocessor = DataPreprocessor()
