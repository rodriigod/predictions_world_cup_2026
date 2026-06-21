"""Feature engineering utilities"""

import pandas as pd
import numpy as np
from typing import List


class FeatureEngineer:
    """Create and engineer features"""
    
    @staticmethod
    def create_polynomial_features(
        df: pd.DataFrame, 
        columns: List[str], 
        degree: int = 2
    ) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Polynomial degree
        
        Returns:
            DataFrame with polynomial features added
        """
        df = df.copy()
        for col in columns:
            for d in range(2, degree + 1):
                df[f"{col}_deg{d}"] = df[col] ** d
        return df
    
    @staticmethod
    def create_interaction_features(
        df: pd.DataFrame, 
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Create interaction features between column pairs
        
        Args:
            df: Input DataFrame
            columns: Columns to create interactions from
        
        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
        return df
    
    @staticmethod
    def create_ratio_features(
        df: pd.DataFrame, 
        numerator_cols: List[str],
        denominator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create ratio features
        
        Args:
            df: Input DataFrame
            numerator_cols: Numerator columns
            denominator_cols: Denominator columns
        
        Returns:
            DataFrame with ratio features added
        """
        df = df.copy()
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                ratio_name = f"{num_col}_ratio_{den_col}"
                # Avoid division by zero
                df[ratio_name] = np.where(
                    df[den_col] != 0,
                    df[num_col] / df[den_col],
                    0
                )
        return df
    
    @staticmethod
    def create_binning_features(
        df: pd.DataFrame, 
        column: str, 
        bins: int = 5,
        labels: List[str] = None
    ) -> pd.DataFrame:
        """
        Create binned/discretized features
        
        Args:
            df: Input DataFrame
            column: Column to bin
            bins: Number of bins
            labels: Custom bin labels
        
        Returns:
            DataFrame with binned feature added
        """
        df = df.copy()
        df[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=labels)
        return df
    
    @staticmethod
    def create_aggregation_features(
        df: pd.DataFrame,
        group_by: str,
        agg_column: str,
        agg_methods: List[str] = ["mean", "sum", "std"]
    ) -> pd.DataFrame:
        """
        Create aggregation features based on grouping
        
        Args:
            df: Input DataFrame
            group_by: Column to group by
            agg_column: Column to aggregate
            agg_methods: Aggregation methods
        
        Returns:
            DataFrame with aggregation features added
        """
        df = df.copy()
        for method in agg_methods:
            feature_name = f"{agg_column}_{method}_by_{group_by}"
            df[feature_name] = df.groupby(group_by)[agg_column].transform(method)
        return df


if __name__ == "__main__":
    # Ejemplo de uso
    engineer = FeatureEngineer()
