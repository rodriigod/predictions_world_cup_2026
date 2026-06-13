"""Baseline models for quick evaluation"""

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


class BaselineModel:
    """Simple baseline models for quick testing"""
    
    @staticmethod
    def linear_regression(X_train, y_train, X_test=None):
        """Simple linear regression baseline"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def logistic_regression(X_train, y_train, X_test=None):
        """Simple logistic regression baseline"""
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def decision_tree_classifier(X_train, y_train, X_test=None, max_depth=5):
        """Simple decision tree classifier baseline"""
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def random_forest_classifier(
        X_train, 
        y_train, 
        X_test=None,
        n_estimators=100,
        max_depth=10
    ):
        """Random forest classifier baseline"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def mean_baseline(y_train, y_test=None):
        """Mean baseline for regression"""
        mean_value = np.mean(y_train)
        if y_test is not None:
            y_pred = np.full_like(y_test, mean_value, dtype=float)
            return y_pred
        return mean_value
    
    @staticmethod
    def median_baseline(y_train, y_test=None):
        """Median baseline for regression"""
        median_value = np.median(y_train)
        if y_test is not None:
            y_pred = np.full_like(y_test, median_value, dtype=float)
            return y_pred
        return median_value


if __name__ == "__main__":
    pass
