"""Advanced ML models"""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class AdvancedModels:
    """Advanced machine learning models"""
    
    @staticmethod
    def gradient_boosting_classifier(
        X_train, 
        y_train, 
        X_test=None,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    ):
        """Gradient Boosting Classifier"""
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def support_vector_classifier(
        X_train, 
        y_train, 
        X_test=None,
        kernel='rbf',
        C=1.0
    ):
        """Support Vector Machine Classifier"""
        model = SVC(kernel=kernel, C=C, random_state=42)
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def knn_classifier(
        X_train, 
        y_train, 
        X_test=None,
        n_neighbors=5
    ):
        """K-Nearest Neighbors Classifier"""
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def gradient_boosting_regressor(
        X_train, 
        y_train, 
        X_test=None,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    ):
        """Gradient Boosting Regressor"""
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model
    
    @staticmethod
    def support_vector_regressor(
        X_train, 
        y_train, 
        X_test=None,
        kernel='rbf',
        C=1.0
    ):
        """Support Vector Machine Regressor"""
        model = SVR(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        if X_test is not None:
            y_pred = model.predict(X_test)
            return model, y_pred
        return model


if __name__ == "__main__":
    pass
