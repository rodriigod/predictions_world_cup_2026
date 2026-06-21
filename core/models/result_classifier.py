"""Clasificador multiclase 1X2 — el enfoque de "ML en toda regla".

Formula la predicción como clasificación con 3 etiquetas
[1 = gana el primero, X = empate, 2 = gana el segundo], que es como la
literatura moderna plantea el problema. Entrena tres algoritmos:

- XGBoost (multi:softprob)        : el estándar para datos tabulares
- Random Forest                   : robusto al overfitting
- Regresión Logística Multinomial : el BASELINE contra el que se mide
                                    si el ML complejo realmente aporta

Sirve como verificación cruzada del enfoque Poisson+Dixon-Coles: si las
probabilidades 1X2 del clasificador y las derivadas de las lambdas
coinciden, el pipeline es coherente; el log-loss en holdout dice qué
enfoque está mejor calibrado.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.data.wc_schema import FEATURE_NAMES

CLASSES = ["1", "2", "X"]  # orden alfabético de sklearn


def _make_models(random_state: int) -> dict:
    models = {
        "logistic_baseline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=20,
            random_state=random_state, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob", n_estimators=400,
            learning_rate=0.05, max_depth=4, reg_lambda=1.0,
            random_state=random_state)
    except ImportError:
        pass
    return models


class ResultClassifier:
    """Entrena los 3 algoritmos y se queda con el de menor log-loss."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = _make_models(random_state)
        self.best_name: str | None = None
        self.metrics: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray | None = None) -> pd.DataFrame:
        X = X[FEATURE_NAMES]
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        y_enc = y.astype(str)
        # xgboost requiere etiquetas numéricas
        y_num = y_enc.map({c: i for i, c in enumerate(CLASSES)})

        X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
            X, y_num, sample_weight, test_size=0.2,
            random_state=0, stratify=y_num)

        rows = []
        for name, model in self.models.items():
            if name == "logistic_baseline":
                model.fit(X_tr, y_tr, clf__sample_weight=w_tr)
            else:
                model.fit(X_tr, y_tr, sample_weight=w_tr)
            proba = model.predict_proba(X_te)
            rows.append({
                "model": name,
                "log_loss": log_loss(y_te, proba, sample_weight=w_te),
                "accuracy": accuracy_score(
                    y_te, proba.argmax(axis=1), sample_weight=w_te),
            })
        self.metrics = pd.DataFrame(rows).sort_values("log_loss")
        self.best_name = self.metrics.iloc[0]["model"]

        # reentrenar el mejor con todos los datos
        best = self.models[self.best_name]
        if self.best_name == "logistic_baseline":
            best.fit(X, y_num, clf__sample_weight=sample_weight)
        else:
            best.fit(X, y_num, sample_weight=sample_weight)
        return self.metrics

    def predict_proba_1x2(self, X: pd.DataFrame) -> pd.DataFrame:
        """Probabilidades [p_1, p_X, p_2] del mejor modelo."""
        proba = self.models[self.best_name].predict_proba(X[FEATURE_NAMES])
        out = pd.DataFrame(proba, columns=CLASSES)
        return out[["1", "X", "2"]]
