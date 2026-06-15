"""Stacking de los clasificadores 1X2 (estilo Kaggle / blog cuML de NVIDIA).

Combina las probabilidades de varios modelos base (logística, Random Forest,
XGBoost) con un meta-modelo logístico entrenado sobre predicciones
OUT-OF-FOLD — cada fila del set de meta-entrenamiento se predice con modelos
base que NO la vieron, evitando leakage. Es el patrón de stacking del blog
de NVIDIA cuML, adaptado a sklearn.

Sirve para responder, con el backtest, si ensamblar realmente supera al
mejor modelo individual (en fútbol internacional la señal es casi lineal en
ELO, así que la ganancia esperada es pequeña — pero se mide, no se asume).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.data.wc_schema import FEATURE_NAMES
from src.models.result_classifier import CLASSES, _make_models

_LABEL = {c: i for i, c in enumerate(CLASSES)}   # "1"->0, "2"->1, "X"->2


def _reorder(proba: np.ndarray, classes) -> np.ndarray:
    """Reordena columnas de predict_proba al orden canónico CLASSES."""
    pos = {c: i for i, c in enumerate(classes)}
    return proba[:, [pos[i] for i in range(len(CLASSES))]]


class StackedResultClassifier:
    """Ensamble por stacking; expone predict_proba_1x2 como ResultClassifier."""

    def __init__(self, random_state: int = 42, n_folds: int = 5):
        self.random_state = random_state
        self.n_folds = n_folds
        self.base = _make_models(random_state)
        self.meta = LogisticRegression(max_iter=2000, C=1.0)

    def _fit_base(self, name, model, X, y, w) -> None:
        if name == "logistic_baseline":
            model.fit(X, y, clf__sample_weight=w)
        else:
            model.fit(X, y, sample_weight=w)

    def _base_proba(self, model, X) -> np.ndarray:
        return _reorder(model.predict_proba(X), model.classes_)

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray | None = None) -> "StackedResultClassifier":
        X = X[FEATURE_NAMES].reset_index(drop=True)
        y = pd.Series(y).astype(str).map(_LABEL).to_numpy()
        n = len(X)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight)

        # meta-features OUT-OF-FOLD
        oof = np.zeros((n, len(self.base) * len(CLASSES)))
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=0)
        for tr, va in skf.split(X, y):
            for j, (name, model) in enumerate(self.base.items()):
                m = clone(model)
                self._fit_base(name, m, X.iloc[tr], y[tr], w[tr])
                oof[va, j * 3:j * 3 + 3] = self._base_proba(m, X.iloc[va])
        self.meta.fit(oof, y, sample_weight=w)

        # reentrenar los base con todos los datos (para inferencia)
        for name, model in self.base.items():
            self._fit_base(name, model, X, y, w)
        return self

    def predict_proba_1x2(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURE_NAMES]
        feats = np.hstack([self._base_proba(m, X) for m in self.base.values()])
        proba = _reorder(self.meta.predict_proba(feats), self.meta.classes_)
        # proba en orden CLASSES = ["1","2","X"]; devolver [1, X, 2]
        out = pd.DataFrame(proba, columns=CLASSES)
        return out[["1", "X", "2"]]
