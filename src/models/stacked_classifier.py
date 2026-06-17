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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from src.data.wc_schema import FEATURE_NAMES
from src.models.result_classifier import CLASSES, _make_models

_LABEL = {c: i for i, c in enumerate(CLASSES)}   # "1"->0, "2"->1, "X"->2


def _reorder(proba: np.ndarray, classes) -> np.ndarray:
    """Reordena columnas de predict_proba al orden canónico CLASSES."""
    pos = {c: i for i, c in enumerate(classes)}
    return proba[:, [pos[i] for i in range(len(CLASSES))]]


class StackedResultClassifier:
    """Ensamble por stacking; expone predict_proba_1x2 como ResultClassifier.

    `cv`: "time" (TimeSeriesSplit, por defecto) genera las predicciones
    out-of-fold respetando el ORDEN TEMPORAL — cada fold de validación es
    posterior a su entrenamiento, sin leakage hacia el futuro. El dataset
    histórico ya viene ordenado por fecha, así que basta el orden de filas.
    "kfold" usa StratifiedKFold (aleatorio) como referencia.

    `calibrate`: aplica calibración isotónica por clase sobre las
    probabilidades del meta-modelo (ajustada sobre las predicciones OOF) y
    renormaliza — pule la calibración antes de usar las probabilidades.
    """

    def __init__(self, random_state: int = 42, n_folds: int = 5,
                 cv: str = "time", gap: int = 10, calibrate: bool = True):
        self.random_state = random_state
        self.n_folds = n_folds
        self.cv = cv
        self.gap = gap
        self.calibrate = calibrate
        self.base = _make_models(random_state)
        self.meta = LogisticRegression(max_iter=2000, C=1.0)
        self.calibrators: list[IsotonicRegression] | None = None

    def _splitter(self, X, y):
        if self.cv == "time":
            return TimeSeriesSplit(n_splits=self.n_folds, gap=self.gap).split(X)
        return StratifiedKFold(self.n_folds, shuffle=True,
                               random_state=0).split(X, y)

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

        # meta-features OUT-OF-FOLD (TimeSeriesSplit: respeta el tiempo)
        oof = np.zeros((n, len(self.base) * len(CLASSES)))
        seen = np.zeros(n, dtype=bool)
        for tr, va in self._splitter(X, y):
            seen[va] = True
            for j, (name, model) in enumerate(self.base.items()):
                m = clone(model)
                self._fit_base(name, m, X.iloc[tr], y[tr], w[tr])
                oof[va, j * 3:j * 3 + 3] = self._base_proba(m, X.iloc[va])
        # con TimeSeriesSplit las primeras filas nunca son validación -> excluir
        self.meta.fit(oof[seen], y[seen], sample_weight=w[seen])

        # calibración isotónica por clase sobre las predicciones OOF del meta
        if self.calibrate:
            meta_oof = _reorder(self.meta.predict_proba(oof[seen]),
                                self.meta.classes_)
            self.calibrators = []
            for c in range(len(CLASSES)):
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(meta_oof[:, c], (y[seen] == c).astype(float))
                self.calibrators.append(iso)

        # reentrenar los base con todos los datos (para inferencia)
        for name, model in self.base.items():
            self._fit_base(name, model, X, y, w)
        return self

    def predict_proba_1x2(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURE_NAMES]
        feats = np.hstack([self._base_proba(m, X) for m in self.base.values()])
        proba = _reorder(self.meta.predict_proba(feats), self.meta.classes_)
        if self.calibrators is not None:
            cal = np.column_stack([
                self.calibrators[c].transform(proba[:, c])
                for c in range(len(CLASSES))])
            proba = cal / cal.sum(axis=1, keepdims=True)
        # proba en orden CLASSES = ["1","2","X"]; devolver [1, X, 2]
        out = pd.DataFrame(proba, columns=CLASSES)
        return out[["1", "X", "2"]]
