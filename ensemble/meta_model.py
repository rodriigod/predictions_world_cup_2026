"""Meta-modelo de stacking: logística multinomial regularizada (L2).

Apila las probabilidades de core/ y microsim/ + las señales del LLM y aprende
el peso de cada input para predecir el resultado 1X2. Deliberadamente SIMPLE
(no gradient boosting): con pocos partidos de Mundial, un modelo lineal
regularizado es más robusto y, además, sus coeficientes son interpretables —
justo lo que se necesita para diagnosticar qué input aporta y cuál mete ruido.

- Validación temporal: la fuerza de regularización C se elige por
  TimeSeriesSplit (mismo esquema anti-leakage que core/), minimizando el RPS.
- Features estandarizadas: los coeficientes quedan en una escala comparable
  (efecto por desviación estándar), para leer "cuánto pesa cada uno".
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.utils.metrics import ModelMetrics
from ensemble.features import GROUP_OF, META_COLUMNS

CLASSES = ["1", "X", "2"]
IDX = {c: i for i, c in enumerate(CLASSES)}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def _proba_1x2(model: Pipeline, X) -> np.ndarray:
    """predict_proba reordenado a columnas [1, X, 2]."""
    p = model.predict_proba(X)
    pos = {c: i for i, c in enumerate(model.named_steps["clf"].classes_)}
    return p[:, [pos["1"], pos["X"], pos["2"]]]


def _make_pipeline(C: float) -> Pipeline:
    # solver lbfgs => regularización L2 por defecto (no se pasa `penalty=` para
    # evitar el DeprecationWarning de sklearn>=1.8; el default sigue siendo L2).
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=2000)),
    ])


class StackingMetaModel:
    """Logística multinomial L2 sobre META_COLUMNS."""

    def __init__(self, C: Optional[float] = None,
                 feature_names: Optional[list[str]] = None):
        self.C = C                       # None -> se elige por CV temporal
        self.feature_names = feature_names or META_COLUMNS
        self.pipeline: Optional[Pipeline] = None
        self.C_: Optional[float] = None  # C finalmente usado
        self.cv_rps_: Optional[float] = None

    # ------------------------------- ajuste ----------------------------------
    def _select_C(self, X: pd.DataFrame, y: pd.Series, n_splits: int) -> float:
        """Elige C por TimeSeriesSplit minimizando RPS (X debe venir ordenado
        por fecha). Si hay muy pocos datos, cae a C=1.0."""
        if len(X) < (n_splits + 1) * 5:
            return 1.0
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_C, best_rps = 1.0, np.inf
        for C in C_GRID:
            rps_folds = []
            for tr, va in tscv.split(X):
                if len(np.unique(y.iloc[tr])) < 3:
                    continue
                pipe = _make_pipeline(C).fit(X.iloc[tr], y.iloc[tr])
                proba = _proba_1x2(pipe, X.iloc[va])
                idx = y.iloc[va].map(IDX).to_numpy()
                rps_folds.append(ModelMetrics.rps(idx, proba))
            if rps_folds:
                mean_rps = float(np.mean(rps_folds))
                if mean_rps < best_rps:
                    best_rps, best_C = mean_rps, C
        self.cv_rps_ = best_rps if np.isfinite(best_rps) else None
        return best_C

    def fit(self, X: pd.DataFrame, y: pd.Series, *, n_splits: int = 5
            ) -> "StackingMetaModel":
        """Entrena. Si C es None, lo selecciona por TimeSeriesSplit (RPS).
        X debe venir ORDENADO POR FECHA para que el split temporal sea válido."""
        X = X[self.feature_names]
        self.C_ = self.C if self.C is not None else self._select_C(X, y, n_splits)
        self.pipeline = _make_pipeline(self.C_).fit(X, y)
        return self

    # ------------------------------ predicción -------------------------------
    def predict_proba_1x2(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("meta-modelo no entrenado")
        return _proba_1x2(self.pipeline, X[self.feature_names])

    # ------------------------------ coeficientes -----------------------------
    def coefficients(self) -> pd.DataFrame:
        """Coeficientes por clase (escala estandarizada, comparables) + grupo
        (core/microsim/llm) + |peso| promedio. Diagnostica qué input aporta."""
        if self.pipeline is None:
            raise RuntimeError("meta-modelo no entrenado")
        clf = self.pipeline.named_steps["clf"]
        coef = clf.coef_                          # [n_clases, n_features]
        classes = list(clf.classes_)
        df = pd.DataFrame({"feature": self.feature_names})
        df["group"] = df["feature"].map(GROUP_OF)
        for ci, cls in enumerate(classes):
            df[f"coef_{cls}"] = coef[ci]
        coef_cols = [f"coef_{c}" for c in classes]
        df["abs_mean"] = df[coef_cols].abs().mean(axis=1)
        return df.sort_values("abs_mean", ascending=False).reset_index(drop=True)

    # ------------------------------ persistencia -----------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "StackingMetaModel":
        with open(path, "rb") as f:
            return pickle.load(f)
