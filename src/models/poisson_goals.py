"""Modelo de goles esperados (lambda) por equipo-partido.

Dos backends intercambiables:
- "poisson":  GLM PoissonRegressor (log-link) — interpretable, robusto
              con pocos datos, el baseline recomendado.
- "gbm":      HistGradientBoostingRegressor(loss="poisson") — equivalente
              funcional a XGBoost con objective='count:poisson', captura
              no-linealidades e interacciones. (Se usa sklearn porque
              xgboost requiere libomp en macOS; si está disponible,
              "xgb" lo usa con el mismo contrato.)

Ambos predicen log(lambda) implícitamente y exponen predict_lambda().
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.wc_schema import FEATURE_NAMES

LAMBDA_FLOOR, LAMBDA_CEIL = 0.15, 4.5  # límites sanos para selecciones


class PoissonGoalsModel:
    def __init__(self, backend: str = "poisson", random_state: int = 42):
        self.backend = backend
        if backend == "poisson":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("glm", PoissonRegressor(alpha=1e-3, max_iter=1000)),
            ])
        elif backend == "gbm":
            self.model = HistGradientBoostingRegressor(
                loss="poisson", max_iter=400, learning_rate=0.06,
                max_depth=4, l2_regularization=1.0,
                random_state=random_state)
        elif backend == "xgb":
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                objective="count:poisson", n_estimators=400,
                learning_rate=0.06, max_depth=4, reg_lambda=1.0,
                random_state=random_state)
        else:
            raise ValueError(f"backend desconocido: {backend}")

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray | None = None) -> dict:
        """Entrena y devuelve métricas de holdout (deviance de Poisson).

        `sample_weight` implementa el decaimiento temporal de Dixon-Coles
        (1997): partidos recientes/importantes pesan más en el ajuste.
        """
        X = X[FEATURE_NAMES]
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
            X, y, sample_weight, test_size=0.2, random_state=0)
        self._fit_weighted(X_tr, y_tr, w_tr)
        pred = np.clip(self.model.predict(X_te), LAMBDA_FLOOR, LAMBDA_CEIL)
        naive = np.full_like(pred, y_tr.mean())
        metrics = {
            "backend": self.backend,
            "holdout_poisson_deviance": mean_poisson_deviance(
                y_te, pred, sample_weight=w_te),
            "naive_poisson_deviance": mean_poisson_deviance(
                y_te, naive, sample_weight=w_te),
            "mean_lambda_pred": float(pred.mean()),
            "mean_goals_actual": float(y_te.mean()),
        }
        # reentrenar con todo el dataset para producción
        self._fit_weighted(X, y, sample_weight)
        return metrics

    def _fit_weighted(self, X, y, w) -> None:
        if self.backend == "poisson":
            self.model.fit(X, y, glm__sample_weight=w)
        else:
            self.model.fit(X, y, sample_weight=w)

    def predict_lambda(self, X: pd.DataFrame) -> np.ndarray:
        lam = self.model.predict(X[FEATURE_NAMES])
        return np.clip(lam, LAMBDA_FLOOR, LAMBDA_CEIL)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PoissonGoalsModel":
        return joblib.load(path)
