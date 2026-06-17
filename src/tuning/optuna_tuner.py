"""Optimización de hiperparámetros con Optuna + TimeSeriesSplit (leak-free).

Principio anti-leakage (igual que el stacking):
- La búsqueda usa TimeSeriesSplit con gap: cada fold de validación es
  POSTERIOR a su entrenamiento, nunca al revés.
- Se tunea sobre partidos GENERALES (no-Mundial); el test de los 7 Mundiales
  del backtest queda intacto como evaluación final independiente.

Modelos: XGBoost y logística (1X2), y el half-life del decaimiento del
Poisson. CatBoost queda fuera a propósito (dependencia pesada); LightGBM se
usa solo si está instalado. La métrica objetivo es el RPS (menor = mejor).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# etiquetas en orden 1/X/2 -> 0/1/2 (predict_proba de sklearn queda alineado)
LABEL = {"1": 0, "X": 1, "2": 2}


def rps(y_idx: np.ndarray, proba: np.ndarray) -> float:
    proba = np.asarray(proba, float)
    cum_p = np.cumsum(proba, axis=1)
    cum_t = np.cumsum(np.eye(3)[np.asarray(y_idx)], axis=1)
    return float(np.mean(np.sum((cum_p - cum_t) ** 2, axis=1) / 2))


def _cv_rps(make_model, X, y, w, n_splits=5, gap=15) -> float:
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    for tr, va in tscv.split(X):
        m = make_model()
        if isinstance(m, Pipeline):
            m.fit(X.iloc[tr], y[tr], clf__sample_weight=w[tr])
        else:
            m.fit(X.iloc[tr], y[tr], sample_weight=w[tr])
        scores.append(rps(y[va], m.predict_proba(X.iloc[va])))
    return float(np.mean(scores))


def _xgb_factory(params: dict):
    from xgboost import XGBClassifier
    base = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                tree_method="hist", random_state=42, n_jobs=-1)
    return lambda: XGBClassifier(**{**base, **params})


def _logistic_factory(params: dict):
    return lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, **params))])


def tune_classifier(kind: str, X, y_idx, w, n_trials=40,
                    n_splits=5, gap=15) -> optuna.Study:
    """Tunea 'xgb' o 'logistic' por RPS con TimeSeriesSplit."""
    def objective(trial: optuna.Trial) -> float:
        if kind == "xgb":
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            )
            make = _xgb_factory(params)
        elif kind == "logistic":
            params = dict(C=trial.suggest_float("C", 1e-3, 100.0, log=True))
            make = _logistic_factory(params)
        else:
            raise ValueError(kind)
        return _cv_rps(make, X, y_idx, w, n_splits, gap)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def default_cv_rps(kind: str, X, y_idx, w, n_splits=5, gap=15) -> float:
    """RPS con la MISMA CV de los defaults actuales de producción."""
    if kind == "xgb":
        make = _xgb_factory(dict(n_estimators=400, max_depth=4,
                                 learning_rate=0.05, reg_lambda=1.0))
    else:
        make = _logistic_factory(dict(C=1.0))
    return _cv_rps(make, X, y_idx, w, n_splits, gap)
