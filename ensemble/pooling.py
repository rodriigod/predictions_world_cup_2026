"""Combinación de pronósticos SIN entrenamiento: log-pooling y extremizing.

Alternativa al stacking (regresión logística entrenada). Dos métodos clásicos de
agregación de pronósticos probabilísticos:

  - log_pool : media GEOMÉTRICA normalizada de las probabilidades de core y
    microsim (a.k.a. "logarithmic opinion pool"). A diferencia del promedio
    aritmético, es la regla de agregación "externamente bayesiana" y trata 0s con
    dureza. Sin parámetros que entrenar (peso 0.5/0.5 por defecto).

  - extremize : tras agrupar, empuja las probabilidades hacia los extremos
    (Satopää et al. 2014). La media de pronósticos tiende a ser sub-confiada
    (regresa a 1/k); extremizar lo corrige elevando a una potencia a>1 y
    renormalizando:  p_i^a / Σ_j p_j^a. a=1 no hace nada; a>1 afila. El parámetro
    a se ajusta por CROSS-VALIDATION (no a ojo).

Notar la simetría con la calibración: extremizar con potencia a equivale a
temperature scaling con T=1/a. La diferencia conceptual es el ORDEN: aquí se
agrega primero (log-pool) y luego se extremiza el consenso.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

EPS = 1e-12


def _renorm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def log_pool(core: np.ndarray, micro: np.ndarray, w: float = 0.5) -> np.ndarray:
    """Media geométrica normalizada (log-pool) de dos juegos de probs (n,3).
    `w` = peso de core (1-w para microsim). w=0.5 = pesos iguales."""
    core = np.clip(np.asarray(core, float), EPS, 1.0)
    micro = np.clip(np.asarray(micro, float), EPS, 1.0)
    g = np.exp(w * np.log(core) + (1 - w) * np.log(micro))
    return _renorm(g)


def extremize(proba: np.ndarray, a: float) -> np.ndarray:
    """Extremizing de Satopää: p_i^a / Σ p_j^a. a>1 empuja a los extremos."""
    p = np.clip(np.asarray(proba, float), EPS, 1.0) ** a
    return _renorm(p)


def fit_extremize_a(proba: np.ndarray, y_idx: np.ndarray, *, n_splits: int = 5,
                    seed: int = 0, grid: tuple[float, ...] = (
                        0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
                    ) -> float:
    """Elige `a` por CV (StratifiedKFold) minimizando el RPS promedio held-out.
    Incluye a<1 (suavizar) por si el consenso resultara SOBRE-confiado: honesto,
    no se asume que extremizar siempre ayuda."""
    from core.utils.metrics import ModelMetrics
    proba = np.asarray(proba, float)
    y = np.asarray(y_idx)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_a, best_rps = 1.0, np.inf
    for a in grid:
        rps_folds = []
        for _, va in skf.split(proba, y):
            rps_folds.append(ModelMetrics.rps(y[va], extremize(proba[va], a)))
        mean_rps = float(np.mean(rps_folds))
        if mean_rps < best_rps:
            best_rps, best_a = mean_rps, a
    return best_a


def cv_extremized_oof(proba: np.ndarray, y_idx: np.ndarray, *, n_splits: int = 5,
                      seed: int = 0) -> tuple[np.ndarray, float]:
    """Extremizing OUT-OF-FOLD: `a` se ajusta en el train de cada fold y se aplica
    al fold de validación. Devuelve (probs_oof, a_global_para_referencia)."""
    proba = np.asarray(proba, float)
    y = np.asarray(y_idx)
    out = np.full_like(proba, np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(proba, y):
        a = fit_extremize_a(proba[tr], y[tr], n_splits=4, seed=seed)
        out[va] = extremize(proba[va], a)
    a_global = fit_extremize_a(proba, y, n_splits=n_splits, seed=seed)
    return out, a_global
