"""Temperature scaling — calibración de probabilidades 1X2.

Aprende UN solo parámetro T: divide los log-odds del modelo por T y renormaliza
(softmax). T<1 "afila" (más confianza), T>1 "suaviza". Es el método estándar
para calibrar sin deformar el ranking de clases (no cambia el argmax). Se ajusta
minimizando la NLL (log-loss) sobre predicciones OUT-OF-FOLD (leak-free).

El diagnóstico (scripts/ensemble_calibration.py) mostró que core está
SUB-confiado en favoritos -> se espera T<1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

EPS = 1e-12


def _apply_T(proba: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(np.asarray(proba, float), EPS, 1.0)) / T
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


class TemperatureScaler:
    """Calibrador de 1 parámetro. `fit` sobre probs OOF + clase real (índice)."""

    def __init__(self, T: float = 1.0):
        self.T_ = float(T)

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> "TemperatureScaler":
        proba = np.asarray(proba, float)
        y = np.asarray(y_idx)
        logits = np.log(np.clip(proba, EPS, 1.0))

        def nll(T: float) -> float:
            z = logits / T
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            return float(-np.mean(np.log(np.clip(
                p[np.arange(len(y)), y], EPS, 1.0))))

        res = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
        self.T_ = float(res.x)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return _apply_T(proba, self.T_)

    # --------------------------- persistencia --------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"T": self.T_}), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TemperatureScaler":
        T = json.loads(Path(path).read_text(encoding="utf-8"))["T"]
        return TemperatureScaler(T)
