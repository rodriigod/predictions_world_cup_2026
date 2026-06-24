"""Calibración de probabilidades 1X2: isotónica y Platt (multiclase).

Complementa a `ensemble.calibrate.TemperatureScaler` (1 parámetro) con dos
métodos más expresivos, evaluados de forma HONESTA por cross-validation para no
sobre-ajustar la calibración misma:

  - IsotonicCalibrator : una IsotonicRegression por clase (one-vs-rest) sobre la
    probabilidad predicha de esa clase, luego renormaliza a suma 1. No paramétrica
    y monótona: puede corregir sub/sobre-confianza no lineal, pero con pocos datos
    es la que más riesgo de overfit tiene -> SIEMPRE se valida con CV.

  - PlattCalibrator : Platt scaling generalizado a multiclase. Ajusta una
    logística multinomial sobre el LOGIT de las probabilidades (features =
    log p_1, log p_X, log p_2). Equivale a "vector scaling": más flexible que
    temperature (que es el caso de 1 parámetro escalar) pero igualmente suave.

Ambos exponen .fit(proba, y_idx) / .transform(proba) y son serializables (JSON).
`cv_calibrated_oof` produce predicciones calibradas OUT-OF-FOLD (cada fila la
predice un calibrador que NO la vio) — así la comparación RPS/Brier es leak-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

EPS = 1e-12
N_CLASSES = 3


def _renorm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, None)
    return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
class IsotonicCalibrator:
    """Isotónica por clase (one-vs-rest) + renormalización."""

    def __init__(self):
        self.models_: list[IsotonicRegression] = []

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> "IsotonicCalibrator":
        proba = np.asarray(proba, float)
        y = np.asarray(y_idx)
        self.models_ = []
        for c in range(proba.shape[1]):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(proba[:, c], (y == c).astype(float))
            self.models_.append(iso)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, float)
        out = np.column_stack([m.predict(proba[:, c])
                               for c, m in enumerate(self.models_)])
        return _renorm(out)


# --------------------------------------------------------------------------- #
class PlattCalibrator:
    """Platt multiclase: logística multinomial sobre el logit de las probs."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.clf_: LogisticRegression | None = None
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _features(proba: np.ndarray) -> np.ndarray:
        return np.log(np.clip(np.asarray(proba, float), EPS, 1.0))

    def fit(self, proba: np.ndarray, y_idx: np.ndarray) -> "PlattCalibrator":
        X = self._features(proba)
        y = np.asarray(y_idx)
        self.clf_ = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000)
        self.clf_.fit(X, y)
        self.classes_ = self.clf_.classes_
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        if self.clf_ is None:
            raise RuntimeError("PlattCalibrator no ajustado")
        p = self.clf_.predict_proba(self._features(proba))
        # reordena a [0,1,2] por si faltara alguna clase en el train
        out = np.zeros((len(p), N_CLASSES))
        for j, c in enumerate(self.classes_):
            out[:, int(c)] = p[:, j]
        return _renorm(out)


# --------------------------------------------------------------------------- #
_CALIBRATORS = {"isotonic": IsotonicCalibrator, "platt": PlattCalibrator}


def make_calibrator(method: str):
    return _CALIBRATORS[method]()


def cv_calibrated_oof(proba: np.ndarray, y_idx: np.ndarray, method: str, *,
                      n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Predicciones calibradas OUT-OF-FOLD (leak-free): para cada fold, ajusta el
    calibrador en el resto y predice el fold. Así se mide la calibración SIN que
    el calibrador haya visto la fila que evalúa (evita el auto-engaño de A.1)."""
    proba = np.asarray(proba, float)
    y = np.asarray(y_idx)
    out = np.full_like(proba, np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(proba, y):
        cal = make_calibrator(method).fit(proba[tr], y[tr])
        out[va] = cal.transform(proba[va])
    return out


# ------------------------------- persistencia ------------------------------- #
def save_calibrator(cal, path: str | Path) -> None:
    """Serializa un calibrador entrenado a JSON (portable, sin pickle)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(cal, IsotonicCalibrator):
        payload = {"method": "isotonic", "models": [
            {"x": m.f_.x.tolist(), "y": m.f_.y.tolist()} for m in cal.models_]}
    elif isinstance(cal, PlattCalibrator):
        payload = {"method": "platt", "C": cal.C,
                   "classes": np.asarray(cal.classes_).astype(int).tolist(),
                   "coef": cal.clf_.coef_.tolist(),
                   "intercept": cal.clf_.intercept_.tolist()}
    else:
        raise TypeError(f"calibrador no serializable: {type(cal)}")
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_calibrator(path: str | Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["method"] == "isotonic":
        cal = IsotonicCalibrator()
        for m in d["models"]:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(m["x"], m["y"])          # re-ajuste exacto sobre los nudos
            cal.models_.append(iso)
        return cal
    if d["method"] == "platt":
        cal = PlattCalibrator(C=d.get("C", 1.0))
        clf = LogisticRegression()
        clf.classes_ = np.array(d["classes"])
        clf.coef_ = np.array(d["coef"])
        clf.intercept_ = np.array(d["intercept"])
        cal.clf_ = clf
        cal.classes_ = clf.classes_
        return cal
    raise ValueError(f"método desconocido: {d['method']}")
