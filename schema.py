"""Contrato de salida común para TODOS los modelos de predicción del repo.

Cada modelo de predicción (el pipeline estadístico de `core/`, la microsim
de plantillas de `microsim/`, las features de LLM de `llm_features/`, etc.)
debe devolver sus predicciones como `MatchPrediction`. Así el `ensemble/`
puede combinar modelos heterogéneos sin conocer sus tripas: todos hablan el
mismo idioma de salida.

Diseño:
- dataclass (igual que el resto del repo: `core/data/wc_schema.py`,
  `core/simulation/monte_carlo.py`), sin dependencias nuevas.
- Lo MÍNIMO obligatorio son las probabilidades 1X2 y las lambdas; todo lo
  demás (matriz de marcadores, confianza) es opcional para no forzar a un
  modelo a producir algo que no calcula.
- `__post_init__` valida que las probabilidades sean un simplex (suman 1,
  no-negativas) para que un modelo mal-portado falle temprano y no
  contamine el ensemble.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

PROB_SUM_TOL = 1e-6


@dataclass
class MatchPrediction:
    """Predicción de UN partido, en el idioma común del repo.

    `home`/`away` son el equipo nominal local/visitante del fixture
    (`team_a`/`team_b`). En sedes neutrales del Mundial no hay localía real,
    pero se conserva la designación del fixture para poder casar partidos
    entre modelos sin ambigüedad.
    """

    # --- Identidad del partido (obligatorio) ---
    team_home: str
    team_away: str
    match_date: str                      # ISO 'YYYY-MM-DD' (col `date` del fixture)

    # --- Resultado 1X2 (obligatorio) ---
    prob_home: float
    prob_draw: float
    prob_away: float

    # --- Goles esperados (obligatorio) ---
    lambda_home: float
    lambda_away: float

    # --- Procedencia del modelo (obligatorio) ---
    model_name: str
    model_version: str

    # --- Opcionales: solo si el modelo los produce ---
    score_matrix: Optional[np.ndarray] = None   # P(goles_home, goles_away)
    confidence: Optional[float] = None           # incertidumbre del modelo, si la estima

    def __post_init__(self) -> None:
        probs = np.array([self.prob_home, self.prob_draw, self.prob_away], dtype=float)
        if np.any(probs < -PROB_SUM_TOL):
            raise ValueError(
                f"{self.model_name}: probabilidades 1X2 negativas {tuple(probs)} "
                f"({self.team_home} vs {self.team_away})")
        s = float(probs.sum())
        if abs(s - 1.0) > 1e-3:
            raise ValueError(
                f"{self.model_name}: las probabilidades 1X2 suman {s:.6f}, no 1.0 "
                f"({self.team_home} vs {self.team_away})")
        if self.score_matrix is not None:
            self.score_matrix = np.asarray(self.score_matrix, dtype=float)

    # --- Utilidades de interoperación ---
    @property
    def probs(self) -> np.ndarray:
        """Vector 1X2 como np.array, conveniente para el ensemble."""
        return np.array([self.prob_home, self.prob_draw, self.prob_away], dtype=float)

    @property
    def match_key(self) -> tuple[str, str, str]:
        """Clave para casar el MISMO partido entre modelos distintos."""
        return (self.match_date, self.team_home, self.team_away)

    def to_dict(self) -> dict:
        d = asdict(self)
        # la matriz no es serializable a un valor escalar de DataFrame; se
        # resume con su forma y se omite el array completo en la vista tabular.
        d["score_matrix"] = None if self.score_matrix is None else self.score_matrix.shape
        return d


def predictions_to_dataframe(preds: list["MatchPrediction"]) -> pd.DataFrame:
    """Vista tabular de una lista de predicciones (omite la score_matrix)."""
    return pd.DataFrame([p.to_dict() for p in preds])
