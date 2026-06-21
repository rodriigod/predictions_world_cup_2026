"""Capa de adaptación: salida del pipeline de `core/` -> `MatchPrediction`.

Esto NO es modelado. Es una traducción pura de la tabla `matches` que ya
produce `GroupStageSimulator.run()` (Poisson GLM + Dixon-Coles + Monte Carlo)
al contrato común `schema.MatchPrediction`, para que el `ensemble/` pueda
combinar este modelo con otros sin conocer su implementación.

No se toca ningún hiperparámetro ni la lógica: se leen las columnas que el
pipeline ya calcula y, como mucho, se reconstruye la matriz Dixon-Coles a
partir de las lambdas usando la MISMA función del modelo (`_dixon_coles_matrix`),
de modo que la `score_matrix` es exactamente la que el modelo ya usa por dentro.
"""

from __future__ import annotations

import pandas as pd

from core.simulation.monte_carlo import DC_RHO, _dixon_coles_matrix
from schema import MatchPrediction

MODEL_NAME = "poisson_dc_montecarlo"
MODEL_VERSION = "1.0.0"


def matches_to_predictions(
    matches: pd.DataFrame,
    *,
    prob_source: str = "mc",
    rho: float = DC_RHO,
    include_score_matrix: bool = True,
    model_name: str = MODEL_NAME,
    model_version: str = MODEL_VERSION,
) -> list[MatchPrediction]:
    """Convierte la tabla `matches` de `GroupStageSimulator.run()` en una lista
    de `MatchPrediction`.

    prob_source:
      - "mc" (default): usa el 1X2 muestreado del Monte Carlo
        (`p_win_a`/`p_draw`/`p_win_b`), que es el titular de producción:
        incorpora jitter de lambdas e incentivos de la fecha 3.
      - "dc": usa el 1X2 ANALÍTICO de la matriz Dixon-Coles base
        (`p_win_a_dc`/...), el 1X2 "puro" pre-incentivos.

    La `score_matrix` (si se pide) se reconstruye con `_dixon_coles_matrix` a
    partir de `lambda_a`/`lambda_b`, idéntica a la que el modelo usa internamente.
    `confidence` queda en None: el pipeline GLM+MC no estima incertidumbre
    epistémica nativa (queda disponible para modelos que sí la produzcan).
    """
    if prob_source == "mc":
        ph, pd_, pa = "p_win_a", "p_draw", "p_win_b"
    elif prob_source == "dc":
        ph, pd_, pa = "p_win_a_dc", "p_draw_dc", "p_win_b_dc"
    else:
        raise ValueError(f"prob_source debe ser 'mc' o 'dc', no {prob_source!r}")

    preds: list[MatchPrediction] = []
    for row in matches.itertuples(index=False):
        d = row._asdict()
        lam_h = float(d["lambda_a"])
        lam_a = float(d["lambda_b"])
        score_matrix = (
            _dixon_coles_matrix(lam_h, lam_a, rho) if include_score_matrix else None
        )
        preds.append(
            MatchPrediction(
                team_home=str(d["team_a"]),
                team_away=str(d["team_b"]),
                match_date=str(d["date"]),
                prob_home=float(d[ph]),
                prob_draw=float(d[pd_]),
                prob_away=float(d[pa]),
                lambda_home=lam_h,
                lambda_away=lam_a,
                model_name=model_name,
                model_version=model_version,
                score_matrix=score_matrix,
            )
        )
    return preds
