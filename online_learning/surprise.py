"""D. Detección automática de sorpresas.

Para cada partido 2026 calcula el "surprise score": cuánto MENOS que el azar
(1/3) le daba core/ al resultado que realmente ocurrió.

    prob_core_pre = P_core(resultado real)          (1X2 pre-torneo de core/)
    surprise_score = 1/3 − prob_core_pre

    >0  = sorpresa (core le daba menos que el azar; ej. P=0.05 -> +0.283 grande)
    <0  = esperado (core le daba más que el azar)

Se registra en data/surprise_log.csv y se usa para (a) listar los partidos donde
el modelo más falló y (b) detectar SESGO SISTEMÁTICO: si core sobreestima a los
favoritos en el formato de 48 equipos, es cuantificable y corregible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from online_learning.dataset import load_results
from online_learning.priors import canon, to_es

LOG = Path(__file__).resolve().parent / "data" / "surprise_log.csv"
LOG_COLS = ["date", "equipos", "home", "away", "resultado_real",
            "prob_core_pre", "surprise_score"]


def _core_probs(home_en: str, away_en: str) -> tuple[float, float, float]:
    """1X2 pre-torneo de core/ (lectura, no modifica producción)."""
    from ensemble.predict import _context
    return _context().core_probs(home_en, away_en)


def build_surprise_log(*, save: bool = True) -> pd.DataFrame:
    """Calcula el surprise log de todos los partidos 2026 jugados."""
    rows = []
    for r in load_results().itertuples():
        a, b = r.home_team, r.away_team
        try:
            p1, pX, p2 = _core_probs(a, b)
        except Exception:
            continue
        gh, ag = int(r.home_goals), int(r.away_goals)
        res = "1" if gh > ag else ("2" if gh < ag else "X")
        p_real = {"1": p1, "X": pX, "2": p2}[res]
        rows.append({
            "date": pd.Timestamp(r.date).date().isoformat(),
            "equipos": f"{to_es(a)} - {to_es(b)}", "home": a, "away": b,
            "resultado_real": f"{gh}-{ag} ({res})",
            "prob_core_pre": round(float(p_real), 4),
            "surprise_score": round(1 / 3 - float(p_real), 4)})
    df = pd.DataFrame(rows, columns=LOG_COLS)
    df = df.sort_values("surprise_score", ascending=False).reset_index(drop=True)
    if save:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(LOG, index=False)
    return df


def detect_favorite_bias(threshold: float = 0.5) -> dict:
    """¿core sobreestima a los favoritos? Sobre los partidos con un favorito claro
    (P_max ≥ threshold), compara la probabilidad media que core daba al favorito
    contra la frecuencia REAL con que el favorito ganó. Gap>0 = sobreestimación.
    Con pocos partidos es solo indicativo (se reporta n y el IC binomial)."""
    fav_pred, fav_won = [], []
    for r in load_results().itertuples():
        a, b = r.home_team, r.away_team
        try:
            probs = _core_probs(a, b)
        except Exception:
            continue
        k = int(np.argmax(probs))
        if probs[k] < threshold:
            continue
        gh, ag = int(r.home_goals), int(r.away_goals)
        res = 0 if gh > ag else (2 if gh < ag else 1)
        fav_pred.append(float(probs[k]))
        fav_won.append(1.0 if res == k else 0.0)
    n = len(fav_pred)
    if n == 0:
        return {"n": 0, "pred_mean": None, "won_rate": None, "gap": None}
    pred_mean = float(np.mean(fav_pred))
    won_rate = float(np.mean(fav_won))
    se = float(np.sqrt(won_rate * (1 - won_rate) / n)) if n else None
    return {"n": n, "pred_mean": pred_mean, "won_rate": won_rate,
            "gap": pred_mean - won_rate, "se": se, "threshold": threshold}
